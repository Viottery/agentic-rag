from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from app.rag.embeddings import embed_texts
from app.rag.qdrant_store import QdrantStore
from app.rag.schemas import DocumentChunk


# 当前阶段先采用基于字符窗口的轻量切块配置。
# 这让 indexing 保持简单稳定，后续如果要切到 token-based splitter、
# markdown/HTML-aware splitter 或语义切块，可以集中替换这里的默认策略。
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_BATCH_SIZE = 64
INDEX_MANIFEST_FILE_NAME = "_index_manifest.json"


def _emit_progress(enabled: bool, message: str) -> None:
    if not enabled:
        return
    print(message, file=sys.stderr, flush=True)


def _build_progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[unknown]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _emit_progress_bar(enabled: bool, prefix: str, current: int, total: int) -> None:
    if not enabled:
        return
    bar = _build_progress_bar(current, total)
    print(f"\r{prefix} {bar} {current}/{total}", end="", file=sys.stderr, flush=True)
    if current >= total:
        print(file=sys.stderr, flush=True)


def _path_to_posix(path: Path) -> str:
    return path.as_posix()


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index_manifest(root_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(root_dir) / INDEX_MANIFEST_FILE_NAME
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_index_manifest(root_dir: str | Path, payload: dict[str, Any]) -> str:
    manifest_path = Path(root_dir) / INDEX_MANIFEST_FILE_NAME
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(manifest_path)


def load_source_manifest(root_dir: str | Path) -> dict[str, dict[str, Any]]:
    source_manifest_path = Path(root_dir) / "_manifest.json"
    if not source_manifest_path.exists():
        return {}

    payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for item in payload.get("pages", []):
        relative_path = str(item.get("relative_path", "")).strip()
        if relative_path:
            result[relative_path] = item
    return result


def build_file_hierarchy_metadata(
    file_path: str | Path,
    *,
    root_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    根据本地文件路径构建层次化 metadata。

    目标：
    - 让文件系统中的目录树可以稳定映射到向量库 payload
    - 为后续 coarse-to-fine routing 提供直接可消费的结构字段
    """
    path_obj = Path(file_path)
    if root_dir is not None:
        root_path = Path(root_dir)
        relative_path = path_obj.relative_to(root_path)
        root_name = root_path.name
    else:
        relative_path = Path(path_obj.name)
        root_name = path_obj.parent.name

    path_segments = list(relative_path.parent.parts) if relative_path.parent != Path(".") else []
    hierarchy_path = "/".join(path_segments)
    hierarchy_prefixes = [
        "/".join(path_segments[:index])
        for index in range(1, len(path_segments) + 1)
    ]

    return {
        "file_path": str(path_obj),
        "source_root": root_name,
        "relative_path": _path_to_posix(relative_path),
        "file_name": path_obj.name,
        "file_stem": path_obj.stem,
        "knowledge_path": path_segments,
        "knowledge_path_text": " / ".join(path_segments),
        "path_segments": path_segments,
        "hierarchy_path": hierarchy_path,
        "hierarchy_level": len(path_segments),
        "hierarchy_prefixes": hierarchy_prefixes,
        "root_group": path_segments[0] if path_segments else "",
        "leaf_group": path_segments[-1] if path_segments else "",
        "top_level_group": path_segments[0] if path_segments else "",
        "parent_path": "/".join(path_segments[:-1]) if len(path_segments) > 1 else "",
    }


def build_document_id_from_path(
    file_path: str | Path,
    *,
    root_dir: str | Path | None = None,
) -> str:
    """
    使用相对路径生成更稳定的 document_id。

    相比只用 file stem：
    - 可避免不同目录下同名文件冲突
    - 更适合层次化知识库重建
    """
    path_obj = Path(file_path)
    if root_dir is not None:
        relative_path = path_obj.relative_to(Path(root_dir))
    else:
        relative_path = Path(path_obj.name)

    normalized = relative_path.with_suffix("").as_posix()
    return normalized.replace("/", "::")


def normalize_text(text: str) -> str:
    """
    统一文本格式，减少无意义空白对切块的影响。

    这是一个偏保守的清洗步骤：
    - 保留段落边界
    - 不尝试重写标点、标题层级、列表结构

    当前实现故意保持朴素，避免过早引入针对特定文档格式的规则。
    如果后续接入 markdown / html / pdf 解析，建议在进入本函数之前完成更强的结构化预处理。
    """
    lines = [line.strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs).strip()


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    按字符窗口切块，并尽量在段落边界截断。

    说明：
    - 当前使用字符级切块，优点是依赖少、行为稳定、便于调试
    - 这是一个有意保留的 naive 实现，不处理 token 长度、标题继承、表格等复杂结构

    后续可替换方向：
    - token-based chunking
    - 按 markdown 标题/代码块切块
    - query-aware 或语义切块
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    chunks: list[dict[str, Any]] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            # 当前阶段优先在段落分隔符处截断，避免简单字符截断带来的语义撕裂。
            # 这里仍然是启发式规则，不保证找到最佳边界。
            split_at = normalized.rfind("\n\n", start, end)
            if split_at > start:
                end = split_at

        content = normalized[start:end].strip()
        if content:
            chunks.append(
                {
                    "content": content,
                    "start": start,
                    "end": end,
                }
            )

        if end >= text_length:
            break

        next_start = max(end - chunk_overlap, 0)

        # 避免在短段落 + overlap 场景下反复命中同一分隔点，导致切块游标不再前进。
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def build_document_chunks(
    *,
    document_id: str,
    source_name: str,
    title: str,
    text: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """
    将单篇文档转换为可入库的 chunk 列表。
    """
    if not document_id.strip():
        raise ValueError("document_id 不能为空")

    base_metadata = metadata or {}
    split_chunks = split_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    total_chunks = len(split_chunks)
    if not total_chunks:
        return []

    chunks: list[DocumentChunk] = []
    padding = max(3, len(str(total_chunks)))

    for index, item in enumerate(split_chunks):
        # chunk_id 采用稳定可预测的命名，方便：
        # - 重建索引时幂等覆盖
        # - 调试时快速定位原始文档与切块序号
        # 后续如果需要跨数据源全局唯一，也可以切换到 UUID/内容哈希策略。
        chunk_id = f"{document_id}::chunk::{index:0{padding}d}"
        chunk_metadata = {
            **base_metadata,
            "chunk_index": index,
            "chunk_start": item["start"],
            "chunk_end": item["end"],
            "chunk_count": total_chunks,
        }

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                source_name=source_name,
                title=title,
                content=item["content"],
                metadata=chunk_metadata,
            )
        )

    return chunks


def index_chunks(
    chunks: list[DocumentChunk],
    *,
    store: QdrantStore | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: bool = False,
    progress_prefix: str = "[index-chunks]",
) -> int:
    """
    将 chunk 批量 embedding 后写入 Qdrant。
    """
    if not chunks:
        return 0

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    rag_store = store or QdrantStore()

    total = len(chunks)
    num_batches = math.ceil(total / batch_size)

    for batch_index in range(num_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, total)
        batch_chunks = chunks[start:end]
        batch_texts = [chunk.content for chunk in batch_chunks]
        _emit_progress_bar(
            progress,
            progress_prefix,
            batch_index + 1,
            num_batches,
        )

        # 当前采用串行批处理：
        # - 实现简单，失败边界清晰
        # - 便于后续加入重试、监控、进度日志
        # 如果数据量进一步增大，再考虑并行 embedding / 异步 upsert。
        batch_vectors = embed_texts(batch_texts)
        rag_store.upsert_chunks(batch_chunks, batch_vectors)

    return total


def index_document(
    *,
    document_id: str,
    source_name: str,
    title: str,
    text: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = True,
    store: QdrantStore | None = None,
    progress: bool = False,
) -> list[DocumentChunk]:
    """
    完成单篇文档的切块、向量化与入库。
    """
    rag_store = store or QdrantStore()

    chunks = build_document_chunks(
        document_id=document_id,
        source_name=source_name,
        title=title,
        text=text,
        metadata=metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if recreate:
        # 当前策略是“同 document_id 全量重建”，适合本阶段的离线索引流程。
        # 若后续需要增量更新，可在这里引入：
        # - 基于内容哈希的跳过逻辑
        # - 按 chunk 级别 diff/upsert
        # - 文档版本管理
        rag_store.delete_by_document_id(document_id)

    index_chunks(
        chunks,
        store=rag_store,
        batch_size=batch_size,
        progress=progress,
        progress_prefix=f"[index-document] upsert {title}",
    )
    return chunks


def load_text_file(path: str | Path) -> str:
    """
    读取 UTF-8 文本文件。
    """
    file_path = Path(path)
    return file_path.read_text(encoding="utf-8")


def index_text_file(
    path: str | Path,
    *,
    source_name: str = "local_files",
    document_id: str | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    root_dir: str | Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = True,
    store: QdrantStore | None = None,
    progress: bool = False,
) -> list[DocumentChunk]:
    """
    将单个文本文件索引进 Qdrant。
    """
    file_path = Path(path)
    text = load_text_file(file_path)

    hierarchy_metadata = build_file_hierarchy_metadata(
        file_path,
        root_dir=root_dir,
    )

    resolved_document_id = document_id or build_document_id_from_path(
        file_path,
        root_dir=root_dir,
    )
    resolved_title = title or file_path.name
    resolved_metadata = {
        **hierarchy_metadata,
        **(metadata or {}),
    }

    return index_document(
        document_id=resolved_document_id,
        source_name=source_name,
        title=resolved_title,
        text=text,
        metadata=resolved_metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
        recreate=recreate,
        store=store,
        progress=progress,
    )


def index_directory(
    directory: str | Path,
    *,
    pattern: str = "**/*.txt",
    source_name: str = "local_files",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = True,
    store: QdrantStore | None = None,
    progress: bool = False,
) -> dict[str, int]:
    """
    批量索引目录下的文本文件。
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {dir_path}")

    # 当前先只处理文本文件批量导入，保持入口清晰。
    # 如果后续接入多格式文档，建议不要在这里堆 if/else，
    # 而是改成独立 loader/parser 层，再统一产出 text + metadata。
    files = sorted(path for path in dir_path.glob(pattern) if path.is_file())
    rag_store = store or QdrantStore()
    existing_index_manifest = load_index_manifest(dir_path)
    previous_documents = existing_index_manifest.get("documents", {})
    source_manifest = load_source_manifest(dir_path)

    indexed_documents = 0
    indexed_chunks = 0
    skipped_documents = 0
    deleted_documents = 0
    current_documents: dict[str, Any] = {}
    total_files = len(files)

    _emit_progress(progress, f"[index-dir] start files={total_files} root={dir_path}")

    for file_index, file_path in enumerate(files, start=1):
        _emit_progress_bar(
            progress,
            "[index-dir] files",
            file_index,
            total_files,
        )
        relative_path = _path_to_posix(file_path.relative_to(dir_path))
        content_hash = sha256_file(file_path)
        source_item = source_manifest.get(relative_path, {})
        resolved_document_id = str(
            source_item.get("document_id")
            or build_document_id_from_path(file_path, root_dir=dir_path)
        )
        resolved_title = str(source_item.get("title") or file_path.name)
        resolved_metadata = {
            **(source_item or {}),
        }

        previous_record = previous_documents.get(relative_path, {})
        if (
            previous_record
            and previous_record.get("content_hash") == content_hash
            and previous_record.get("document_id") == resolved_document_id
        ):
            skipped_documents += 1
            current_documents[relative_path] = previous_record
            _emit_progress(
                progress,
                f"[index-dir] skip unchanged file={relative_path}",
            )
            continue

        _emit_progress(
            progress,
            f"[index-dir] indexing file={relative_path}",
        )
        chunks = index_text_file(
            file_path,
            source_name=source_name,
            document_id=resolved_document_id,
            title=resolved_title,
            metadata=resolved_metadata,
            root_dir=dir_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            recreate=recreate,
            store=rag_store,
            progress=progress,
        )
        if chunks:
            indexed_documents += 1
            indexed_chunks += len(chunks)
            current_documents[relative_path] = {
                "document_id": resolved_document_id,
                "title": resolved_title,
                "content_hash": content_hash,
                "chunk_count": len(chunks),
            }
        else:
            current_documents[relative_path] = {
                "document_id": resolved_document_id,
                "title": resolved_title,
                "content_hash": content_hash,
                "chunk_count": 0,
            }

    current_document_ids = {
        str(item.get("document_id", "")).strip()
        for item in current_documents.values()
        if str(item.get("document_id", "")).strip()
    }

    for relative_path, previous_record in previous_documents.items():
        if relative_path in current_documents:
            continue
        document_id = str(previous_record.get("document_id", "")).strip()
        if document_id:
            if document_id in current_document_ids:
                continue
            rag_store.delete_by_document_id(document_id)
            deleted_documents += 1
            _emit_progress(
                progress,
                f"[index-dir] deleted stale document_id={document_id}",
            )

    manifest_path = save_index_manifest(
        dir_path,
        {
            "root_dir": str(dir_path),
            "source_name": source_name,
            "documents": current_documents,
        },
    )

    result = {
        "documents": indexed_documents,
        "chunks": indexed_chunks,
        "skipped_documents": skipped_documents,
        "deleted_documents": deleted_documents,
        "manifest_path": manifest_path,
    }
    _emit_progress(
        progress,
        (
            "[index-dir] done "
            f"indexed_documents={indexed_documents} "
            f"indexed_chunks={indexed_chunks} "
            f"skipped_documents={skipped_documents} "
            f"deleted_documents={deleted_documents}"
        ),
    )
    return result
