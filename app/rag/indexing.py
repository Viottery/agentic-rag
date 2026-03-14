from __future__ import annotations

import math
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
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = True,
    store: QdrantStore | None = None,
) -> list[DocumentChunk]:
    """
    将单个文本文件索引进 Qdrant。
    """
    file_path = Path(path)
    text = load_text_file(file_path)

    resolved_document_id = document_id or file_path.stem
    resolved_title = title or file_path.name
    resolved_metadata = {
        "file_path": str(file_path),
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

    indexed_documents = 0
    indexed_chunks = 0

    for file_path in files:
        chunks = index_text_file(
            file_path,
            source_name=source_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            recreate=recreate,
            store=rag_store,
        )
        if chunks:
            indexed_documents += 1
            indexed_chunks += len(chunks)

    return {
        "documents": indexed_documents,
        "chunks": indexed_chunks,
    }
