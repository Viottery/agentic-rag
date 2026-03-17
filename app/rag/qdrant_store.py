from __future__ import annotations

import os
from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.rag.schemas import DocumentChunk, RetrievedItem


DEFAULT_COLLECTION_NAME = "agentic_rag_docs"


FILTERABLE_METADATA_KEYS = {
    "source_root",
    "relative_path",
    "file_name",
    "file_stem",
    "knowledge_path",
    "knowledge_path_text",
    "path_segments",
    "hierarchy_path",
    "hierarchy_level",
    "hierarchy_prefixes",
    "root_group",
    "leaf_group",
    "top_level_group",
    "parent_path",
}


def build_structure_summary_from_payloads(
    payloads: list[dict[str, Any]],
    *,
    collection_name: str,
    collection_exists: bool,
    point_count: int,
    max_groups: int = 8,
    max_paths_per_group: int = 5,
) -> dict[str, Any]:
    """根据 payload 样本归纳当前知识库的层次结构摘要。"""
    source_documents: dict[str, set[str]] = defaultdict(set)
    group_documents: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    path_documents: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )

    for index, payload in enumerate(payloads):
        document_id = str(
            payload.get("document_id")
            or payload.get("chunk_id")
            or payload.get("title")
            or f"payload-{index}"
        ).strip()
        if not document_id:
            continue

        source_name = str(payload.get("source_name") or "unknown").strip() or "unknown"
        top_level_group = str(
            payload.get("top_level_group")
            or payload.get("root_group")
            or "(ungrouped)"
        ).strip() or "(ungrouped)"
        hierarchy_path = str(
            payload.get("hierarchy_path")
            or payload.get("parent_path")
            or ""
        ).strip()

        source_documents[source_name].add(document_id)
        group_documents[source_name][top_level_group].add(document_id)

        effective_path = hierarchy_path or (top_level_group if top_level_group != "(ungrouped)" else "")
        if effective_path:
            path_documents[source_name][top_level_group][effective_path].add(document_id)

    rendered_sources: list[dict[str, Any]] = []
    for source_name, document_ids in sorted(
        source_documents.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        groups: list[dict[str, Any]] = []
        for group_name, group_doc_ids in sorted(
            group_documents[source_name].items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[:max_groups]:
            path_names = [
                path_name
                for path_name, _ in sorted(
                    path_documents[source_name][group_name].items(),
                    key=lambda item: (-len(item[1]), item[0]),
                )[:max_paths_per_group]
            ]
            groups.append(
                {
                    "group_name": group_name,
                    "doc_count": len(group_doc_ids),
                    "paths": path_names,
                }
            )

        rendered_sources.append(
            {
                "source_name": source_name,
                "doc_count": len(document_ids),
                "groups": groups,
            }
        )

    sampled_document_count = len(
        {
            document_id
            for documents in source_documents.values()
            for document_id in documents
        }
    )

    return {
        "collection_name": collection_name,
        "collection_exists": collection_exists,
        "point_count": point_count,
        "sampled_document_count": sampled_document_count,
        "sources": rendered_sources,
    }


def render_structure_summary(summary: dict[str, Any]) -> str:
    """将知识库结构摘要渲染为适合 planner 阅读的短文本。"""
    collection_name = str(summary.get("collection_name", DEFAULT_COLLECTION_NAME))
    if not summary.get("collection_exists", False):
        return f"Local KB unavailable: collection `{collection_name}` is not initialized."

    sources = summary.get("sources", [])
    if not sources:
        return (
            f"Local KB collection `{collection_name}` exists, but no structured documents "
            "were found in the sampled payloads."
        )

    lines = [
        (
            f"collection={collection_name} "
            f"points={summary.get('point_count', 0)} "
            f"sampled_documents={summary.get('sampled_document_count', 0)}"
        )
    ]
    for source in sources:
        group_parts: list[str] = []
        for group in source.get("groups", []):
            paths = [path for path in group.get("paths", []) if path]
            if paths:
                group_parts.append(
                    f"{group.get('group_name', '(ungrouped)')} "
                    f"({group.get('doc_count', 0)} docs; scopes: {', '.join(paths)})"
                )
            else:
                group_parts.append(
                    f"{group.get('group_name', '(ungrouped)')} "
                    f"({group.get('doc_count', 0)} docs)"
                )

        lines.append(
            f"- source={source.get('source_name', 'unknown')} "
            f"docs={source.get('doc_count', 0)} "
            f"groups={'; '.join(group_parts) if group_parts else 'None'}"
        )

    return "\n".join(lines)


def to_qdrant_point_id(chunk_id: str) -> str:
    """
    将业务侧 chunk_id 映射为 Qdrant 可接受的 point id。

    当前项目里的 chunk_id 形如 `document::chunk::001`，可读性很好，
    但并不符合部分 Qdrant 版本对 point id 的要求。
    这里用稳定的 UUID5 做一层映射，保证：
    - 同一个 chunk_id 每次生成相同 point id
    - 不影响业务侧继续使用原始 chunk_id 做 payload/source_id
    """
    return str(uuid5(NAMESPACE_URL, chunk_id))


class QdrantStore:
    """
    Qdrant 向量存储封装。

    提供：
    - collection 初始化
    - chunk upsert
    - 相似度检索
    - 按 document_id 删除
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))

        try:
            from qdrant_client import QdrantClient
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "缺少 qdrant-client 依赖，无法连接向量库。"
                "请先安装 requirements.txt 中的依赖。"
            ) from exc

        self.client = QdrantClient(host=self.host, port=self.port)

    def collection_exists(self) -> bool:
        """
        判断目标 collection 是否已经存在。

        这个显式检查让 indexing / retrieval 在“尚未建库”的早期阶段
        可以更平滑地返回空结果，而不是把异常直接暴露到上层工作流。
        """
        collections = self.client.get_collections().collections
        existing_names = {item.name for item in collections}
        return self.collection_name in existing_names

    def ensure_collection(self, vector_size: int) -> None:
        """
        如果 collection 不存在，则创建。
        如果已存在，则直接复用。
        """
        if self.collection_exists():
            return

        from qdrant_client.models import Distance, VectorParams

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def count_points(self) -> int:
        """
        返回当前 collection 中的点数量。

        用于快速确认：
        - collection 是否存在
        - indexing 是否真的写入了数据
        """
        if not self.collection_exists():
            return 0

        response = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )
        return int(response.count)

    def _payload_to_item(self, payload: dict[str, Any], *, fallback_id: str = "") -> RetrievedItem:
        return RetrievedItem(
            chunk_id=str(payload.get("chunk_id", fallback_id)),
            document_id=str(payload.get("document_id", "")),
            source_name=str(payload.get("source_name", "qdrant_kb")),
            title=str(payload.get("title", "")),
            content=str(payload.get("content", "")),
            score=0.0,
            metadata=dict(payload.get("metadata", {})),
        )

    def describe_structure(
        self,
        *,
        sample_limit: int = 2000,
        page_size: int = 256,
        max_groups: int = 8,
        max_paths_per_group: int = 5,
    ) -> dict[str, Any]:
        """
        读取当前 collection 的 payload 样本，归纳一份层次结构摘要。

        这份摘要主要给 planner / router 使用，帮助它知道当前本地知识库中
        已有哪些 source、顶层分组和层次路径，而不是盲目创建检索任务。
        """
        if not self.collection_exists():
            return build_structure_summary_from_payloads(
                [],
                collection_name=self.collection_name,
                collection_exists=False,
                point_count=0,
                max_groups=max_groups,
                max_paths_per_group=max_paths_per_group,
            )

        payloads: list[dict[str, Any]] = []
        offset: Any = None

        while len(payloads) < sample_limit:
            current_limit = min(page_size, sample_limit - len(payloads))
            response = self.client.scroll(
                collection_name=self.collection_name,
                limit=current_limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if isinstance(response, tuple):
                points, offset = response
            else:
                points = list(getattr(response, "points", []))
                offset = getattr(response, "next_page_offset", None)

            if not points:
                break

            for point in points:
                payload = getattr(point, "payload", None) or {}
                payloads.append(dict(payload))
                if len(payloads) >= sample_limit:
                    break

            if offset is None:
                break

        return build_structure_summary_from_payloads(
            payloads,
            collection_name=self.collection_name,
            collection_exists=True,
            point_count=self.count_points(),
            max_groups=max_groups,
            max_paths_per_group=max_paths_per_group,
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        """
        将 chunk 与对应向量写入 Qdrant。
        """
        if not chunks:
            return

        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")

        vector_size = len(vectors[0])
        self.ensure_collection(vector_size=vector_size)

        from qdrant_client.models import PointStruct

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            filterable_metadata = {
                key: value
                for key, value in chunk.metadata.items()
                if key in FILTERABLE_METADATA_KEYS
            }
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source_name": chunk.source_name,
                "title": chunk.title,
                "content": chunk.content,
                **filterable_metadata,
                "metadata": chunk.metadata,
            }

            points.append(
                PointStruct(
                    id=to_qdrant_point_id(chunk.chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def _build_filter(
        self,
        *,
        source_name: str | None = None,
        top_level_group: str | None = None,
        hierarchy_scope: str | None = None,
        document_id: str | None = None,
    ) -> "Filter | None":
        must_conditions: list[Any] = []

        if source_name:
            from qdrant_client.models import FieldCondition, MatchValue

            must_conditions.append(
                FieldCondition(
                    key="source_name",
                    match=MatchValue(value=source_name),
                )
            )

        if top_level_group:
            from qdrant_client.models import FieldCondition, MatchValue

            must_conditions.append(
                FieldCondition(
                    key="top_level_group",
                    match=MatchValue(value=top_level_group),
                )
            )

        # 对层次化路径范围，当前先使用 hierarchy_prefixes 里的精确命中。
        # 这让上层可以用“顶层分组”或“更细层级路径”做粗过滤。
        if hierarchy_scope:
            from qdrant_client.models import FieldCondition, MatchValue

            must_conditions.append(
                FieldCondition(
                    key="hierarchy_prefixes",
                    match=MatchValue(value=hierarchy_scope),
                )
            )

        if document_id:
            from qdrant_client.models import FieldCondition, MatchValue

            must_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            )

        if not must_conditions:
            return None

        from qdrant_client.models import Filter

        return Filter(must=must_conditions)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        *,
        source_name: str | None = None,
        top_level_group: str | None = None,
        hierarchy_scope: str | None = None,
        document_id: str | None = None,
    ) -> list[RetrievedItem]:
        """
        执行向量相似度检索。
        """
        if not query_vector:
            return []

        # 在知识库尚未初始化时，当前阶段优先返回空结果。
        # 这样 agent 仍然可以走完回答链路，而不是因为 collection 不存在直接失败。
        if not self.collection_exists():
            return []

        query_filter = self._build_filter(
            source_name=source_name,
            top_level_group=top_level_group,
            hierarchy_scope=hierarchy_scope,
            document_id=document_id,
        )

        # qdrant-client 在不同版本之间检索接口有差异：
        # - 新版常见为 query_points(...)
        # - 旧版常见为 search(...)
        # 这里统一兼容，避免当前项目被客户端版本细节卡住。
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                query_filter=query_filter,
            )
            results = response.points
        elif hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
            )
        else:
            raise RuntimeError("当前 qdrant-client 版本不支持 query_points/search 检索接口。")

        retrieved: list[RetrievedItem] = []
        for item in results:
            payload = item.payload or {}
            retrieved_item = self._payload_to_item(payload, fallback_id=str(item.id))
            retrieved_item.score = float(item.score)
            retrieved.append(retrieved_item)

        return retrieved

    def scroll_items(
        self,
        *,
        limit: int = 1000,
        source_name: str | None = None,
        top_level_group: str | None = None,
        hierarchy_scope: str | None = None,
        document_id: str | None = None,
    ) -> list[RetrievedItem]:
        """
        在当前 collection 内按过滤条件滚动读取候选项。

        主要用于本地 lexical/BM25 召回与调试，不依赖向量检索。
        """
        if limit <= 0:
            return []
        if not self.collection_exists():
            return []

        query_filter = self._build_filter(
            source_name=source_name,
            top_level_group=top_level_group,
            hierarchy_scope=hierarchy_scope,
            document_id=document_id,
        )

        collected: list[RetrievedItem] = []
        offset: Any = None
        page_size = min(limit, 256)

        while len(collected) < limit:
            current_limit = min(page_size, limit - len(collected))
            response = self.client.scroll(
                collection_name=self.collection_name,
                limit=current_limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=query_filter,
            )

            if isinstance(response, tuple):
                points, offset = response
            else:
                points = list(getattr(response, "points", []))
                offset = getattr(response, "next_page_offset", None)

            if not points:
                break

            for point in points:
                payload = getattr(point, "payload", None) or {}
                collected.append(self._payload_to_item(dict(payload), fallback_id=str(getattr(point, "id", ""))))
                if len(collected) >= limit:
                    break

            if offset is None:
                break

        return collected

    def delete_by_document_id(self, document_id: str) -> None:
        """
        按 document_id 删除所有相关 chunk。
        """
        if not document_id.strip():
            return

        # 全新环境下可能还没有 collection；这里直接跳过，保证重建索引幂等。
        if not self.collection_exists():
            return

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
