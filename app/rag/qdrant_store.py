from __future__ import annotations

import os
from uuid import NAMESPACE_URL, uuid5
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.rag.schemas import DocumentChunk, RetrievedItem


DEFAULT_COLLECTION_NAME = "agentic_rag_docs"


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
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source_name": chunk.source_name,
                "title": chunk.title,
                "content": chunk.content,
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

    def search(self, query_vector: list[float], top_k: int = 3) -> list[RetrievedItem]:
        """
        执行向量相似度检索。
        """
        if not query_vector:
            return []

        # 在知识库尚未初始化时，当前阶段优先返回空结果。
        # 这样 agent 仍然可以走完回答链路，而不是因为 collection 不存在直接失败。
        if not self.collection_exists():
            return []

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
            )
            results = response.points
        elif hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )
        else:
            raise RuntimeError("当前 qdrant-client 版本不支持 query_points/search 检索接口。")

        retrieved: list[RetrievedItem] = []
        for item in results:
            payload = item.payload or {}

            retrieved.append(
                RetrievedItem(
                    chunk_id=str(payload.get("chunk_id", item.id)),
                    document_id=str(payload.get("document_id", "")),
                    source_name=str(payload.get("source_name", "qdrant_kb")),
                    title=str(payload.get("title", "")),
                    content=str(payload.get("content", "")),
                    score=float(item.score),
                    metadata=dict(payload.get("metadata", {})),
                )
            )

        return retrieved

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
