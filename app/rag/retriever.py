from __future__ import annotations

from typing import Any

from app.rag.embeddings import embed_query
from app.rag.qdrant_store import QdrantStore
from app.rag.schemas import RetrievedItem


DEFAULT_TOP_K = 3


def retrieve(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    store: QdrantStore | None = None,
) -> list[RetrievedItem]:
    """
    执行一次基础向量检索。

    当前职责保持尽量单一：
    - query embedding
    - 调用向量库搜索
    - 返回标准化的 RetrievedItem

    当前实现是基础版 retriever，不包含：
    - query rewrite
    - hybrid search
    - reranking
    - evidence filtering / dedup

    这些能力更适合后续逐步补到 retrieval subgraph，而不是堆在这个入口里。
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    rag_store = store or QdrantStore()
    query_vector = embed_query(cleaned_query)
    return rag_store.search(query_vector, top_k=top_k)


def items_to_docs(items: list[RetrievedItem]) -> list[str]:
    """
    将检索结果转换为 responder 友好的纯文本列表。

    当前 responder 仍然消费简单的字符串列表，所以保留这个轻量适配层。
    如果后续 responder 升级为直接读取结构化 evidence，这里可以弱化甚至移除。
    """
    return [item.content for item in items if item.content.strip()]


def items_to_sources(items: list[RetrievedItem]) -> list[str]:
    """
    提取检索来源标识。

    当前默认使用 chunk_id，因为它在现有索引策略下稳定且可定位。
    后续如果希望上层展示 document 级来源，可以切换成 document_id/title。
    """
    return [item.chunk_id for item in items if item.chunk_id.strip()]


def items_to_evidence(
    items: list[RetrievedItem],
    *,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """
    将 RetrievedItem 转换为 agent 层可消费的 evidence 结构。

    这里故意返回通用 dict，而不是直接依赖 agent 模块类型，
    目的是保持 rag 层与 agent 层解耦，避免未来出现反向依赖。
    """
    evidence: list[dict[str, Any]] = []

    for rank, item in enumerate(items, start=1):
        metadata = {
            **item.metadata,
            "rank": rank,
        }
        if query is not None:
            metadata["query"] = query

        evidence.append(
            {
                "source_type": "local_kb",
                "source_name": item.source_name,
                "source_id": item.chunk_id,
                "title": item.title,
                "content": item.content,
                "score": item.score,
                "metadata": metadata,
            }
        )

    return evidence


def retrieve_as_context(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    store: QdrantStore | None = None,
) -> dict[str, Any]:
    """
    为 agent 节点提供一站式检索结果组装。

    返回字段与当前 AgentState 的消费方式对齐：
    - retrieved_items: 原始结构化结果
    - retrieved_docs: 文本片段列表
    - retrieved_sources: 来源标识列表
    - evidence: 结构化证据

    这是一个面向当前项目状态的薄封装。
    后续如果 retrieval subgraph 变复杂，可以让 agent 改为直接消费更细的中间结果。
    """
    items = retrieve(
        query,
        top_k=top_k,
        store=store,
    )

    return {
        "retrieved_items": items,
        "retrieved_docs": items_to_docs(items),
        "retrieved_sources": items_to_sources(items),
        "evidence": items_to_evidence(items, query=query),
    }
