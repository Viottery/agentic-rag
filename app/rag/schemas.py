from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """
    文档切块后的最小存储单元。

    用于：
    - embedding
    - qdrant upsert
    - 检索结果映射
    """

    chunk_id: str
    document_id: str
    source_name: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedItem:
    """
    检索返回结果。

    用于：
    - retriever -> agent node
    - 构造 evidence
    """

    chunk_id: str
    document_id: str
    source_name: str
    title: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)