from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.retriever import _bm25_retrieve, _document_focused_retrieve, _fuse_results, _rank_documents
from app.rag.schemas import RetrievedItem


def test_bm25_retrieve_prioritizes_title_matches() -> None:
    candidates = [
        RetrievedItem(
            chunk_id="a",
            document_id="doc-a",
            source_name="kb",
            title="8924-坚雷.txt",
            content="这是一段完全无关的介绍文本。",
            score=0.0,
            metadata={"file_stem": "8924-坚雷", "relative_path": "干员/8924-坚雷.txt"},
        ),
        RetrievedItem(
            chunk_id="b",
            document_id="doc-b",
            source_name="kb",
            title="9903-真理.txt",
            content="真理是一名来自乌萨斯学生自治团的辅助干员。",
            score=0.0,
            metadata={"file_stem": "9903-真理", "relative_path": "干员/9903-真理.txt"},
        ),
    ]

    results = _bm25_retrieve("干员真理 介绍", candidates, top_k=2)

    assert results
    assert results[0].chunk_id == "b"
    assert results[0].metadata["retrieval_channel"] == "bm25"


def test_fuse_results_keeps_items_from_both_channels_and_merges_scores() -> None:
    vector_items = [
        RetrievedItem(
            chunk_id="a",
            document_id="doc-a",
            source_name="kb",
            title="A.txt",
            content="alpha",
            score=0.8,
            metadata={},
        ),
        RetrievedItem(
            chunk_id="b",
            document_id="doc-b",
            source_name="kb",
            title="B.txt",
            content="beta",
            score=0.7,
            metadata={},
        ),
    ]
    bm25_items = [
        RetrievedItem(
            chunk_id="b",
            document_id="doc-b",
            source_name="kb",
            title="B.txt",
            content="beta",
            score=4.0,
            metadata={"retrieval_channel": "bm25"},
        ),
        RetrievedItem(
            chunk_id="c",
            document_id="doc-c",
            source_name="kb",
            title="C.txt",
            content="gamma",
            score=3.0,
            metadata={"retrieval_channel": "bm25"},
        ),
    ]

    fused = _fuse_results(vector_items, bm25_items, top_k=3)

    assert len(fused) == 3
    assert fused[0].chunk_id == "b"
    assert fused[0].metadata["retrieval_channels"] == ["bm25", "vector"]
    assert fused[0].metadata["bm25_score"] == 4.0
    assert fused[0].metadata["vector_score"] == 0.7


def test_rank_documents_prefers_document_with_strong_title_match() -> None:
    items = [
        RetrievedItem(
            chunk_id="truth-1",
            document_id="doc-truth",
            source_name="kb",
            title="1765-真理.txt",
            content="干员信息",
            score=0.016,
            metadata={"retrieval_channels": ["bm25"]},
        ),
        RetrievedItem(
            chunk_id="other-1",
            document_id="doc-other",
            source_name="kb",
            title="19822-特米米.txt",
            content="角色资料",
            score=0.016,
            metadata={"retrieval_channels": ["vector"]},
        ),
    ]

    ranked_documents = _rank_documents("干员真理 介绍", items)

    assert ranked_documents
    assert ranked_documents[0][0] == "doc-truth"


class _FakeStore:
    def __init__(self, items: list[RetrievedItem]) -> None:
        self._items = items

    def scroll_items(self, **kwargs):  # noqa: ANN003
        document_id = kwargs.get("document_id")
        return [item for item in self._items if item.document_id == document_id]


def test_document_focused_retrieve_prefers_matching_chunks_inside_selected_document() -> None:
    store = _FakeStore(
        [
            RetrievedItem(
                chunk_id="truth-1",
                document_id="doc-truth",
                source_name="kb",
                title="1765-真理.txt",
                content="真理是乌萨斯学生自治团出身的辅助干员。",
                score=0.0,
                metadata={"chunk_index": 1, "file_stem": "1765-真理", "relative_path": "干员/1765-真理.txt"},
            ),
            RetrievedItem(
                chunk_id="truth-2",
                document_id="doc-truth",
                source_name="kb",
                title="1765-真理.txt",
                content="技能二：文学风暴。攻击范围扩大。",
                score=0.0,
                metadata={"chunk_index": 7, "file_stem": "1765-真理", "relative_path": "干员/1765-真理.txt"},
            ),
        ]
    )

    results = _document_focused_retrieve(
        "干员真理的背景信息",
        top_k=2,
        rag_store=store,  # type: ignore[arg-type]
        document_rankings=[("doc-truth", 3.0)],
    )

    assert results
    assert results[0].chunk_id == "truth-1"
    assert results[0].metadata["retrieval_channel"] == "document_focus"
