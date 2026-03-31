from __future__ import annotations

import math
import logging
import re
from collections import Counter
from typing import Any

from app.rag.embeddings import embed_query
from app.rag.qdrant_store import QdrantStore
from app.rag.reranker import rerank_pairs
from app.rag.schemas import RetrievedItem


logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
VECTOR_CANDIDATE_MULTIPLIER = 4
BM25_CANDIDATE_LIMIT = 4000
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60.0
DOCUMENT_FOCUS_TOP_DOCS = 2
DOCUMENT_FOCUS_CANDIDATE_LIMIT = 256


def _apply_semantic_reranker(
    query: str,
    items: list[RetrievedItem],
    *,
    top_k: int,
) -> list[RetrievedItem]:
    if not query.strip() or not items:
        return items[:top_k]

    from app.core.config import get_settings

    settings = get_settings()
    if not settings.reranker_enabled:
        return items[:top_k]

    candidate_limit = max(top_k, settings.reranker_max_candidates)
    candidates = items[:candidate_limit]

    try:
        scores = rerank_pairs(query, [item.content for item in candidates])
    except Exception as exc:
        logger.warning("Semantic reranker failed; keeping heuristic ranking. Reason: %s", exc)
        return items[:top_k]

    rescored: list[tuple[float, int]] = []
    for index, (item, score) in enumerate(zip(candidates, scores)):
        rescored.append((float(score), index))

    rescored.sort(key=lambda entry: entry[0], reverse=True)

    reranked_items: list[RetrievedItem] = []
    for rank, (score, index) in enumerate(rescored[:top_k], start=1):
        item = candidates[index]
        reranked_items.append(
            RetrievedItem(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                source_name=item.source_name,
                title=item.title,
                content=item.content,
                score=score,
                metadata={
                    **item.metadata,
                    "rank": rank,
                    "reranker_model": settings.reranker_model,
                    "reranker_score": score,
                },
            )
        )

    return reranked_items


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _tokenize_for_bm25(text: str) -> list[str]:
    cleaned = text.lower()
    tokens: list[str] = []

    for token in re.findall(r"[a-z0-9_/-]{2,}", cleaned):
        tokens.append(token)

    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if len(chunk) <= 4:
            tokens.append(chunk)
        else:
            tokens.append(chunk[:4])
        if len(chunk) >= 2:
            tokens.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))

    return tokens


def _title_text(item: RetrievedItem) -> str:
    file_stem = str(item.metadata.get("file_stem", "")).strip()
    relative_path = str(item.metadata.get("relative_path", "")).strip()
    parts = [item.title.strip(), file_stem, relative_path]
    return " ".join(part for part in parts if part)


def _section_text(item: RetrievedItem) -> str:
    section_title = str(item.metadata.get("section_title", "")).strip()
    section_path_text = str(item.metadata.get("section_path_text", "")).strip()
    return " ".join(part for part in (section_title, section_path_text) if part)


def _query_tokens(query: str) -> list[str]:
    return _tokenize_for_bm25(query)


def _title_match_strength(query_tokens: list[str], item: RetrievedItem) -> float:
    title_text = _title_text(item).lower()
    if not title_text:
        return 0.0

    score = 0.0
    seen_tokens: set[str] = set()
    for token in query_tokens:
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        if len(token) <= 1:
            continue
        if token.lower() in title_text:
            score += 1.0
    return score


def _section_match_strength(query_tokens: list[str], item: RetrievedItem) -> float:
    section_text = _section_text(item).lower()
    if not section_text:
        return 0.0

    score = 0.0
    seen_tokens: set[str] = set()
    for token in query_tokens:
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        if len(token) <= 1:
            continue
        if token.lower() in section_text:
            score += 1.0
    return score


def _is_document_level_section(item: RetrievedItem) -> bool:
    section_title = str(item.metadata.get("section_title", "")).strip().lower()
    if not section_title:
        return False

    title = item.title.rsplit(".", 1)[0].strip().lower()
    file_stem = str(item.metadata.get("file_stem", "")).strip().lower()
    return section_title in {title, file_stem}


def _looks_like_structured_query(query: str) -> bool:
    lowered = query.lower()
    signals = [
        "compare",
        "comparison",
        "versus",
        "vs",
        "better",
        "difference",
        "ability",
        "abilities",
        "role",
        "strength",
        "对比",
        "比较",
        "差异",
        "哪个",
        "更强",
        "厉害",
        "能力",
        "定位",
        "作用",
    ]
    if any(signal in lowered for signal in signals):
        return True
    if _contains_cjk(query):
        return len(_tokenize_for_bm25(query)) >= 8
    return len(_tokenize_for_bm25(query)) >= 6


def _chunk_information_density(item: RetrievedItem) -> float:
    content = item.content.strip()
    if not content:
        return 0.0

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
    if not paragraphs:
        paragraphs = [content]

    short_paragraphs = sum(1 for paragraph in paragraphs if len(paragraph) <= 20)
    numeric_markers = len(re.findall(r"\d+[%./]?\d*", content))
    label_markers = len(re.findall(r"[：:]", content))

    score = min(numeric_markers * 0.08, 0.4) + min(label_markers * 0.04, 0.2)

    if len(paragraphs) >= 5 and short_paragraphs / len(paragraphs) >= 0.7:
        score -= 0.45

    script_families = 0
    for pattern in (r"[A-Za-z]", r"[\u4e00-\u9fff]", r"[\u3040-\u30ff]", r"[\uac00-\ud7af]"):
        if re.search(pattern, content):
            script_families += 1
    if script_families >= 3 and len(paragraphs) >= 4:
        score -= 0.3

    return score


def _structured_section_bonus(query: str, item: RetrievedItem) -> float:
    section_title = str(item.metadata.get("section_title", "")).strip()
    if not section_title or _is_document_level_section(item):
        return 0.0

    bonus = 0.2
    if _looks_like_structured_query(query):
        bonus += 0.55
    return bonus


def _compute_bm25_scores(
    query_tokens: list[str],
    documents: list[list[str]],
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[float]:
    if not query_tokens or not documents:
        return [0.0 for _ in documents]

    document_count = len(documents)
    document_frequencies: Counter[str] = Counter()
    document_term_frequencies: list[Counter[str]] = []
    document_lengths: list[int] = []

    for tokens in documents:
        token_counter = Counter(tokens)
        document_term_frequencies.append(token_counter)
        document_lengths.append(len(tokens))
        for token in token_counter.keys():
            document_frequencies[token] += 1

    average_length = sum(document_lengths) / document_count if document_count else 0.0
    scores: list[float] = []

    for token_counter, doc_length in zip(document_term_frequencies, document_lengths):
        score = 0.0
        normalization = k1 * (1 - b + b * (doc_length / average_length)) if average_length > 0 else k1
        for token in query_tokens:
            tf = token_counter.get(token, 0)
            if tf <= 0:
                continue
            df = document_frequencies.get(token, 0)
            idf = math.log(1 + ((document_count - df + 0.5) / (df + 0.5)))
            score += idf * ((tf * (k1 + 1)) / (tf + normalization))
        scores.append(score)

    return scores


def _bm25_retrieve(
    query: str,
    candidates: list[RetrievedItem],
    *,
    top_k: int,
) -> list[RetrievedItem]:
    query_tokens = _tokenize_for_bm25(query)
    if not query_tokens or not candidates:
        return []

    title_documents = [_tokenize_for_bm25(_title_text(item)) for item in candidates]
    content_documents = [_tokenize_for_bm25(item.content) for item in candidates]

    title_scores = _compute_bm25_scores(query_tokens, title_documents)
    content_scores = _compute_bm25_scores(query_tokens, content_documents)

    ranked: list[tuple[float, int]] = []
    for index, item in enumerate(candidates):
        title_score = title_scores[index]
        content_score = content_scores[index]
        combined_score = (2.5 * title_score) + content_score
        if combined_score <= 0:
            continue
        ranked.append((combined_score, index))

    ranked.sort(key=lambda entry: entry[0], reverse=True)

    results: list[RetrievedItem] = []
    for score, index in ranked[:top_k]:
        item = candidates[index]
        results.append(
            RetrievedItem(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                source_name=item.source_name,
                title=item.title,
                content=item.content,
                score=score,
                metadata={
                    **item.metadata,
                    "retrieval_channel": "bm25",
                    "title_text": _title_text(item),
                },
            )
        )

    return results


def _fuse_results(
    vector_items: list[RetrievedItem],
    bm25_items: list[RetrievedItem],
    *,
    top_k: int,
) -> list[RetrievedItem]:
    if top_k <= 0:
        return []

    combined: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(vector_items, start=1):
        entry = combined.setdefault(
            item.chunk_id,
            {
                "item": item,
                "rrf_score": 0.0,
                "channels": set(),
                "vector_score": 0.0,
                "bm25_score": 0.0,
            },
        )
        entry["rrf_score"] += 1.0 / (RRF_K + rank)
        entry["channels"].add("vector")
        entry["vector_score"] = max(entry["vector_score"], item.score)
        if item.score > entry["item"].score:
            entry["item"] = item

    for rank, item in enumerate(bm25_items, start=1):
        entry = combined.setdefault(
            item.chunk_id,
            {
                "item": item,
                "rrf_score": 0.0,
                "channels": set(),
                "vector_score": 0.0,
                "bm25_score": 0.0,
            },
        )
        entry["rrf_score"] += 1.0 / (RRF_K + rank)
        entry["channels"].add("bm25")
        entry["bm25_score"] = max(entry["bm25_score"], item.score)
        if item.score > entry["item"].score:
            entry["item"] = item

    ranked_entries = sorted(
        combined.values(),
        key=lambda entry: (
            entry["rrf_score"],
            entry["bm25_score"],
            entry["vector_score"],
        ),
        reverse=True,
    )

    fused: list[RetrievedItem] = []
    for rank, entry in enumerate(ranked_entries[:top_k], start=1):
        item = entry["item"]
        fused.append(
            RetrievedItem(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                source_name=item.source_name,
                title=item.title,
                content=item.content,
                score=entry["rrf_score"],
                metadata={
                    **item.metadata,
                    "rank": rank,
                    "retrieval_channels": sorted(entry["channels"]),
                    "vector_score": entry["vector_score"],
                    "bm25_score": entry["bm25_score"],
                    "fused_score": entry["rrf_score"],
                },
            )
        )

    return fused


def _rank_documents(
    query: str,
    items: list[RetrievedItem],
) -> list[tuple[str, float]]:
    query_tokens = _query_tokens(query)
    document_scores: dict[str, float] = {}

    for rank, item in enumerate(items, start=1):
        if not item.document_id:
            continue
        base_score = 1.0 / (RRF_K + rank)
        title_score = _title_match_strength(query_tokens, item)
        retrieval_channels = item.metadata.get("retrieval_channels", [])
        channel_bonus = 0.0
        if isinstance(retrieval_channels, list) and "bm25" in retrieval_channels:
            channel_bonus += 0.2
        if isinstance(retrieval_channels, list) and "vector" in retrieval_channels:
            channel_bonus += 0.1
        channel_bonus += min(_section_match_strength(query_tokens, item) * 0.35, 1.2)
        channel_bonus += _structured_section_bonus(query, item)
        channel_bonus += _chunk_information_density(item) * 0.25

        document_scores[item.document_id] = document_scores.get(item.document_id, 0.0) + base_score + title_score + channel_bonus

    return sorted(document_scores.items(), key=lambda item: item[1], reverse=True)


def _document_focused_retrieve(
    query: str,
    *,
    top_k: int,
    rag_store: QdrantStore,
    document_rankings: list[tuple[str, float]],
    source_name: str | None = None,
    top_level_group: str | None = None,
    hierarchy_scope: str | None = None,
) -> list[RetrievedItem]:
    if not document_rankings:
        return []

    query_tokens = _query_tokens(query)
    focused_results: list[RetrievedItem] = []

    for document_id, document_score in document_rankings[:DOCUMENT_FOCUS_TOP_DOCS]:
        candidates = rag_store.scroll_items(
            limit=DOCUMENT_FOCUS_CANDIDATE_LIMIT,
            source_name=source_name,
            top_level_group=top_level_group,
            hierarchy_scope=hierarchy_scope,
            document_id=document_id,
        )
        if not candidates:
            continue

        title_documents = [_tokenize_for_bm25(_title_text(item)) for item in candidates]
        content_documents = [_tokenize_for_bm25(item.content) for item in candidates]
        title_scores = _compute_bm25_scores(query_tokens, title_documents)
        content_scores = _compute_bm25_scores(query_tokens, content_documents)

        chunk_ranked: list[tuple[float, int]] = []
        for index, item in enumerate(candidates):
            title_score = title_scores[index]
            content_score = content_scores[index]
            section_score = _section_match_strength(query_tokens, item)
            chunk_index = int(item.metadata.get("chunk_index", 0) or 0)
            position_prior = max(0.0, 1.0 - (0.03 * chunk_index))
            section_bonus = _structured_section_bonus(query, item)
            information_density = _chunk_information_density(item)
            combined_score = (
                (3.0 * title_score)
                + content_score
                + (1.4 * section_score)
                + document_score
                + position_prior
                + section_bonus
                + information_density
            )
            if combined_score <= 0:
                continue
            chunk_ranked.append((combined_score, index))

        chunk_ranked.sort(key=lambda entry: entry[0], reverse=True)

        for score, index in chunk_ranked[:top_k]:
            item = candidates[index]
            focused_results.append(
                RetrievedItem(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    source_name=item.source_name,
                    title=item.title,
                    content=item.content,
                    score=score,
                    metadata={
                        **item.metadata,
                        "retrieval_channel": "document_focus",
                        "document_focus_score": score,
                    },
                )
            )

    return focused_results


def retrieve(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    source_name: str | None = None,
    top_level_group: str | None = None,
    hierarchy_scope: str | None = None,
    store: QdrantStore | None = None,
) -> list[RetrievedItem]:
    """
    执行本地混合检索。

    当前 pipeline：
    - 向量召回
    - lexical/BM25 召回（显式纳入标题、文件名、路径）
    - Reciprocal Rank Fusion 融合排序
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    rag_store = store or QdrantStore()
    vector_top_k = max(top_k * VECTOR_CANDIDATE_MULTIPLIER, top_k)
    query_vector = embed_query(cleaned_query)

    vector_items = rag_store.search(
        query_vector,
        top_k=vector_top_k,
        source_name=source_name,
        top_level_group=top_level_group,
        hierarchy_scope=hierarchy_scope,
    )

    bm25_candidates = rag_store.scroll_items(
        limit=max(vector_top_k, BM25_CANDIDATE_LIMIT),
        source_name=source_name,
        top_level_group=top_level_group,
        hierarchy_scope=hierarchy_scope,
    )
    bm25_items = _bm25_retrieve(
        cleaned_query,
        bm25_candidates,
        top_k=vector_top_k,
    )
    fused_items = _fuse_results(
        vector_items,
        bm25_items,
        top_k=vector_top_k,
    )

    document_rankings = _rank_documents(cleaned_query, fused_items)
    focused_items = _document_focused_retrieve(
        cleaned_query,
        top_k=vector_top_k,
        rag_store=rag_store,
        document_rankings=document_rankings,
        source_name=source_name,
        top_level_group=top_level_group,
        hierarchy_scope=hierarchy_scope,
    )

    if focused_items:
        rerank_candidates = _fuse_results(
            fused_items,
            focused_items,
            top_k=max(top_k, vector_top_k),
        )
        return _apply_semantic_reranker(cleaned_query, rerank_candidates, top_k=top_k)

    return _apply_semantic_reranker(cleaned_query, fused_items, top_k=top_k)


def items_to_docs(items: list[RetrievedItem]) -> list[str]:
    return [item.content for item in items if item.content.strip()]


def items_to_sources(items: list[RetrievedItem]) -> list[str]:
    return [item.chunk_id for item in items if item.chunk_id.strip()]


def items_to_evidence(
    items: list[RetrievedItem],
    *,
    query: str | None = None,
) -> list[dict[str, Any]]:
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
    source_name: str | None = None,
    top_level_group: str | None = None,
    hierarchy_scope: str | None = None,
    store: QdrantStore | None = None,
) -> dict[str, Any]:
    items = retrieve(
        query,
        top_k=top_k,
        source_name=source_name,
        top_level_group=top_level_group,
        hierarchy_scope=hierarchy_scope,
        store=store,
    )

    return {
        "retrieved_items": items,
        "retrieved_docs": items_to_docs(items),
        "retrieved_sources": items_to_sources(items),
        "evidence": items_to_evidence(items, query=query),
    }
