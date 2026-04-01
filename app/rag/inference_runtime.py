from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from typing import Any

from app.core.config import get_settings
from app.rag.embeddings import get_embedding_model
from app.rag.reranker import get_reranker_model


logger = logging.getLogger(__name__)


@dataclass
class _EmbeddingTask:
    query: str
    future: asyncio.Future[list[float]]


@dataclass
class _RerankTask:
    query: str
    candidates: list[str]
    future: asyncio.Future[list[float]]


class LocalModelRuntime:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedding_queue: asyncio.Queue[_EmbeddingTask | None] = asyncio.Queue()
        self._rerank_queue: asyncio.Queue[_RerankTask | None] = asyncio.Queue()
        self._embedding_worker_task: asyncio.Task[None] | None = None
        self._rerank_worker_task: asyncio.Task[None] | None = None
        self._embedding_model: Any | None = None
        self._reranker_model: Any | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        if self._started:
            return

        self._embedding_model = await asyncio.to_thread(get_embedding_model)
        if self._settings.reranker_enabled:
            self._reranker_model = await asyncio.to_thread(get_reranker_model)

        self._embedding_worker_task = asyncio.create_task(self._embedding_worker())
        self._rerank_worker_task = asyncio.create_task(self._rerank_worker())
        self._started = True
        logger.warning("LocalModelRuntime started with preloaded embedding/reranker models.")

    async def stop(self) -> None:
        if not self._started:
            return

        await self._embedding_queue.put(None)
        await self._rerank_queue.put(None)

        for task in (self._embedding_worker_task, self._rerank_worker_task):
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._embedding_worker_task = None
        self._rerank_worker_task = None
        self._embedding_model = None
        self._reranker_model = None
        self._started = False

    async def embed_query(self, query: str) -> list[float]:
        if not self._started:
            raise RuntimeError("LocalModelRuntime is not started")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[float]] = loop.create_future()
        await self._embedding_queue.put(_EmbeddingTask(query=query, future=future))
        return await future

    async def rerank_pairs(self, query: str, candidates: list[str]) -> list[float]:
        if not self._started:
            raise RuntimeError("LocalModelRuntime is not started")
        if not candidates:
            return []
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[float]] = loop.create_future()
        await self._rerank_queue.put(
            _RerankTask(query=query, candidates=candidates, future=future)
        )
        return await future

    async def _embedding_worker(self) -> None:
        while True:
            task = await self._embedding_queue.get()
            if task is None:
                break

            batch = [task]
            batch_limit = max(1, self._settings.local_rag_embedding_batch_size)
            while len(batch) < batch_limit:
                try:
                    next_task = self._embedding_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if next_task is None:
                    await self._embedding_queue.put(None)
                    break
                batch.append(next_task)

            try:
                queries = [item.query for item in batch]
                vectors = await asyncio.to_thread(
                    self._encode_queries,
                    queries,
                )
                for item, vector in zip(batch, vectors):
                    if not item.future.done():
                        item.future.set_result(vector)
            except Exception as exc:  # noqa: BLE001
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

    async def _rerank_worker(self) -> None:
        while True:
            task = await self._rerank_queue.get()
            if task is None:
                break

            batch = [task]
            batch_limit = max(1, self._settings.local_rag_rerank_batch_tasks)
            while len(batch) < batch_limit:
                try:
                    next_task = self._rerank_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if next_task is None:
                    await self._rerank_queue.put(None)
                    break
                batch.append(next_task)

            try:
                scores_per_task = await asyncio.to_thread(
                    self._predict_rerank_scores,
                    batch,
                )
                for item, scores in zip(batch, scores_per_task):
                    if not item.future.done():
                        item.future.set_result(scores)
            except Exception as exc:  # noqa: BLE001
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

    def _encode_queries(self, queries: list[str]) -> list[list[float]]:
        model = self._embedding_model
        if model is None:
            raise RuntimeError("embedding model is not loaded")
        vectors = model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def _predict_rerank_scores(self, tasks: list[_RerankTask]) -> list[list[float]]:
        model = self._reranker_model
        if model is None:
            return [[] for _ in tasks]

        pairs: list[list[str]] = []
        slices: list[tuple[int, int]] = []
        for task in tasks:
            start = len(pairs)
            pairs.extend([[task.query, candidate] for candidate in task.candidates])
            end = len(pairs)
            slices.append((start, end))

        scores = model.predict(pairs)
        if hasattr(scores, "tolist"):
            normalized_scores = [float(item) for item in scores.tolist()]
        else:
            normalized_scores = [float(item) for item in scores]

        results: list[list[float]] = []
        for start, end in slices:
            results.append(normalized_scores[start:end])
        return results


_RUNTIME: LocalModelRuntime | None = None


def get_local_model_runtime() -> LocalModelRuntime | None:
    return _RUNTIME


def ensure_local_model_runtime() -> LocalModelRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = LocalModelRuntime()
    return _RUNTIME
