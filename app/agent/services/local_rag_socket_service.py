from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from app.agent.services.local_rag_program import (
    LocalRAGProgramRequest,
    LocalRAGProgramResponse,
    _model_dump,
    _model_validate,
    run_local_rag_program_async,
)
from app.agent.llm import get_chat_model
from app.core.config import get_settings
from app.rag.inference_runtime import ensure_local_model_runtime


logger = logging.getLogger(__name__)


@dataclass
class _QueuedRetrieveRequest:
    request: LocalRAGProgramRequest
    future: asyncio.Future[LocalRAGProgramResponse]


class LocalRAGSocketService:
    def __init__(self) -> None:
        settings = get_settings()
        self._socket_path = Path(settings.local_rag_socket_path)
        self._retrieve_worker_count = max(1, settings.local_rag_retrieve_workers)
        self._retrieve_queue: asyncio.Queue[_QueuedRetrieveRequest | None] = asyncio.Queue()
        self._retrieve_workers: list[asyncio.Task[None]] = []
        self._server: asyncio.AbstractServer | None = None
        self._runtime = ensure_local_model_runtime()
        self._started = False

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        if self._started:
            return

        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self._socket_path.exists():
            self._socket_path.unlink()

        await asyncio.to_thread(get_chat_model)
        await self._runtime.start()
        self._retrieve_workers = [
            asyncio.create_task(self._retrieve_worker(index))
            for index in range(self._retrieve_worker_count)
        ]
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self._socket_path),
        )
        self._started = True
        logger.warning(
            "LocalRAGSocketService started at %s with %s retrieve workers.",
            self._socket_path,
            self._retrieve_worker_count,
        )

    async def stop(self) -> None:
        if not self._started:
            return

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for _ in self._retrieve_workers:
            await self._retrieve_queue.put(None)
        for task in self._retrieve_workers:
            with suppress(asyncio.CancelledError):
                await task
        self._retrieve_workers = []

        await self._runtime.stop()

        if self._socket_path.exists():
            self._socket_path.unlink()
        self._started = False

    async def submit(self, request: LocalRAGProgramRequest) -> LocalRAGProgramResponse:
        if not self._started:
            raise RuntimeError("LocalRAGSocketService is not started")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[LocalRAGProgramResponse] = loop.create_future()
        await self._retrieve_queue.put(_QueuedRetrieveRequest(request=request, future=future))
        return await future

    async def _retrieve_worker(self, worker_index: int) -> None:
        while True:
            item = await self._retrieve_queue.get()
            if item is None:
                break
            try:
                response = await run_local_rag_program_async(item.request)
                if not item.future.done():
                    item.future.set_result(response)
            except Exception as exc:  # noqa: BLE001
                if not item.future.done():
                    item.future.set_exception(exc)
                logger.exception("local rag retrieve worker %s failed: %s", worker_index, exc)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw_line = await reader.readline()
            if not raw_line:
                return
            envelope = json.loads(raw_line.decode("utf-8"))
            command = str(envelope.get("command", "")).strip()

            if command == "health":
                response = {
                    "ok": True,
                    "status": "ok",
                    "socket_path": str(self._socket_path),
                    "runtime_started": self._runtime.started,
                }
            elif command == "retrieve":
                request = _model_validate(
                    LocalRAGProgramRequest,
                    dict(envelope.get("payload", {})),
                )
                result = await self.submit(request)
                response = {
                    "ok": True,
                    "result": _model_dump(result),
                }
            else:
                response = {
                    "ok": False,
                    "error": f"unknown command: {command or '(empty)'}",
                }
        except Exception as exc:  # noqa: BLE001
            response = {
                "ok": False,
                "error": str(exc),
            }

        writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()


_SERVICE: LocalRAGSocketService | None = None


def get_local_rag_socket_service() -> LocalRAGSocketService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = LocalRAGSocketService()
    return _SERVICE
