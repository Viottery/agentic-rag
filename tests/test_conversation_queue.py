from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "pydantic_settings",
    types.SimpleNamespace(BaseSettings=object, SettingsConfigDict=lambda **kwargs: kwargs),
)

from app.runtime.conversation_queue import ConversationQueueManager


def test_same_conversation_jobs_are_serialized() -> None:
    async def runner(question: str, conversation_id: str, turn_id: str, job_id: str) -> dict:
        nonlocal_active[0] += 1
        max_active[0] = max(max_active[0], nonlocal_active[0])
        await asyncio.sleep(0.05)
        nonlocal_active[0] -= 1
        return {"answer": question, "status": "finished"}

    async def main() -> None:
        manager = ConversationQueueManager(runner=runner, max_concurrent_conversations=4)
        await asyncio.gather(
            manager.submit(question="first", conversation_id="conv-a", mode="wait"),
            manager.submit(question="second", conversation_id="conv-a", mode="wait"),
        )

    nonlocal_active = [0]
    max_active = [0]
    asyncio.run(main())

    assert max_active[0] == 1


def test_different_conversations_can_run_concurrently() -> None:
    async def runner(question: str, conversation_id: str, turn_id: str, job_id: str) -> dict:
        nonlocal_active[0] += 1
        max_active[0] = max(max_active[0], nonlocal_active[0])
        await asyncio.sleep(0.05)
        nonlocal_active[0] -= 1
        return {"answer": question, "status": "finished"}

    async def main() -> None:
        manager = ConversationQueueManager(runner=runner, max_concurrent_conversations=4)
        await asyncio.gather(
            manager.submit(question="first", conversation_id="conv-a", mode="wait"),
            manager.submit(question="second", conversation_id="conv-b", mode="wait"),
        )

    nonlocal_active = [0]
    max_active = [0]
    asyncio.run(main())

    assert max_active[0] >= 2


def test_background_job_returns_job_record_and_result() -> None:
    async def runner(question: str, conversation_id: str, turn_id: str, job_id: str) -> dict:
        await asyncio.sleep(0.01)
        return {"answer": question.upper(), "status": "finished"}

    async def main() -> None:
        manager = ConversationQueueManager(runner=runner, max_concurrent_conversations=2)
        job = await manager.submit(question="agentic", conversation_id="conv-bg", mode="background")
        assert job.status in {"queued", "running"}

        deadline = time.monotonic() + 1.0
        while job.status not in {"finished", "failed"} and time.monotonic() < deadline:
            await asyncio.sleep(0.02)

        assert job.status == "finished"
        assert job.result is not None
        assert job.result["answer"] == "AGENTIC"
        assert job.result["conversation_id"] == "conv-bg"

    asyncio.run(main())
