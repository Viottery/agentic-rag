from __future__ import annotations

from typing import Any

from app.agent.services.local_rag_process_client import (
    ainvoke_local_rag_via_subprocess,
    invoke_local_rag_via_subprocess,
)


def invoke_local_rag_via_bash(
    payload: dict[str, Any],
    *,
    async_mode: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    return invoke_local_rag_via_subprocess(
        payload,
        async_mode=async_mode,
        timeout_seconds=timeout_seconds,
    )


async def ainvoke_local_rag_via_bash(
    payload: dict[str, Any],
    *,
    async_mode: bool = True,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    return await ainvoke_local_rag_via_subprocess(
        payload,
        async_mode=async_mode,
        timeout_seconds=timeout_seconds,
    )
