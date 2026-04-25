from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from app.core.config import get_settings
from app.runtime.process_runner import arun_process, run_process


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _python_executable() -> str:
    return sys.executable


def _build_args(request_path: Path, output_path: Path) -> list[str]:
    return [
        _python_executable(),
        "-m",
        "app.agent.services.local_rag_client",
        "invoke",
        "--request-file",
        str(request_path),
        "--output-file",
        str(output_path),
    ]


def invoke_local_rag_via_subprocess(
    payload: dict[str, Any],
    *,
    async_mode: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    timeout = timeout_seconds or settings.local_rag_subprocess_timeout_seconds

    with tempfile.TemporaryDirectory(prefix="local-rag-program-") as tempdir:
        request_path = Path(tempdir) / "request.json"
        output_path = Path(tempdir) / "response.json"
        request_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        completed = run_process(
            _build_args(request_path, output_path),
            cwd=_repo_root(),
            timeout_seconds=timeout,
        )
        if completed.timed_out:
            raise RuntimeError(f"local rag program timed out after {timeout} seconds")
        if completed.exit_code != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or f"exit={completed.exit_code}"
            raise RuntimeError(f"local rag program failed: {details}")
        if not output_path.exists():
            raise RuntimeError("local rag program did not produce an output file")
        return json.loads(output_path.read_text(encoding="utf-8"))


async def ainvoke_local_rag_via_subprocess(
    payload: dict[str, Any],
    *,
    async_mode: bool = True,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    timeout = timeout_seconds or settings.local_rag_subprocess_timeout_seconds

    with tempfile.TemporaryDirectory(prefix="local-rag-program-") as tempdir:
        request_path = Path(tempdir) / "request.json"
        output_path = Path(tempdir) / "response.json"
        request_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        completed = await arun_process(
            _build_args(request_path, output_path),
            cwd=_repo_root(),
            timeout_seconds=timeout,
        )
        if completed.timed_out:
            raise RuntimeError(f"local rag program timed out after {timeout} seconds")
        if completed.exit_code != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or f"exit={completed.exit_code}"
            raise RuntimeError(f"local rag program failed: {details}")
        if not output_path.exists():
            raise RuntimeError("local rag program did not produce an output file")
        return json.loads(output_path.read_text(encoding="utf-8"))
