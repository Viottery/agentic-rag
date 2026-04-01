from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any

from app.core.config import get_settings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _python_executable() -> str:
    return sys.executable


def _build_command(request_path: Path, output_path: Path, *, async_mode: bool) -> str:
    python_executable = shlex.quote(_python_executable())
    request_arg = shlex.quote(str(request_path))
    output_arg = shlex.quote(str(output_path))
    return (
        f"{python_executable} -m app.agent.services.local_rag_client "
        f"invoke --request-file {request_arg} --output-file {output_arg}"
    )


def invoke_local_rag_via_bash(
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
        command = _build_command(request_path, output_path, async_mode=async_mode)
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or f"exit={completed.returncode}"
            raise RuntimeError(f"local rag program failed: {details}")
        if not output_path.exists():
            raise RuntimeError("local rag program did not produce an output file")
        return json.loads(output_path.read_text(encoding="utf-8"))


async def ainvoke_local_rag_via_bash(
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
        command = _build_command(request_path, output_path, async_mode=async_mode)
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            command,
            cwd=str(_repo_root()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise RuntimeError(
                f"local rag program timed out after {timeout} seconds",
            ) from None
        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            details = stderr_text or stdout_text or f"exit={process.returncode}"
            raise RuntimeError(f"local rag program failed: {details}")
        if not output_path.exists():
            raise RuntimeError("local rag program did not produce an output file")
        return json.loads(output_path.read_text(encoding="utf-8"))
