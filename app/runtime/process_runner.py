from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from collections.abc import Sequence


@dataclass(frozen=True)
class ProcessResult:
    args: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_process(
    args: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: int,
) -> ProcessResult:
    command_args = tuple(str(item) for item in args)
    resolved_cwd = str(Path(cwd).resolve())
    started_at = time.time()
    try:
        completed = subprocess.run(
            list(command_args),
            cwd=resolved_cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return ProcessResult(
            args=command_args,
            cwd=resolved_cwd,
            exit_code=int(completed.returncode),
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration_seconds=round(max(0.0, time.time() - started_at), 3),
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessResult(
            args=command_args,
            cwd=resolved_cwd,
            exit_code=-2,
            stdout=_decode_output(exc.stdout),
            stderr=_decode_output(exc.stderr),
            duration_seconds=round(max(0.0, time.time() - started_at), 3),
            timed_out=True,
        )


async def arun_process(
    args: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: int,
) -> ProcessResult:
    command_args = tuple(str(item) for item in args)
    resolved_cwd = str(Path(cwd).resolve())
    started_at = time.time()
    process = await asyncio.create_subprocess_exec(
        *command_args,
        cwd=resolved_cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
        return ProcessResult(
            args=command_args,
            cwd=resolved_cwd,
            exit_code=int(process.returncode or 0),
            stdout=_decode_output(stdout_bytes),
            stderr=_decode_output(stderr_bytes),
            duration_seconds=round(max(0.0, time.time() - started_at), 3),
        )
    except TimeoutError:
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        return ProcessResult(
            args=command_args,
            cwd=resolved_cwd,
            exit_code=-2,
            stdout=_decode_output(stdout_bytes),
            stderr=_decode_output(stderr_bytes),
            duration_seconds=round(max(0.0, time.time() - started_at), 3),
            timed_out=True,
        )
