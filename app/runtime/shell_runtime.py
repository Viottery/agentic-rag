from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import time

from app.core.config import get_settings


_READ_ONLY_PREFIXES = (
    "pwd",
    "ls",
    "find",
    "rg",
    "grep",
    "cat",
    "sed",
    "head",
    "tail",
    "wc",
    "stat",
    "git status",
    "git diff",
    "git show",
    "python -m pytest",
    "pytest",
)

_DANGEROUS_PATTERNS = (
    r"(^|[\s;&|])sudo(\s|$)",
    r"rm\s+-rf\s+/",
    r"mkfs(\.| )",
    r"(^|[\s;&|])(shutdown|reboot|halt|poweroff)(\s|$)",
    r"dd\s+if=",
    r"(:\(\)\s*\{\s*:\|\:&\s*\};:)",
    r"curl\b[^|]*\|\s*(sh|bash)",
    r"wget\b[^|]*\|\s*(sh|bash)",
    r">\s*/dev/sd[a-z]",
    r"chmod\s+-R\s+777\s+/",
    r"chown\s+-R\s+/\b",
    r"(^|[\s;&|])(ssh|scp|rsync)\s+",
)


@dataclass
class ShellExecutionResult:
    allowed: bool
    command: str
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    risk_level: str
    policy_reason: str
    truncated: bool


def _normalize_command(command: str) -> str:
    return re.sub(r"\s+", " ", command).strip()


def _classify_risk(command: str) -> str:
    normalized = _normalize_command(command).lower()
    if any(normalized.startswith(prefix) for prefix in _READ_ONLY_PREFIXES):
        return "low"
    return "medium"


def _policy_decision(command: str) -> tuple[bool, str, str]:
    settings = get_settings()
    if not settings.shell_runtime_enabled:
        return False, "shell runtime is disabled by configuration.", "blocked"

    normalized = _normalize_command(command)
    if not normalized:
        return False, "empty shell command.", "blocked"

    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return False, "shell command blocked by dangerous-pattern policy.", "blocked"

    risk_level = _classify_risk(normalized)
    policy_mode = settings.shell_policy_mode.strip().lower() or "workspace-write"
    if policy_mode == "disabled":
        return False, "shell runtime policy mode is disabled.", "blocked"
    if policy_mode == "read-only" and risk_level != "low":
        return False, "shell command requires write-capable policy mode.", risk_level
    return True, "allowed", risk_level


def _truncate_output(text: str) -> tuple[str, bool]:
    settings = get_settings()
    cleaned = text.strip()
    if len(cleaned) <= settings.shell_max_output_chars:
        return cleaned, False
    limit = max(0, settings.shell_max_output_chars - 1)
    return cleaned[:limit].rstrip() + "…", True


def run_shell_command(command: str, *, cwd: str | None = None) -> ShellExecutionResult:
    settings = get_settings()
    allowed, policy_reason, risk_level = _policy_decision(command)
    resolved_cwd = str(Path(cwd or ".").resolve())
    if not allowed:
        return ShellExecutionResult(
            allowed=False,
            command=_normalize_command(command),
            cwd=resolved_cwd,
            exit_code=-1,
            stdout="",
            stderr="",
            duration_seconds=0.0,
            risk_level=risk_level,
            policy_reason=policy_reason,
            truncated=False,
        )

    started_at = time.time()
    try:
        completed = subprocess.run(
            [settings.shell_program, "-lc", command],
            cwd=resolved_cwd,
            capture_output=True,
            text=True,
            timeout=settings.shell_command_timeout_seconds,
            check=False,
        )
        stdout, stdout_truncated = _truncate_output(completed.stdout or "")
        stderr, stderr_truncated = _truncate_output(completed.stderr or "")
        return ShellExecutionResult(
            allowed=True,
            command=_normalize_command(command),
            cwd=resolved_cwd,
            exit_code=int(completed.returncode),
            stdout=stdout,
            stderr=stderr,
            duration_seconds=round(max(0.0, time.time() - started_at), 3),
            risk_level=risk_level,
            policy_reason=policy_reason,
            truncated=stdout_truncated or stderr_truncated,
        )
    except subprocess.TimeoutExpired as exc:
        stdout, stdout_truncated = _truncate_output(exc.stdout or "")
        stderr, stderr_truncated = _truncate_output(exc.stderr or "")
        return ShellExecutionResult(
            allowed=True,
            command=_normalize_command(command),
            cwd=resolved_cwd,
            exit_code=-2,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=round(max(0.0, time.time() - started_at), 3),
            risk_level=risk_level,
            policy_reason=f"command timed out after {settings.shell_command_timeout_seconds}s.",
            truncated=stdout_truncated or stderr_truncated,
        )


async def arun_shell_command(command: str, *, cwd: str | None = None) -> ShellExecutionResult:
    settings = get_settings()
    allowed, policy_reason, risk_level = _policy_decision(command)
    resolved_cwd = str(Path(cwd or ".").resolve())
    if not allowed:
        return ShellExecutionResult(
            allowed=False,
            command=_normalize_command(command),
            cwd=resolved_cwd,
            exit_code=-1,
            stdout="",
            stderr="",
            duration_seconds=0.0,
            risk_level=risk_level,
            policy_reason=policy_reason,
            truncated=False,
        )

    started_at = time.time()
    try:
        process = await asyncio.create_subprocess_exec(
            settings.shell_program,
            "-lc",
            command,
            cwd=resolved_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=settings.shell_command_timeout_seconds,
        )
        stdout, stdout_truncated = _truncate_output(stdout_bytes.decode("utf-8", errors="replace"))
        stderr, stderr_truncated = _truncate_output(stderr_bytes.decode("utf-8", errors="replace"))
        return ShellExecutionResult(
            allowed=True,
            command=_normalize_command(command),
            cwd=resolved_cwd,
            exit_code=int(process.returncode or 0),
            stdout=stdout,
            stderr=stderr,
            duration_seconds=round(max(0.0, time.time() - started_at), 3),
            risk_level=risk_level,
            policy_reason=policy_reason,
            truncated=stdout_truncated or stderr_truncated,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return ShellExecutionResult(
            allowed=True,
            command=_normalize_command(command),
            cwd=resolved_cwd,
            exit_code=-2,
            stdout="",
            stderr="",
            duration_seconds=round(max(0.0, time.time() - started_at), 3),
            risk_level=risk_level,
            policy_reason=f"command timed out after {settings.shell_command_timeout_seconds}s.",
            truncated=False,
        )
