from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "pydantic_settings",
    types.SimpleNamespace(BaseSettings=object, SettingsConfigDict=lambda **kwargs: kwargs),
)

from app.runtime.shell_runtime import run_shell_command


def test_shell_runtime_blocks_obviously_dangerous_command() -> None:
    result = run_shell_command("sudo rm -rf /")

    assert result.allowed is False
    assert result.exit_code == -1
    assert "blocked" in result.policy_reason


def test_shell_runtime_executes_simple_read_only_command() -> None:
    result = run_shell_command("printf 'agentic-rag'")

    assert result.allowed is True
    assert result.exit_code == 0
    assert "agentic-rag" in result.stdout
