from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "pydantic_settings",
    types.SimpleNamespace(BaseSettings=object, SettingsConfigDict=lambda **kwargs: kwargs),
)

import app.runtime.shell_runtime as shell_runtime_module
from app.runtime.shell_runtime import (
    get_pending_shell_approval,
    reject_shell_approval,
    run_shell_command,
)


def _settings(**overrides):
    values = {
        "shell_runtime_enabled": True,
        "shell_program": "bash",
        "shell_policy_mode": "workspace-write",
        "shell_workspace_root": str(Path.cwd()),
        "shell_allowed_extra_roots": "",
        "shell_protected_paths": ".git,.env,data/memory/conversations.db",
        "shell_allow_destructive_commands": False,
        "shell_approval_mode": "high-risk",
        "shell_approval_ttl_seconds": 900,
        "shell_command_timeout_seconds": 10,
        "shell_max_output_chars": 6000,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


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


def test_shell_runtime_blocks_cwd_outside_workspace(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    monkeypatch.setattr(
        shell_runtime_module,
        "get_settings",
        lambda: _settings(shell_workspace_root=str(workspace)),
    )

    result = run_shell_command("pwd", cwd=str(outside))

    assert result.allowed is False
    assert "outside" in result.policy_reason
    assert "cwd_outside_workspace" in (result.policy_violations or [])


def test_shell_runtime_blocks_path_reference_outside_workspace(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        shell_runtime_module,
        "get_settings",
        lambda: _settings(shell_workspace_root=str(workspace)),
    )

    result = run_shell_command("cat /etc/passwd", cwd=str(workspace))

    assert result.allowed is False
    assert "outside" in result.policy_reason
    assert "path_outside_workspace" in (result.policy_violations or [])


def test_shell_runtime_blocks_protected_workspace_path(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("SECRET=1\n", encoding="utf-8")
    monkeypatch.setattr(
        shell_runtime_module,
        "get_settings",
        lambda: _settings(shell_workspace_root=str(workspace)),
    )

    result = run_shell_command("cat .env", cwd=str(workspace))

    assert result.allowed is False
    assert "protected" in result.policy_reason
    assert "protected_path" in (result.policy_violations or [])


def test_shell_runtime_blocks_python_path_call_to_protected_path(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("SECRET=1\n", encoding="utf-8")
    monkeypatch.setattr(
        shell_runtime_module,
        "get_settings",
        lambda: _settings(shell_workspace_root=str(workspace)),
    )

    result = run_shell_command(
        "python -c \"from pathlib import Path; print(Path('.env').read_text())\"",
        cwd=str(workspace),
    )

    assert result.allowed is False
    assert "protected" in result.policy_reason
    assert "protected_path" in (result.policy_violations or [])


def test_shell_runtime_allows_workspace_write_and_records_it(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        shell_runtime_module,
        "get_settings",
        lambda: _settings(shell_workspace_root=str(workspace)),
    )

    result = run_shell_command("printf 'ok' > notes.txt", cwd=str(workspace))

    assert result.allowed is True
    assert result.exit_code == 0
    assert result.write_detected is True
    assert str(workspace / "notes.txt") in (result.touched_paths or [])
    assert (workspace / "notes.txt").read_text(encoding="utf-8") == "ok"


def test_shell_runtime_blocks_write_in_read_only_mode(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        shell_runtime_module,
        "get_settings",
        lambda: _settings(
            shell_workspace_root=str(workspace),
            shell_policy_mode="read-only",
        ),
    )

    result = run_shell_command("printf 'ok' > notes.txt", cwd=str(workspace))

    assert result.allowed is False
    assert result.write_detected is True
    assert "write-capable" in result.policy_reason
    assert not (workspace / "notes.txt").exists()


def test_shell_runtime_requires_approval_for_destructive_commands_in_workspace(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "notes.txt"
    target.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(
        shell_runtime_module,
        "get_settings",
        lambda: _settings(shell_workspace_root=str(workspace)),
    )

    result = run_shell_command("rm notes.txt", cwd=str(workspace))

    assert result.allowed is False
    assert result.exit_code == -3
    assert result.approval_required is True
    assert result.approval_id
    assert "approval required" in result.policy_reason
    assert "destructive_command" in (result.policy_violations or [])
    assert "approval_required" in (result.policy_violations or [])
    assert str(target) in (result.touched_paths or [])
    assert target.exists()
    assert reject_shell_approval(result.approval_id) is True


def test_shell_runtime_executes_destructive_command_after_approval(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "notes.txt"
    target.write_text("remove me", encoding="utf-8")
    monkeypatch.setattr(
        shell_runtime_module,
        "get_settings",
        lambda: _settings(shell_workspace_root=str(workspace)),
    )

    pending = run_shell_command("rm notes.txt", cwd=str(workspace))
    approval = get_pending_shell_approval(pending.approval_id)
    assert approval is not None
    assert approval["command"] == "rm notes.txt"

    result = run_shell_command(
        str(approval["command"]),
        cwd=str(approval["cwd"]),
        approval_id=pending.approval_id,
    )

    assert result.allowed is True
    assert result.exit_code == 0
    assert result.risk_level == "high"
    assert result.policy_reason == "allowed by approval"
    assert result.write_detected is True
    assert str(target) in (result.touched_paths or [])
    assert not target.exists()
    assert get_pending_shell_approval(pending.approval_id) is None
