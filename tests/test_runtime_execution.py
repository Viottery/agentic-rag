from __future__ import annotations

from pathlib import Path
import sys

import app.runtime.platform as platform_module
import app.runtime.shell_providers as shell_provider_module
from app.runtime.platform import build_local_rag_endpoint, default_local_rag_transport
from app.runtime.process_runner import run_process
from app.runtime.shell_providers import resolve_shell_provider


def test_default_local_rag_transport_uses_tcp_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(platform_module, "is_windows", lambda: True)

    assert default_local_rag_transport("auto") == "tcp"


def test_default_local_rag_transport_uses_unix_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(platform_module, "is_windows", lambda: False)

    assert default_local_rag_transport("auto") == "unix"


def test_build_local_rag_endpoint_keeps_explicit_tcp_settings() -> None:
    endpoint = build_local_rag_endpoint(
        transport="tcp",
        socket_path="",
        host="0.0.0.0",
        port=9001,
    )

    assert endpoint.transport == "tcp"
    assert endpoint.host == "0.0.0.0"
    assert endpoint.port == 9001


def test_shell_provider_auto_uses_bash_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(shell_provider_module, "is_windows", lambda: False)
    monkeypatch.setattr(shell_provider_module.shutil, "which", lambda name: f"/mock/{name}")

    provider = resolve_shell_provider(provider_name="auto", shell_program="")

    assert provider.name == "bash"
    assert provider.program == "/mock/bash"
    assert provider.spawn_args("printf ok") == ["/mock/bash", "-lc", "printf ok"]


def test_shell_provider_auto_uses_powershell_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(shell_provider_module, "is_windows", lambda: True)
    monkeypatch.setattr(
        shell_provider_module.shutil,
        "which",
        lambda name: f"/mock/{name}" if name == "pwsh" else None,
    )

    provider = resolve_shell_provider(provider_name="auto", shell_program="")

    assert provider.name == "powershell"
    assert provider.program == "/mock/pwsh"
    assert provider.spawn_args("Write-Output ok") == [
        "/mock/pwsh",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "Write-Output ok",
    ]


def test_shell_provider_uses_explicit_program_for_selected_provider() -> None:
    provider = resolve_shell_provider(provider_name="powershell", shell_program="C:/pwsh/pwsh.exe")

    assert provider.name == "powershell"
    assert provider.program == "C:/pwsh/pwsh.exe"


def test_process_runner_executes_direct_program_without_shell() -> None:
    result = run_process(
        [sys.executable, "-c", "print('process-runner-ok')"],
        cwd=Path.cwd(),
        timeout_seconds=10,
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stdout.strip() == "process-runner-ok"
