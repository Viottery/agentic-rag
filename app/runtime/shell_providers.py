from __future__ import annotations

from dataclasses import dataclass
import os
import shutil

from app.runtime.platform import is_windows


@dataclass(frozen=True)
class ShellProvider:
    name: str
    program: str
    args: tuple[str, ...]

    def spawn_args(self, command: str) -> list[str]:
        return [self.program, *self.args, command]


def _first_available(candidates: list[str], fallback: str) -> str:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return fallback


def resolve_shell_provider(
    *,
    provider_name: str | None,
    shell_program: str | None,
) -> ShellProvider:
    requested = (provider_name or "auto").strip().lower()
    configured_program = (shell_program or "").strip()

    if requested in {"auto", ""}:
        requested = "powershell" if is_windows() else "bash"

    if requested in {"powershell", "pwsh"}:
        program = configured_program or _first_available(
            ["pwsh", "powershell.exe", "powershell"],
            "powershell.exe" if os.name == "nt" else "pwsh",
        )
        return ShellProvider(
            name="powershell",
            program=program,
            args=("-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"),
        )

    if requested == "bash":
        program = configured_program or _first_available(["bash"], "bash")
        return ShellProvider(name="bash", program=program, args=("-lc",))

    raise ValueError(f"unsupported shell provider: {provider_name}")
