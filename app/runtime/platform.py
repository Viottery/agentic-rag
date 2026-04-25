from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import tempfile


def platform_name() -> str:
    return platform.system().lower()


def is_windows() -> bool:
    return platform_name().startswith("win")


def default_workspace_root(configured: str | None = None) -> Path:
    cleaned = (configured or "").strip()
    if cleaned:
        candidate = Path(cleaned).expanduser()
        if candidate.exists():
            return candidate.resolve()
        if cleaned == "/workspace":
            return Path.cwd().resolve()
        return candidate.resolve()

    env_root = os.getenv("AGENTIC_RAG_WORKSPACE_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.cwd().resolve()


def default_local_rag_transport(configured: str | None = None) -> str:
    cleaned = (configured or "auto").strip().lower()
    if cleaned in {"unix", "tcp"}:
        return cleaned
    return "tcp" if is_windows() else "unix"


def default_local_rag_socket_path(configured: str | None = None) -> Path:
    cleaned = (configured or "").strip()
    if cleaned:
        return Path(cleaned).expanduser()
    return Path(tempfile.gettempdir()) / "agentic-rag-local-rag.sock"


@dataclass(frozen=True)
class LocalRAGEndpoint:
    transport: str
    socket_path: Path
    host: str
    port: int


def build_local_rag_endpoint(
    *,
    transport: str,
    socket_path: str,
    host: str,
    port: int,
) -> LocalRAGEndpoint:
    resolved_transport = default_local_rag_transport(transport)
    return LocalRAGEndpoint(
        transport=resolved_transport,
        socket_path=default_local_rag_socket_path(socket_path),
        host=(host or "127.0.0.1").strip() or "127.0.0.1",
        port=int(port or 8765),
    )
