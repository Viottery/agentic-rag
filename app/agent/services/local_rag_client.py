from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings


async def _send_envelope(
    envelope: dict[str, Any],
    *,
    socket_path: str,
) -> dict[str, Any]:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    writer.write((json.dumps(envelope, ensure_ascii=False) + "\n").encode("utf-8"))
    await writer.drain()
    raw_line = await reader.readline()
    writer.close()
    await writer.wait_closed()
    if not raw_line:
        raise RuntimeError("local rag service returned an empty response")
    return json.loads(raw_line.decode("utf-8"))


async def invoke_local_rag_client(
    payload: dict[str, Any],
    *,
    socket_path: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    resolved_socket_path = socket_path or settings.local_rag_socket_path
    response = await _send_envelope(
        {
            "command": "retrieve",
            "payload": payload,
        },
        socket_path=resolved_socket_path,
    )
    if not response.get("ok", False):
        raise RuntimeError(str(response.get("error", "local rag service failed")))
    return dict(response.get("result", {}))


async def health_local_rag_client(*, socket_path: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    resolved_socket_path = socket_path or settings.local_rag_socket_path
    response = await _send_envelope(
        {"command": "health"},
        socket_path=resolved_socket_path,
    )
    if not response.get("ok", False):
        raise RuntimeError(str(response.get("error", "local rag service failed")))
    return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Client for the local rag socket service.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    invoke_parser = subparsers.add_parser("invoke", help="Invoke local rag service")
    invoke_parser.add_argument("--request-file", required=True, help="Path to request JSON")
    invoke_parser.add_argument("--output-file", default="", help="Path to response JSON")
    invoke_parser.add_argument("--socket-path", default="", help="Override local rag socket path")

    health_parser = subparsers.add_parser("health", help="Check local rag service health")
    health_parser.add_argument("--socket-path", default="", help="Override local rag socket path")
    return parser


def _load_payload(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_payload(payload: dict[str, Any], output_path: str | None) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        Path(output_path).write_text(body, encoding="utf-8")
        return
    print(body)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "health":
        response = asyncio.run(
            health_local_rag_client(socket_path=args.socket_path or None),
        )
        _write_payload(response, None)
        return 0

    payload = _load_payload(args.request_file)
    result = asyncio.run(
        invoke_local_rag_client(
            payload,
            socket_path=args.socket_path or None,
        )
    )
    _write_payload(result, args.output_file or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
