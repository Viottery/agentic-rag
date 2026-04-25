from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.runtime.platform import build_local_rag_endpoint


async def _send_envelope(
    envelope: dict[str, Any],
    *,
    transport: str,
    socket_path: str,
    host: str,
    port: int,
) -> dict[str, Any]:
    if transport == "tcp":
        reader, writer = await asyncio.open_connection(host, port)
    else:
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
    transport: str | None = None,
    socket_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    endpoint = build_local_rag_endpoint(
        transport=transport or settings.local_rag_transport,
        socket_path=socket_path or settings.local_rag_socket_path,
        host=host or settings.local_rag_host,
        port=port or settings.local_rag_port,
    )
    response = await _send_envelope(
        {
            "command": "retrieve",
            "payload": payload,
        },
        transport=endpoint.transport,
        socket_path=str(endpoint.socket_path),
        host=endpoint.host,
        port=endpoint.port,
    )
    if not response.get("ok", False):
        raise RuntimeError(str(response.get("error", "local rag service failed")))
    return dict(response.get("result", {}))


async def health_local_rag_client(
    *,
    transport: str | None = None,
    socket_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    endpoint = build_local_rag_endpoint(
        transport=transport or settings.local_rag_transport,
        socket_path=socket_path or settings.local_rag_socket_path,
        host=host or settings.local_rag_host,
        port=port or settings.local_rag_port,
    )
    response = await _send_envelope(
        {"command": "health"},
        transport=endpoint.transport,
        socket_path=str(endpoint.socket_path),
        host=endpoint.host,
        port=endpoint.port,
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
    invoke_parser.add_argument("--transport", default="", choices=["", "auto", "unix", "tcp"], help="Override local rag transport")
    invoke_parser.add_argument("--socket-path", default="", help="Override local rag socket path")
    invoke_parser.add_argument("--host", default="", help="Override local rag TCP host")
    invoke_parser.add_argument("--port", type=int, default=0, help="Override local rag TCP port")

    health_parser = subparsers.add_parser("health", help="Check local rag service health")
    health_parser.add_argument("--transport", default="", choices=["", "auto", "unix", "tcp"], help="Override local rag transport")
    health_parser.add_argument("--socket-path", default="", help="Override local rag socket path")
    health_parser.add_argument("--host", default="", help="Override local rag TCP host")
    health_parser.add_argument("--port", type=int, default=0, help="Override local rag TCP port")
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
            health_local_rag_client(
                transport=args.transport or None,
                socket_path=args.socket_path or None,
                host=args.host or None,
                port=args.port or None,
            ),
        )
        _write_payload(response, None)
        return 0

    payload = _load_payload(args.request_file)
    result = asyncio.run(
        invoke_local_rag_client(
            payload,
            transport=args.transport or None,
            socket_path=args.socket_path or None,
            host=args.host or None,
            port=args.port or None,
        )
    )
    _write_payload(result, args.output_file or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
