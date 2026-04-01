from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_QUESTION = "请根据知识库内容介绍一下这个项目的目标和整体架构。"


@dataclass
class ProbeResult:
    index: int
    conversation_id: str
    question: str
    started_at: float
    finished_at: float
    status_code: int
    ok: bool
    response_status: str
    answer_preview: str
    output_full: str
    error: str
    raw_response: dict[str, Any]

    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "conversation_id": self.conversation_id,
            "question": self.question,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration,
            "status_code": self.status_code,
            "ok": self.ok,
            "response_status": self.response_status,
            "answer_preview": self.answer_preview,
            "output_full": self.output_full,
            "error": self.error,
            "raw_response": self.raw_response,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe whether /chat runs multiple conversations concurrently.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="FastAPI service base URL",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Number of concurrent conversations to launch",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Question to send. Repeat to provide one per conversation. If omitted, a default RAG question is used.",
    )
    parser.add_argument(
        "--mode",
        choices=["wait", "background"],
        default="wait",
        help="Chat request mode. Use wait to measure end-to-end concurrent processing directly.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval for background mode in seconds",
    )
    parser.add_argument(
        "--sequential-baseline",
        action="store_true",
        help="Also run the same requests sequentially for comparison.",
    )
    parser.add_argument(
        "--same-conversation",
        action="store_true",
        help="Reuse one conversation_id for all requests, useful for verifying serialization inside a single conversation.",
    )
    return parser


def _pick_question(index: int, questions: list[str]) -> str:
    if not questions:
        return DEFAULT_QUESTION
    if index < len(questions):
        return questions[index]
    return questions[index % len(questions)]


def _preview_answer(payload: dict[str, Any]) -> str:
    output = _extract_output_text(payload)
    if len(output) <= 100:
        return output
    return output[:99].rstrip() + "…"


def _extract_output_text(payload: dict[str, Any]) -> str:
    for key in (
        "answer",
        "grounded_answer",
        "answer_draft",
        "result",
        "observation",
        "aggregated_context",
        "error",
    ):
        value = payload.get(key, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _request_json(url: str, *, data: dict[str, Any] | None = None, timeout: float = 180.0) -> tuple[int, dict[str, Any]]:
    encoded = None
    headers = {"Accept": "application/json"}
    if data is not None:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=encoded, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = int(getattr(response, "status", 200))
            body = response.read().decode("utf-8")
            return status_code, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:  # noqa: BLE001
            payload = {"error": body}
        return int(exc.code), payload


async def _wait_for_background_job(
    *,
    base_url: str,
    job_id: str,
    timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    url = f"{base_url}/chat/jobs/{job_id}"
    while time.monotonic() < deadline:
        _, payload = await asyncio.to_thread(_request_json, url, timeout=timeout)
        if payload.get("status") in {"finished", "failed"}:
            return payload
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"background job {job_id} did not finish within {timeout:.1f}s")


async def _run_single_probe(
    *,
    base_url: str,
    index: int,
    question: str,
    timeout: float,
    mode: str,
    poll_interval: float,
    forced_conversation_id: str | None = None,
) -> ProbeResult:
    conversation_id = forced_conversation_id or f"probe-conv-{index + 1}-{uuid4().hex[:8]}"
    payload = {
        "question": question,
        "conversation_id": conversation_id,
        "mode": mode,
    }

    started_at = time.monotonic()
    response_status = "failed"
    answer_preview = ""
    output_full = ""
    error = ""
    status_code = 0
    ok = False
    raw_response: dict[str, Any] = {}

    try:
        status_code, body = await asyncio.to_thread(
            _request_json,
            f"{base_url}/chat",
            data=payload,
            timeout=timeout,
        )

        if status_code >= 400:
            raise RuntimeError(str(body))

        if mode == "background":
            job_id = str(body.get("job_id", "")).strip()
            if not job_id:
                raise RuntimeError(f"background mode did not return job_id: {body}")
            job_payload = await _wait_for_background_job(
                base_url=base_url,
                job_id=job_id,
                timeout=timeout,
                poll_interval=poll_interval,
            )
            result_payload = job_payload.get("result", {})
            ok = job_payload.get("status") == "finished"
            response_status = str(result_payload.get("status", job_payload.get("status", "")))
            answer_preview = _preview_answer(result_payload)
            output_full = _extract_output_text(result_payload)
            error = str(job_payload.get("error", ""))
            raw_response = dict(job_payload)
        else:
            ok = True
            response_status = str(body.get("status", ""))
            answer_preview = _preview_answer(body)
            output_full = _extract_output_text(body)
            error = str(body.get("error", ""))
            raw_response = dict(body)

    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        raw_response = {"error": error}

    finished_at = time.monotonic()
    return ProbeResult(
        index=index,
        conversation_id=conversation_id,
        question=question,
        started_at=started_at,
        finished_at=finished_at,
        status_code=status_code,
        ok=ok,
        response_status=response_status,
        answer_preview=answer_preview,
        output_full=output_full,
        error=error,
        raw_response=raw_response,
    )


async def _run_batch(
    *,
    base_url: str,
    count: int,
    questions: list[str],
    timeout: float,
    mode: str,
    poll_interval: float,
    concurrent: bool,
    same_conversation: bool,
) -> list[ProbeResult]:
    shared_conversation_id = f"probe-shared-{uuid4().hex[:8]}" if same_conversation else None
    if concurrent:
        tasks = [
            _run_single_probe(
                base_url=base_url,
                index=index,
                question=_pick_question(index, questions),
                timeout=timeout,
                mode=mode,
                poll_interval=poll_interval,
                forced_conversation_id=shared_conversation_id,
            )
            for index in range(count)
        ]
        return await asyncio.gather(*tasks)

    results: list[ProbeResult] = []
    for index in range(count):
        result = await _run_single_probe(
            base_url=base_url,
            index=index,
            question=_pick_question(index, questions),
            timeout=timeout,
            mode=mode,
            poll_interval=poll_interval,
            forced_conversation_id=shared_conversation_id,
        )
        results.append(result)
    return results


def _render_summary(label: str, results: list[ProbeResult], *, log_file: str | None = None) -> dict[str, Any]:
    if not results:
        return {
            "label": label,
            "count": 0,
            "wall_time_seconds": 0.0,
            "sum_individual_seconds": 0.0,
            "concurrency_factor": 0.0,
            "overlap_detected": False,
            "log_file": log_file or "",
            "results": [],
        }

    started_at = min(item.started_at for item in results)
    finished_at = max(item.finished_at for item in results)
    wall_time = finished_at - started_at
    sum_individual = sum(item.duration for item in results)
    concurrency_factor = (sum_individual / wall_time) if wall_time > 0 else 0.0

    return {
        "label": label,
        "count": len(results),
        "wall_time_seconds": round(wall_time, 3),
        "sum_individual_seconds": round(sum_individual, 3),
        "concurrency_factor": round(concurrency_factor, 3),
        "overlap_detected": concurrency_factor > 1.25,
        "log_file": log_file or "",
        "results": [
            {
                "index": item.index,
                "conversation_id": item.conversation_id,
                "question": item.question,
                "duration_seconds": round(item.duration, 3),
                "status_code": item.status_code,
                "ok": item.ok,
                "response_status": item.response_status,
                "error": item.error,
                "answer_preview": item.answer_preview,
                "output_full": item.output_full,
            }
            for item in results
        ],
    }


def _write_log_file(
    *,
    label: str,
    results: list[ProbeResult],
    directory: Path,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}.json"
    payload = {
        "label": label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [item.to_log_dict() for item in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _main_async(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    log_dir = Path(tempfile.mkdtemp(prefix="chat-probe-logs-"))

    concurrent_results = await _run_batch(
        base_url=base_url,
        count=args.count,
        questions=args.question,
        timeout=args.timeout,
        mode=args.mode,
        poll_interval=args.poll_interval,
        concurrent=True,
        same_conversation=args.same_conversation,
    )
    concurrent_log = _write_log_file(
        label="concurrent",
        results=concurrent_results,
        directory=log_dir,
    )
    print(
        json.dumps(
            _render_summary(
                "concurrent",
                concurrent_results,
                log_file=str(concurrent_log),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.sequential_baseline:
        sequential_results = await _run_batch(
            base_url=base_url,
            count=args.count,
            questions=args.question,
            timeout=args.timeout,
            mode=args.mode,
            poll_interval=args.poll_interval,
            concurrent=False,
            same_conversation=args.same_conversation,
        )
        sequential_log = _write_log_file(
            label="sequential",
            results=sequential_results,
            directory=log_dir,
        )
        print(
            json.dumps(
                _render_summary(
                    "sequential",
                    sequential_results,
                    log_file=str(sequential_log),
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(json.dumps({"log_dir": str(log_dir)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
