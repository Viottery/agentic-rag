from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def wait_for_health(base_url: str, timeout_seconds: int = 120) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5.0) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
            if response.status == 200 and data.get("status") == "ok":
                print("[ok] health check passed")
                return
            last_error = f"unexpected health response: {response.status} {body}"
        except Exception as exc:  # pragma: no cover - smoke script path
            last_error = str(exc)

        time.sleep(2)

    raise RuntimeError(f"health check did not become ready in time: {last_error}")


def call_chat(base_url: str, question: str) -> dict[str, Any]:
    payload = json.dumps({"question": question}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180.0) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat request failed: {exc.code} {error_body}") from exc

    data = json.loads(body)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def assert_common_shape(data: dict[str, Any]) -> None:
    required_keys = [
        "question",
        "subtasks",
        "planner_control",
        "current_task",
        "aggregated_context",
        "evidence",
        "retrieved_docs",
        "retrieved_sources",
        "used_tools",
        "answer_draft",
        "grounded_answer",
        "citations",
        "verification_result",
        "checker_result",
        "trace_summary",
        "iteration_count",
        "max_iterations",
        "error",
        "answer",
        "status",
    ]

    missing = [key for key in required_keys if key not in data]
    if missing:
        raise AssertionError(f"missing response keys: {missing}")

    if not isinstance(data["subtasks"], list):
        raise AssertionError("subtasks must be a list")
    if not isinstance(data["evidence"], list):
        raise AssertionError("evidence must be a list")
    if not isinstance(data["citations"], list):
        raise AssertionError("citations must be a list")
    if not isinstance(data["verification_result"], dict):
        raise AssertionError("verification_result must be a dict")
    if not isinstance(data["checker_result"], dict):
        raise AssertionError("checker_result must be a dict")
    if data["status"] != "finished":
        raise AssertionError(f"workflow did not finish, status={data['status']}")
    if not data["answer"]:
        raise AssertionError("final answer is empty")
    if not data["grounded_answer"]:
        raise AssertionError("grounded_answer is empty")
    if data["evidence"] and not data["citations"]:
        raise AssertionError("citations should not be empty when evidence exists")


def assert_task_type_present(data: dict[str, Any], expected_task_type: str) -> None:
    task_types = [task.get("task_type", "") for task in data.get("subtasks", [])]
    if expected_task_type not in task_types:
        raise AssertionError(
            f"expected task_type={expected_task_type}, actual task types={task_types}"
        )


def run_case(base_url: str, name: str, question: str, expected_task_type: str) -> None:
    print(f"\n=== Running case: {name} ===")
    print(f"question: {question}")
    data = call_chat(base_url, question)
    assert_common_shape(data)
    assert_task_type_present(data, expected_task_type)
    print(f"[ok] case passed: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end smoke test for /chat")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wait_for_health(args.base_url)

    cases = [
        (
            "rag",
            "请根据知识库内容介绍一下这个项目的目标和整体架构。",
            "rag",
        ),
        (
            "search",
            "请通过当前 search agent 获取信息，并基于结果给出简短总结；如果当前只是 mock 搜索，请明确说明限制。",
            "search",
        ),
        (
            "action",
            "请执行一个简单操作：把字符串 'agentic rag' 转成大写并返回结果。",
            "action",
        ),
    ]

    for name, question, expected_task_type in cases:
        run_case(args.base_url, name, question, expected_task_type)

    print("\n[ok] all end-to-end chat cases passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - smoke script path
        print(f"[error] {exc}", file=sys.stderr)
        raise
