from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any


logger = logging.getLogger(__name__)

SUPPORTED_RERANKER_DEVICES = {"auto", "cpu", "cuda", "xpu", "gpu"}


def _normalize_device_name(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized not in SUPPORTED_RERANKER_DEVICES:
        raise ValueError(
            f"不支持的 reranker device: {value!r}。"
            f"可选值: {', '.join(sorted(SUPPORTED_RERANKER_DEVICES))}"
        )
    return normalized


def _load_torch_module() -> Any | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _torch_cuda_available() -> bool:
    torch = _load_torch_module()
    return bool(torch and hasattr(torch, "cuda") and torch.cuda.is_available())


def _torch_xpu_available() -> bool:
    torch = _load_torch_module()
    return bool(torch and hasattr(torch, "xpu") and torch.xpu.is_available())


def _default_torch_device() -> str:
    if _torch_cuda_available():
        return "cuda"
    if _torch_xpu_available():
        return "xpu"
    return "cpu"


@lru_cache(maxsize=8)
def resolve_reranker_device(preferred_device: str | None = None) -> str:
    from app.core.config import get_settings

    settings = get_settings()
    device = _normalize_device_name(preferred_device or settings.reranker_device)

    if device == "auto":
        return _default_torch_device()
    if device == "gpu":
        if _torch_cuda_available():
            return "cuda"
        if _torch_xpu_available():
            return "xpu"
        return "cpu"
    return device


@lru_cache(maxsize=8)
def get_reranker_model(
    model_name: str | None = None,
    device: str | None = None,
) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 sentence-transformers 依赖，无法执行 reranker。"
            "请先安装 requirements.txt 中的依赖。"
        ) from exc

    from app.core.config import get_settings

    settings = get_settings()
    resolved_model_name = (model_name or settings.reranker_model).strip()
    resolved_device = resolve_reranker_device(device)

    return CrossEncoder(
        resolved_model_name,
        device=resolved_device,
    )


def rerank_pairs(
    query: str,
    candidates: list[str],
    *,
    model_name: str | None = None,
    device: str | None = None,
) -> list[float]:
    if not query.strip() or not candidates:
        return []

    model = get_reranker_model(model_name, device)
    pairs = [[query, candidate] for candidate in candidates]
    scores = model.predict(pairs)

    if hasattr(scores, "tolist"):
        return [float(item) for item in scores.tolist()]
    return [float(item) for item in scores]
