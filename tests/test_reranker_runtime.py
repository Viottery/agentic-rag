from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "pydantic_settings",
    types.SimpleNamespace(BaseSettings=object, SettingsConfigDict=lambda **kwargs: kwargs),
)

import app.core.config as config_module
import app.rag.reranker as reranker_module


class _Settings:
    reranker_model = "test-reranker"
    reranker_device = "auto"


def _clear_reranker_caches() -> None:
    reranker_module.resolve_reranker_device.cache_clear()
    reranker_module.get_reranker_model.cache_clear()


def test_resolve_reranker_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(reranker_module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(reranker_module, "_torch_xpu_available", lambda: False)
    _clear_reranker_caches()

    assert reranker_module.resolve_reranker_device() == "cuda"


def test_resolve_reranker_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(reranker_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(reranker_module, "_torch_xpu_available", lambda: False)
    _clear_reranker_caches()

    assert reranker_module.resolve_reranker_device() == "cpu"


def test_rerank_pairs_uses_cross_encoder_predict(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _FakeCrossEncoder:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            calls.append({"args": args, "kwargs": kwargs})

        def predict(self, pairs):  # noqa: ANN001, ANN201
            assert pairs == [["query", "doc-a"], ["query", "doc-b"]]
            return [0.2, 0.8]

    monkeypatch.setattr(config_module, "get_settings", lambda: _Settings())
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder),
    )
    monkeypatch.setattr(reranker_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(reranker_module, "_torch_xpu_available", lambda: False)
    _clear_reranker_caches()

    scores = reranker_module.rerank_pairs("query", ["doc-a", "doc-b"])

    assert scores == [0.2, 0.8]
    assert calls
    assert calls[0]["kwargs"]["device"] == "cpu"
