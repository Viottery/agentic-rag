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
import app.rag.embeddings as embeddings_module


class _Settings:
    embedding_model = "test-model"
    embedding_backend = "auto"
    embedding_device = "auto"


def _clear_embedding_caches() -> None:
    embeddings_module.resolve_embedding_runtime.cache_clear()
    embeddings_module.get_embedding_model.cache_clear()


def test_resolve_embedding_runtime_prefers_openvino_gpu(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(embeddings_module, "_openvino_gpu_available", lambda: True)
    monkeypatch.setattr(embeddings_module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(embeddings_module, "_torch_xpu_available", lambda: True)
    _clear_embedding_caches()

    backend, device = embeddings_module.resolve_embedding_runtime()

    assert backend == "openvino"
    assert device == "GPU"


def test_resolve_embedding_runtime_falls_back_to_cuda_when_no_openvino(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(embeddings_module, "_openvino_gpu_available", lambda: False)
    monkeypatch.setattr(embeddings_module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(embeddings_module, "_torch_xpu_available", lambda: False)
    _clear_embedding_caches()

    backend, device = embeddings_module.resolve_embedding_runtime()

    assert backend == "torch"
    assert device == "cuda"


def test_resolve_embedding_runtime_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(embeddings_module, "_openvino_gpu_available", lambda: False)
    monkeypatch.setattr(embeddings_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(embeddings_module, "_torch_xpu_available", lambda: False)
    _clear_embedding_caches()

    backend, device = embeddings_module.resolve_embedding_runtime()

    assert backend == "torch"
    assert device == "cpu"


def test_resolve_embedding_runtime_honors_explicit_torch_cpu(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "get_settings", lambda: _Settings())
    _clear_embedding_caches()

    backend, device = embeddings_module.resolve_embedding_runtime("torch", "cpu")

    assert backend == "torch"
    assert device == "cpu"


def test_get_embedding_model_uses_openvino_backend(monkeypatch) -> None:
    calls: list[dict] = []

    class _FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(config_module, "get_settings", lambda: _Settings())
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    monkeypatch.setattr(embeddings_module, "_openvino_gpu_available", lambda: True)
    monkeypatch.setattr(embeddings_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(embeddings_module, "_torch_xpu_available", lambda: False)
    _clear_embedding_caches()

    embeddings_module.get_embedding_model()

    assert calls
    assert calls[0]["kwargs"]["backend"] == "openvino"
    assert calls[0]["kwargs"]["model_kwargs"] == {"device": "GPU"}
    model = embeddings_module.get_embedding_model()
    assert getattr(model, "_agentic_runtime_backend") == "openvino"
    assert getattr(model, "_agentic_runtime_device") == "GPU"


def test_get_embedding_model_falls_back_to_torch_when_openvino_init_fails(monkeypatch) -> None:
    calls: list[dict] = []

    class _FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            calls.append({"args": args, "kwargs": kwargs})
            if kwargs.get("backend") == "openvino":
                raise RuntimeError("openvino init failed")

    monkeypatch.setattr(config_module, "get_settings", lambda: _Settings())
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    monkeypatch.setattr(embeddings_module, "_openvino_gpu_available", lambda: True)
    monkeypatch.setattr(embeddings_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(embeddings_module, "_torch_xpu_available", lambda: False)
    _clear_embedding_caches()

    embeddings_module.get_embedding_model()

    assert len(calls) == 2
    assert calls[0]["kwargs"]["backend"] == "openvino"
    assert calls[1]["kwargs"]["device"] == "cpu"


def test_describe_active_embedding_runtime_reports_fallback_result(monkeypatch) -> None:
    class _FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            if kwargs.get("backend") == "openvino":
                raise RuntimeError("openvino init failed")

    monkeypatch.setattr(config_module, "get_settings", lambda: _Settings())
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    monkeypatch.setattr(embeddings_module, "_openvino_gpu_available", lambda: True)
    monkeypatch.setattr(embeddings_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(embeddings_module, "_torch_xpu_available", lambda: False)
    _clear_embedding_caches()

    runtime = embeddings_module.describe_active_embedding_runtime()

    assert runtime["backend"] == "torch"
    assert runtime["device"] == "cpu"
