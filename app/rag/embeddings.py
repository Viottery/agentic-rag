from __future__ import annotations

from functools import lru_cache
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = Any


logger = logging.getLogger(__name__)

SUPPORTED_EMBEDDING_BACKENDS = {"auto", "torch", "openvino"}
SUPPORTED_EMBEDDING_DEVICES = {"auto", "gpu", "cpu", "cuda", "xpu"}


def _normalize_backend_name(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized not in SUPPORTED_EMBEDDING_BACKENDS:
        raise ValueError(
            f"不支持的 embedding backend: {value!r}。"
            f"可选值: {', '.join(sorted(SUPPORTED_EMBEDDING_BACKENDS))}"
        )
    return normalized


def _normalize_device_name(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized not in SUPPORTED_EMBEDDING_DEVICES:
        raise ValueError(
            f"不支持的 embedding device: {value!r}。"
            f"可选值: {', '.join(sorted(SUPPORTED_EMBEDDING_DEVICES))}"
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


def _openvino_gpu_available() -> bool:
    try:
        from openvino import Core
    except ModuleNotFoundError:
        return False

    try:
        available_devices = {str(device).upper() for device in Core().available_devices}
    except Exception:
        return False

    return any(device.startswith("GPU") for device in available_devices)


def _default_torch_device() -> str:
    if _torch_cuda_available():
        return "cuda"
    if _torch_xpu_available():
        return "xpu"
    return "cpu"


@lru_cache(maxsize=8)
def resolve_embedding_runtime(
    preferred_backend: str | None = None,
    preferred_device: str | None = None,
) -> tuple[str, str]:
    """
    解析当前 embedding 推理后端与设备偏好。

    这里返回的是“应优先尝试的 runtime”，不是最终保证成功的 runtime。
    某些模型即便探测到 Intel GPU 可用，也可能在 OpenVINO 导出或 GPU 编译阶段失败，
    此时 `get_embedding_model()` 会负责做实际回退。

    优先级：
    1. Intel GPU: OpenVINO GPU
    2. Nvidia GPU: Torch CUDA
    3. 其他可用 Torch 设备，例如 XPU
    4. CPU
    """
    from app.core.config import get_settings

    settings = get_settings()
    backend = _normalize_backend_name(preferred_backend or settings.embedding_backend)
    device = _normalize_device_name(preferred_device or settings.embedding_device)

    if backend == "openvino":
        if device in {"auto", "gpu"}:
            return ("openvino", "GPU" if _openvino_gpu_available() else "CPU")
        if device == "cpu":
            return ("openvino", "CPU")
        raise ValueError("OpenVINO backend 仅支持 device=auto|gpu|cpu。")

    if backend == "torch":
        if device == "auto":
            return ("torch", _default_torch_device())
        if device == "gpu":
            if _torch_cuda_available():
                return ("torch", "cuda")
            if _torch_xpu_available():
                return ("torch", "xpu")
            return ("torch", "cpu")
        return ("torch", device)

    if device == "cpu":
        return ("torch", "cpu")
    if device == "cuda":
        return ("torch", "cuda")
    if device == "xpu":
        return ("torch", "xpu")
    if device == "gpu":
        if _openvino_gpu_available():
            return ("openvino", "GPU")
        if _torch_cuda_available():
            return ("torch", "cuda")
        if _torch_xpu_available():
            return ("torch", "xpu")
        return ("torch", "cpu")

    if _openvino_gpu_available():
        return ("openvino", "GPU")
    return ("torch", _default_torch_device())


@lru_cache(maxsize=12)
def get_embedding_model(
    model_name: str | None = None,
    backend: str | None = None,
    device: str | None = None,
) -> SentenceTransformer:
    """
    返回全局复用的 embedding 模型实例。

    根据当前环境自动尝试 Intel GPU / Nvidia / CPU，
    同时允许通过配置或显式参数覆盖默认行为。

    注意：
    - `resolve_embedding_runtime()` 只负责设备探测与优先级决策
    - 真正的模型初始化可能因导出、编译或驱动问题失败
    - 当 backend 处于 auto 模式时，这里会自动回退到 Torch
    """
    try:
        from sentence_transformers import SentenceTransformer as SentenceTransformerModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 sentence-transformers 依赖，无法执行 embedding。"
            "请先安装 requirements.txt 中的依赖。"
        ) from exc

    from app.core.config import get_settings

    settings = get_settings()
    resolved_model_name = (model_name or settings.embedding_model).strip()
    resolved_backend, resolved_device = resolve_embedding_runtime(backend, device)

    if resolved_backend == "openvino":
        try:
            return SentenceTransformerModel(
                resolved_model_name,
                backend="openvino",
                model_kwargs={"device": resolved_device},
            )
        except Exception as exc:
            if backend not in {None, "", "auto"}:
                raise
            logger.warning(
                "OpenVINO backend init failed for model %s on device %s; "
                "falling back to Torch. Reason: %s",
                resolved_model_name,
                resolved_device,
                exc,
            )
            resolved_backend, resolved_device = ("torch", _default_torch_device())

    return SentenceTransformerModel(
        resolved_model_name,
        device=resolved_device,
    )


def embed_texts(
    texts: list[str],
    *,
    model_name: str | None = None,
    backend: str | None = None,
    device: str | None = None,
) -> list[list[float]]:
    """
    对一组文本生成 embedding。
    """
    if not texts:
        return []

    model = get_embedding_model(model_name, backend, device)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(
    query: str,
    *,
    model_name: str | None = None,
    backend: str | None = None,
    device: str | None = None,
) -> list[float]:
    """
    对单条查询生成 embedding。
    """
    if not query.strip():
        return []

    model = get_embedding_model(model_name, backend, device)
    vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return vector.tolist()
