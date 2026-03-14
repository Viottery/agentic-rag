from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = Any


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    返回全局复用的 embedding 模型实例。

    使用缓存避免重复加载模型。
    """
    try:
        from sentence_transformers import SentenceTransformer as SentenceTransformerModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 sentence-transformers 依赖，无法执行 embedding。"
            "请先安装 requirements.txt 中的依赖。"
        ) from exc

    return SentenceTransformerModel(model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    对一组文本生成 embedding。
    """
    if not texts:
        return []

    model = get_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """
    对单条查询生成 embedding。
    """
    if not query.strip():
        return []

    model = get_embedding_model()
    vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return vector.tolist()
