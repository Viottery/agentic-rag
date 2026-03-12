import os

from langchain_openai import ChatOpenAI


def get_chat_model() -> ChatOpenAI:
    """
    返回项目使用的 Chat 模型实例。

    通过环境变量控制：
    - OPENAI_API_KEY
    - OPENAI_BASE_URL
    - OPENAI_MODEL
    - OPENAI_TEMPERATURE
    """

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )