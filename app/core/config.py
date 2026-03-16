from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic RAG"
    app_env: str = "dev"
    debug: bool = True
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    agent_max_iterations: int = 6
    agent_max_duration_seconds: int = 90
    tavily_api_key: str = ""
    tavily_search_depth: str = "basic"
    tavily_max_results: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
