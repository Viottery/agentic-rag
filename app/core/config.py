from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic RAG"
    app_env: str = "dev"
    debug: bool = True
    qdrant_mode: str = "server"
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_local_path: str = "data/qdrant-local"
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_backend: str = "torch"
    embedding_device: str = "cpu"
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "auto"
    reranker_max_candidates: int = 12
    agent_max_iterations: int = 6
    agent_max_duration_seconds: int = 90
    agent_max_concurrent_conversations: int = 4
    conversation_store_path: str = "data/memory/conversations.db"
    conversation_recent_turns: int = 4
    conversation_summary_turns: int = 8
    conversation_summary_max_chars: int = 2400
    conversation_memory_candidate_limit: int = 3
    conversation_memory_recall_limit: int = 4
    conversation_memory_note_max_chars: int = 280
    local_rag_transport: str = "auto"
    local_rag_socket_path: str = ""
    local_rag_host: str = "127.0.0.1"
    local_rag_port: int = 8765
    local_rag_enabled: bool = True
    local_rag_retrieve_workers: int = 2
    local_rag_embedding_batch_size: int = 8
    local_rag_rerank_batch_tasks: int = 4
    local_rag_subprocess_timeout_seconds: int = 150
    shell_runtime_enabled: bool = True
    shell_provider: str = "auto"
    shell_program: str = ""
    shell_policy_mode: str = "workspace-write"
    shell_workspace_root: str = ""
    shell_allowed_extra_roots: str = ""
    shell_protected_paths: str = ".git,.env,data/memory/conversations.db"
    shell_allow_destructive_commands: bool = False
    shell_approval_mode: str = "high-risk"
    shell_approval_ttl_seconds: int = 900
    shell_command_timeout_seconds: int = 60
    shell_max_output_chars: int = 6000
    tavily_api_key: str = ""
    tavily_search_depth: str = "basic"
    tavily_max_results: int = 5
    tavily_extract_top_k: int = 2
    tavily_extract_depth: str = "basic"
    tavily_chunk_size: int = 900
    tavily_chunk_overlap: int = 120
    tavily_max_chunks_per_search: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
