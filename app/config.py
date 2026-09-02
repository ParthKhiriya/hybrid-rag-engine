from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"

    llm_provider: str = "local"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "hybrid_rag_chunks"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "hybrid-rag-demo"
    neo4j_database: str = "neo4j"

    embedding_dim: int = 384
    top_k_vector: int = 8
    top_k_graph: int = 8
    rrf_k: int = 60
    graph_max_hops: int = 3
    allow_in_memory_fallback: bool = True

    otel_enabled: bool = False
    otel_service_name: str = "hybrid-rag-engine"

    # RBAC: comma-separated API keys per role. Empty by default — no key
    # grants any access until explicitly configured. Higher roles should
    # NOT be included in lower-role lists; resolve_role() already checks
    # admin > writer > reader so a single admin key works everywhere.
    api_keys_reader: str = ""
    api_keys_writer: str = ""
    api_keys_admin: str = ""

    # Content safety
    block_quarantined_ingest: bool = False  # if True, reject flagged docs outright
    exclude_quarantined_from_context: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
