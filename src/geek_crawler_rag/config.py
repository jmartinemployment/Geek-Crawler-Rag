from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mongo_crawler_url: str = "mongodb://localhost:27017"
    mongo_db_name: str = "geek_crawler"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "geek_crawler_chunks"
    qdrant_ad_templates_collection: str = "geek_ad_templates"
    qdrant_upsert_delay_seconds: float = 0.0

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    openai_embedding_tokens_per_minute: int = 700_000
    openai_embedding_max_batch_tokens: int = 50_000
    openai_embedding_max_retries: int = 8
    openai_embedding_retry_max_seconds: float = 60.0

    # Legacy uniform chunk defaults (still used as fallback knobs).
    chunk_size_tokens: int = 650
    chunk_overlap_tokens: int = 80

    # Parent/child indexing (Phase B2).
    child_chunk_size_tokens: int = 200
    child_chunk_overlap_tokens: int = 40
    parent_chunk_size_tokens: int = 1000
    parent_chunk_overlap_tokens: int = 80

    page_batch_size: int = 25
    embed_batch_size: int = 32

    # Durable smallest-first indexing scheduler.
    index_scheduler_enabled: bool = True
    index_scheduler_interval_seconds: int = 7_200
    index_scheduler_poll_seconds: int = 60
    index_job_lease_seconds: int = 900
    index_job_heartbeat_seconds: int = 60
    index_scheduler_max_attempts: int = 5
    index_scheduler_retry_seconds: int = 7_200

    # Hybrid retrieval + rerank (Phase B4/B5).
    hybrid_dense_limit: int = 30
    hybrid_lexical_limit: int = 30
    rerank_pool_size: int = 25
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-english-v3.0"
    rerank_enabled: bool = True

    # Optional shared secret for index/query (GeekAPI / gcc-v2). Empty = open (dev).
    api_key: str | None = None

    # Push index status to GeekAPI → SignalR (no UI polling).
    index_status_webhook_url: str | None = None
    index_status_webhook_key: str | None = None

    # Citeable generate (Phase 2) — OpenAI chat models; empty key → generate returns warning.
    openai_longform_model: str = "o3"
    openai_standard_model: str = "gpt-4o"
    generate_min_quality: float = 0.35
    generate_max_pages: int = 8
    generate_markdown_chars: int = 12000
    generate_enabled: bool = True

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
