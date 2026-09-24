from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    mongo_crawler_url: str = "mongodb://localhost:27017"
    mongo_db_name: str = "geek_crawler"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "geek_crawler_chunks"
    qdrant_ad_templates_collection: str = "geek_ad_templates"
    qdrant_upsert_delay_seconds: float = 0.5

    # Hybrid retrieval over the named sparse vector. Off by default, and deliberately a setting
    # rather than a constant: turning it on requires the collection to carry the sparse vector AND
    # its points to carry values (scripts/migrate_sparse_vectors.py). Flipping it before the
    # backfill scores the sparse branch against vectors nothing wrote, and _query_hybrid converts
    # any resulting error into chunks=[] on HTTP 200 -- an empty corpus the writer cannot tell from
    # a real one. That is the failure this setting exists to keep out of the default path.
    hybrid_retrieval_enabled: bool = False

    # Sparse (SPLADE) model for the hybrid branch. One setting because three places have to agree:
    # llama_engine encodes queries with it, scripts/migrate_sparse_vectors.py encodes the backfill
    # with it, and the indexer encodes new chunks with it. Two different SPLADE models produce
    # weights over different vocabularies, so a mismatch does not error -- it scores query tokens
    # against an index built from other tokens and quietly returns the wrong passages.
    sparse_model: str = "prithivida/Splade_PP_en_v1"

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    # Stay well under the account ceiling. Setting this AT the limit
    # (1,000,000 TPM for text-embedding-3-small) caused sustained runs to
    # return HTTP 500 server_error instead of clean 429s — see
    # plans/embedding-cache-and-duplicate-results.md.
    openai_embedding_tokens_per_minute: int = 400_000
    openai_embedding_max_batch_tokens: int = 50_000
    # Embedding calls are fail-closed (no in-process OpenAI retries).
    openai_embedding_max_retries: int = 0
    openai_embedding_retry_max_seconds: float = 0.0
    embedding_quarantine_dir: str = (
        "/var/lib/geek-crawler-rag/embedding_quarantine"
    )

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
    index_scheduler_interval_seconds: int = 300
    index_scheduler_poll_seconds: int = 60
    index_job_lease_seconds: int = 900
    index_job_heartbeat_seconds: int = 60
    index_scheduler_max_attempts: int = 5
    index_scheduler_retry_seconds: int = 300

    # Hybrid retrieval + rerank (Phase B4/B5).
    hybrid_dense_limit: int = 30
    hybrid_lexical_limit: int = 30
    rerank_pool_size: int = 40
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-english-v3.0"
    rerank_enabled: bool = True

    # Service authentication is mandatory unless an explicit test-only mode is set.
    api_key: str | None = None
    local_test_mode: bool = False
    crawler_owner_id: str = "system:crawler"
    crawler_visibility: str = "service"
    # Comma-separated hosts whose indexed pages default to sourceRights=consented
    # (P1.5 smoke corpus: ApprovalMax / Plooto). All others default unknown.
    source_rights_consented_hosts: str = "approvalmax.com,plooto.com"

    # GeekAPI-owned manifest signing keys. No manifest/catalog authority is stored here.
    context_manifest_signing_keys: dict[str, str] = {}
    context_asset_indexing_enabled: bool = True
    context_manifest_query_enabled: bool = True

    # Push index status to GeekAPI → SignalR (no UI polling).
    index_status_webhook_url: str | None = None
    index_status_webhook_key: str | None = None

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
