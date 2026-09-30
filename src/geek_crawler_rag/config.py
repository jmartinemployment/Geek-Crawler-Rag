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
    # Zero. This fired after every flush, and at batch 64 on a large run that is
    # tens of minutes of pure sleeping for nothing measured: the VPS profile put
    # the api container at 86% of ONE core against a 3.0-core limit with Qdrant
    # at 0.17%, so Qdrant was never the thing needing to be slowed down. The
    # guard in indexer.py stays, so this remains an escape hatch if Qdrant ever
    # does need backpressure - it just is not paid by default any more.
    qdrant_upsert_delay_seconds: float = 0.0

    # Hybrid retrieval over the named sparse vector.
    #
    # On. The precondition it guards is met by construction rather than by backfill: the collection
    # is created with both vectors (QdrantStore.ensure_collection) and every chunk indexed into it
    # gets SPLADE weights at write time, so there is no window in which the sparse branch queries
    # values nothing wrote. That window is what this flag was false for.
    #
    # It stays a setting because the guarantee is per collection, not per deploy. Pointed at a
    # collection created before the unified schema -- one with no "text-sparse" vector, or with the
    # vector but unpopulated points -- a hybrid query errors, and _query_hybrid converts that into
    # chunks=[] on HTTP 200, an empty corpus the writer cannot tell from a real one. Set
    # HYBRID_RETRIEVAL_ENABLED=false to fall back to dense without a deploy.
    hybrid_retrieval_enabled: bool = True

    # Sparse model for the hybrid branch. One setting because three places have to agree:
    # llama_engine encodes queries with it, scripts/migrate_sparse_vectors.py encodes the backfill
    # with it, and the indexer encodes new chunks with it. Two different sparse models produce
    # weights over different token spaces, so a mismatch does not error -- it scores query tokens
    # against an index built from other tokens and quietly returns the wrong passages.
    #
    # BM25, not SPLADE, since 2026-09-30. SPLADE was 87% of indexing wall time and it failed the job
    # the sparse channel exists for: it shatters a literal into WordPiece fragments, encoding
    # XJ-4420-B as `x ##j 44 ##20 b` and collapsing 7.3.1 to `7`, four of which XJ-4425-B also
    # produces. It retrieves but cannot discriminate. BM25 emits one entry per literal token -- 22 for
    # 22 words on the same text -- which is identity. The collection's sparse vector must carry
    # `modifier: "idf"` for this to weight anything: fastembed returns uniform values and Qdrant
    # applies IDF at query time. That modifier is creation-only, and it would be WRONG with SPLADE.
    sparse_model: str = "Qdrant/bm25"

    # Dense embeddings, computed locally. No API key, no rate limit, no per-chunk cost and no external
    # service in the indexing path -- see plans/go-local-embeddings.md. The model's own sequence limit
    # (512 for this one) is read from its tokenizer at startup rather than declared here, so a swap
    # cannot leave a stale ceiling behind.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    # ONNX intra-op threads for dense inference. The box has 4 vCPU; BM25 needs almost none, so the
    # dense model gets them. Inference runs in a worker thread, never on the event loop.
    embedding_threads: int = 4
    embedding_max_batch_tokens: int = 50_000
    # No retry setting, and there is no second mechanism: an embed call is made once. One sat here for
    # OpenAI and was removed on 2026-09-28 along with the loop it fed -- it made a failed attempt
    # invisible when the next one succeeded, and it re-sent tokens the account may already have been
    # billed for. Nothing replaces it: a local model has no transient network failure to ride out, and
    # a model that will not load is a startup failure rather than something to retry per batch.
    #
    # The quarantine stays. It is about embeddings generally, not about OpenAI: a batch that cannot be
    # embedded is written to disk and the circuit opens, rather than the run continuing with a hole
    # in the corpus that nothing records.
    embedding_quarantine_dir: str = (
        "/var/lib/geek-crawler-rag/embedding_quarantine"
    )

    # Legacy uniform chunk defaults (still used as fallback knobs).
    chunk_size_tokens: int = 650
    chunk_overlap_tokens: int = 80

    # Parent/child indexing (Phase B2).
    child_chunk_size_tokens: int = 200
    child_chunk_overlap_tokens: int = 40
    # 480, not 1000, since 2026-09-30. bge-small truncates at 512 tokens and does so silently, and a
    # parent point IS embedded when it differs from its child (llama_nodes). Measured: where a parent
    # is genuinely wider than its child it runs ~3x the child's length at p50, so a 200-token child
    # implies a ~590-token parent -- already past the ceiling at the old setting. 480 keeps the median
    # case whole. The chunker counts tiktoken BPE and the model counts WordPiece, which runs longer, so
    # this is a budget rather than a guarantee: LocalDenseEmbedding counts and logs anything that still
    # reaches the limit, and that count is how this number gets tuned rather than guessed again.
    parent_chunk_size_tokens: int = 480
    parent_chunk_overlap_tokens: int = 80

    page_batch_size: int = 25
    # 64, and the same 64 in deploy/hostinger-compose.yml. This was 32 here, 56 in
    # compose and 64 in the README - three answers to one question. 64 is the one
    # the container is sized for ("EMBED_BATCH_SIZE=64 peaks near ~3.9 GiB"), and
    # 128 was OOM-killed, so that is the ceiling rather than a target.
    #
    # It drives three things at once: the indexer's flush threshold, the OpenAI
    # batch cap, and the Qdrant batch. Every per-flush cost - a Qdrant retrieve, a
    # collection_exists round trip, the upsert, a Mongo write - is paid half as
    # often at 64 as at 32.
    embed_batch_size: int = 64

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
