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
    # Seconds the Qdrant HTTP client waits for a response. qdrant_client defaults to 5, and we
    # never set it until 2026-10-01, when four of the batch's largest runs died on it in a row:
    # stripe at 7,461 chunks, liveplan at 22,369, ramp and datarails before writing anything.
    #
    # Qdrant was not slow. Its own access log shows those same PUTs served in 10-70ms, memory at
    # 28% and status green. The API container was drawing 299% CPU on ONNX inference, which starves
    # the asyncio event loop that has to read the response -- so the deadline expired on our side,
    # not theirs. The bigger the run, the more draws against that 5-second odds, which is exactly
    # why only large runs failed and ramp's 110 sequential resume lookups failed before any work.
    #
    # 60s is far above any observed Qdrant latency here and still fails if Qdrant genuinely dies.
    # Raising it does not mask a slow database; it stops a busy client from calling a fast one dead.
    qdrant_timeout_seconds: int = 60
    qdrant_collection: str = "geek_crawler_chunks"
    qdrant_ad_templates_collection: str = "geek_ad_templates"
    # Zero. This fired after every flush, and at batch 64 on a large run that is
    # tens of minutes of pure sleeping for nothing measured: the VPS profile put
    # the api container at 86% of ONE core against a 3.0-core limit with Qdrant
    # at 0.17%, so Qdrant was never the thing needing to be slowed down. The
    # guard in indexer.py stays, so this remains an escape hatch if Qdrant ever
    # does need backpressure - it just is not paid by default any more.
    qdrant_upsert_delay_seconds: float = 0.0

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
    # a metered remote API and was removed on 2026-09-28 along with the loop it fed -- it made a failed
    # invisible when the next one succeeded, and it re-sent tokens the account may already have been
    # billed for. Nothing replaces it: a local model has no transient network failure to ride out, and
    # a model that will not load is a startup failure rather than something to retry per batch.
    #
    # The quarantine stays. It is about embeddings generally, not about any one provider: a batch that
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
    # 500, and it is now a GUARANTEE rather than a budget.
    #
    # bge-small truncates at 512 tokens, silently, and a parent point IS embedded when it differs
    # from its child (llama_nodes). This was 1000, then 480, and both were guesses -- because the
    # chunker measured tiktoken BPE while the model counted WordPiece, which runs far longer on code
    # and technical identifiers. Measured on the live corpus 2026-09-30: 136 of 1,984 parents crossed
    # 512 anyway, worst case 638 tokens against a 480 setting, with the discarded tail being Python
    # source. No value here could have been correct, because the two sides were not measuring the
    # same thing.
    #
    # Since 2026-09-30 the chunker measures with the model's own tokenizer (chunk_tokenizer) and
    # chunk_text REFUSES a size_tokens above sequence_limit minus special_token_overhead -- 512 - 2 =
    # 510 for this model. So 500 content tokens reach inference as 502 and cannot be truncated by
    # arithmetic, not by estimate. Raising this above 510 is a startup error, not a silent cut.
    #
    # LocalDenseEmbedding still counts anything that reaches the ceiling. A non-zero count now means
    # a real defect -- a caller bypassing the chunker -- not a number in need of tuning.
    parent_chunk_size_tokens: int = 500
    parent_chunk_overlap_tokens: int = 80

    page_batch_size: int = 25
    # 64, and the same 64 in deploy/hostinger-compose.yml. This was 32 here, 56 in
    # compose and 64 in the README - three answers to one question. 64 is the one
    # the container is sized for ("EMBED_BATCH_SIZE=64 peaks near ~3.9 GiB"), and
    # 128 was OOM-killed, so that is the ceiling rather than a target.
    #
    # It drives three things at once: the indexer's flush threshold, the embedder's
    # batch cap, and the Qdrant batch. Every per-flush cost - a Qdrant retrieve, a
    # collection_exists round trip, the upsert, a Mongo write - is paid half as
    # often at 64 as at 32.
    embed_batch_size: int = 64

    # Durable indexing scheduler, oldest-content-ready first (see mongo.find_oldest_content_ready_run).
    # Off unless the environment turns it on. Indexing is triggered by POST /v1/index, which GeekAPI
    # calls when a run completes. Production sets INDEX_SCHEDULER_ENABLED explicitly in the box's
    # compose file, so this default decides nothing there; it decides what a fresh deploy does.
    index_scheduler_enabled: bool = False
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
