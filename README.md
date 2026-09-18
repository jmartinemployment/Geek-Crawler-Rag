# Geek-Crawler-Rag

Standalone **Python** retrieval product for the Geek-Crawler Mongo corpus.
**FastAPI** exposes `v1/*`; **LlamaIndex** owns ingest/embed/dense retrieval into Qdrant
(parent/child chunks, hybrid BM25 RRF, optional Cohere rerank).

See [`architecture.md`](./architecture.md) and [`plans/geek-crawler-rag.md`](./plans/geek-crawler-rag.md).
Historical Phase U generate contract fixture (models only; no `/v1/generate` endpoint): [`contracts/phase-u/generate.section.v1.json`](./contracts/phase-u/generate.section.v1.json).

## Product overview

> ## ⛔ One corpus format. No converters.
>
> **The corpus body is clean semantic `contentHtml` plus typed `blocks`** — that is
> the only representation, for storage, for verification and on the wire between
> services. The crawl path contains no text-format conversion step
> (`Geek-Crawler-v2/plans/corpus-rebuild.md`) and nothing here may add one.
>
> | Concern | The method |
> |---|---|
> | Corpus body | typed **`blocks`** (`heading`+`level`, `paragraph`, `listItem`, `quote`, `code`, `row`+`cells`, `term`, `definition`; each with `text`/`cells`, `html`, `anchors`) |
> | Display / audit | **`contentHtml`** |
> | Page as a string | **one** projection — `block_text.derive_plaintext_from_blocks` |
> | Quote verification | `citation_verify.quote_in_text(quote, plain_text, blocks)` against that same string |
> | Page read API | `GET /v1/pages…` → `PageTextResponse.text` |
> | Run readiness | **`ContentReadyAt`** |
>
> Chunk text and verification text come from the same projection deliberately: a
> quote is taken from a retrieved chunk and matched against the page, so two
> "join the blocks" implementations make correct citations fail.
>
> **Why this is a hard rule.** The crawler changed corpus format and this service
> did not. Every page then classified as having no usable body, and
> `_delete_unusable` removed it **along with its Qdrant points**. Eight runs,
> 5,274 pages, 0 chunks upserted, corpus destroyed on 2026-09-18. A second
> representation is not a convenience; it is the seam the two halves drift apart
> along.
>
> **Migration state — complete in this service.** `extract.py`, `block_text.py`,
> `citation_verify.quote_in_text`, `unusable.py` (`no_content` only) and
> `indexer._skip_unusable` (counts, never deletes) are block-based; the page API
> returns `PageTextResponse`; the run filter and its covering index are
> `ContentReadyAt` / `ix_crawl_runs_content_ready`; chunks are cut at heading
> blocks with per-section anchors; `parserId` is `crawler-blocks`. The ops
> scripts that existed only to serve the old format are deleted.
>
> Two operational items remain, neither in code — confirm the Mongo key casing
> for `ContentReadyAt` with `scripts/verify_ingest_fields.py` on the VPS, and
> drop the superseded legacy readiness index once the image carrying `78c143b` is
> running (the name is in §4's drop command). Detail:
> [`plans/retire-legacy-corpus-format.md`](./plans/retire-legacy-corpus-format.md).
>
> **Chunks indexed before `b9fadcc` need a reindex** to gain section titles and
> per-section anchors; they are otherwise valid.
>
> **Operator-supplied assets are a separate concern.** `asset_context.parse_asset`
> accepts `text/plain` and `text/html` only, and a generated report a human reads
> is not corpus. Neither path feeds retrieval or verification.

Geek-Crawler-Rag turns partner and competitor website crawls into searchable evidence with verified block-text reads. It combines semantic and keyword retrieval, hierarchical context, entity-aware filtering, and quote-level verification for downstream content systems.

### Capabilities

- English-only parent/child chunking for pinpoint and section-level context
- OpenAI embeddings and deterministic Qdrant vector records
- LlamaIndex dense retrieval combined with BM25/text search and reciprocal-rank fusion
- Optional Cohere reranking
- Entity, source, category, quality, host, and chunk-role filters
- Graph-style entity/category/co-occurrence themes
- Few-shot advertising-template indexing and retrieval
- Full-page block-text reads for source verification (`PageTextResponse.text`)
- Quote verification helpers (chunk and citation text must appear in the page's block-text projection)
- Idempotent run-level reindexing. **Indexing no longer deletes anything** — the crawler owns the reject taxonomy, and this service re-adjudicating it is what destroyed 5,274 pages on 2026-09-18. A page this run cannot use is counted (`pagesSkippedUnusable`) and left alone

### Technology

Python, FastAPI, Pydantic, LlamaIndex, MongoDB, Qdrant, OpenAI, BM25, Cohere, Readability, Docker, and GHCR.

## Place in the Geek content platform

```text
Geek-Crawler-v2 → MongoDB → Geek-Crawler-Rag/Qdrant
                                  ↓
                         GeekAPI → Content Creator v2
```

**Geek-Crawler-v2** produces the clean crawl corpus. This service owns indexing, hybrid/graph retrieval, themes, and run-scoped block-text reads. **Content Creator v2** owns generation, operator writing workflows, and publishing.

## What this is / is not

| Is | Is not |
|----|--------|
| Index + query for `geek_crawler` pages | A crawler |
| FastAPI + LlamaIndex + Qdrant on Hostinger | Cloud vector DB / pgvector |
| English-only embed (`text-embedding-3-small`) | Spanish indexing |
| Owned by this repo | Logic inside phi or GeekAPI |

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Mongo + Qdrant liveness (`engine`, `features`) |
| `POST` | `/v1/index` | Enqueue full-run index `{ "runId": "…" }` |
| `GET` | `/v1/index/{runId}` | Index job status (ops/debug; UI uses SignalR, not polling) |
| `GET` | `/v1/index-scheduler` | Persisted scheduler cadence, next run, and last enqueue |
| `POST` | `/v1/query` | Hybrid or graph retrieve (see below) |
| `POST` | `/v1/templates/index` | Upsert ad-template exemplars (Content Creator owns corpus) |
| `POST` | `/v1/templates/query` | Retrieve few-shot templates by need (+ channel/framework/tags) |
| `GET` | `/v1/pages/{pageId}?runId=…` | Run-scoped block-text projection for citation reads (`PageTextResponse`) |
| `GET` | `/v1/pages?runId=&url=` | Same lookup by run + URL |

RAG in this repository is **library-only**: index, query, and page block text. There is no `POST /v1/generate` endpoint.

`POST /v1/query` body (camelCase; new fields optional / backward compatible):

```json
{
  "need": "…",
  "runId": "…",
  "crawlType": "partner",
  "host": null,
  "topK": 8,
  "preferParent": true,
  "preferChild": false,
  "chunkRole": "child",
  "sourceTypes": ["partner"],
  "entityNames": ["acme.com"],
  "categories": ["pricing"],
  "minQuality": 0.4,
  "retrievalMode": "hybrid"
}
```

- Default `retrievalMode`: `hybrid` (LlamaIndex dense + BM25/text RRF + optional Cohere).
- **Results are de-duplicated by text.** A heading section shorter than the child
  window produces a child chunk identical to its parent, and identical vectors
  score identically, so the pair would otherwise occupy adjacent slots. Exact
  repeated text is dropped before the `topK` cap is applied, and the candidate
  pool over-fetches so `topK` is filled with distinct results.
- `preferParent` / `preferChild` select which text a hit returns. They no longer
  affect de-duplication, which is unconditional. Setting `preferParent: true`
  additionally collapses sibling children that share one parent.
- `retrievalMode: "graph"`: parent-biased hybrid plus `themes[]` (entity / category / co-occurrence) for slides/strategy.
- Index writes parent + child LlamaIndex nodes with `parentText` / `childText` and entity metadata.
- `GET /health` includes `"engine": "llamaindex"` and `"features": ["hybrid","graph","ad-templates"]`.

`POST /v1/templates/index` example:

```json
{
  "templates": [
    {
      "id": "pas-linkedin",
      "name": "PAS LinkedIn",
      "channel": "linkedin",
      "framework": "pas",
      "tone": "professional",
      "body": "Problem… Agitate… Solution…",
      "entityTags": ["acme"]
    }
  ]
}
```

`POST /v1/templates/query`: `{ "need": "…", "topK": 5, "channel": "linkedin", "framework": "pas" }`.

Service authentication is mandatory: set `API_KEY` and send `X-Api-Key`.
Unauthenticated operation is available only with the explicit test-only
`LOCAL_TEST_MODE=true` setting.

Governed Knowledge endpoints are service-only:

- `POST /v1/assets/index` indexes one exact manifest-authorized revision/resource.
- `POST /v1/assets/delete` deletes one exact manifest-authorized revision/resource.
- `POST /v1/assets/query` searches only exact entries in a signed manifest.

Configure GeekAPI manifest verification keys with
`CONTEXT_MANIFEST_SIGNING_KEYS` (a JSON key-ID-to-secret map). The service keeps
only rebuildable vectors; asset, revision, resource, and manifest authority
remains in GeekRepository.

Index concurrency is **1**. Rebuild deletes all Qdrant points for `runId`, then reindexes.
At index start the service logs **`mongoPageCount`**. Runs with `mongoPageCount` above **50 000** are skipped (Hostinger safety cap).

### Indexing trigger and OpenAI rate limits

**The index scheduler is deprecated.** Indexing is triggered by `POST /v1/index`.
`INDEX_SCHEDULER_ENABLED` is `false`; `scheduler.py` and
`mongo.find_smallest_content_ready_run` remain in the tree without being the live route.

Keep the flag set in the environment rather than relying on a default — `config.py:50` and
`deploy/hostinger-compose.yml:45` (`${INDEX_SCHEDULER_ENABLED:-true}`) both default it on. Until
`78c143b` it also could not have selected a run, because the filter matched nothing; the flag is now
the only thing holding it off. If it ever did run it would cost embedding spend, not corpus —
indexing is read-only (`indexer._skip_unusable`).

When it did run, the scheduler persisted its next due time in Mongo, took an atomic lease, and chose the
smallest completed run whose crawl-level `ContentReadyAt` confirms every persisted page carries extracted
content and which is not already indexed (`INDEX_SCHEDULER_INTERVAL_SECONDS`, 300s in production).

Index jobs — scheduled or manual — use Mongo leases, heartbeats, and stale-job recovery. Failed jobs are
**not** auto-retried in-process; they fail closed, and an operator or a new enqueue starts a fresh
attempt. See [`plans/rules.md`](./plans/rules.md) §3a (**No Retries. No Fallbacks. No Crappy Code.**).

All corpus, query, and ad-template embeddings pass through one rolling
token-per-minute limiter, sequentially partitioned by item and token count.
Embedding calls are **fail-closed with no in-process retries**
(`OPENAI_EMBEDDING_MAX_RETRIES=0`): the first HTTP 500 or empty-input 400
quarantines the batch and fails the job with already upserted points preserved
(see [`docs/embedding-circuit-recovery.md`](./docs/embedding-circuit-recovery.md)
and [`plans/rules.md`](./plans/rules.md) §3a).

**Keep the throttle well under the account ceiling.** Your OpenAI TPM limit is
returned in `x-ratelimit-limit-tokens` on any embeddings response. Setting the
throttle *at* that limit rather than below it causes sustained runs to receive
HTTP 500 `server_error` instead of clean 429s — a 1,000,000 setting against a
1,000,000 ceiling killed multi-hour runs until it was lowered to 400,000.

- `OPENAI_EMBEDDING_TOKENS_PER_MINUTE=400000` (40% of a 1,000,000 ceiling)
- `OPENAI_EMBEDDING_MAX_BATCH_TOKENS=50000`
- `EMBED_BATCH_SIZE=32`
- `OPENAI_EMBEDDING_MAX_RETRIES=0` (fail-closed by law; do not raise — [`plans/rules.md`](./plans/rules.md) §3a)
- `QDRANT_UPSERT_DELAY_SECONDS=0.5`
- `INDEX_SCHEDULER_INTERVAL_SECONDS=300`

Re-running a failed job resumes rather than restarting: point IDs are
deterministic, so chunks already committed to Qdrant are skipped. Identical
strings within a batch are embedded once and the vector reused for every point
that shares that text (short heading sections yield a child identical to its
parent, ~29% of calls on marketing pages).

`GET /health` reports current throttle counters and scheduler state. Per-job
status includes `attempt`, `trigger`, `embeddingRateLimitRetries`, and
`embeddingWaitSeconds`; scheduler status includes `lastSelectionReason`.

Run readiness is stamped by GeekAPI at ingest (`ContentReadyAt`), so this service
has no reconciliation script — the one that backfilled the old marker was deleted
with the format it served. To check what Mongo actually holds, per field and per
casing:

```bash
uv run python scripts/verify_ingest_fields.py
uv run python scripts/verify_ingest_fields.py --run-id <guid>
```

### Index status push (no UI polling)

On status transitions the API POSTs to GeekAPI (optional):

- `INDEX_STATUS_WEBHOOK_URL` — e.g. `https://api.geekatyourspot.com/api/geek-crawler/internal/rag/index-status`
- `INDEX_STATUS_WEBHOOK_KEY` — `GEEK_BACKEND_API_KEY` (falls back to `API_KEY` if unset)

GeekAPI fans out SignalR **`GeekCrawlerRagIndexEvent`**. The Geek-Crawler UI listens on that hub method; it does **not** poll `GET /v1/index`.

## Local run

```bash
cp .env.example .env   # set MONGO_CRAWLER_URL, OPENAI_API_KEY
docker compose up -d qdrant
uv sync
uv run geek-crawler-rag
```

Or full stack:

```bash
docker compose up --build -d
```

## Hostinger

Live stack on KVM 2 (alongside Mongo):

- Health: `http://2.24.101.90:8080/health`
- Image: `ghcr.io/jmartinemployment/geek-crawler-rag:latest`
- Compose: [`deploy/hostinger-compose.yml`](./deploy/hostinger-compose.yml) (Qdrant + API; Mongo via `host.docker.internal`)

Caps: Qdrant `mem_limit: 3g`, `cpus: 0.5`, `MAX_SEARCH_THREADS=1`; API
`mem_limit: 2g`. Corpus indexing pauses two seconds between Qdrant batches.
The Hostinger compose enables the two-hour scheduler by default.
Point `MONGO_CRAWLER_URL` at the existing Hostinger Mongo `geek_crawler` database. Normal indexing reads the corpus; controlled cleanup procedures may delete unusable crawl pages and related links.

GeekAPI: set `GEEK_CRAWLER_RAG_URL` / optional `GEEK_CRAWLER_RAG_API_KEY`.

### Deploy to the VPS

CI builds and pushes `ghcr.io/jmartinemployment/geek-crawler-rag:latest` on every
push to `main`, but **it does not deploy**. Updating the live box is manual.

The server layout differs from this repo: there is no `deploy/` directory on the
VPS. The compose file is `/docker/geek-crawler-rag/docker-compose.yml`, with all
values inline (no `env_file`). Passing `-f deploy/hostinger-compose.yml` fails
with `no such file or directory`.

```bash
ssh -i ~/.ssh/hostinger_rag_ed25519 root@2.24.101.90 \
  'cd /docker/geek-crawler-rag && docker compose pull && docker compose up -d'
```

Verify:

```bash
curl -s http://2.24.101.90:8080/health
```

Qdrant is pinned to `v1.13.4`, so only the API container is recreated. A deploy
interrupts any in-flight index run; the scheduler's lease/heartbeat recovery
re-claims it on the next tick and no Qdrant points are wiped.

A green `/health` alone does **not** prove the new image is live — confirm the
API container's uptime reset via `docker ps`.

## Troubleshooting OpenAI errors

When indexing fails with OpenAI HTTP 500 errors, the service captures OpenAI's
request ID and error message for diagnosis. See [`docs/embedding-circuit-recovery.md`](./docs/embedding-circuit-recovery.md)
for quarantine workflow, examining error details, and recovery procedures.

Key points:
- Embedding failures (HTTP 400 or 500) write quarantine JSON to `EMBEDDING_QUARANTINE_DIR`
- Quarantine files include `openaiDiagnostics` with `requestId`, `errorType`, and `message`
- Use the request ID to check [OpenAI status](https://status.openai.com/) and distinguish genuine outages from request-shape issues
- Failed runs are marked `FAILED` with no Qdrant wipe (points are preserved for recovery)
- Re-index is manual; use operator decision after examining the quarantine

## Consumers

- **gcc-v2 WRITE** (via GeekAPI): HTTP query client — resolve `runId`, call `/v1/query`, inject chunks; on miss **notify-and-skip**.
- Thin trigger: `POST /v1/index` when a crawl reaches **`complete`**.
- Progress: webhook → GeekAPI → SignalR (see above).
