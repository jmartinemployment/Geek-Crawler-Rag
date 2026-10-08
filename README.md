# Geek-Crawler-Rag

Standalone **Python** retrieval product for the Geek-Crawler Mongo corpus.
**FastAPI** exposes `v1/*`; **LlamaIndex** owns ingest/embed/dense retrieval into Qdrant
(parent/child chunks, hybrid dense + BM25 fused by relative score, optional Cohere rerank).

See [`architecture.md`](./architecture.md) and [`plans/geek-crawler-rag.md`](./plans/geek-crawler-rag.md).
`contracts/` holds cross-repo wire contracts, both live and historical:

- **Live, CI-enforced** — [`contracts/rag-index-status/webhook.v1.json`](./contracts/rag-index-status/webhook.v1.json):
  every field the index-status webhook posts to GeekAPI. GeekBackend keeps a copy whose `fields`
  map must match; a test on each side pins its own half and `webhook-contract-agrees` compares the
  two. It exists because five fields were posted on every webhook and bound by nothing, so a corpus
  gutted by 4xx error pages reported identically to a clean one.
- **Historical** — [`contracts/phase-u/generate.section.v1.json`](./contracts/phase-u/generate.section.v1.json):
  Phase U generate fixture, models only. There is no `/v1/generate` endpoint and never will be.

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
> | Quote verification | `citation_verify.quote_in_text(quote, plain_text, blocks)` against that same string, served by `POST /v1/verify` |
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
- Local dense + sparse embeddings and deterministic Qdrant vector records
- Hybrid retrieval in Qdrant: dense (meaning) and sparse BM25 (keyword) halves fused by relative score (alpha 0.5); `keyword` on the request gives the keyword half its own text. The older in-process BM25 re-rank + RRF over the candidates was deleted on 2026-10-08
- Optional Cohere reranking
- Entity, source, category, quality, host, and chunk-role filters
- Graph-style entity/category/co-occurrence themes
- Few-shot advertising-template indexing and retrieval
- Full-page block-text reads for source verification (`PageTextResponse.text`)
- Quote verification helpers (chunk and citation text must appear in the page's block-text projection)
- Idempotent run-level reindexing. **Indexing no longer deletes anything** — the crawler owns the reject taxonomy, and this service re-adjudicating it is what destroyed 5,274 pages on 2026-09-18. A page this run cannot use is counted (`pagesSkippedUnusable`) and left alone

### Technology

Python, FastAPI, Pydantic, LlamaIndex, MongoDB, Qdrant, fastembed/ONNX, BM25, Cohere, Docker, and GHCR.

**Readability was removed and is not a dependency** — no import in `src/`, no entry in
`pyproject.toml`. It is an *article* extractor, and most crawled pages are product, pricing, feature
and solution pages. Measured against visible prose it returned 9%, 44% and 175% of three live pages;
the crawler's selector-based extractor measures 101-103% on the same three
(`Geek-Crawler-v2/src/crawl/extract-content.ts` header). The corpus arrives as `contentHtml` plus
typed `blocks`; nothing here re-extracts it.

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
| English-only embed (`bge-small-en-v1.5`) | Spanish indexing |
| Owned by this repo | Logic inside phi or GeekAPI |

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Mongo + Qdrant liveness (`engine`, `features`) |
| `POST` | `/v1/index` | Enqueue full-run index `{ "runId": "…" }` |
| `GET` | `/v1/index/{runId}` | Index job status (ops/debug; UI uses SignalR, not polling) |
| `GET` | `/v1/index-scheduler` | Persisted scheduler cadence, next run, and last enqueue |
| `POST` | `/v1/index/hosts` | Which run is indexed for each URL's host: `{ urls, crawlType }`. A host is not a run — a site on the partner and competitor lists has two — so send the type; an untyped host indexed under more than one type is refused with the types named, never guessed |
| `POST` | `/v1/query` | Hybrid or graph retrieve (see below) |
| `POST` | `/v1/templates/index` | Upsert ad-template exemplars — a derived index; the records are GeekRepository's |
| `POST` | `/v1/templates/query` | Retrieve few-shot templates by need (+ channel/framework/tags) |
| `GET` | `/v1/pages/{pageId}?runId=…` | Run-scoped block-text projection for citation reads (`PageTextResponse`) |
| `GET` | `/v1/pages?runId=&url=` | Same lookup by run + URL |
| `POST` | `/v1/verify` | Is each quote on its page? `{ runId, quotes: [{ pageId, quote }] }` → `found`, `reason`, `sourceDigest` per quote |

RAG in this repository is **library-only**: index, query, and page block text. There is no `POST /v1/generate` endpoint.

`POST /v1/query` body (camelCase; new fields optional / backward compatible):

```json
{
  "need": "…",
  "keyword": "…",
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

- Default `retrievalMode`: `hybrid` (Qdrant dense + sparse BM25, fused by relative score, then the pool cut, optional Cohere rerank, and page-diverse selection). The in-process BM25 re-rank + RRF that used to follow the hybrid query was deleted on 2026-10-08: it scored the keyword a second time once the hybrid's own keyword half ran. `git log -S bm25_rank_indices` finds it.
- **`keyword` (optional, 2026-10-08).** `need` is embedded for the meaning half. When `keyword` is given it is the text the sparse BM25 half searches instead of `need`, so `need` can be a paragraph describing the reader's situation while the keyword half matches a few distinctive terms. Absent, both halves get `need`.
- **Results are de-duplicated by text.** A heading section shorter than the child
  window produces a child chunk identical to its parent, and identical vectors
  score identically, so the pair would otherwise occupy adjacent slots. Exact
  repeated text is dropped before the `topK` cap is applied, and the candidate
  pool over-fetches so `topK` is filled with distinct results.
- **Results are page-diverse (2026-10-08).** Passages are ranked, then pages are
  selected: every page's best passage first (pages in rank order of those best
  passages), then every page's second-best, and so on until `topK` is filled. A
  page can contribute several passages only after every page with an admissible
  passage has had its first. Before this, the top `topK` came straight off the
  ranked list and one page could take most of the slots — on Stampli the bare
  keyword put 13 of 32 passages on one blog post. The returned order is the
  selection order, so a consumer that truncates keeps the diversity.
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

### Indexing trigger

Indexing is triggered by `POST /v1/index`, which GeekAPI calls when a crawl completes. The route
refuses a run with no `ContentReadyAt` (409, nothing queued) — the same readiness the scheduler's
candidate query filters on, checked by `IndexService.readiness_refusal` at both entrances before the
claim, so a refused run leaves no job row.

`INDEX_SCHEDULER_ENABLED` defaults to `true` in `config.py` and `deploy/hostinger-compose.yml`, and
production sets it `true` in the box's own compose file (`/docker/geek-crawler-rag/docker-compose.yml`,
not in git). It is the only automatic catch-up for a lost enqueue: GeekAPI's `POST /v1/index` fails
closed and only logs when the Library is unreachable, and every 300s the scheduler indexes the oldest
content-ready run that has no job row. It logs `found no eligible content-ready run` when there is
nothing to do. To stop it, use `POST /v1/index-scheduler/pause` with a reason (recorded in Mongo, no
restart) and `/resume`; editing the flag means recreating the container, which kills any index job
in flight.
If it runs it costs embedding spend, not corpus — indexing is read-only (`indexer._skip_unusable`).

When it runs, the scheduler persists its next due time in Mongo, takes an atomic lease, and chooses the
completed run with the OLDEST `ContentReadyAt` — the crawl-level marker confirming every persisted page
carries extracted content — that is not already indexed (`INDEX_SCHEDULER_INTERVAL_SECONDS`, 300s in production).

Index jobs — scheduled or manual — use Mongo leases and heartbeats. There is **no** automatic
stale-job recovery: `Indexer.start()` deliberately does not `claim_recoverable`. A graceful stop
(a deploy) marks in-flight jobs `failed`, and the scheduler skips failed runs, so they wait for an
operator to re-post them. Failed jobs are **not** auto-retried in-process; they fail closed, and an
operator or a new enqueue starts a fresh attempt. See
[`plans/rules.md`](./plans/rules.md) §3a (**No Retries. No Fallbacks. No Crappy Code.**) and
[`docs/index-job-recovery-after-restart.md`](./docs/index-job-recovery-after-restart.md) for the
re-post procedure and the two states that look like success but are not.

**Embeddings are local and in-process.** Dense is `BAAI/bge-small-en-v1.5` (384-d) and sparse is
`Qdrant/bm25`, both through fastembed/ONNX — no API key, no rate limit, no per-chunk cost and no
external service in the indexing path. Inference runs in a worker thread, never on the event loop.

There is no throttle and no retry. An embed call is made **once**; a batch that cannot be embedded is
quarantined and the job fails closed, with already upserted points preserved (see
[`docs/embedding-circuit-recovery.md`](./docs/embedding-circuit-recovery.md) and
[`plans/rules.md`](./plans/rules.md) §3a). The fail-closed policy is unchanged; what was removed with
the remote API was the tokens-per-minute limiter it needed.

- `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`
- `EMBEDDING_DIMENSIONS=384` — must equal the collection's declared `size`; startup refuses a mismatch
- `EMBEDDING_THREADS=4` — ONNX intra-op threads; BM25 needs almost none, so dense gets them
- `EMBEDDING_MAX_BATCH_TOKENS=50000`
- `EMBED_BATCH_SIZE=64`
- `FASTEMBED_CACHE_PATH=/tmp/fastembed_cache` — pinned so the named volume keeps catching model weights
- `QDRANT_UPSERT_DELAY_SECONDS=0` — no pause between upserts; `config.py` and the compose default agree
- `INDEX_SCHEDULER_INTERVAL_SECONDS=300`

**The model truncates at 512 tokens, silently.** `LocalDenseEmbedding` counts each input with the
model's own tokenizer and logs every one that reaches the limit, reporting the total as
`truncatedInputs`. It does not raise. The chunker measures in that same tokenizer
(`chunk_tokenizer.ChunkTokenizer`, a required argument with no default) and rejects a chunk budget
above the model's usable sequence limit at the boundary (`chunk.py`), so a chunk inside its configured
budget no longer crosses the ceiling — the tiktoken-versus-WordPiece gap that silently truncated 6.9%
of parent chunks was closed on 2026-09-30. A non-zero `truncatedInputs` is a defect to look at, not an
expected cost.

**`EMBED_BATCH_SIZE=128` OOM-killed the container — do not set it there again.** On
2026-09-25, 32 → 128 took the api container from a steady 2.4 GiB to past its **6 GiB**
limit in about two minutes: `docker events` recorded `oom` then `die exitCode=137`, and
each kill emptied the in-memory index queue and orphaned the `running` row, which reads
as a hang rather than a kill. Note `docker inspect` reported `OOMKilled=false` afterwards
— it reflects the state after the restart, so the event log is the authority, not
`inspect`. 64 with `mem_limit: 7g` is the supported setting (the compose says why not 8g:
api 7 + qdrant 3 + mongo 4 + caddy 0.25 already commit 14.25 of the host's 15 GiB); treat
128 as known-bad at 6 GiB. Idle is ~0.9 GiB, so the batch-dependent share is roughly linear: ~1.5 GiB at 32,
~3 GiB at 64.

**Retry policy, one place: there is none.** The SDK's `max_retries` stays 0 and
`LlamaIndexEngine._embed_batch` makes one attempt — it fails or it does not. The bounded
retry that used to live there was removed on 2026-09-28; the method's docstring records the
three things that were wrong with it. See [`plans/rules.md`](./plans/rules.md) §3a.

**Why the delay and batch size were revisited, 2026-09-25.** Indexing was assumed CPU-bound on sparse
encoding. Measured mid-run on the VPS it was not: the api container sat at **86% of one
core** against a 3.0-core limit, Qdrant at 0.17%, Mongo at 0.6%, with 9 GB of the host's
16 GB free and the embedding throttle at 32k tokens of its 400k window
(`rateLimitRetries: 0`). Nothing was saturated — the indexer is **latency-bound on
serialized round trips**, so raising the box's allocation would have bought nothing.

What it was actually spending time on, per batch of 32: embed, upsert, Mongo persist,
a 0.5s sleep, and the status webhook **twice**. Batch size 32 → 128 quarters the number
of cycles, so it quarters that fixed cost per chunk; the sleep is gone; and the duplicate
webhook was removed in `indexer.py` (`_persist` already notifies). The 0.5s sleep was
backpressure from `a92def3`, sized for the old 8 GiB box where Qdrant's 3 GiB limit sat
inside ~1 GiB of headroom — vectors are `on_disk` now and the headroom is ~9 GB, so the
condition it guarded is no longer the live one. If Qdrant RSS climbs toward its limit
during ingest again, restore the delay first.

Re-running a job replaces the run's index: every attempt deletes the run's points
before writing, so nothing from an earlier attempt survives. Embeddings are reused
through `rag_vector_cache`, so a re-run mostly costs Qdrant writes. Identical
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
cp .env.example .env   # set MONGO_CRAWLER_URL, API_KEY
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
`mem_limit: 7g`, `cpus: 3.5`. There is no pause between Qdrant batches
(`QDRANT_UPSERT_DELAY_SECONDS` defaults to 0).
The scheduler is on in the repo compose and on the box (see *Indexing trigger*).
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
interrupts any in-flight index run and nothing re-claims it: the queue is in-process, and
`Indexer.start()` does not reclaim leases. No Qdrant points are wiped. Re-post the run — see
[`docs/index-job-recovery-after-restart.md`](./docs/index-job-recovery-after-restart.md). Do not
push while an index run is in flight.

A green `/health` alone does **not** prove the new image is live — confirm the
API container's uptime reset via `docker ps`.

## Troubleshooting embedding failures

An embed call is made once. A batch that cannot be embedded is written to
`EMBEDDING_QUARANTINE_DIR` and the job fails closed; points already upserted are kept.

- The quarantine file carries `reason` (`inference_failed` or `empty_input`), `model`, `batchSize`,
  `tokenCount`, per-item previews, and `detail` — the exception chain, innermost first, because ONNX
  and tokenizer errors put the useful text on the innermost exception.
- `reason: empty_input` does not mean inference broke. It means the chunker or
  `embedding_sanitize` let an empty string through, and both are supposed to make it unreachable —
  so it points at a hole in one of them.
- Recovery is a deliberate re-post, never automatic. See
  [`docs/embedding-circuit-recovery.md`](./docs/embedding-circuit-recovery.md).

There is no provider status page to check and no request ID to quote: embedding runs in this process.
If it fails, the cause is local — the model, the input, or the host.

## Consumers

- **gcc-v2 WRITE** (via GeekAPI): HTTP query client — resolve `runId`, call `/v1/query`, inject chunks; on miss **notify-and-skip**.
- Thin trigger: `POST /v1/index` when a crawl reaches **`complete`**.
- Progress: webhook → GeekAPI → SignalR (see above).
