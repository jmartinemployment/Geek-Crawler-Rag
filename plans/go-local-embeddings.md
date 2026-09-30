# Go local: bge-small dense + BM25 sparse, drop and recreate

All numbers measured on the live VPS, 2026-09-30.

## Why

**SPLADE is 87% of indexing wall time.** py-spy, 40 samples of the event-loop thread: 70% ONNX
inference, 17.5% post-processing, 10% all awaits combined, 2.5% numpy. Corroborated from outside —
5.1–5.6 chunks/s across a 2000× range in job size, `embeddingWaitSeconds` 0.0 on every job. The dense
side was never the constraint.

**And SPLADE fails the job it was added for.** Exact term binding is the requirement. `XJ-4420-B`
encodes to `x ##j 44 ##20 b`; `7.3.1` collapses to `7`. Four of those five fragments are also produced
by `XJ-4425-B`, so SPLADE retrieves but cannot discriminate. BM25 gives 22 entries for 22 words.

**OpenAI buys nothing measurable.** bge-small and text-embedding-3-small sit at roughly the same MTEB
retrieval score, and the dense channel only needs semantic proximity once BM25+IDF carries identity.
Against that: ~$4.60 per rebuild on an $11 balance, a key exposed in plain text this session, and a
subsystem (`EmbeddingThrottle`, most of `embedding_circuit.py`) that exists only to survive it.

Corpus is test data. No migration.

## Decisions

| | |
|---|---|
| Dense | `BAAI/bge-small-en-v1.5`, 384-d, local. **13.2 texts/s** at `threads=4`; bge-base managed 4.3, slower than today's whole pipeline, so it is out |
| Sparse | `Qdrant/bm25` + **`modifier: "idf"`** — 10 MB, no neural inference (0.20s/600 chunks) |
| Parent tier | emit a parent point **only when it differs from its child** |
| Expected | ~10.4 chunks/s, ~2× today, at $0 — arithmetic from measured parts, never run as a pipeline |

**`modifier` and `datatype` are creation-only.** A `PATCH` of `datatype` returns `200 ok` and silently
changes nothing. Wrong at creation means another rebuild.

**IDF is coupled to the model** — correct with BM25, wrong with SPLADE (it would double-penalise an
already-contextualised neural vector).

### The parent tier

`llama_nodes.py:95-126` emits a `chunk_role="parent"` point per parent; `:127-156` a child point.
Live: 101,556 points, 45,944 parent (45.2%), 55,612 child. Sampling 3,000 child points against the
`parentText` in their own payload:

```
parent == child:  71.4%   ← byte-identical duplicate
parent wider:     28.6%   ← ratio p50=2.96  p90=5.18  max=15.00
```

71.4% are pure duplication and the source of IDF contamination — `log(N / df(term))` inflates both
terms together, depressing IDF hardest for the rare literals BM25 exists to weight. Payloads do *not*
contaminate IDF; only the indexed sparse vector does.

But 28.6% are genuinely wider, typically 3×. That quarter has meaning spread across a window no
200-token child contains: pricing matrices, multi-step integration docs, scoping qualifiers far from
the feature they qualify. Dropping the tier outright would silently lose it.

So: gate the emission on `unit.parent_text.strip() != unit.child_text.strip()`. ~32% fewer points, no
retrieval risk. **`parent_chunk_size_tokens` 1000 → 480** — a surviving wide parent at p50 is ~590
tokens, already past bge-small's 512 ceiling.

Nothing depends on parent points existing for every unit: retrieval never filters `chunkRole`, and
`query.py:243,318,345,391` all read `payload.get("parentText")`, which every child carries (`:135`).

## Changes

**`local_embedding.py`** — done, committed. fastembed-backed `BaseEmbedding`, inference offloaded with
`asyncio.to_thread` so it never blocks the loop the way SPLADE did. 7 tests.

**`llama_engine.py`** — construct `LocalDenseEmbedding`; delete the `OPENAI_API_KEY` guard (`:66-67`,
today a hard startup failure), the two `httpx` clients (`:70-73`), `EmbeddingThrottle` (`:85-87`) and
both `acquire()` calls (`:391`, `:419`). Add `threads`/`batch_size` to `fastembed_sparse_encoder`
(`:104`) — neither is passed today, which is why the process draws 147% CPU while ORT sees 4 cores.
Rewrite the SPLADE comment block (`:92-104`). `:23` imports two names never referenced.
**`embedding_stats()` must keep its keys, zeroed** — see the follow-on below.

**`config.py`** — `sparse_model` → `Qdrant/bm25`; `embedding_dimensions` → 384;
`parent_chunk_size_tokens` → 480; new dense `embedding_model`; drop the four `openai_*` settings and
the TPM comment blocks (`:50-53`, `:56-62`).

**`embedding_circuit.py`** — keep `EmbeddingCircuitOpen`, `quarantine_embedding_batch`,
`open_embedding_circuit` (generic; drop the `request_id`/`openai_message` kwargs and the
`openaiDiagnostics` key). Delete `describe_openai_error`, `is_openai_http_500`,
`is_empty_embedding_input_error`. `should_quarantine_embedding_error` keeps its shape but both
predicates it delegates to are OpenAI-only — it needs ONNX/model-load classification or the quarantine
path is unreachable. The empty-text guards it backstops (`llama_engine.py:166-172`, `:329-340`) stay:
fastembed embeds `""` into a garbage vector without complaint.

**`embedding_throttle.py`** — `EmbeddingThrottle` dies. `is_retryable_rate_limit` and
`retry_after_seconds` are already dead in `src/`. But `embedding_token_count` and
`partition_embedding_batches` are live (`llama_engine.py:346`, `:415`) and **tiktoken-based** — wrong
tokenizer for bge-small's 512-token WordPiece limit. Re-base, don't delete: the `ValueError` at
`:103-107` is the only per-item length enforcement.

**`llama_nodes.py`** — the parent gate at `:95`; new model name into `embeddingModel` (`:120`, `:150`).
Also `asset_context.py:177,229`. `context_models.py:402` reads it as required (`Field(...)`, no
default) and `asset_context.py:403` does `payload["embeddingModel"]`, so it must keep being written.

**Delete `embed.py`** — nothing in `src/` imports it.

**`pyproject.toml`** — add nothing; `fastembed>=0.8.1` is already pinned. **Drop every OpenAI package**:
`llama-index-embeddings-openai:21`, `openai:25`, `llama-index-llms-openai:32`. No hedge on the last —
grep shows zero LLM usage, consistent with CLAUDE.md §1 (RAG never generates). `tiktoken:30` goes with
`partition_embedding_batches`' re-basing.

**Two collections, not one** — `ad_templates.py:68` builds `geek_ad_templates` from
`embedding_dimensions` too.

**Tests** — 33 `openai_api_key="test"` callsites across 9 files exist only to clear the startup guard;
there is no `conftest.py`, so each wires its own. Add one. Rewrite `test_embedding_throttle.py` and the
OpenAI half of `test_embedding_circuit.py`; fix stale `_vector_size = 1536` in three tests and the
`embeddingModel` fixture at `test_governed_context.py:194`.

**Docs** — `README.md:110,222-243,309`; `architecture.md:122`;
`docs/embedding-circuit-recovery.md:24,45-58,65,84,95,110,112`; `.env.example:15-17,44-52`;
`.env.hostinger`; `deploy/hostinger-compose.yml:37-42,62`. Delete
`docs/index-job-recovery-after-restart.md:74`, which claims `/tmp/fastembed_cache` is not a volume — it
is (`hostinger-compose.yml:68`, declared `:98`).

## Schema — both collections

```json
{ "vectors": { "size": 384, "distance": "Cosine", "datatype": "float16", "on_disk": false },
  "sparse_vectors": { "text-sparse": { "modifier": "idf", "index": { "on_disk": false } } },
  "hnsw_config": { "m": 16, "ef_construct": 200, "full_scan_threshold": 10000, "on_disk": false },
  "optimizers_config": { "indexing_threshold": 0, "default_segment_number": 2,
                         "max_optimization_threads": 2 },
  "on_disk_payload": true, "quantization_config": null }
```

`indexing_threshold: 0` is correct only because nothing queries during the load; restore to 20000
after. **19** payload indexes, not 26 — seven hold zero points (`assetId`, `assetVersionId`,
`freshness`, `lifecycle`, `manifestEligible`, `resourceDigest`, `resourceId`). `chunkRole` stays: both
values survive, and it is the filter the verification work uses.

**Container:** `QDRANT__PERFORMANCE__MAX_SEARCH_THREADS: "2"`; `mem_limit` stays 3g (384-d float16 at
500k points is ~380 MB against 386 MiB used today); drop the `OPENAI_*` lines; **pin
`FASTEMBED_CACHE_PATH=/tmp/fastembed_cache`** — the volume currently catches models by coincidence of
fastembed's built-in default, not configuration.

## Sequence

1. ~~Kill the queued index jobs.~~ Done — 8 dequeued, 1 stopping.
2. Code above; `uv run pytest` green.
3. Drop both collections and the `rag_index_jobs` rows. Optionally clear `rag_vector_cache` (436 rows
   become unreachable — the key includes the model and writes are `$setOnInsert`).
4. Deploy. **Window for the two commits held for it**: GeekBackend `2123a1b` then this repo's
   `406a77c`. Receiver first.

   `.env.hostinger` and `hostinger-compose.yml` are **already migrated** — `OPENAI_*` gone,
   `EMBEDDING_*` and `FASTEMBED_CACHE_PATH` in. Note the consequence: a `docker compose up -d` with
   the *old* image would now fail, because the old code reads settings the compose file no longer
   passes. The next `up -d` must be the new image. (Editing the env file cannot disturb the running
   container — compose reads it at `up` time and the current container's environment is already
   fixed.)
5. Recreate both collections with their payload indexes.
6. Index 2–3 runs. **Gate, then the rest.**
7. Restore `indexing_threshold: 20000`; poll `optimizer_status` on `GET /collections/{name}` — not
   `/cluster`, which has no such field.
8. `scripts/list_unindexed_runs.py --requeue`; unpause intake.

## The gate

**This is a production contract, not a formality.** Content Creator's declared-URL validation does not
stop at counting: it queries the RAG (`POST /v1/query`), verifies each quote against page text
(`GET /v1/pages/{id}` — `GccPartnerExtractionModels.cs:77`, *"True only after quote↔page-text verify"*),
and **refuses to write** when nothing citable comes back (`GccGroundingResolver`, refusal text
*"citable passage"*).

Quote verification itself is model-independent — it is string matching against corpus text, so it
cannot degrade. What can degrade is whether `/v1/query` surfaces the passage at all. A retrieval
regression therefore presents as **a refusal to produce content**, which is fail-closed and correct,
but it means a URL that validates as usable today can stop validating.

So the gate runs the real workflow, not a synthetic proxy:

1. **Baseline first, on the current OpenAI+SPLADE index.** Take the partner/competitor URLs that
   validate as usable today and record, per URL: does `/v1/query` return passages, do the quotes
   verify, does validation confirm usability.
2. Rebuild.
3. **Same URLs, same validation path.** The gate passes only if the set that confirms usable is the
   same size or larger.

Supplement it with a stratified query set, because the URL set alone may not exercise the risk:
sample **wide-parent units (ratio ≥ 2)** across the three shapes where a 200-token child cannot
carry the meaning — matrix-relational ("which tier includes multi-currency consolidation"),
multi-step synthesis, and distant negation (a qualifier hundreds of tokens from its subject). A
uniform sample draws ~71% from the region where the change provably costs nothing.

Threshold set before running, not after.

### The counts gate is safe, checked

`GccDeclaredUrlEvidence` requires `MinIndexedPages = 25` and `MinIndexedChunks = 250`.

- `RagPagesEnglish` is **untouched** — the parent gate removes duplicate points, not pages.
- `RagChunksUpserted` falls ~32%, and the smallest currently-passing run has 2,366 chunks → ~1,608,
  still 6.4× the threshold. No run in the corpus flips.

One narrow exposure to re-check after the rebuild: a site with 25+ pages but under ~370 chunks today
would cross. That needs fewer than ~15 chunks/page against the current average of 30.9, so it is
sparse-page sites only. None exist today.

## Verification

1. The gate above — go/no-go.
2. `uv run pytest` — count holds or rises.
3. Throughput from `rag_index_jobs` against the 5.2 chunks/s baseline.
4. Re-profile with py-spy: the 87% sparse frame must be gone.
5. `scripts/inspect_qdrant_vectors.py` reads `size` live, so it verifies 384 without a hardcoded
   expectation.
6. **Zero OpenAI, proved.** All four must come back empty — no exemption for comments, since the test
   is whether an occurrence *can be read as evidence*, not whether it executes:
   - `grep -rni openai --include='*.py' src/`
   - `grep -rni openai pyproject.toml uv.lock`
   - `grep -rni OPENAI .env.example deploy/ docs/ README.md architecture.md`
   - any request to `api.openai.com` during a full index run

## Follow-on: two permanently-zero wire fields

`embeddingRateLimitRetries` and `embeddingWaitSeconds` are pinned in
`contracts/rag-index-status/webhook.v1.json:21-22` with a byte-matching GeekBackend copy that
`.github/workflows/cross-repo-citation-contract.yml` diffs. Without the throttle they can only be zero
— which is the same shape as the defect the contract was written to prevent, so it is a deviation, not
a non-issue. Accepted here only because closing it needs a cross-repo window that should not be spent
while the throttle still has a live producer.

Survey: **zero references in `content-creator-v2`**, and the contract's own `$notes.persisted` confirms
neither reaches `crawl_runs` — they stop at SignalR and the `rag-index` projection.

- **Delete `embeddingRateLimitRetries`.** No local analogue; with `openai_embedding_max_retries: 0` it
  was already near-permanently zero by policy.
- **Rename `embeddingWaitSeconds` → `embeddingSecondsTotal`**, carrying real dense-inference seconds
  per run. More useful than what it replaces, and already plumbed end to end. Renamed, not repurposed:
  reporting inference time under `...WaitSeconds` is a quiet lie the next reader takes at face value.
- **Removal is runtime-safe in either merge order** (absent field → zero; unknown member ignored by
  `System.Text.Json`), so only the CI diff forces sequencing. **The rename is not** — a sender emitting
  a field the receiver does not bind is the 2026-09-29 defect exactly. GeekBackend first, red window on
  `main` between merges, GeekAPI deployed before the RAG.
- Sender sites: `models.py:45-46`, `indexer.py:489-506`, `status_store.py:497-500`,
  `llama_engine.py:130-132`, `app.py:299`, `tests/test_index_status_webhook_contract.py:133-134`, plus
  three `embedding_stats` mocks and `README.md:284-285`. Receiver:
  `GeekCrawlerRagWebhookController.cs:125-126,180,182`, `GeekCrawlerController.cs:220-221`,
  `HttpGeekCrawlerRagClient.cs:212-213,342-343,474`, and its contract test.
- While in there: `$notes.persisted` in both copies describes the receiver swallowing a persist failure
  and returning `Accepted`. GeekBackend `2123a1b` makes that false.

## Known unknowns

- **~10.4 chunks/s is arithmetic**, composed from measured parts, never run as a pipeline.
- **The token-weighted saving from the parent gate is unmeasured** — 32% is the point reduction.
- **No retrieval quality number for bge-small on this corpus.** MTEB parity says nothing about your
  data. This is the gate, and the thing that could send the plan back.

## Out of scope

- Re-crawling for composition (72–76% editorial). Separate variable; the tiering fix already shipped.
- Quantization. 380 MB against 7.7 GiB free. Trigger at ~60% of `mem_limit`, then scalar int8 with
  `always_ram` and rescoring — never binary, since a recall miss here is a failed quote.
- BM42 as the sparse fallback if BM25 ranks poorly: identity too (22/22) with learned weights, but
  Qdrant built it for short documents and it stacks a second local model onto 4 vCPUs.
- Backups. Single node, one volume, no Qdrant-side snapshots. Confirm what the existing backup covers.
- Rotating the exposed OpenAI key. Required regardless.
