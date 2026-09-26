# Qdrant stability plan (from qdrant-advisor / skills.qdrant.tech)

## Problem framing (current deployment)
- **Deploy:** Docker Compose self-hosted on Hostinger VPS (~8 GiB RAM, 2 CPUs), Qdrant **v1.13.4**, `mem_limit: 3g`, API `2g`, Mongo ~2.7 GiB colocated.
- **Corpus:** `geek_crawler_chunks` ~**760k+** points, **1536-d** Cosine (`text-embedding-3-small`), `on_disk_payload: true` already; many **KEYWORD** + **TEXT** payload indexes (`text` / `childText` / `parentText`).
- **Symptoms seen:** RocksDB **too many open files** (fixed `nofile=65535`); then **process Killed** under write load; collection **yellow** while optimizer runs; multi-hour partner indexes; progress wiped on retry.

## Capacity check (skills: sizing)
Rough resident estimate (vectors in RAM):

`vectors × dims × 4 × 1.5` ≈ `760_000 × 1536 × 4 × 1.5` ≈ **~7 GiB** for vectors alone — **before** HNSW, ID tracker, payload indexes, optimizer temp segments, Mongo, or API.

**Most likely cause (priority order, per monitoring skill):**
1. **Working set / resident RAM exceeds Qdrant’s 3 GiB cap** during ingest + optimize → SIGKILL / OOM-class failure.
2. **Optimizer competing with writes** (yellow status, high CPU/iowait) → slow ingest and unstable latency.
3. **Heavy TEXT payload indexes** on long fields → extra RAM + slower HNSW/filterable-index builds.
4. Client/process issues we already hit (nofile, no mid-run job progress, OpenAI 500 not retried) — secondary but real.

Canonical refs: [Sizing](https://skills.qdrant.tech/search?query=capacity+planning+memory+estimate+vectors), [Memory usage](https://skills.qdrant.tech/search?query=on_disk+memmap+cold+memory+tier+vectors), [Bulk upload](https://skills.qdrant.tech/md/documentation/manage-data/bulk-upload/), [Indexing performance](https://skills.qdrant.tech/search?query=qdrant+indexing+performance+optimizer+memory), [Monitoring/debug](https://skills.qdrant.tech/search?query=high+memory+usage+OOM+RAM).

## What NOT to do (from skills)
- Do **not** size as `points × dims × 4` only.
- Do **not** change collection/optimizer knobs **while** a heavy optimize is mid-flight unless necessary (cascading re-optimizations).
- Do **not** add more TEXT payload indexes on long body fields.
- Do **not** raise Qdrant `mem_limit` without freeing host RAM (Mongo+API already consume most of 8 GiB).
- Do **not** treat “page cache filled RAM” alone as a leak; watch **container RSS / anon** and kills.
- Do **not** recommend dual concurrent indexers on this box.

## Target architecture (stay on Hostinger for now)
Keep self-hosted Docker (matches “you own ops”). Make the collection **disk-first** so a 3 GiB container can survive ~1M×1536 vectors; keep HNSW searchable with controlled RAM.

## Implementation phases

### Phase 0 — Finish reliability patch already in flight (app)
Already hot-patched / in working tree; **commit + rebuild image** so it is not lost on recreate:
- Mid-run Mongo persist of `pagesSeen` / `chunksUpserted` (`src/geek_crawler_rag/indexer.py`)
- Retry OpenAI **5xx / connection / timeout** (`src/geek_crawler_rag/llama_engine.py`)
  — **DONE 2026-09-26.** Blocked for weeks because it contradicted
  `.cursor/rules/no-retries-no-fallbacks.mdc` (`alwaysApply: true`), which rejects
  "retry on 502/500" by name. Resolved by amending §3a rather than smuggling it past the
  rule: implemented in `LlamaIndexEngine._embed_batch`, bounded and logged, narrowed to
  failures with no cause in this repo. 400 and 429 are still never retried. Read the
  amendment in `plans/rules.md` §3a before touching this.
- Skip Qdrant delete-by-runId when `attempt > 1` (deterministic point IDs)
- Keep `nofile=65535`, **4 GiB swap**, `QDRANT_UPSERT_DELAY_SECONDS=0.5`, search threads **1**
- Remove debug `_agent_dbg` instrumentation after verification

### Phase 1 — Make collection fit RAM (Qdrant config, v1.13.4 APIs)
On **v1.13.4** use `on_disk` (not 1.19 `memory:` tiers):

1. Snapshot / confirm backups before mutation.
2. Update collection vectors to **`on_disk: true`** (and HNSW `on_disk: true` only if still memory-bound after vectors move — skill warns latency cost; prefer vectors-on-disk first on NVMe).
3. Revisit payload indexes in `qdrant_store._ensure_payload_indexes`:
   - Keep **KEYWORD** indexes used in filters (`runId`, `host`, `chunkRole`, …).
   - **Stop creating TEXT indexes** on `text` / `childText` / `parentText` for new collections (or mark `on_disk: true` if kept). These dominate RAM/optimizer cost and are not required for your dense+rerank path.
4. Optionally enable **scalar quantization** with quantized vectors in RAM for search, originals on disk — validate recall before relying on it.
5. Verify via `/collections/geek_crawler_chunks`, `/metrics`, container RSS **&lt; ~80% of 3 GiB** under ingest.

### Phase 2 — Bulk ingest profile (while backfilling partners)
Per bulk-upload + indexing skills:

- Client batches **64–128** points (today `embed_batch_size=32`) — raise carefully given 3 GiB cap.
- Keep **one** upload stream (do **not** use 2–4 parallel uploads on this VPS).
- During large backfill: temporarily raise `indexing_threshold_kb` so HNSW builds after load, then restore; or accept yellow optimizer and **pause query SLAs** during backfill.
- Lower `optimizer_cpu_budget` if search must stay responsive during writes.

### Phase 3 — Observability (skills: self-host must monitor)
- Scrape Qdrant `/metrics` + collection `optimizer_status` (and `/collections/.../optimizations` if upgrading past 1.17).
- Alert: container restart/OOM, RSS &gt; 85% of `mem_limit`, `optimizer_status` error, upsert 5xx rate.
- Surface mid-run index job counters in GeekAPI/SignalR (already webhook-capable).

### Phase 4 — Version / sizing decision (later)
- Align **server and client** (now server 1.13.4 vs client 1.19 warning).
- If corpus grows toward multi-million 1536-d points: either **larger VPS / dedicated Qdrant host**, or **Qdrant Cloud** (skill: choose Cloud when you do not want to own ops). Re-run sizing calculator before expanding.

## Success criteria
- No Qdrant `Killed` / red collection for 24h of continuous indexing.
- Qdrant RSS stays under ~**2.4 GiB** of 3 GiB during partner upserts.
- At least one mid-size run (e.g. Zoho ~946 pages) reaches **`complete`** with non-zero Mongo progress throughout.
- Remaining pending runs drain under scheduler without wipe-on-retry.

## Suggested first execution order
1. Let current Zoho run finish (or fail cleanly) under mid-run persist.
2. Commit/rebuild API with reliability fixes; remove debug instrumentation.
3. Apply **vectors `on_disk: true`** + trim TEXT indexes.
4. Resume pending partners; watch `/metrics` + RSS.
5. Only then consider quantization / HNSW-on-disk / Cloud move.

## Todos
- [ ] Commit/rebuild API: mid-run persist, OpenAI 5xx retry, skip delete on attempt>1; remove agent debug logs
- [ ] Update geek_crawler_chunks to vectors on_disk (v1.13); verify RSS under ingest
- [ ] Stop TEXT indexes on text/childText/parentText (keep keyword filters); document migration
- [ ] Tune embed batch 64–128, keep concurrency=1, optional indexing_threshold during backfill
- [ ] Add Qdrant /metrics + optimizer/RSS alerts; confirm one full partner complete
