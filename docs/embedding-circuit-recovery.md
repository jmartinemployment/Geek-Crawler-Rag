# Embedding circuit recovery runbook

## State machine

```
Job starts (attempt N)
  Batch 1..K: HTTP 200 -> upsert (points kept; deterministic IDs)
  Batch K+1: HTTP 500 -> quarantine JSON -> EmbeddingCircuitOpen
  Job FAILED immediately (no further batches; no Qdrant wipe)
```

- **Per-batch failure aborts the whole job** (not "skip and continue").
- **Recovery is deliberate**, not automatic in-process retry.
- `OPENAI_EMBEDDING_MAX_RETRIES=0` — exponential backoff is off by design.
- On `attempt > 1`, indexer **skips delete-by-runId**; upserts overwrite the same point IDs.

## Where quarantine files live

| Env | Path |
|-----|------|
| Hostinger Docker volume `embedding_quarantine` | `/var/lib/geek-crawler-rag/embedding_quarantine` |
| Config / env | `EMBEDDING_QUARANTINE_DIR` |

Files survive API container recreate. Default retain hint: **14 days**. Prune:

```bash
docker exec "$(docker ps -qf name=geek-crawler-rag-api)" \
  find /var/lib/geek-crawler-rag/embedding_quarantine -type f -mtime +14 -delete
```

## Alerting

On circuit open the API emits a structured ERROR log line:

`embedding_circuit_open {"event":"embedding_circuit_open","runId":...,"quarantinePath":...,"batchSize":...,"statusCode":500,...}`

Wire log aggregation (or Grep Docker logs) on `embedding_circuit_open`. Index job status also becomes `failed` with the quarantine path in `error`, and the existing GeekAPI index-status webhook fires when configured.

## Recovery steps (operator)

1. Confirm OpenAI is healthy (billing / status) and that the failure was 500 not 429.
2. Inspect quarantine JSON on the volume (pageId/chunkId previews). Fix payload issues if sanitizer missed something.
3. Requeue the same `runId` via the normal index enqueue/scheduler path (do **not** wipe Qdrant manually).
4. Because point IDs are deterministic and attempt>1 skips wipe, already-good chunks are overwritten idempotently; the previously failing batch is retried as part of the full run.
5. After success, delete or archive the quarantine file.

There is no separate `retry_quarantine` job yet — requeue the crawl run.

## What "clean payloads" means

Sanitizer (`embedding_sanitize.py`) only strips **encoding/control junk**:

- Null bytes and C0 controls (keeps tab/LF/CR)
- Lone UTF-8 surrogates → replacement
- Zero-width / BOM / soft hyphen

It does **not** strip HTML tags, truncate for tokens, or rewrite semantics. Batch token limits remain enforced by `partition_embedding_batches`.

## Phase C notes (vectors on_disk)

- Target: Qdrant container RSS under ~**80% of 3 GiB** (~2.4 GiB) during ingest — safety margin before Docker `mem_limit` OOM/kill (host already swap-backed).
- Expect higher search I/O latency with `on_disk: true` on 1536-d; acceptable tradeoff on this 8 GiB Hostinger box. Re-measure P95 after optimizer returns **green**.
- Server is Qdrant **v1.13.4** (`VectorParamsDiff.on_disk`).
