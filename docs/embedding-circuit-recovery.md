# Quarantine workflow for failed index runs

## Workflow

```
Run fails
  -> 1. Quarantine (do not auto-requeue; keep Qdrant points)
  -> 2. Examine for anomalies
  -> 3. Can salvage/fix? (empty text cleaning, delete bad chunk, …)
       Yes -> fix issue -> manual requeue that runId
       No  -> usable data exists -> keep data, report, do not requeue
```

The indexer continues other runs as soon as step 1 completes. Steps 2–3 are operator decisions.

### 1. Fail → quarantine

- Job state becomes `FAILED` with the **real** error text.
- Embed batch failures (HTTP **500** or empty-input **400**) also write `failed_embedding_*.json` under `EMBEDDING_QUARANTINE_DIR`.
- **No** Qdrant wipe on fail/cancel/stop (so “usable data exists” is still true when it should be).
- **No** auto-requeue: no `nextRetryAtUtc` on FAILED; FAILED/SKIPPED excluded from scheduler; API start does **not** `claim_recoverable`.

### 2. Examine

| Situation | Where |
|-----------|--------|
| Embed 400/500 after this ships | Quarantine dump (`items[]`, `statusCode`, page/chunk ids) + job `error` (includes path) |
| Cancel / shutdown | API logs + job counters + Qdrant point count for `runId` |
| Before dump existed | API docker logs only |

Park stub files that only say “stopped requeue” are not examination.

### 3. Salvage

| Answer | Action |
|--------|--------|
| **Yes** | Delete/fix the issue (e.g. empty embed texts are skipped before OpenAI) → manual `POST /v1/index` |
| **No** | Usable data exists → keep points, report, do not requeue |

`OPENAI_EMBEDDING_MAX_RETRIES=0` — no in-process backoff loop.

On `attempt > 1`, indexer skips delete-by-runId at start; upserts overwrite deterministic point IDs.

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

On quarantine dump the API emits:

`embedding_circuit_open {"event":"embedding_circuit_open","runId":...,"quarantinePath":...,"batchSize":...,"statusCode":400|500,...}`

Job `error` includes `Quarantine: <path>`. GeekAPI index-status webhook fires when configured.

## What "clean payloads" means

Sanitizer (`embedding_sanitize.py`) strips encoding/control junk only. Empty/whitespace texts are **skipped** before OpenAI (not sent). Token limits remain in `partition_embedding_batches`.

## Phase C notes (vectors on_disk)

- Target: Qdrant RSS under ~**80% of 3 GiB** during ingest.
- `on_disk: true` on 1536-d; Qdrant **v1.13.4**.
