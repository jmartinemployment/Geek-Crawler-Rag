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
- **Every** embed failure writes `failed_embedding_*.json` under `EMBEDDING_QUARANTINE_DIR`, carrying
  `reason`, `model`, `batchSize`, `tokenCount`, per-item previews, and `detail` — the exception chain,
  innermost first, because ONNX and tokenizer errors put the useful text on the innermost exception.
- **No** Qdrant wipe on fail/cancel/stop (so “usable data exists” is still true when it should be).
- **No** auto-requeue: no `nextRetryAtUtc` on FAILED; FAILED/SKIPPED excluded from scheduler; API start does **not** `claim_recoverable`.

### 2. Examine

| Situation | Where |
|-----------|--------|
| Any embed failure | Quarantine dump (`items[]`, `reason`, `detail`, page/chunk ids) + job `error` (includes the path) |
| Cancel / shutdown | API logs + job counters + Qdrant point count for `runId` |
| Before dump existed | API docker logs only |

Park stub files that only say “stopped requeue” are not examination.

## Diagnosing an embed failure

Embeddings are computed in this process. There is no provider status page to check and no request id
to quote — if it failed, the cause is local: the model, the input, or the host.

1. **Quarantine JSON** (`EMBEDDING_QUARANTINE_DIR/failed_embedding_*.json`):
   ```json
   {
     "reason": "inference_failed",
     "excType": "RuntimeError",
     "model": "BAAI/bge-small-en-v1.5",
     "batchSize": 64,
     "detail": "RuntimeError: inference failed <- ValueError: tokenizer vocab missing"
   }
   ```

2. **API log event** (`embedding_circuit_open`):
   ```
   embedding_circuit_open {"runId":"...","reason":"inference_failed","batchSize":64,"detail":"..."}
   ```

**`reason` is the first thing to read.**

| `reason` | What it means | Where to look |
|---|---|---|
| `inference_failed` | ONNX could not produce vectors for the batch | `detail`'s innermost exception; then host memory and the model cache volume |
| `empty_input` | An empty string reached the embedder | Not an inference fault. `embedding_sanitize` and the empty-string guard in `llama_engine` are supposed to make this unreachable, so it points at a hole in one of them |

**A truncated input is not a failure and does not quarantine.** The model silently truncates at 512
tokens; `LocalDenseEmbedding` counts every input that reaches the limit, logs it with its token count,
and reports the total as `truncatedInputs` in `embedding_stats`. It does not raise, because the chunker
sizes chunks with tiktoken BPE while the model counts WordPiece — which runs longer on technical text —
so a chunk inside its configured budget can legitimately cross the ceiling. A rising count is the
signal to lower `PARENT_CHUNK_SIZE_TOKENS`, not an incident.

### 3. Salvage

| Answer | Action |
|--------|--------|
| **Yes** | Delete/fix the issue (e.g. an empty embed text that should have been filtered) → manual `POST /v1/index` |
| **No** | Usable data exists → keep points, report, do not requeue |

**Nothing retries an embed call.** A bounded retry lived in `LlamaIndexEngine._embed_batch`
between 2026-09-26 and 2026-09-28 and was removed: it made a failed attempt invisible
whenever the next one succeeded, and it re-sent tokens the account may already have been
billed for. A quarantine therefore means what it has always meant — the call failed, once.
The failures it was added for had a cause, found on 2026-09-10: the throttle was set to the
account's exact token ceiling. Keep it well under. See `plans/rules.md` §3a.

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

On quarantine dump the API emits a structured log event:

`embedding_circuit_open {"event":"embedding_circuit_open","runId":...,"quarantinePath":...,"batchSize":...,"tokenCount":...,"model":...,"reason":"inference_failed|empty_input","detail":"...","excType":...}`

Fields:
- `reason`: `inference_failed` or `empty_input` — a slug, not an HTTP status. There is no transport
  to carry one, and the field it replaced invited a reader to look for an API response that does not
  exist.
- `detail` (when available): the exception chain, innermost last, joined with `<-`
- `model`: which embedding model produced the failure, so a mixed-model corpus is diagnosable
- `excType`: Exception class name (e.g., `InternalServerError`, `BadRequestError`)

Job `error` includes `Quarantine: <path>`. GeekAPI index-status webhook fires when configured.

## What "clean payloads" means

Sanitizer (`embedding_sanitize.py`) strips encoding/control junk only. Empty and whitespace-only texts
are **skipped** before the embedder, never passed to it — fastembed returns a vector for `""` without
raising, so an empty chunk would otherwise be stored as a real, retrievable point. Batch token limits
remain in `partition_embedding_batches` (`embedding_batching.py`), whose counts are tiktoken BPE and
therefore size batches without being an authority on whether one item fits the model.

## Phase C notes (vectors on_disk)

- Target: Qdrant RSS under ~**80% of 3 GiB** during ingest.
- Dense 384-d (`bge-small-en-v1.5`) held in RAM, sparse `Qdrant/bm25` with the `idf` modifier;
  Qdrant **v1.13.4**.
