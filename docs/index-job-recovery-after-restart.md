# Re-posting index jobs after an API restart

Nothing re-drives a stranded index job. An operator must re-post it. This is the
procedure.

## Why nothing recovers on its own

Two deliberate decisions, not gaps:

- `Indexer.start()` does **not** call `claim_recoverable` — *"that path was
  re-queuing cancelled deploy jobs. Operator must re-enqueue deliberately."*
- The scheduler is deprecated; `INDEX_SCHEDULER_ENABLED` stays `false`, so
  `find_smallest_content_ready_run` never runs.

The worker queue is an in-process `asyncio.Queue` at concurrency 1. **A restart
empties it.** The Mongo rows in `rag_index_jobs` survive, so they are the only
record of what was supposed to happen — and a row alone re-drives nothing.

## What a restart leaves behind

| How it stopped | Job row lands as | Re-post |
|---|---|---|
| `docker stop` / compose restart (SIGTERM → `lifespan` → `indexer.stop()`) | `failed`, error *"Indexer shut down while job was in flight"* | Immediately |
| SIGKILL / OOM / exit 137 (`stop()` never runs) | stays `pending`/`running`, lease alive up to 900s from the last 60s heartbeat | **Wait for the lease**, then re-post |
| RAG unreachable when the crawl finished | **no row at all** — `EnqueueIndexAsync` fails closed and returns null | Immediately |

That third row is not hypothetical. On 2026-09-24 eleven enqueues returned null
while RAG was 500ing on a dropped collection; every crawl closed looking healthy
and eight runs sat complete, content-ready and unindexed. GeekAPI now logs the
null (`GeekBackend@ fix(rag): a lost index enqueue leaves a trace against the
run`) — but logging it does not index it.

## A lapsed lease does not mean stranded

The trap. A job waiting in a **healthy** queue looks identical in Mongo to one
stranded by a restart: `pending`, lease lapsed. Concurrency is 1, so a queued job
burns its 900s lease waiting its turn. Observed 2026-09-24 18:16 — six queued
jobs all read *"lease lapsed 3s ago"* while the worker indexed happily ahead of
them. Re-posting those would have indexed every one of them twice.

`31a68af` is why the queued ones are fine: `_run_claimed_job` calls `reacquire()`
when it reaches a job, so a lease that lapsed while queued is simply retaken.

**The discriminator is the process, not the lease.** The queue dies with the
process, so compare `claimedAtUtc` against the app's start time (`/proc/1`):

- `claimedAtUtc >= process start` → in the live worker's queue. Leave it alone.
- `claimedAtUtc < process start` → the process that queued it is gone. Stranded.

## A hung job is not a stranded job — do not re-post it

Learned the hard way on 2026-09-25. A job can sit `running` while advancing nothing, and
re-posting it makes things **worse**: `claim(force=True)` rewrites the Mongo row but does
**not** cancel the running asyncio task. The zombie keeps holding
`LlamaIndexEngine._embedding_call_lock`, which every embed call needs, so the new attempt
sails through the resume-skip batches (skips need no embedding) and then blocks forever on
a lock nobody will release. Two jobs, one lock, no progress, and nothing in the log
because a lock acquisition prints nothing.

**Tell them apart by the heartbeat, not the state.** A live job renews its lease every 60s.

```bash
# leaseUntil == claimedAtUtc + 900s exactly  =>  the heartbeat never ran  =>  hung
```

| Symptom | Fix |
|---|---|
| `pending`, no worker behind it | re-post (below) |
| `running`, `leaseUntil` advancing, chunk count climbing | leave it alone |
| `running`, `leaseUntil` frozen at `claimedAt + 900s` | **restart the process, then re-post** |

Restart with `docker restart geek-crawler-rag-api-1`, **not** `docker compose up -d`: a
restart keeps the container's filesystem and therefore the 509 MB SPLADE model in
`/tmp/fastembed_cache`, which is not a volume and is re-downloaded on every recreate.

And check `docker events` before theorising. An OOM kill looks exactly like a hang from
the job's side — the queue dies with the process and the row is left `running`:

```bash
docker events --since '<ISO time>' --until "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  --filter container=geek-crawler-rag-api-1 --filter event=oom --filter event=die
```

`docker inspect` is **not** authoritative here: after the restart it reports
`OOMKilled=false` even when the event log records `oom` then `die exitCode=137`.

## The procedure

`scripts/list_unindexed_runs.py` does both halves. `scripts/` is not in the
image, so pipe it from a checkout:

```bash
# 1. Look. Read-only.
ssh -i ~/.ssh/hostinger_rag_ed25519 root@<vps> \
  'docker exec -i geek-crawler-rag-api-1 python -' < scripts/list_unindexed_runs.py

# 2. Act. Re-posts every re-postable run, smallest first.
ssh -i ~/.ssh/hostinger_rag_ed25519 root@<vps> \
  'docker exec -i geek-crawler-rag-api-1 python - --requeue' < scripts/list_unindexed_runs.py
```

It classifies every content-ready run as `COMPLETE`, `QUEUED_LIVE` (leave alone),
`NEVER_QUEUED` / `FAILED` / `STRANDED` (re-post), or `HELD_DEAD` (wait, then
re-post) and re-posts only the middle group.

`--skip-failed` leaves `FAILED` runs alone. Use it for anything unattended: a failed job
may be quarantined, and [`embedding-circuit-recovery.md`](./embedding-circuit-recovery.md)
says a person examines a quarantine before requeueing it. `NEVER_QUEUED` and `STRANDED`
carry no such decision — nothing is waiting on a human — so those still go.

### Unattended (installed on the VPS, 2026-09-25)

`/docker/geek-crawler-rag/requeue-stranded.sh`, on a `*/30` crontab, logging to
`/var/log/rag-requeue.log`. It runs the same script with `--requeue --skip-failed`, so it
only ever *adds* to the queue and never stops, cancels or restarts anything. It fixes
stranding; it does **not** rescue a hung job (see above), which is deliberate — detecting
a hang reliably means auto-restarting the container, and a long resume-scan is
indistinguishable from a hang for minutes at a time.

```bash
tail -40 /var/log/rag-requeue.log     # "re-postable: 0" repeated = nothing needed rescuing
```

Remove it with `crontab -e`. Note the copy at `/docker/geek-crawler-rag/` is a copy, not a
symlink — update it when this script changes.

By hand, the same call:

```bash
# Note: 8080, not 8000. And there is no curl in the image.
docker exec geek-crawler-rag-api-1 python -c '
import json, os, urllib.request
r = urllib.request.Request("http://127.0.0.1:8080/v1/index",
    data=json.dumps({"runId": "<guid>"}).encode(),
    headers={"Content-Type": "application/json", "X-Api-Key": os.environ["API_KEY"]},
    method="POST")
print(urllib.request.urlopen(r, timeout=30).read().decode())'
```

## Two things that will mislead you

**A refused claim looks like a successful one.** `claim(force=True)` will not
take a row that is `pending`/`running` with a live lease, but `_enqueue` discards
its accepted flag — so the POST returns **HTTP 200 carrying the stale row** and
queues nothing. `state=pending` in the response is not proof anything was
accepted. Check `attempt` incremented, or the log line `Enqueued index job for
runId=… trigger=manual`.

**Re-posting resumes; it does not duplicate.** Point IDs are deterministic and
`attempt > 1` skips the delete-by-runId at start, so committed chunks are
skipped. A first attempt *does* delete the run's existing points first
(`Deleted Qdrant points for runId=…`) — that is the idempotent rebuild, not data
loss.

## Verifying it took

```bash
ssh -i ~/.ssh/hostinger_rag_ed25519 root@<vps> \
  'docker logs --since 5m geek-crawler-rag-api-1 2>&1 | grep -E "Enqueued index job|Index complete|stale index worker"'
```

`Index complete for runId=… chunksUpserted=…` is the finish line. GeekAPI also
mirrors the outcome onto the run itself (`RagState`, `RagChunksUpserted`,
`RagPagesEnglish`) via the status webhook, so a `RagState` of `pending` on a run
whose job row says `complete` means the webhook, not the indexer, is what failed.

Related: [`embedding-circuit-recovery.md`](./embedding-circuit-recovery.md) for a
job that failed on an embedding batch rather than a restart.
