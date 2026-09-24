#!/usr/bin/env python3
"""Find content-ready crawls that no index job is going to pick up, and optionally re-post them.

Read-only by default. `--requeue` is the only writing path and it writes nothing
to Mongo or Qdrant directly — it POSTs /v1/index, the same call an operator makes
by hand.

Why this exists
---------------
Nothing re-drives a stranded index job. That is deliberate in two separate
places:

  * `Indexer.start()` does not call `claim_recoverable` -- "that path was
    re-queuing cancelled deploy jobs. Operator must re-enqueue deliberately."
  * The scheduler is deprecated and `INDEX_SCHEDULER_ENABLED` stays false, so
    `find_smallest_content_ready_run` never runs.

The worker queue is an in-process `asyncio.Queue`. A container restart empties
it and leaves the Mongo rows behind, so `rag_index_jobs` is the only durable
record of what was supposed to happen. Two incidents on 2026-09-24 produced
exactly that state: eleven enqueues returned null while RAG was 500ing on a
dropped collection (`GeekBackend@fix(rag): a lost index enqueue leaves a trace`),
and ten queued jobs died of lease loss on reaching the worker
(`31a68af`). Eight runs sat complete, content-ready and unindexed with no one
watching.

A lapsed lease does not mean stranded
-------------------------------------
This is the part that is easy to get wrong, and getting it wrong double-indexes.

Concurrency is 1, so a queued job waits for everything ahead of it while its
lease -- taken at enqueue, 900s -- runs down. A job sitting in a healthy queue
behind a long run therefore looks *identical in Mongo* to a job stranded by a
restart: state `pending`, lease lapsed. Observed live on 2026-09-24 18:16:27,
six queued jobs all read "lease lapsed 3s ago" while the worker was perfectly
happily indexing ahead of them. Re-posting those would have queued every one of
them a second time.

`31a68af` is why the queued ones are fine: `_run_claimed_job` calls
`reacquire()` when it reaches a job, so a lease that lapsed while waiting is
simply retaken.

The discriminator is not in the lease, it is the process. The in-process queue
dies with the process, so:

  claimedAtUtc >= this process's start  -> the job is in THIS worker's queue.
                                          It will run. Leave it alone.
  claimedAtUtc <  this process's start  -> the process that queued it is gone
                                          and nothing re-drives it. Stranded.

Process start is read from `/proc/1` (PID 1 is the app under tini), which is
namespace-correct inside the container. This assumes ONE api replica, which is
what the Hostinger compose runs; with two, `leaseOwner` would have to be
compared against each live worker's owner uuid instead.

The six states, and which are re-postable
-----------------------------------------
`IndexStatusStore.claim(force=True)` -- what a manual POST uses -- matches a row
whose state is not pending/running, OR one that is pending/running with an
expired or absent lease. Crossed with the liveness test above:

  NEVER_QUEUED  no job row                          -> re-post (lost-enqueue case)
  FAILED        state=failed/skipped                -> re-post (graceful shutdown
                                                       lands here: `stop()` marks
                                                       in-flight jobs failed)
  STRANDED      pending/running, queued by a dead
                process, lease lapsed               -> re-post (SIGKILL lands here)
  HELD_DEAD     pending/running, queued by a dead
                process, lease still live           -> WAIT, then re-post. A POST
                                                       now returns HTTP 200 with
                                                       the stale row and enqueues
                                                       nothing.
  QUEUED_LIVE   pending/running, queued by the
                running process                     -> leave alone, it will run
  COMPLETE      state=complete                      -> nothing to do

HELD_DEAD is the trap, because the refusal is silent: `_enqueue` discards its
accepted flag, so `enqueue()` returns the existing row and the caller cannot
tell a fresh claim from a refusal. This script reads the lease and the process
clock itself rather than trusting the response.

Usage
-----
The image does not ship `scripts/`, so pipe this in from a checkout:

  ssh -i ~/.ssh/hostinger_rag_ed25519 root@<vps> \
    'docker exec -i geek-crawler-rag-api-1 python -' < scripts/list_unindexed_runs.py

  # and to act on it, passing the flag through to the piped script:
  ssh -i ~/.ssh/hostinger_rag_ed25519 root@<vps> \
    'docker exec -i geek-crawler-rag-api-1 python - --requeue' < scripts/list_unindexed_runs.py

Locally, where the package is importable and Mongo is reachable:

  uv run python scripts/list_unindexed_runs.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

from geek_crawler_rag.config import get_settings
from geek_crawler_rag.mongo import MongoCorpus
from geek_crawler_rag.status_store import COLLECTION as JOBS_COLLECTION

# The states TERMINAL_CRAWL_STATUSES names, kept local so this script reports on
# exactly what the scheduler filter would have considered indexable.
INDEXABLE_CRAWL_STATUSES = ("complete", "external")

NEVER_QUEUED = "NEVER_QUEUED"
FAILED = "FAILED"
STRANDED = "STRANDED"
HELD_DEAD = "HELD_DEAD"
QUEUED_LIVE = "QUEUED_LIVE"
COMPLETE = "COMPLETE"

REPOSTABLE = (NEVER_QUEUED, FAILED, STRANDED)


def _as_utc(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


def process_started_at() -> datetime | None:
    """When the process holding the in-memory queue started.

    PID 1 is the app (tini -> uvicorn), and /proc is namespaced, so this is the
    container's app start even though the script runs as a second process beside
    it. None when /proc is unreadable, which makes every liveness call below
    report unknown rather than guess.
    """
    try:
        return datetime.fromtimestamp(os.stat("/proc/1").st_ctime, timezone.utc)
    except Exception:  # noqa: BLE001 - a diagnostic reports, it does not raise
        return None


def classify(
    job: dict | None, now: datetime, started: datetime | None
) -> tuple[str, str]:
    """The row's state, and a human reason. Never raises -- a diagnostic reports."""
    if job is None:
        return NEVER_QUEUED, "no rag_index_jobs row -- the enqueue never landed"

    state = str(job.get("state") or "")
    if state == "complete":
        return COMPLETE, f"indexed, chunksUpserted={job.get('chunksUpserted')}"
    if state in ("failed", "skipped"):
        return FAILED, f"state={state} error={job.get('error')!r}"
    if state not in ("pending", "running"):
        return NEVER_QUEUED, f"unrecognised state={state!r}"

    lease = _as_utc(job.get("leaseUntil"))
    claimed = _as_utc(job.get("claimedAtUtc"))
    lease_note = (
        "no lease"
        if lease is None
        else f"lease lapsed {int((now - lease).total_seconds())}s ago"
        if lease <= now
        else f"lease live for another {int((lease - now).total_seconds())}s"
    )

    # Whose queue is it in? That, not the lease, decides whether anything will
    # ever pick this job up again.
    if started is None or claimed is None:
        return (
            STRANDED,
            f"state={state}, {lease_note}; could not establish liveness "
            f"(processStart={started} claimedAtUtc={claimed}) -- treating as stranded",
        )

    if claimed >= started:
        return (
            QUEUED_LIVE,
            f"state={state}, {lease_note}; queued by the running worker "
            f"({int((claimed - started).total_seconds())}s after it started) -- it will run",
        )

    if lease is not None and lease > now:
        return (
            HELD_DEAD,
            f"state={state}, {lease_note}, but queued before this process started "
            f"-- a POST would be silently refused; wait "
            f"{int((lease - now).total_seconds())}s",
        )

    return (
        STRANDED,
        f"state={state}, {lease_note}; queued by a process that is gone -- nothing re-drives it",
    )


def post_index(run_id: str, *, base_url: str, api_key: str) -> str:
    """POST /v1/index. Returns a one-line outcome; never raises."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/index",
        data=json.dumps({"runId": run_id}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Api-Key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.load(response)
        return f"accepted state={body.get('state')} attempt={body.get('attempt')}"
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:200]
        return f"HTTP {err.code}: {detail}"
    except Exception as err:  # noqa: BLE001 - a diagnostic reports, it does not raise
        return f"{type(err).__name__}: {err}"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requeue",
        action="store_true",
        help="POST /v1/index for every re-postable run, smallest first. Off by default.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8080"),
        help="Where the API listens. Inside the container that is port 8080.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50_000,
        help="Skip runs larger than this, matching the scheduler's safety cap.",
    )
    args = parser.parse_args()

    settings = get_settings()
    corpus = MongoCorpus(settings.mongo_crawler_url, settings.mongo_db_name)
    runs = corpus.db["crawl_runs"]
    pages = corpus.db["crawl_pages"]
    jobs = corpus.db[JOBS_COLLECTION]
    now = datetime.now(timezone.utc)
    started = process_started_at()

    rows: list[dict] = []
    cursor = runs.find(
        {
            "Status": {"$in": list(INDEXABLE_CRAWL_STATUSES)},
            "ContentReadyAt": {"$exists": True, "$nin": [None, ""]},
            "Id": {"$type": "string", "$ne": ""},
        },
        {"_id": 0, "Id": 1, "Status": 1, "CrawlType": 1, "ContentReadyAt": 1},
    )
    async for doc in cursor:
        run_id = str(doc.get("Id") or "")
        if not run_id:
            continue
        page_count = await pages.count_documents({"RunId": run_id})
        job = await jobs.find_one({"runId": run_id})
        state, reason = classify(job, now, started)
        rows.append(
            {
                "runId": run_id,
                "crawlType": str(doc.get("CrawlType") or ""),
                "pages": page_count,
                "state": state,
                "reason": reason,
            }
        )

    await corpus.close()

    if not rows:
        print("No content-ready runs found. Nothing to report.")
        return 0

    # Smallest first: the queue is FIFO at concurrency 1, so this is the execution
    # order. A pipeline fault then surfaces on the cheapest run rather than after
    # the most expensive one.
    rows.sort(key=lambda row: row["pages"])

    print(f"{len(rows)} content-ready run(s), {now.isoformat()}")
    print()
    for row in rows:
        print(
            f"  {row['runId']}  pages={row['pages']:>5}  "
            f"{row['crawlType']:<13} {row['state']:<12} {row['reason']}"
        )

    repostable = [r for r in rows if r["state"] in REPOSTABLE and r["pages"] > 0]
    oversized = [r for r in rows if r["state"] in REPOSTABLE and r["pages"] > args.max_pages]
    empty = [r for r in rows if r["state"] in REPOSTABLE and r["pages"] <= 0]
    held = [r for r in rows if r["state"] == HELD_DEAD]
    repostable = [r for r in repostable if r["pages"] <= args.max_pages]

    print()
    print(f"re-postable: {len(repostable)}  held: {len(held)}  "
          f"oversized: {len(oversized)}  zero-page: {len(empty)}  "
          f"total pages to index: {sum(r['pages'] for r in repostable)}")

    if held:
        print()
        print("HELD_DEAD runs are skipped: a POST would return 200 and enqueue nothing.")
        for row in held:
            print(f"  {row['runId']}  {row['reason']}")

    if not args.requeue:
        print()
        print("Read-only. Re-run with --requeue to POST /v1/index for the re-postable runs.")
        return 0

    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        print()
        print("FAILED: API_KEY is not set in this environment, so /v1/index would 401.",
              file=sys.stderr)
        return 2

    print()
    print(f"Re-posting {len(repostable)} run(s), smallest first:")
    for row in repostable:
        outcome = post_index(row["runId"], base_url=args.base_url, api_key=api_key)
        print(f"  {row['runId']}  pages={row['pages']:>5}  -> {outcome}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
