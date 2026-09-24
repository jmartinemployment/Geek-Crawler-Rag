#!/usr/bin/env python3
"""Delete stale `rag_index_jobs` documents by runId.

A job left `pending` or `running` by a killed container never finishes and never clears: the worker
re-claims it, and it sits in the queue counting against anything that asks whether indexing is
idle. Removing the document is how the queue forgets it.

Identity note, because getting this wrong deletes nothing while looking like it worked: the `_id`
of a job document is an ObjectId. The run's GUID lives in `runId`. A filter of
`{"_id": "<guid>"}` matches zero documents and reports a clean `deleted_count` of 0.

Refuses to delete a job that is genuinely in flight -- `running` with a lease that has not expired
-- because that one is not stale, it is working. Everything else is fair game.

Usage:
  uv run python scripts/purge_stale_job.py <runId> [<runId> ...]
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from pymongo import MongoClient

from geek_crawler_rag.config import Settings
from geek_crawler_rag.status_store import COLLECTION as INDEX_JOBS_COLLECTION


def is_live_job(job: dict, now: datetime) -> bool:
    """True when the job is running under a lease that has not expired."""
    if str(job.get("state") or "") != "running":
        return False
    lease_until = job.get("leaseUntil")
    if not isinstance(lease_until, datetime):
        return False
    if lease_until.tzinfo is None:
        lease_until = lease_until.replace(tzinfo=timezone.utc)
    return lease_until > now


def describe(job: dict) -> str:
    return (
        f"state={job.get('state')} "
        f"claimedAtUtc={job.get('claimedAtUtc')} "
        f"leaseUntil={job.get('leaseUntil')} "
        f"leaseOwner={job.get('leaseOwner')} "
        f"pagesSeen={job.get('pagesSeen') or 0} "
        f"chunksUpserted={job.get('chunksUpserted') or 0}"
    )


def main(argv: list[str]) -> int:
    run_ids = argv[1:]
    if not run_ids:
        print("usage: purge_stale_job.py <runId> [<runId> ...]", file=sys.stderr)
        return 2

    settings = Settings()
    client: MongoClient | None = None
    try:
        client = MongoClient(settings.mongo_crawler_url, serverSelectionTimeoutMS=10_000)
        jobs = client[settings.mongo_db_name][INDEX_JOBS_COLLECTION]
        now = datetime.now(timezone.utc)
        purged = 0

        for run_id in run_ids:
            job = jobs.find_one({"runId": run_id})
            if job is None:
                print(f"{run_id}  NOT FOUND (no job document with this runId)", file=sys.stderr)
                continue

            print(f"{run_id}  {describe(job)}")
            if is_live_job(job, now):
                print(
                    f"{run_id}  REFUSED: running under a live lease; this job is not stale",
                    file=sys.stderr,
                )
                continue

            result = jobs.delete_one({"_id": job["_id"]})
            print(f"{run_id}  deleted_count={result.deleted_count}")
            if result.deleted_count == 1:
                purged += 1

        print("")
        print(f"Purged {purged} of {len(run_ids)} requested job(s).")
        return 0 if purged == len(run_ids) else 1
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
