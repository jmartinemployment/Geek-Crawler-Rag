#!/usr/bin/env python3
"""Report crawl runs that are content-ready but have no index to show for it.

GeekAPI enqueues an index on crawl completion, fire-and-forget. EnqueueIndexAsync fails closed by
returning null, so a RAG outage loses that enqueue with nothing left behind that ties the loss to a
run. On 2026-09-24 eleven crawls completed that way and the first visible symptom, hours later, was
a 502 on an unrelated endpoint.

Nothing watches for that gap. This does: a run whose `ContentReadyAt` is older than the grace
period and which has no index job document at all, or whose job ended `failed`, is a lost enqueue.

Reports only. Enqueuing lives in scripts/trigger_manual_index.py and stays there -- two ways to
decide what needs indexing is two answers to drift apart.

Exit codes: 0 nothing lost, 1 lost runs found, 2 the check could not run.

Usage:
  uv run python scripts/check_index_reconciliation.py
  uv run python scripts/check_index_reconciliation.py --grace-minutes 60
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import MongoClient

from geek_crawler_rag.config import Settings
from geek_crawler_rag.status_store import COLLECTION as INDEX_JOBS_COLLECTION

DEFAULT_GRACE_MINUTES = 30


def parse_ready_at(value: Any) -> datetime | None:
    """`ContentReadyAt` as an aware datetime, or None when it cannot be read.

    GeekAPI writes it as a Postgres-export string -- "2026-09-24 12:32:44.213000+00" -- with a
    space separator and a two-digit offset that fromisoformat rejects. Mongo can also hand back a
    real datetime. Both shapes arrive here, and an unreadable one is not a crisis: it just cannot
    be aged, so the caller leaves that run alone rather than guessing.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace(" ", "T")
    if text.endswith("+00"):
        text = text[:-3] + "+00:00"
    elif text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def classify(run: dict, jobs: list[dict], now: datetime, grace: timedelta) -> str | None:
    """Why this run counts as a lost enqueue, or None when it is fine.

    Within the grace period nothing is wrong yet -- the enqueue is asynchronous and a job document
    appears a moment after the crawl closes.
    """
    ready_at = parse_ready_at(run.get("ContentReadyAt"))
    if ready_at is None or now - ready_at < grace:
        return None
    if not jobs:
        return "no index job was ever created"
    states = {str(job.get("state") or "") for job in jobs}
    if states == {"failed"}:
        return "its only index job failed"
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grace-minutes",
        type=int,
        default=DEFAULT_GRACE_MINUTES,
        help=f"how long a run may sit content-ready before it counts as lost (default {DEFAULT_GRACE_MINUTES})",
    )
    args = parser.parse_args(argv[1:])
    if args.grace_minutes < 0:
        print("FAILED: --grace-minutes must not be negative", file=sys.stderr)
        return 2

    settings = Settings()
    client: MongoClient | None = None
    try:
        client = MongoClient(settings.mongo_crawler_url, serverSelectionTimeoutMS=10_000)
        db = client[settings.mongo_db_name]
        now = datetime.now(timezone.utc)
        grace = timedelta(minutes=args.grace_minutes)

        runs = list(
            db["crawl_runs"].find(
                {
                    "Status": {"$in": ["complete", "external"]},
                    "ContentReadyAt": {"$exists": True, "$nin": [None, ""]},
                    "RagIndexedAtUtc": None,
                },
                {"_id": 0, "Id": 1, "ContentReadyAt": 1, "SeedUrlsJson": 1},
            )
        )
        jobs_by_run: dict[str, list[dict]] = {}
        for job in db[INDEX_JOBS_COLLECTION].find({}, {"_id": 0, "runId": 1, "state": 1}):
            jobs_by_run.setdefault(str(job.get("runId") or ""), []).append(job)

        lost = []
        for run in runs:
            run_id = str(run.get("Id") or "")
            if not run_id:
                continue
            reason = classify(run, jobs_by_run.get(run_id, []), now, grace)
            if reason:
                lost.append((run_id, reason, str(run.get("SeedUrlsJson") or "")))

        print(f"content-ready, not yet indexed: {len(runs)}")
        print(f"grace period:                   {args.grace_minutes} minutes")
        print(f"lost enqueues:                  {len(lost)}")
        if not lost:
            print("")
            print("OK: every content-ready run has an index job.")
            return 0

        print("")
        for run_id, reason, seed in sorted(lost):
            seed = seed.replace("[", "").replace("]", "").replace('"', "")[:60]
            print(f"  {run_id}  {reason}  {seed}")
        print("")
        print("Enqueue them with: python scripts/trigger_manual_index.py " + " ".join(r for r, _, _ in lost))
        return 1
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
