#!/usr/bin/env python3
"""Verify and optionally stamp run-level Markdown readiness.

Dry-run is the default. Validation is deliberately sequential and throttled so
historical reconciliation cannot recreate the scheduler's former Mongo load.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient


@dataclass
class ReconcileCounts:
    scanned: int = 0
    ready: int = 0
    incomplete: int = 0
    zero_pages: int = 0
    marked: int = 0
    cleared: int = 0


def missing_markdown_query(run_id: str) -> dict[str, Any]:
    return {
        "RunId": run_id,
        "$and": [
            {
                "$or": [
                    {"Markdown": {"$exists": False}},
                    {"Markdown": None},
                    {"Markdown": ""},
                ]
            },
            {
                "$or": [
                    {"markdown": {"$exists": False}},
                    {"markdown": None},
                    {"markdown": ""},
                ]
            },
        ],
    }


def reconcile(
    *,
    mongo_url: str,
    db_name: str,
    run_ids: list[str],
    write: bool,
    max_runs: int | None,
    delay_seconds: float,
) -> ReconcileCounts:
    client = MongoClient(
        mongo_url,
        serverSelectionTimeoutMS=15_000,
        connectTimeoutMS=15_000,
        socketTimeoutMS=300_000,
        maxPoolSize=2,
    )
    db = client[db_name]
    runs = db["crawl_runs"]
    pages = db["crawl_pages"]
    query: dict[str, Any] = {
        "Status": {"$in": ["complete", "external"]},
        "Id": {"$type": "string", "$ne": ""},
    }
    if run_ids:
        query["Id"]["$in"] = run_ids
    cursor = runs.find(
        query,
        {"Id": 1, "Status": 1, "SeedUrlsJson": 1, "MarkdownReadyAt": 1, "_id": 0},
    ).sort("CompletedAtUtc", -1)
    if max_runs is not None:
        cursor = cursor.limit(max_runs)

    counts = ReconcileCounts()
    now = datetime.now(timezone.utc).isoformat()
    for run in cursor:
        run_id = str(run.get("Id") or "")
        total = pages.count_documents({"RunId": run_id})
        missing = pages.count_documents(missing_markdown_query(run_id), limit=1)
        current_ready = bool(run.get("MarkdownReadyAt"))
        counts.scanned += 1
        if total == 0:
            classification = "zero_pages"
            counts.zero_pages += 1
        elif missing:
            classification = "incomplete"
            counts.incomplete += 1
        else:
            classification = "ready"
            counts.ready += 1

        action = "none"
        if classification == "ready" and not current_ready:
            action = "mark" if write else "would_mark"
            if write:
                runs.update_one(
                    {"Id": run_id},
                    {"$set": {"MarkdownReadyAt": now}},
                )
                counts.marked += 1
        elif classification != "ready" and current_ready:
            action = "clear" if write else "would_clear"
            if write:
                runs.update_one(
                    {"Id": run_id},
                    {"$unset": {"MarkdownReadyAt": ""}},
                )
                counts.cleared += 1

        print(
            f"runId={run_id} status={classification} pages={total} "
            f"marker={current_ready} action={action}",
            flush=True,
        )
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    client.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify crawl-run Markdown readiness (dry-run by default)."
    )
    parser.add_argument("--mongo-url", default=None)
    parser.add_argument("--db", default="geek_crawler")
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    counts = reconcile(
        mongo_url=(
            args.mongo_url
            or os.environ.get("MONGO_CRAWLER_URL")
            or "mongodb://localhost:27017"
        ).strip(),
        db_name=args.db,
        run_ids=args.run_id,
        write=args.write,
        max_runs=args.max_runs,
        delay_seconds=max(0.0, args.delay_seconds),
    )
    print(
        "summary "
        + " ".join(f"{key}={value}" for key, value in vars(counts).items()),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
