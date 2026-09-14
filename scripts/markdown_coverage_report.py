#!/usr/bin/env python3
"""Report Markdown coverage and runs ready to reindex (no backfill).

Dry read-only against Mongo. Pages missing Markdown should be **deleted**
(cleanup / index sweeper) and **re-crawled** — never backfilled.

Usage:
  uv run python scripts/markdown_coverage_report.py
  uv run python scripts/markdown_coverage_report.py --run-id <guid>
  uv run python scripts/markdown_coverage_report.py --limit-runs 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pymongo import MongoClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-url", default=os.environ.get("MONGO_CRAWLER_URL", "mongodb://localhost:27017"))
    parser.add_argument("--db", default=os.environ.get("MONGO_DB_NAME", "geek_crawler"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit-runs", type=int, default=30)
    args = parser.parse_args()

    client = MongoClient(args.mongo_url, serverSelectionTimeoutMS=8000)
    db = client[args.db]
    pages = db["crawl_pages"]
    runs = db["crawl_runs"]

    run_filter: dict = {}
    if args.run_id:
        run_filter["Id"] = args.run_id
    else:
        run_filter["Status"] = {"$in": ["complete", "external"]}

    cursor = runs.find(run_filter, {"Id": 1, "CrawlType": 1, "Status": 1, "FinishedAtUtc": 1}).sort(
        "FinishedAtUtc", -1
    ).limit(args.limit_runs if not args.run_id else 1)

    print("runId\tcrawlType\tstatus\tpages\twithMarkdown\tmissingMarkdown\taction")
    needs_reindex: list[str] = []
    needs_delete: list[str] = []
    for run in cursor:
        rid = str(run.get("Id") or "")
        if not rid:
            continue
        total = pages.count_documents({"RunId": rid})
        with_md = pages.count_documents(
            {
                "RunId": rid,
                "$or": [
                    {"Markdown": {"$type": "string", "$ne": ""}},
                    {"markdown": {"$type": "string", "$ne": ""}},
                ],
            }
        )
        missing = max(0, total - with_md)
        if missing > 0:
            action = "delete-missing-then-recrawl"
            needs_delete.append(rid)
        elif with_md > 0:
            action = "reindex"
            needs_reindex.append(rid)
        else:
            action = "no-pages"
        print(
            f"{rid}\t{run.get('CrawlType')}\t{run.get('Status')}\t"
            f"{total}\t{with_md}\t{missing}\t{action}"
        )

    print()
    if needs_delete:
        print("Runs with pages missing Markdown (delete via cleanup, then re-crawl):")
        for rid in needs_delete:
            print(f"  {rid}")
    if needs_reindex:
        print("Suggested reindex (Markdown present on all pages):")
        for rid in needs_reindex:
            print(
                f'  curl -X POST "$RAG_URL/v1/index" -H "Content-Type: application/json" '
                f"-d '{{\"runId\":\"{rid}\"}}'"
            )
    if not needs_reindex and not needs_delete:
        print("No runs with pages to act on.")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
