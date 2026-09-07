#!/usr/bin/env python3
"""Report Markdown coverage and runs that need reindex after citeable-rag deploy.

Dry read-only against Mongo. Use after backfill / before POST /v1/index.

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

    print("runId\tcrawlType\tstatus\tpages\twithMarkdown\tmissingMarkdown\treindex?")
    needs_reindex: list[str] = []
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
        # Approximate "has Html but no Markdown"
        with_html = pages.count_documents(
            {
                "RunId": rid,
                "$or": [
                    {"Html": {"$type": "string", "$ne": ""}},
                    {"html": {"$type": "string", "$ne": ""}},
                ],
            }
        )
        missing = max(0, with_html - with_md)
        reindex = "yes" if with_md > 0 else "skip-empty"
        if missing > 0:
            reindex = "backfill-first"
        elif with_md > 0:
            needs_reindex.append(rid)
            reindex = "yes"
        print(
            f"{rid}\t{run.get('CrawlType')}\t{run.get('Status')}\t"
            f"{total}\t{with_md}\t{missing}\t{reindex}"
        )

    print()
    if needs_reindex:
        print("Suggested reindex (Markdown present):")
        for rid in needs_reindex:
            print(f'  curl -X POST "$RAG_URL/v1/index" -H "Content-Type: application/json" -d \'{{"runId":"{rid}"}}\'')
    else:
        print("No runs ready for reindex (backfill Markdown first if missingMarkdown > 0).")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
