#!/usr/bin/env python3
"""Confirm GeekAPI actually persists the corpus fields the Library reads.

Dry read-only against Mongo. The Library reads `ContentHtml` and `Blocks` per
page and schedules on the run-level `ContentReadyAt`. The crawler sends all
three; this confirms they survive the hop through GeekAPI, because a field
silently dropped on ingest reads back exactly like a page that never had content.

Reports field presence and casing, because the run filter and the scheduler
index cannot hedge across casings the way the page projections do.

Usage (on the VPS, where Mongo lives):
  uv run python scripts/verify_ingest_fields.py
  uv run python scripts/verify_ingest_fields.py --run-id <guid>
  MONGO_CRAWLER_URL=mongodb://host:27017 uv run python scripts/verify_ingest_fields.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pymongo import MongoClient

PAGE_FIELDS = [
    ("ContentHtml", "contentHtml"),
    ("Blocks", "blocks"),
    ("Text", "text"),
    ("Html", "html"),
    ("Title", "title"),
]
RUN_FIELDS = [
    ("ContentReadyAt", "contentReadyAt"),
]


def present(doc: dict, names: tuple[str, str]) -> str:
    for name in names:
        if name in doc and doc[name] not in (None, "", [], {}):
            return name
    for name in names:
        if name in doc:
            return f"{name} (empty)"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mongo-url",
        default=os.environ.get("MONGO_CRAWLER_URL", "mongodb://localhost:27017"),
    )
    parser.add_argument("--db", default=os.environ.get("MONGO_DB_NAME", "geek_crawler"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    client = MongoClient(args.mongo_url, serverSelectionTimeoutMS=5000)
    db = client[args.db]

    page_filter = {"RunId": args.run_id} if args.run_id else {}
    page = db["crawl_pages"].find_one(page_filter)
    if page is None:
        print("no crawl_pages document found — crawl something first", file=sys.stderr)
        return 1

    print(f"crawl_pages sample: {page.get('Url') or page.get('url')}\n")
    print(f"{'field':<26}{'stored as':<26}{'verdict'}")
    blocking = False
    for names in PAGE_FIELDS:
        found = present(page, names)
        required = names[0] in ("ContentHtml", "Blocks")
        verdict = "ok" if found else ("MISSING — blocks the plan" if required else "absent")
        if required and not found:
            blocking = True
        print(f"{names[0]:<26}{found or '-':<26}{verdict}")

    blocks = page.get("Blocks") or page.get("blocks")
    if isinstance(blocks, list) and blocks:
        kinds = sorted({b.get("kind") for b in blocks if isinstance(b, dict)})
        print(f"\nblocks: {len(blocks)} on this page, kinds={kinds}")
        sample = next((b for b in blocks if isinstance(b, dict) and b.get("kind") == "row"), None)
        if sample:
            print(f"row block survives BSON: cells={sample.get('cells')}")
    elif blocks is not None:
        print(f"\nBlocks present but not a list: {type(blocks).__name__}")

    run_filter = {"Id": args.run_id} if args.run_id else {}
    run = db["crawl_runs"].find_one(run_filter)
    print()
    if run is None:
        print("no crawl_runs document found")
    else:
        print(f"crawl_runs sample: {run.get('Id') or run.get('id')}")
        for names in RUN_FIELDS:
            found = present(run, names)
            print(f"{names[0]:<26}{found or '-':<26}{'ok' if found else 'absent'}")
        if not present(run, RUN_FIELDS[0]):
            print("\nContentReadyAt absent — the scheduler index would match nothing.")
            blocking = True

    print()
    print("BLOCKED: fix GeekAPI persistence first" if blocking else "clear to proceed")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
