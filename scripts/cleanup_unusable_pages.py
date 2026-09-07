#!/usr/bin/env python3
"""Delete unusable crawl_pages that leaked past the crawler.

Reasons (aligned with Geek-Crawler-v2 reject plan):
  - locale URL paths
  - FailureReason / robots-denied / challenge
  - extract_empty (and related backfill skip marks)

Also deletes crawl_links for removed PageIds.

Dry-run by default; pass --write to delete.

Usage:
  uv run python scripts/cleanup_unusable_pages.py --dry-run --limit 500
  uv run python scripts/cleanup_unusable_pages.py --write
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pymongo import MongoClient  # noqa: E402

from geek_crawler_rag.unusable import should_exclude_locale_path  # noqa: E402


@dataclass
class CleanupCounts:
    scanned: int = 0
    deleted_pages: int = 0
    deleted_links: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scanned": self.scanned,
            "deleted_pages": self.deleted_pages,
            "deleted_links": self.deleted_links,
        }
        for k, v in sorted(self.by_reason.items()):
            out[f"reason_{k}"] = v
        return out


def _mongo_url(cli: str | None) -> str:
    return (cli or os.environ.get("MONGO_CRAWLER_URL") or "mongodb://localhost:27017").strip()


PROJECTION = {
    "Id": 1,
    "Url": 1,
    "FinalUrl": 1,
    "FailureReason": 1,
    "MarkdownBackfillSkip": 1,
    "RobotsAllowed": 1,
}


SKIP_MARKS = [
    "locale",
    "failure",
    "extract_empty",
    "extract_error",
    "fetch_error",
    "no_html",
    "robots",
]


def _delete_ids(pages, links, ids: list[Any], *, write: bool, counts: CleanupCounts) -> None:
    if not ids:
        return
    if write:
        link_res = links.delete_many({"PageId": {"$in": ids}})
        counts.deleted_links += int(link_res.deleted_count)
        page_res = pages.delete_many({"Id": {"$in": ids}})
        counts.deleted_pages += int(page_res.deleted_count)
    else:
        counts.deleted_pages += len(ids)


def cleanup(
    *,
    mongo_url: str,
    db_name: str,
    write: bool,
    limit: int | None,
    batch_size: int,
    run_id: str | None,
    skip_locale_scan: bool,
) -> CleanupCounts:
    counts = CleanupCounts()
    client = MongoClient(
        mongo_url,
        serverSelectionTimeoutMS=15_000,
        connectTimeoutMS=15_000,
        socketTimeoutMS=180_000,
        retryWrites=True,
    )
    db = client[db_name]
    pages = db["crawl_pages"]
    links = db["crawl_links"]

    run_filter: dict[str, Any] = {"RunId": run_id} if run_id else {}

    def scoped(extra: dict[str, Any]) -> dict[str, Any]:
        if run_filter:
            return {"$and": [run_filter, extra]}
        return extra

    # --- Fast path: field-based deletes (no full corpus classify) ---
    for label, q in [
        ("failure", scoped({"FailureReason": {"$exists": True, "$type": "string", "$ne": ""}})),
        ("failure", scoped({"RobotsAllowed": False})),
        (
            "marked",
            scoped({"MarkdownBackfillSkip": {"$in": SKIP_MARKS}}),
        ),
    ]:
        if limit is not None and counts.deleted_pages >= limit:
            break
        while True:
            if limit is not None and counts.deleted_pages >= limit:
                break
            take = batch_size
            if limit is not None:
                take = min(batch_size, limit - counts.deleted_pages)
            batch = list(pages.find(q, {"Id": 1, "MarkdownBackfillSkip": 1}).limit(take))
            if not batch:
                break
            ids = []
            for doc in batch:
                counts.scanned += 1
                skip = doc.get("MarkdownBackfillSkip")
                if label == "marked":
                    if skip == "locale":
                        counts.by_reason["locale"] += 1
                    elif skip in ("failure", "robots"):
                        counts.by_reason["failure"] += 1
                    else:
                        counts.by_reason["extract_empty"] += 1
                else:
                    counts.by_reason[label] += 1
                if doc.get("Id") is not None:
                    ids.append(doc["Id"])
            _delete_ids(pages, links, ids, write=write, counts=counts)
            print(
                f"progress fast label={label} scanned={counts.scanned} "
                f"deleted_pages={counts.deleted_pages} deleted_links={counts.deleted_links} "
                f"reasons={dict(counts.by_reason)}",
                flush=True,
            )
            if not write:
                break

    # --- Locale URL scan (remaining pages) ---
    if not skip_locale_scan and (limit is None or counts.deleted_pages < limit):
        pending: list[Any] = []
        cursor = pages.find(run_filter, PROJECTION).batch_size(batch_size)
        for doc in cursor:
            if limit is not None and counts.deleted_pages + len(pending) >= limit:
                break
            counts.scanned += 1
            # Already-deleted failure rows won't appear when write=True
            fr = doc.get("FailureReason")
            if isinstance(fr, str) and fr.strip():
                continue
            if doc.get("RobotsAllowed") is False:
                continue
            url = str(doc.get("FinalUrl") or doc.get("Url") or "")
            if not should_exclude_locale_path(url) and not should_exclude_locale_path(
                str(doc.get("Url") or "")
            ):
                continue
            counts.by_reason["locale"] += 1
            if doc.get("Id") is not None:
                pending.append(doc["Id"])
            if len(pending) >= batch_size:
                _delete_ids(pages, links, pending, write=write, counts=counts)
                pending = []
                print(
                    f"progress locale_scan scanned={counts.scanned} "
                    f"deleted_pages={counts.deleted_pages} deleted_links={counts.deleted_links} "
                    f"reasons={dict(counts.by_reason)}",
                    flush=True,
                )
                if not write and limit is None:
                    break
        if pending:
            _delete_ids(pages, links, pending, write=write, counts=counts)
            print(
                f"progress locale_scan scanned={counts.scanned} "
                f"deleted_pages={counts.deleted_pages} deleted_links={counts.deleted_links} "
                f"reasons={dict(counts.by_reason)}",
                flush=True,
            )

    client.close()
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delete unusable crawl_pages from Mongo.")
    parser.add_argument("--mongo-url", default=None)
    parser.add_argument("--db", default="geek_crawler")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-locale-scan",
        action="store_true",
        help="Only delete FailureReason / RobotsAllowed / MarkdownBackfillSkip matches",
    )
    args = parser.parse_args(argv)
    write = bool(args.write)
    mode = "WRITE" if write else "DRY-RUN"
    print(
        f"cleanup_unusable_pages mode={mode} run_id={args.run_id or '*'} "
        f"limit={args.limit or '*'} locale_scan={not args.skip_locale_scan}"
    )
    try:
        counts = cleanup(
            mongo_url=_mongo_url(args.mongo_url),
            db_name=args.db,
            write=write,
            limit=args.limit,
            batch_size=args.batch_size,
            run_id=args.run_id,
            skip_locale_scan=args.skip_locale_scan,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for k, v in counts.as_dict().items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
