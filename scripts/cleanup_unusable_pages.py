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


def _delete_ids(
    pages,
    links,
    docs: list[dict[str, Any]],
    *,
    write: bool,
    counts: CleanupCounts,
    delete_links: bool,
) -> None:
    if not docs:
        return
    oids = [d["_id"] for d in docs if d.get("_id") is not None]
    page_ids = [d["Id"] for d in docs if d.get("Id") is not None]
    n = len(oids) if oids else len(page_ids)
    if write:
        if delete_links and page_ids:
            link_res = links.delete_many({"PageId": {"$in": page_ids}})
            counts.deleted_links += int(link_res.deleted_count)
        if oids:
            page_res = pages.delete_many({"_id": {"$in": oids}})
            counts.deleted_pages += int(page_res.deleted_count)
        elif page_ids:
            page_res = pages.delete_many({"Id": {"$in": page_ids}})
            counts.deleted_pages += int(page_res.deleted_count)
    else:
        counts.deleted_pages += n


def cleanup(
    *,
    mongo_url: str,
    db_name: str,
    write: bool,
    limit: int | None,
    batch_size: int,
    run_id: str | None,
    skip_locale_scan: bool,
    delete_links: bool = True,
) -> CleanupCounts:
    counts = CleanupCounts()
    client = MongoClient(
        mongo_url,
        serverSelectionTimeoutMS=30_000,
        connectTimeoutMS=30_000,
        socketTimeoutMS=300_000,
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

    # --- Fast path: field-based deletes (queries shaped for indexes) ---
    # Prefer $gt:"" over $type so FailureReason_1 / MarkdownBackfillSkip_1 can be used.
    fast_steps: list[tuple[str, dict[str, Any], str | None]] = [
        ("failure", scoped({"FailureReason": {"$gt": ""}}), "FailureReason_1"),
        ("failure", scoped({"RobotsAllowed": False}), "RobotsAllowed_1"),
        (
            "marked",
            scoped({"MarkdownBackfillSkip": {"$in": SKIP_MARKS}}),
            "MarkdownBackfillSkip_1",
        ),
    ]
    for label, q, hint in fast_steps:
        if limit is not None and counts.deleted_pages >= limit:
            break
        while True:
            if limit is not None and counts.deleted_pages >= limit:
                break
            take = batch_size
            if limit is not None:
                take = min(batch_size, limit - counts.deleted_pages)
            cursor = pages.find(q, {"_id": 1, "Id": 1, "MarkdownBackfillSkip": 1}).limit(take)
            if hint:
                try:
                    cursor = cursor.hint(hint)
                except Exception as exc:
                    print(f"WARN hint {hint} skipped: {exc}", flush=True)
            batch = list(cursor)
            if not batch:
                break
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
            _delete_ids(
                pages, links, batch, write=write, counts=counts, delete_links=delete_links
            )
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
        pending: list[dict[str, Any]] = []
        cursor = pages.find(run_filter, {"_id": 1, "Id": 1, "Url": 1, "FinalUrl": 1, "FailureReason": 1, "RobotsAllowed": 1}).batch_size(batch_size)
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
            pending.append(doc)
            if len(pending) >= batch_size:
                _delete_ids(
                    pages, links, pending, write=write, counts=counts, delete_links=delete_links
                )
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
            _delete_ids(
                pages, links, pending, write=write, counts=counts, delete_links=delete_links
            )
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
    parser.add_argument(
        "--skip-link-delete",
        action="store_true",
        help="Delete pages only (faster). Orphan crawl_links can be purged later.",
    )
    args = parser.parse_args(argv)
    write = bool(args.write)
    mode = "WRITE" if write else "DRY-RUN"
    print(
        f"cleanup_unusable_pages mode={mode} run_id={args.run_id or '*'} "
        f"limit={args.limit or '*'} locale_scan={not args.skip_locale_scan} "
        f"delete_links={not args.skip_link_delete}"
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
            delete_links=not args.skip_link_delete,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for k, v in counts.as_dict().items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
