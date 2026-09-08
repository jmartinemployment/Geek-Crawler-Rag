#!/usr/bin/env python3
"""One-time Readability → markdown backfill for Mongo crawl_pages.

Mirrors Geek-Crawler-v2 extract-content.ts + locale-path.ts.
Does NOT re-crawl. Dry-run by default; pass --write to persist.

Unusable pages (locale / FailureReason / extract-empty) are **deleted**
(not skip-marked) so they do not linger in the corpus.

Usage:
  uv run python scripts/backfill_markdown.py --dry-run
  uv run python scripts/backfill_markdown.py --run-id <guid> --write
  uv run python scripts/backfill_markdown.py --write --limit 500
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_md
from pymongo import MongoClient, UpdateOne
from readability import Document

from geek_crawler_rag.unusable import should_exclude_locale_path

MAX_MARKDOWN_CHARS = 500_000

# Re-export for tests that import from this module
__all__ = [
    "extract_clean_content",
    "should_exclude_locale_path",
    "backfill",
    "main",
]


@dataclass
class Counts:
    scanned: int = 0
    updated: int = 0
    deleted_locale: int = 0
    deleted_failure: int = 0
    deleted_extract_empty: int = 0
    deleted_links: int = 0
    skipped_has_markdown: int = 0
    skipped_no_html: int = 0
    already_markdown: int = 0
    missing_doc: int = 0
    runs_marked_ready: int = 0
    errors: int = 0
    complete_pass: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "scanned": self.scanned,
            "updated": self.updated,
            "deleted_locale": self.deleted_locale,
            "deleted_failure": self.deleted_failure,
            "deleted_extract_empty": self.deleted_extract_empty,
            "deleted_links": self.deleted_links,
            "already_markdown": self.already_markdown,
            "missing_doc": self.missing_doc,
            "runs_marked_ready": self.runs_marked_ready,
            "errors": self.errors,
            "complete_pass": self.complete_pass,
        }


def extract_clean_content(html: str, page_url: str) -> tuple[str | None, str | None, str | None]:
    """Return (title, markdown, excerpt). Mirrors extract-content.ts."""
    if not html or len(html) < 40:
        return None, None, None
    try:
        doc = Document(html)
        title = (doc.title() or "").strip() or None
        content_html = doc.summary(html_partial=True)
        excerpt = None
        if not content_html or len(content_html) < 20:
            soup = BeautifulSoup(html, "lxml")
            main = soup.find("article") or soup.find("main") or soup.body
            if main is None:
                return None, None, None
            if not title:
                t_el = soup.find("title") or soup.find("h1")
                title = t_el.get_text(strip=True) if t_el else None
            content_html = str(main)
        else:
            if not title:
                soup = BeautifulSoup(html, "lxml")
                t_el = soup.find("title") or soup.find("h1")
                title = t_el.get_text(strip=True) if t_el else None

        markdown = html_to_md(content_html, heading_style="ATX").strip()
        if len(markdown) > MAX_MARKDOWN_CHARS:
            markdown = markdown[:MAX_MARKDOWN_CHARS]
        if not markdown:
            return title, None, excerpt
        return title, markdown, excerpt
    except Exception:
        return None, None, None


def _has_markdown(doc: dict[str, Any]) -> bool:
    for key in ("Markdown", "markdown"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return True
    return False


def _mongo_url_from_env(cli_url: str | None) -> str:
    import os

    return (cli_url or os.environ.get("MONGO_CRAWLER_URL") or "mongodb://localhost:27017").strip()


def _missing_markdown_query(run_id: str) -> dict[str, Any]:
    return {
        "RunId": run_id,
        "$expr": {
            "$and": [
                {
                    "$eq": [
                        {"$trim": {"input": {"$ifNull": ["$Markdown", ""]}}},
                        "",
                    ]
                },
                {
                    "$eq": [
                        {"$trim": {"input": {"$ifNull": ["$markdown", ""]}}},
                        "",
                    ]
                },
            ]
        },
    }


def backfill(
    *,
    mongo_url: str,
    db_name: str,
    run_id: str | None,
    write: bool,
    limit: int | None,
    batch_size: int,
) -> Counts:
    counts = Counts()
    client = MongoClient(
        mongo_url,
        serverSelectionTimeoutMS=15_000,
        connectTimeoutMS=15_000,
        socketTimeoutMS=300_000,
        retryWrites=True,
        maxPoolSize=8,
    )
    db = client[db_name]
    pages = db["crawl_pages"]
    links = db["crawl_links"]
    runs = db["crawl_runs"]
    if write and run_id:
        runs.update_one({"Id": run_id}, {"$unset": {"MarkdownReadyAt": ""}})

    query: dict[str, Any] = {
        "Html": {"$exists": True, "$type": "string"},
        **_missing_markdown_query(run_id or ""),
    }
    if not run_id:
        query.pop("RunId", None)

    projection = {
        "Id": 1,
        "RunId": 1,
        "Url": 1,
        "FinalUrl": 1,
        "Html": 1,
        "Markdown": 1,
        "markdown": 1,
        "Title": 1,
        "RobotsAllowed": 1,
        "FailureReason": 1,
        "MarkdownBackfilledAt": 1,
    }
    now = datetime.now(timezone.utc)
    last_log = 0
    last_object_id: Any | None = None

    def delete_page(page_id: Any, reason: str) -> None:
        if reason == "locale":
            counts.deleted_locale += 1
        elif reason == "failure":
            counts.deleted_failure += 1
        else:
            counts.deleted_extract_empty += 1
        if not write or page_id is None:
            return
        try:
            link_res = links.delete_many({"PageId": page_id})
            counts.deleted_links += int(link_res.deleted_count)
            pages.delete_one({"Id": page_id})
        except Exception as exc:
            counts.errors += 1
            print(f"WARN delete failed id={page_id}: {exc}", flush=True)

    while True:
        if limit is not None and counts.scanned >= limit:
            break
        take = batch_size
        if limit is not None:
            take = min(batch_size, limit - counts.scanned)
        # One round-trip per batch (include Html) — avoids per-page find_one over WAN.
        try:
            batch_query = dict(query)
            if last_object_id is not None:
                batch_query["_id"] = {"$gt": last_object_id}
            batch_docs = list(
                pages.find(batch_query, projection).sort("_id", 1).limit(take)
            )
        except Exception as exc:
            counts.errors += 1
            print(f"WARN batch fetch failed: {exc}", flush=True)
            break
        if not batch_docs:
            counts.complete_pass = 1
            break
        last_object_id = batch_docs[-1]["_id"]

        pending_updates: list[UpdateOne] = []
        for doc in batch_docs:
            if limit is not None and counts.scanned >= limit:
                break
            counts.scanned += 1
            page_id = doc.get("Id")

            url = str(doc.get("FinalUrl") or doc.get("Url") or "")
            if doc.get("RobotsAllowed") is False:
                delete_page(page_id, "failure")
                continue
            failure = doc.get("FailureReason")
            if isinstance(failure, str) and failure.strip():
                delete_page(page_id, "failure")
                continue
            if should_exclude_locale_path(url) or should_exclude_locale_path(
                str(doc.get("Url") or "")
            ):
                delete_page(page_id, "locale")
                continue

            if _has_markdown(doc):
                counts.already_markdown += 1
                if write:
                    try:
                        pages.update_one(
                            {"Id": page_id},
                            {
                                "$set": {"MarkdownBackfilledAt": now},
                                "$unset": {"MarkdownBackfillSkip": ""},
                            },
                        )
                    except Exception as exc:
                        counts.errors += 1
                        print(f"WARN mark existing md failed id={page_id}: {exc}", flush=True)
                continue

            html = doc.get("Html")
            if not isinstance(html, str) or len(html.strip()) < 40:
                delete_page(page_id, "extract_empty")
                continue

            try:
                title, markdown, excerpt = extract_clean_content(html, url)
            except Exception as exc:
                print(f"WARN extract failed id={page_id}: {exc}", flush=True)
                delete_page(page_id, "extract_empty")
                continue
            if not markdown:
                delete_page(page_id, "extract_empty")
                continue

            counts.updated += 1
            if write:
                pending_updates.append(
                    UpdateOne(
                        {"Id": page_id},
                        {
                            "$set": {
                                "Title": title,
                                "Markdown": markdown,
                                "Excerpt": excerpt,
                                "MarkdownBackfilledAt": now,
                            },
                            "$unset": {"MarkdownBackfillSkip": ""},
                        },
                    )
                )

            if counts.scanned - last_log >= 50 or counts.updated in (1, 5, 10, 25):
                last_log = counts.scanned
                print(
                    f"progress scanned={counts.scanned} updated={counts.updated} "
                    f"del_locale={counts.deleted_locale} del_fail={counts.deleted_failure} "
                    f"del_extract={counts.deleted_extract_empty}",
                    flush=True,
                )

        if write and pending_updates:
            try:
                pages.bulk_write(pending_updates, ordered=False)
            except Exception as exc:
                counts.errors += 1
                print(
                    f"WARN batch update failed count={len(pending_updates)}: {exc}",
                    flush=True,
                )
                counts.updated -= len(pending_updates)
                break

    if run_id and limit is None and counts.complete_pass and counts.errors == 0:
        total_pages = pages.count_documents({"RunId": run_id})
        missing_markdown = pages.count_documents(
            _missing_markdown_query(run_id), limit=1
        )
        run = runs.find_one({"Id": run_id}, {"Status": 1, "_id": 0}) or {}
        terminal = str(run.get("Status") or "").lower() == "complete"
        if total_pages > 0 and missing_markdown == 0 and terminal:
            if write:
                result = runs.update_one(
                    {"Id": run_id, "Status": "complete"},
                    {"$set": {"MarkdownReadyAt": now.isoformat()}},
                )
                counts.runs_marked_ready = int(result.modified_count > 0)

    client.close()
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill Markdown from stored Html (Readability).")
    parser.add_argument("--mongo-url", default=None, help="Override MONGO_CRAWLER_URL")
    parser.add_argument("--db", default="geek_crawler", help="Mongo database name")
    parser.add_argument("--run-id", default=None, help="Limit to one RunId")
    parser.add_argument("--limit", type=int, default=None, help="Max pages to scan")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--write", action="store_true", help="Persist updates (default is dry-run)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit dry-run (default). Ignored if --write is set.",
    )
    args = parser.parse_args(argv)

    write = bool(args.write)
    mode = "WRITE" if write else "DRY-RUN"
    print(f"backfill_markdown mode={mode} run_id={args.run_id or '*'} limit={args.limit or '*'}")

    try:
        counts = backfill(
            mongo_url=_mongo_url_from_env(args.mongo_url),
            db_name=args.db,
            run_id=args.run_id,
            write=write,
            limit=args.limit,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for k, v in counts.as_dict().items():
        print(f"{k}={v}")
    return 1 if counts.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
