#!/usr/bin/env python3
"""One-time Readability → markdown backfill for Mongo crawl_pages.

Mirrors Geek-Crawler-v2 extract-content.ts + locale-path.ts.
Does NOT re-crawl. Dry-run by default; pass --write to persist.

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
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_md
from pymongo import MongoClient
from readability import Document

MAX_MARKDOWN_CHARS = 500_000

KEEP_REGION = frozenset({"us"})
DROP_REGION = frozenset({
    "gb", "uk", "au", "nz", "sg", "ae",
    "ca", "ie", "eu", "in", "za", "jp", "kr", "br", "mx", "de", "fr", "es", "it", "nl",
})
NON_ENGLISH_LOCALE = frozenset({
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az",
    "ba", "be", "bg", "bh", "bi", "bm", "bn", "bo", "br", "bs",
    "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy",
    "da", "de", "dv", "dz",
    "ee", "el", "eo", "es", "et", "eu",
    "fa", "ff", "fi", "fj", "fo", "fr", "fy",
    "ga", "gd", "gl", "gn", "gu", "gv",
    "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz",
    "ia", "id", "ie", "ig", "ii", "ik", "io", "is", "it", "iu",
    "ja", "jv",
    "ka", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku", "kv", "kw", "ky",
    "la", "lb", "lg", "li", "ln", "lo", "lt", "lu", "lv",
    "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
    "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny",
    "oc", "oj", "om", "or", "os",
    "pa", "pi", "pl", "ps", "pt",
    "qu",
    "rm", "rn", "ro", "ru", "rw",
    "sa", "sc", "sd", "se", "sg", "si", "sk", "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw",
    "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts", "tt", "tw", "ty",
    "ug", "uk", "ur", "uz",
    "ve", "vi", "vo",
    "wa", "wo",
    "xh",
    "yi", "yo",
    "za", "zh", "zu",
})


@dataclass
class Counts:
    scanned: int = 0
    updated: int = 0
    skipped_no_html: int = 0
    skipped_locale: int = 0
    skipped_has_markdown: int = 0
    skipped_robots: int = 0
    skipped_failure: int = 0
    extract_failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "scanned": self.scanned,
            "updated": self.updated,
            "skipped_no_html": self.skipped_no_html,
            "skipped_locale": self.skipped_locale,
            "skipped_has_markdown": self.skipped_has_markdown,
            "skipped_robots": self.skipped_robots,
            "skipped_failure": self.skipped_failure,
            "extract_failed": self.extract_failed,
        }


def _first_path_segment(pathname: str) -> str | None:
    parts = [p for p in pathname.split("/") if p]
    return parts[0] if parts else None


def _primary_lang(seg: str) -> str:
    return seg.lower().split("-", 1)[0]


def should_exclude_locale_path(url: str) -> bool:
    try:
        pathname = urlparse(url).path or "/"
        seg = _first_path_segment(pathname)
        if not seg:
            return False
        lower = seg.lower()
        if lower in KEEP_REGION:
            return False
        if lower in DROP_REGION:
            return True
        primary = _primary_lang(seg)
        if primary == "en":
            return False
        return primary in NON_ENGLISH_LOCALE
    except Exception:
        return False


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
            # readability-lxml has no excerpt; leave None
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
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=15_000)
    db = client[db_name]
    pages = db["crawl_pages"]

    query: dict[str, Any] = {}
    if run_id:
        query["RunId"] = run_id

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

    cursor = pages.find(query, projection).batch_size(batch_size)
    now = datetime.now(timezone.utc)

    for doc in cursor:
        if limit is not None and counts.scanned >= limit:
            break
        counts.scanned += 1

        if doc.get("MarkdownBackfilledAt") is not None and _has_markdown(doc):
            counts.skipped_has_markdown += 1
            continue
        if _has_markdown(doc):
            counts.skipped_has_markdown += 1
            continue

        html = doc.get("Html")
        if not isinstance(html, str) or len(html.strip()) < 40:
            counts.skipped_no_html += 1
            continue

        if doc.get("RobotsAllowed") is False:
            counts.skipped_robots += 1
            continue

        failure = doc.get("FailureReason")
        if isinstance(failure, str) and failure.strip():
            counts.skipped_failure += 1
            continue

        url = str(doc.get("FinalUrl") or doc.get("Url") or "")
        if should_exclude_locale_path(url):
            counts.skipped_locale += 1
            continue

        title, markdown, excerpt = extract_clean_content(html, url)
        if not markdown:
            counts.extract_failed += 1
            continue

        counts.updated += 1
        if write:
            pages.update_one(
                {"Id": doc["Id"]},
                {
                    "$set": {
                        "Title": title,
                        "Markdown": markdown,
                        "Excerpt": excerpt,
                        "MarkdownBackfilledAt": now,
                    }
                },
            )

    client.close()
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill Markdown from stored Html (Readability).")
    parser.add_argument("--mongo-url", default=None, help="Override MONGO_CRAWLER_URL")
    parser.add_argument("--db", default="geek_crawler", help="Mongo database name")
    parser.add_argument("--run-id", default=None, help="Limit to one RunId")
    parser.add_argument("--limit", type=int, default=None, help="Max pages to scan")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist updates (default is dry-run)",
    )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
