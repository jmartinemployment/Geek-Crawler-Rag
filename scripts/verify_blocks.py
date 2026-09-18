#!/usr/bin/env python3
"""Validate the crawler's cached block corpus against the plaintext projection.

Dry read-only over the crawler's extract cache — the local copy of every page's
typed blocks, written by Geek-Crawler-v2 at
`$DATA_DIR/extract-cache/<runId>/<sha>.blocks.json`. Nothing here touches Mongo
or Qdrant.

What it proves, per run:

  * no structural HTML reaches the projected text (markup is token budget burnt)
  * no anchor markup reaches it either — hrefs belong in node metadata
  * `row` blocks project to their cells rather than to nothing, which is the
    failure mode a naive `block["text"]` map produces silently
  * how much text each page actually yields, so an empty projection is visible

Chunk-level checks (size ceiling, context-prefix duplication) are deliberately
absent: they belong to the chunker, and asserting them here would test a
reimplementation of it rather than the real thing.

Usage:
  uv run python scripts/verify_blocks.py
  uv run python scripts/verify_blocks.py --cache-dir /Volumes/Seagate/geek-crawler-data/extract-cache
  uv run python scripts/verify_blocks.py --run-id 71515c39 --samples 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geek_crawler_rag.block_text import (  # noqa: E402
    ROW_CELL_SEPARATOR,
    derive_plaintext_from_blocks,
    render_block_text,
)

# Structural markup only, and `a` is deliberately absent from the alternation.
# A bare `<tag>` test is a false-positive generator on technical documentation:
# FreshBooks' API pages carry `<accountId>` placeholders and raw XML as visible
# prose (565 matched before narrowing), and `<a token>` in a Bearer-header
# example is not an anchor. An anchor that truly leaked carries an href, so
# that is what ANCHOR_LEAK tests instead of tag shape.
STRUCTURAL_TAG = re.compile(
    r"</?(div|span|ul|ol|li|table|tr|td|th|h[1-6]|br|img|section|article|nav|header|footer)\b",
    re.I,
)
ANCHOR_LEAK = re.compile(r"href\s*=", re.I)
# The defect class that actually exists in this corpus: HTML the *page* embedded
# in its own text as unicode escapes, which extraction faithfully preserved.
# `u003ca href=u0022#videou0022u003e` is `<a href="#video">`. It reaches the
# embedding as literal tokens and can break a quote that spans it.
ESCAPED_MARKUP = re.compile(r"(\\u|u)00(3c|3e|22)", re.I)


def default_cache_dir() -> Path:
    data_dir = os.environ.get("DATA_DIR", "./data")
    return Path(data_dir).expanduser().resolve() / "extract-cache"


def check_page(blocks: list) -> dict:
    """Per-page findings. Silent on success; every key is a count or a sample."""
    text = derive_plaintext_from_blocks(blocks)
    rows = [b for b in blocks if isinstance(b, dict) and b.get("kind") == "row"]
    rows_lost = [b for b in rows if not render_block_text(b)]

    return {
        "blocks": len(blocks),
        "chars": len(text),
        "empty": not text.strip(),
        "html_tags": len(STRUCTURAL_TAG.findall(text)),
        "url_leaks": len(ANCHOR_LEAK.findall(text)),
        "escaped": len(ESCAPED_MARKUP.findall(text)),
        "rows": len(rows),
        "rows_lost": len(rows_lost),
        "text": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--run-id", default=None, help="run id or unique prefix")
    parser.add_argument("--samples", type=int, default=1, help="sample texts to print")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else default_cache_dir()
    if not cache_dir.is_dir():
        print(f"no extract cache at {cache_dir}", file=sys.stderr)
        print("set DATA_DIR or pass --cache-dir", file=sys.stderr)
        return 1

    runs = sorted(d for d in cache_dir.iterdir() if d.is_dir())
    if args.run_id:
        runs = [d for d in runs if d.name.startswith(args.run_id)]
    if not runs:
        print(f"no runs under {cache_dir}", file=sys.stderr)
        return 1

    print(f"extract cache: {cache_dir}  ({len(runs)} run(s))\n")
    totals = {"pages": 0, "empty": 0, "html": 0, "urls": 0, "escaped": 0, "rows": 0, "rows_lost": 0, "chars": 0}
    samples_left = args.samples
    failed = False

    for run in runs:
        files = sorted(run.glob("*.blocks.json"))
        if not files:
            continue

        pages = empty = html = urls = escaped = rows = rows_lost = chars = malformed = 0
        for path in files:
            try:
                blocks = json.loads(path.read_text("utf8"))
            except (OSError, json.JSONDecodeError):
                malformed += 1
                continue
            if not isinstance(blocks, list):
                malformed += 1
                continue

            found = check_page(blocks)
            pages += 1
            chars += found["chars"]
            empty += 1 if found["empty"] else 0
            html += found["html_tags"]
            urls += found["url_leaks"]
            escaped += 1 if found["escaped"] else 0
            rows += found["rows"]
            rows_lost += found["rows_lost"]

            if samples_left > 0 and found["chars"] > 400 and found["rows"] > 0:
                samples_left -= 1
                print(f"  sample — {path.name}")
                print("  " + found["text"][:400].replace("\n", "\n  "))
                print()

        if pages == 0:
            continue
        totals["pages"] += pages
        totals["empty"] += empty
        totals["html"] += html
        totals["urls"] += urls
        totals["escaped"] += escaped
        totals["rows"] += rows
        totals["rows_lost"] += rows_lost
        totals["chars"] += chars

        bad = html or urls or rows_lost or malformed
        failed = failed or bool(bad)
        print(
            f"{run.name[:8]}  {pages:>5} pages  {chars/1e6:>6.2f}M chars  "
            f"rows={rows:<6} lost={rows_lost:<4} html={html:<4} urls={urls:<4} "
            f"escaped={escaped:<4} empty={empty:<4} malformed={malformed}"
        )

    print()
    print(
        f"total: {totals['pages']} pages, {totals['chars']/1e6:.1f}M chars, "
        f"{totals['rows']} table rows"
    )
    print(f"row cells preserved : {totals['rows'] - totals['rows_lost']}/{totals['rows']}")
    print(f"structural markup   : {totals['html']}")
    print(f"anchor markup       : {totals['urls']}")
    print(f"escaped markup pages: {totals['escaped']}")
    print(f"empty projections   : {totals['empty']}")
    print(f"cell separator      : {ROW_CELL_SEPARATOR!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
