#!/usr/bin/env python3
"""Print the text that is actually in the index. Read-only.

The payload carries both tiers of every point -- `childText` is what was embedded,
`parentText` is the wider window the child inherits for grounding -- so the index can
be read back verbatim without a retrieval query. That matters because "what got
crawled" and "what got indexed" are different questions: chunking, the composition
tiering and the parent gate all drop material between them, and only this side of the
boundary tells you what a query can reach.

Qdrant is not published on the VPS host, so this runs where Qdrant is reachable:

  ssh root@<vps> 'docker exec -i geek-crawler-rag-api-1 \
      env QDRANT_URL=http://qdrant:6333 python - --hosts' < scripts/show_indexed_text.py

Modes:
  --hosts                     one line per host: points, distinct pages, chunks/page
  --pages --host X            one line per indexed page on that host
  --page URL                  every chunk of one page, in chunkIndex order
  --grep TEXT                 chunks whose embedded text contains TEXT (case-insensitive)

Filters combine: --host, --run, --role {child,parent}. `--full` prints whole chunks
instead of a 400-character head, and `--parent` adds the parent window under each child.

Nothing is written. Safe against production.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict

DEFAULT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "geek_crawler_chunks")

PAYLOAD_FIELDS = [
    "url",
    "host",
    "runId",
    "title",
    "sectionTitle",
    "chunkIndex",
    "chunkRole",
    "childText",
    "parentText",
    "qualityScore",
    "contentIntent",
]


def scroll(url: str, collection: str, api_key: str | None, flt: dict | None, limit: int):
    """Yield payloads. Qdrant pages with an offset token; exhaust it rather than
    trusting one page, or a large host silently reports as a small one."""
    endpoint = f"{url.rstrip('/')}/collections/{collection}/points/scroll"
    offset = None
    seen = 0
    while True:
        body: dict = {
            "limit": 512,
            "with_payload": PAYLOAD_FIELDS,
            "with_vector": False,
        }
        if flt:
            body["filter"] = flt
        if offset is not None:
            body["offset"] = offset
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["api-key"] = api_key
        request = urllib.request.Request(
            endpoint, data=json.dumps(body).encode("utf-8"), headers=headers
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        result = payload.get("result") or {}
        points = result.get("points") or []
        for point in points:
            yield point.get("payload") or {}
            seen += 1
            if limit and seen >= limit:
                return
        offset = result.get("next_page_offset")
        if offset is None or not points:
            return


def build_filter(host: str | None, run: str | None, role: str | None) -> dict | None:
    must = []
    if host:
        must.append({"key": "host", "match": {"value": host}})
    if run:
        must.append({"key": "runId", "match": {"value": run}})
    if role:
        must.append({"key": "chunkRole", "match": {"value": role}})
    return {"must": must} if must else None


def body_text(item: dict) -> str:
    """The text this point was embedded from.

    A parent point carries its window in ``parentText`` and leaves ``childText`` empty,
    so reading ``childText`` alone reports a parent as a zero-character chunk. Select on
    ``chunkRole``, which is the field that says which tier the point is.
    """
    if (item.get("chunkRole") or "") == "parent":
        return item.get("parentText") or ""
    return item.get("childText") or ""


def clip(text: str, full: bool) -> str:
    text = " ".join((text or "").split())
    if full or len(text) <= 400:
        return text
    return text[:400] + f" … (+{len(text) - 400} chars)"


def report_hosts(payloads) -> None:
    points: dict[str, int] = defaultdict(int)
    pages: dict[str, set] = defaultdict(set)
    chars: dict[str, int] = defaultdict(int)
    for item in payloads:
        host = item.get("host") or "?"
        points[host] += 1
        pages[host].add(item.get("url"))
        chars[host] += len(body_text(item))
    print(f"{'host':32} {'points':>8} {'pages':>7} {'chunks/pg':>10} {'kchars':>8}")
    for host in sorted(points, key=lambda h: -points[h]):
        page_count = len(pages[host]) or 1
        print(
            f"{host[:32]:32} {points[host]:>8} {page_count:>7} "
            f"{points[host] / page_count:>10.1f} {chars[host] / 1000:>8.0f}"
        )


def report_pages(payloads) -> None:
    rows: dict[str, dict] = {}
    for item in payloads:
        url = item.get("url") or "?"
        row = rows.setdefault(url, {"chunks": 0, "chars": 0, "title": item.get("title") or ""})
        row["chunks"] += 1
        row["chars"] += len(body_text(item))
    print(f"{'chunks':>7} {'chars':>8}  url")
    for url in sorted(rows, key=lambda u: -rows[u]["chunks"]):
        row = rows[url]
        print(f"{row['chunks']:>7} {row['chars']:>8}  {url}")
        if row["title"]:
            print(f"{'':17}  {row['title'][:100]}")


def report_chunks(payloads, *, full: bool, with_parent: bool) -> None:
    items = sorted(payloads, key=lambda p: (p.get("url") or "", p.get("chunkIndex") or 0))
    for item in items:
        print("=" * 100)
        print(f"{item.get('url')}  [{item.get('chunkRole')} #{item.get('chunkIndex')}]")
        section = item.get("sectionTitle")
        if section:
            print(f"section: {section}")
        print(
            f"quality={item.get('qualityScore')} intent={item.get('contentIntent')} "
            f"chars={len(body_text(item))}"
        )
        print()
        print(clip(body_text(item), full))
        if with_parent and (item.get("chunkRole") or "") == "child":
            parent = item.get("parentText") or ""
            child = item.get("childText") or ""
            if parent.strip() and parent.strip() != child.strip():
                print()
                print("--- parent window ---")
                print(clip(parent, full))
        print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY"))
    parser.add_argument("--hosts", action="store_true")
    parser.add_argument("--pages", action="store_true")
    parser.add_argument("--page")
    parser.add_argument("--grep")
    parser.add_argument("--host")
    parser.add_argument("--run")
    parser.add_argument("--role", choices=["child", "parent"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--parent", action="store_true")
    args = parser.parse_args()

    flt = build_filter(args.host, args.run, args.role)

    if args.page:
        flt = flt or {"must": []}
        flt["must"].append({"key": "url", "match": {"value": args.page}})
        report_chunks(
            list(scroll(args.url, args.collection, args.api_key, flt, args.limit)),
            full=args.full,
            with_parent=args.parent,
        )
        return 0

    if args.grep:
        needle = args.grep.lower()
        matches = [
            item
            for item in scroll(args.url, args.collection, args.api_key, flt, 0)
            if needle in body_text(item).lower()
        ]
        if args.limit:
            matches = matches[: args.limit]
        print(f"# {len(matches)} chunks contain {args.grep!r}\n")
        report_chunks(matches, full=args.full, with_parent=args.parent)
        return 0

    if args.pages:
        report_pages(scroll(args.url, args.collection, args.api_key, flt, args.limit))
        return 0

    if args.hosts:
        report_hosts(scroll(args.url, args.collection, args.api_key, flt, args.limit))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
