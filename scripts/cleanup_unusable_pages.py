#!/usr/bin/env python3
"""Delete unusable crawl_pages that leaked past the crawler.

Deletes on crawler-owned rejects only:
  - locale URL paths
  - FailureReason / robots-denied / challenge
  - a 4xx or 5xx response, which is the server's error page and not content

**A page with no corpus body is NOT deleted here.** `classify_from_mongo_doc`
reports it as `no_content` and this script passes over it, deliberately: a page
this service cannot read may be perfectly readable to the next one, and
re-adjudicating the crawler's decision is what destroyed 5,274 pages and their
Qdrant points on 2026-09-18. Widening the filter to `no_content` re-arms exactly
that path.

Deletes in this order, and the order is the point: each page's Qdrant points first, then
crawl_links for its PageId, then the page. An orphaned vector is worse than an undeleted
page -- retrieval reads chunk text from the Qdrant payload and filters on runId, which a
surviving point still carries, so the prose stays quotable while /v1/pages 404s for the
deleted row and citation_verify drops every quote from it. If the purge fails, nothing
below it runs.

Dry-run by default; pass --write to delete. --dry-run and --write are mutually exclusive
and argparse enforces it: --dry-run used to be declared and read nowhere, so
`--write --dry-run` deleted.

Usage:
  uv run python scripts/cleanup_unusable_pages.py --dry-run --limit 500
  uv run python scripts/cleanup_unusable_pages.py --write
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pymongo import MongoClient  # noqa: E402
from pymongo.errors import OperationFailure  # noqa: E402

from geek_crawler_rag.unusable import classify_from_mongo_doc  # noqa: E402


@dataclass
class CleanupCounts:
    scanned: int = 0
    deleted_pages: int = 0
    deleted_links: int = 0
    purged_vector_pages: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scanned": self.scanned,
            "deleted_pages": self.deleted_pages,
            "deleted_links": self.deleted_links,
            "purged_vector_pages": self.purged_vector_pages,
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
    "RobotsAllowed": 1,
}


def _purge_vectors(page_ids: list[str], *, counts: CleanupCounts) -> None:
    """Delete each page's Qdrant points. Raises if it cannot, so nothing is deleted after.

    Vectors go before rows, which is the ordering this repo states in writing -- see
    scripts/check_orphaned_crawl_data.py: "Delete vectors for a run with
    DELETE /v1/index/runs/{runId} before its rows, so a surviving point can never outlive
    the page it cites." This script deleted crawl_pages and crawl_links and touched Qdrant
    not at all, which was survivable only while its filters matched nothing. They match now.

    An orphaned point is worse than an undeleted page. Retrieval reads chunk text straight
    from the Qdrant payload and filters on runId, which the point still carries, so the
    prose stays retrievable and quotable -- while /v1/pages 404s for the deleted row, so
    citation_verify drops every quote from it. The generator would be fed the text and be
    unable to cite anything from it.

    Reuses QdrantStore.delete_by_page_id -- it already existed with no production caller --
    rather than rebuilding the filter here. Two implementations of one delete is the drift
    CLAUDE.md names, and this one has to agree with the indexer about ownerId/visibility.
    asyncio.run per batch because that method is async and this script is sync pymongo;
    a fresh loop per batch is cheap next to the deletes themselves.

    No exception is swallowed. If Qdrant refuses, the caller must not proceed to the row
    delete, because that is precisely how the orphan is created.
    """
    if not page_ids:
        return

    from geek_crawler_rag.config import get_settings
    from geek_crawler_rag.qdrant_store import QdrantStore

    settings = get_settings()

    async def run() -> None:
        store = QdrantStore(
            settings.qdrant_url,
            collection=settings.qdrant_collection,
            api_key=settings.qdrant_api_key,
        )
        try:
            for page_id in page_ids:
                await store.delete_by_page_id(
                    page_id,
                    owner_id=settings.crawler_owner_id,
                    visibility=settings.crawler_visibility,
                )
        finally:
            await store.close()

    asyncio.run(run())
    counts.purged_vector_pages += len(page_ids)


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
        # Vectors first. If this raises, nothing below runs and no row is orphaned.
        _purge_vectors([str(pid) for pid in page_ids], counts=counts)
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


def _run_fast_steps(
    pages,
    links,
    fast_steps: list[tuple[str, dict[str, Any], str | None]],
    *,
    write: bool,
    limit: int | None,
    batch_size: int,
    counts: CleanupCounts,
    delete_links: bool,
) -> None:
    """The field-based delete steps, and the dry run that sizes them.

    Extracted from cleanup() so it can be tested without a MongoClient. The two defects
    below were both invisible for the same reason: reaching this loop required a live
    database, so nothing exercised it.
    """
    if not write:
        # A dry run COUNTS. It does not walk one batch and extrapolate.
        #
        # The previous shape ran the same loop as --write and broke after the first batch,
        # while _delete_ids had already added that batch to deleted_pages. So at
        # batch_size 500 the report read deleted_pages=500 per step whether the true figure
        # was 500 or 50,000 -- the number an operator sizes a mass delete from was silently
        # capped at the batch size.
        #
        # It also could not reach the last step. The outer loop breaks once
        # deleted_pages >= limit, and this script's own documented example is
        # `--dry-run --limit 500` with batch_size defaulting to 500, so the first step filled
        # the budget and http_error never ran -- printing no reason_http_error line at all,
        # which reads as "there are none".
        for label, q, _hint in fast_steps:
            matched = pages.count_documents(q)
            counts.scanned += matched
            counts.by_reason[label] += matched
            counts.deleted_pages += matched
            print(f"would-delete fast label={label} matched={matched}", flush=True)

        # Stated rather than left to be discovered: the steps overlap, so a row carrying
        # both a FailureReason and a 403 is counted in two of them. The per-label figures
        # are exact; this is the true row count.
        distinct = pages.count_documents({"$or": [q for _label, q, _h in fast_steps]})
        print(
            f"would-delete fast DISTINCT rows across all steps={distinct} "
            "(the per-label numbers above overlap)",
            flush=True,
        )
        return

    for label, q, hint in fast_steps:
        if limit is not None and counts.deleted_pages >= limit:
            break
        while True:
            if limit is not None and counts.deleted_pages >= limit:
                break
            take = batch_size
            if limit is not None:
                take = min(batch_size, limit - counts.deleted_pages)
            cursor = pages.find(q, {"_id": 1, "Id": 1}).limit(take)
            if hint:
                cursor = cursor.hint(hint)
            # The hint is applied above but validated HERE. cursor.hint() only sets an
            # option on a local pymongo Cursor and never contacts the server, so the
            # try/except that used to wrap it could not fire. Mongo raises OperationFailure
            # on first iteration, for an index that does not exist -- and neither
            # FailureReason_1 nor RobotsAllowed_1 is created by any code in any repo, so on a
            # database where they were not made by hand this killed the whole run on step one
            # while the author's WARN message was unreachable.
            try:
                batch = list(cursor)
            except OperationFailure as exc:
                print(
                    f"WARN hint {hint} unusable ({exc}); retrying {label} unhinted",
                    flush=True,
                )
                batch = list(pages.find(q, {"_id": 1, "Id": 1}).limit(take))
            if not batch:
                break
            for _doc in batch:
                counts.scanned += 1
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
    # Prefer $gt:"" over $type so FailureReason_1 can be used.
    fast_steps: list[tuple[str, dict[str, Any], str | None]] = [
        ("failure", scoped({"FailureReason": {"$gt": ""}}), "FailureReason_1"),
        ("failure", scoped({"RobotsAllowed": False}), "RobotsAllowed_1"),
        # A 4xx or 5xx body is the server's error page, not the site's content.
        # Rows written before the crawler gated on status: nothing anywhere
        # looked at it, and a branded 404 clears every prose floor the pipeline
        # has, so it was stored, chunked and embedded under a URL that does not
        # exist. $gte:400 leaves 0 alone, which is the default for a row written
        # without the field rather than evidence of an error.
        # Matched by $expr, not {"$gte": 400}. GeekAPI stores StatusCode as a STRING, and
        # MongoDB brackets comparisons by BSON type, so a numeric $gte never compares
        # against "404" -- the previous filter returned 0 on a store that held them and
        # reported that indistinguishably from "there were none". $toInt handles both the
        # string shape and a native number, so this keeps working when the encoding is
        # fixed. Non-numeric values are left alone rather than erroring the pipeline: an
        # unreadable status is not evidence of an error page.
        (
            "http_error",
            scoped(
                {
                    "$expr": {
                        "$let": {
                            "vars": {
                                "code": {
                                    "$convert": {
                                        "input": "$StatusCode",
                                        "to": "int",
                                        "onError": 0,
                                        "onNull": 0,
                                    }
                                }
                            },
                            "in": {"$gte": ["$$code", 400]},
                        }
                    }
                }
            ),
            None,
        ),
    ]
    _run_fast_steps(
        pages,
        links,
        fast_steps,
        write=write,
        limit=limit,
        batch_size=batch_size,
        counts=counts,
        delete_links=delete_links,
    )

    # --- Locale scan over the remaining pages ---
    if not skip_locale_scan and (limit is None or counts.deleted_pages < limit):
        pending: list[dict[str, Any]] = []
        cursor = pages.find(
            run_filter,
            {
                "_id": 1,
                "Id": 1,
                "Url": 1,
                "FinalUrl": 1,
                "FailureReason": 1,
                "RobotsAllowed": 1,
                "Blocks": 1,
                "blocks": 1,
            },
        ).batch_size(batch_size)
        for doc in cursor:
            if limit is not None and counts.deleted_pages + len(pending) >= limit:
                break
            counts.scanned += 1
            reason = classify_from_mongo_doc(doc)
            # Fast path already covered failure; only locale deletes here. `no_content`
            # is reported by the classifier and deliberately never deleted.
            if reason != "locale":
                continue
            counts.by_reason[reason] += 1
            pending.append(doc)
            if len(pending) >= batch_size:
                _delete_ids(
                    pages, links, pending, write=write, counts=counts, delete_links=delete_links
                )
                pending = []
                print(
                    f"progress classify_scan scanned={counts.scanned} "
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
                f"progress classify_scan scanned={counts.scanned} "
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
    # Clamped below 1 rather than accepted: pymongo treats Cursor.limit(0) as NO limit
    # ("A limit of 0 is equivalent to no limit"), so --batch-size 0 turned a bounded batch
    # into the entire matching set and bypassed --limit with it. A negative value reduced
    # every pass to one row instead.
    parser.add_argument("--batch-size", type=int, default=500)
    # --write and --dry-run are mutually exclusive, and that is enforced rather than
    # documented. --dry-run was declared here and read nowhere: `write` came from
    # args.write alone, so `--write --dry-run` deleted. On the one script in this repo that
    # issues delete_many against crawl_pages -- and whose docstring is a memorial to 5,274
    # pages lost to a filter nobody expected -- the guard that exists to prevent exactly
    # that had no code behind it. argparse now refuses the combination outright, so the
    # failure is a usage error before any connection is opened.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--write", action="store_true")
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only, delete nothing. The default; cannot be combined with --write.",
    )
    parser.add_argument(
        "--skip-locale-scan",
        action="store_true",
        help="Only delete FailureReason / RobotsAllowed matches",
    )
    parser.add_argument(
        "--skip-link-delete",
        action="store_true",
        help="Delete pages only (faster). Orphan crawl_links can be purged later.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error(
            f"--batch-size must be at least 1 (got {args.batch_size}); "
            "0 means 'no limit' to pymongo and would submit the whole matching set as one "
            "delete"
        )
    # args.dry_run is read here. It cannot be true alongside --write, so this only ever
    # reinforces the default -- but it is read, so a future reader can see that it does
    # something, and a regression that stops honouring it fails the test that pins it.
    write = bool(args.write) and not args.dry_run
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
