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

A dry run counts and a write deletes, and they report under separate keys -- `would_delete`
and `deleted_pages`. Both used to be `deleted_pages`, so the number an operator compared
across the two runs was one label with two meanings. `--limit` is charged against
`CleanupCounts.budget_used`, the sum, so it bounds the same thing in both modes; --limit and
--batch-size are both refused below 1, because 0 means "no limit" to pymongo in one case and
"stop before starting, while printing limit=*" in the other.

No step is skipped, and every step reports even at 0 -- under either mode, whatever --limit
is. A missing `reason_http_error` line reads as "there were no 4xx pages"; a `=0` line is the
only honest way to say "checked, found none".

No index hints. Neither FailureReason_1 nor RobotsAllowed_1 exists in any repo and Mongo
errors on a hint naming a missing index, so the hints were the failure. They are removed
rather than caught -- catching that OperationFailure and re-running the query unhinted is the
fallback CLAUDE.md section 2 forbids, and under --write it proceeded to purge vectors and
delete rows gathered by an unplanned scan.

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

from geek_crawler_rag.unusable import classify_from_mongo_doc  # noqa: E402


@dataclass
class CleanupCounts:
    """Rows removed and rows that WOULD be removed are two numbers, never one.

    `deleted_pages` counted both, so a dry run's report and a write's report used the same
    key for "nothing was touched, this many matched" and "this many rows are gone". An
    operator comparing a --dry-run against the --write that followed it was reading one
    label with two meanings, and a dry run that printed deleted_pages=50019 is the wrong
    thing to have to interpret before a mass delete.
    """

    scanned: int = 0
    deleted_pages: int = 0
    would_delete: int = 0
    deleted_links: int = 0
    purged_vector_pages: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)

    @property
    def budget_used(self) -> int:
        """Rows charged against --limit.

        Exactly one of the two grows in a given mode -- a dry run never deletes and a write
        never records a would-delete -- so the sum is the spend either way. This is a
        separate concept from both, and conflating it with deleted_pages is what made
        --limit mean "rows deleted" under --write and "rows matched" under --dry-run.
        """
        return self.deleted_pages + self.would_delete

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scanned": self.scanned,
            "deleted_pages": self.deleted_pages,
            "would_delete": self.would_delete,
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
        # Not deleted_pages. A dry run removes nothing, so it must not report a number under
        # the key that means "removed".
        counts.would_delete += n


def _run_fast_steps(
    pages,
    links,
    fast_steps: list[tuple[str, dict[str, Any]]],
    *,
    write: bool,
    limit: int | None,
    batch_size: int,
    counts: CleanupCounts,
    delete_links: bool,
) -> None:
    """The field-based delete steps, and the dry run that sizes them.

    Extracted from cleanup() so it can be tested without a MongoClient. Every defect this
    function has carried was invisible for one reason: reaching this loop required a live
    database, so nothing exercised it -- the dry run reporting one batch as the total, the
    index hints that killed the run, the fallback that then "recovered" from them, `--limit`
    meaning two different things, and `--write` never reaching its last step. The stubs in
    tests/test_cleanup_dry_run_counts.py exist so that stops being true.

    A dry run COUNTS -- it does not walk one batch and extrapolate. A write deletes, reaching
    every step, and reports each label even at 0, because an absent `reason_http_error` line
    reads as "there were none" rather than "never checked".
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
        for label, q in fast_steps:
            matched = pages.count_documents(q)
            counts.by_reason[label] += matched
            print(f"would-delete fast label={label} matched={matched}", flush=True)

        # The steps overlap -- a row carrying both a FailureReason and a 403 is matched by two
        # of them -- so the per-label figures above sum to more than the number of rows. They
        # are each exact as a per-reason answer, which is why they are kept; this is the row
        # count, and it is what would_delete and scanned take.
        #
        # Summing them into would_delete was an overcount, and the wrong kind: it inflates the
        # number an operator reads as "rows this will remove" just before a mass delete. A
        # --write run deletes each row once, so the dry run's headline figure has to be the
        # union, not the sum.
        distinct = pages.count_documents({"$or": [q for _label, q in fast_steps]})
        counts.scanned += distinct
        counts.would_delete += distinct
        print(
            f"would-delete fast DISTINCT rows across all steps={distinct} "
            "(the per-label numbers above overlap and sum higher; this is the row count)",
            flush=True,
        )
        return

    for label, q in fast_steps:
        # Every step is entered, and every step reports -- even at 0. The old shape broke out
        # of this loop once the budget was spent, so `--write --limit 500` never ran the
        # http_error step and printed no reason_http_error line, which reads as "there are
        # none" rather than "never checked". Seeding the key here makes a checked-and-empty
        # step visible, and makes the --write report comparable key-for-key with --dry-run.
        counts.by_reason.setdefault(label, 0)
        while True:
            if limit is not None and counts.budget_used >= limit:
                break
            take = batch_size
            if limit is not None:
                take = min(batch_size, limit - counts.budget_used)
                if take < 1:
                    break
            # No hint. The two indexes the old code named -- FailureReason_1 and
            # RobotsAllowed_1 -- are created by no code in any repo, and Mongo errors on a
            # hint naming an index that does not exist. So the hint was the failure, not a
            # protection against one.
            #
            # An earlier version of this caught that OperationFailure and re-ran the query
            # unhinted. That was a fallback, which CLAUDE.md section 2 forbids by name:
            # "If a primary path fails, abort instantly. Never write backup paths, secondary
            # loops, or default to unverified data to salvage the operation." It was also
            # worse than it looked -- under --write it proceeded to _purge_vectors and
            # delete_many on rows gathered by an unplanned scan.
            #
            # Dropping the hint removes the exception rather than catching it, and lets the
            # planner choose. If these queries need an index, it belongs in
            # MongoCorpus.ensure_indexes beside its hint, which is where mongo.py already
            # states the rule: "Mongo errors on a hint naming an index that does not exist,
            # so the two must be changed together."
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
                f"deleted_pages={counts.deleted_pages} would_delete={counts.would_delete} "
                f"deleted_links={counts.deleted_links} "
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

    # --- Fast path: field-based deletes ---
    # $gt:"" rather than $type: it is a range predicate, so it stays index-eligible if an
    # index is ever added for it, where $type cannot be.
    # (label, query). There is no hint column: it held None for every step once the hints
    # came out, and a dead `str | None` named `hint` reads as evidence that hinting is still a
    # thing this script does. It is not, and the name is what gets grepped.
    fast_steps: list[tuple[str, dict[str, Any]]] = [
        ("failure", scoped({"FailureReason": {"$gt": ""}})),
        # Matched as a STRING, not as a boolean. GeekAPI stores this field as "t"/"f", and
        # MongoDB brackets equality by BSON type, so {"RobotsAllowed": False} never compared
        # against "f" -- the step could not match a row, and the dry run printed a confident
        # matched=0 for it. 477ef48 converted every other reader of this field (mongo.py's
        # _as_bool, unusable.py's classify_from_mongo_doc) and left this one, so a
        # robots-denied page was deleted by neither this step nor the locale scan, which skips
        # non-locale reasons on the stated assumption that this step covered them.
        # $in rather than $expr: it is an equality on a small closed set, so it stays
        # index-eligible, and it keeps working when the encoding is fixed to a real boolean.
        (
            "failure",
            scoped({"RobotsAllowed": {"$in": ["f", "false", "0", "n", "no", False]}}),
        ),
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
    if not skip_locale_scan and (limit is None or counts.budget_used < limit):
        pending: list[dict[str, Any]] = []
        # "The remaining pages" is enforced, not assumed. The loop below skips any doc the
        # classifier does not call `locale`, on the stated grounds that "fast path already
        # covered failure" -- but that only holds under --write, where those rows are already
        # deleted. Under --dry-run nothing was deleted, and classify_unusable_page checks
        # locale FIRST (unusable.py:124, before http_error at :137 and failure at :140), so a
        # locale-pathed 404 is reported `locale` here AND counted by the http_error fast step.
        # The dry run's would_delete was an overcount by exactly that intersection.
        #
        # $nor over the same query objects the fast steps used -- not a second copy of their
        # predicates. Two implementations of one filter is the drift CLAUDE.md names, and this
        # one has to agree with the fast steps exactly or the subtraction is wrong. Mongo does
        # the exclusion, so the rows a fast step would take never reach the classifier.
        exclude_fast = {"$nor": [q for _label, q in fast_steps]}
        locale_filter: dict[str, Any] = (
            {"$and": [run_filter, exclude_fast]} if run_filter else exclude_fast
        )
        cursor = pages.find(
            locale_filter,
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
            if limit is not None and counts.budget_used + len(pending) >= limit:
                break
            counts.scanned += 1
            reason = classify_from_mongo_doc(doc)
            # Only locale deletes here. Every fast-step match was excluded by the query above,
            # so a row reaching this point is one no fast step would take. That makes
            # by_reason["locale"] disjoint from every other label, so it adds to the fast
            # steps' distinct count without double-counting -- and would_delete is then the
            # true number of rows a --write would remove. `no_content` is reported by the
            # classifier and deliberately never deleted.
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
                    f"deleted_pages={counts.deleted_pages} would_delete={counts.would_delete} "
                f"deleted_links={counts.deleted_links} "
                    f"reasons={dict(counts.by_reason)}",
                    flush=True,
                )
                # No break here. A dry run used to stop after the first batch of this scan
                # and report it as the total, so an unbounded --dry-run said "batch_size
                # locale pages" whether there were 500 or 50,000 -- the same defect the fast
                # steps had, in the one step that cannot be answered by count_documents
                # because the locale decision is made in Python by classify_from_mongo_doc.
                # So it costs a full cursor walk, which is what an accurate answer costs and
                # what --write already pays. --limit still bounds it.
        if pending:
            _delete_ids(
                pages, links, pending, write=write, counts=counts, delete_links=delete_links
            )
            print(
                f"progress classify_scan scanned={counts.scanned} "
                f"deleted_pages={counts.deleted_pages} would_delete={counts.would_delete} "
                f"deleted_links={counts.deleted_links} "
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
    # Clamped below 1 for the same reason as --batch-size, and displayed without `or`.
    # --limit 0 was accepted, made every budget check true immediately so nothing ran, and
    # then printed `limit=*` -- because `args.limit or '*'` renders 0 as the unlimited
    # marker. A run that deleted nothing announced itself as unbounded.
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
    if args.limit is not None and args.limit < 1:
        parser.error(
            f"--limit must be at least 1 (got {args.limit}); "
            "0 stops every step before it starts and then reports itself as unlimited"
        )
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
        f"limit={'*' if args.limit is None else args.limit} "
        f"locale_scan={not args.skip_locale_scan} "
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
