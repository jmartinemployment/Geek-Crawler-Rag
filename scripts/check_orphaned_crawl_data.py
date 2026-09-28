#!/usr/bin/env python3
"""Report crawl data whose run document does not exist.

The inverse of check_index_reconciliation.py, which finds run documents with no index to
show for them. This finds the other direction: pages, links, index jobs and vectors whose
runId names a run that is not in `crawl_runs`.

Why this needs watching
-----------------------
Nothing else can see it, and nothing else can clean it.

Every cleanup path in the system is keyed off the run document. `DeleteRunAsync` takes a
runId that came from a slot lookup; the commit path retires the superseded run it found by
slot; `purge_runs_except.py` iterates `crawl_runs` and deletes what is outside a keep-set.
All three need a run document to start from. Remove that document while its children
survive and those rows are unreachable by every one of them -- `DeleteRunAsync` says so
itself: "A run document removed while its pages survive leaves rows no slot lookup can ever
reach again."

So the ordinary reassurance does not apply here. A new crawl of the same seed *does*
overwrite its predecessor -- fresh run, then the superseded one retired at commit -- but
that only reaches runs the slot lookup can find. An orphan is never overwritten, never
retired, and never reported. It just stays.

Not observed yet, and the near-miss is the reason this script reads ids the way it does.
On 2026-09-28 fourteen runs looked exactly like this -- rows in `crawl_pages`, nothing in
`crawl_runs` -- and the finding was wrong. `_id` on a run document is a Mongo ObjectId and
the run's GUID lives in `Id`; a probe matching on `_id` finds nothing and reports every run
as an orphan. That is a false positive that ends in an operator deleting a live corpus, so
`known_run_ids` reads both keys.

The Qdrant case is the one that costs more than disk
----------------------------------------------------
Orphaned vectors are reachable by retrieval. Every query is filtered to exactly one runId,
and `/v1/index/hosts` resolves a host to a runId by scanning Qdrant on host alone -- not by
asking which run is published. GeekAPI's GccGroundingResolver feeds that runId straight into
a query. So points belonging to a run that no longer exists can be selected as a host's
grounding evidence and cited. Pages and links merely accumulate; vectors mislead.

Reports only
------------
No --apply, deliberately. Deleting rows that cannot be attributed to a run is how 5,274
pages went on 2026-09-18, and an orphan is exactly the state where attribution has already
been lost. The output names what it found so an operator can act on it, and acting is a
separate, deliberate step.

A live crawl is not an orphan
-----------------------------
Pages arrive before a crawl finishes, so anything recent is excluded rather than reported:
the run document is created before the first page batch, so a healthy in-flight crawl has
one, but a clock skew or a mid-write read must not raise a false alarm. Same discipline as
`list_unindexed_runs.py`'s lease check, where the obvious reading of the data was wrong and
re-posting on it would have double-indexed six runs.

Qdrant needs no grace window: indexing only starts after a crawl commits, so a run being
indexed always has a document.

Exit codes: 0 no orphans, 1 orphans found, 2 the check could not run.

Usage:
  uv run python scripts/check_orphaned_crawl_data.py
  uv run python scripts/check_orphaned_crawl_data.py --grace-minutes 120
  uv run python scripts/check_orphaned_crawl_data.py --skip-qdrant
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import MongoClient
from qdrant_client import QdrantClient

from geek_crawler_rag.config import Settings
from geek_crawler_rag.status_store import COLLECTION as INDEX_JOBS_COLLECTION

DEFAULT_GRACE_MINUTES = 30
QDRANT_SCROLL_PAGE = 1_000
QDRANT_SCAN_CAP = 2_000_000


def as_utc(value: Any) -> datetime | None:
    """A Mongo datetime as an aware UTC datetime, or None when it is not one."""
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def known_run_ids(db: Any) -> set[str]:
    """Every runId in `crawl_runs`, by both `_id` and `Id`.

    Both are read, and this is the most important line in the script. Measured 2026-09-28:
    all 50 run documents carry an ObjectId in `_id` and the run's GUID in `Id`, and the two
    differ in every single one -- `_id=6ab50526319b3e8097720290`,
    `Id=23eb76a1-0e68-49ae-8501-1a629157bba9`. A check matching only `_id` therefore finds
    zero runs and declares every page, link, job and vector in the system an orphan. That
    exact mistake was made by hand the same day and briefly reported fourteen healthy runs
    as orphaned. Reading both keys costs nothing; reading one ends in an operator deleting a
    live corpus on the strength of this script's output.
    """
    ids: set[str] = set()
    for doc in db["crawl_runs"].find({}, {"_id": 1, "Id": 1}):
        for key in ("_id", "Id"):
            value = doc.get(key)
            if value is not None:
                ids.add(str(value))
    return ids


def orphans_in(
    db: Any,
    *,
    collection: str,
    run_field: str,
    time_field: str,
    known: set[str],
    cutoff: datetime,
) -> dict[str, dict[str, Any]]:
    """Group a collection's rows by runId, keeping only unknown runs settled before cutoff.

    A group is skipped when any row in it is newer than the cutoff: that is a crawl still
    writing, not an orphan.
    """
    groups: dict[str, dict[str, Any]] = {}
    for doc in db[collection].find({}, {run_field: 1, time_field: 1}):
        run_id = doc.get(run_field)
        if run_id is None:
            continue
        run_id = str(run_id)
        if run_id in known:
            continue
        stamp = as_utc(doc.get(time_field))
        entry = groups.setdefault(run_id, {"count": 0, "newest": None, "undated": 0})
        entry["count"] += 1
        if stamp is None:
            entry["undated"] += 1
        elif entry["newest"] is None or stamp > entry["newest"]:
            entry["newest"] = stamp
    return {
        run_id: entry
        for run_id, entry in groups.items()
        if entry["newest"] is None or entry["newest"] < cutoff
    }


def qdrant_orphans(settings: Settings, known: set[str]) -> dict[str, int] | None:
    """runId -> point count for points whose run is not in `crawl_runs`.

    None when the collection is absent: no collection means no vectors, which is not an
    orphan condition and must not read as one.
    """
    client = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=120
    )
    try:
        if not client.collection_exists(settings.qdrant_collection):
            return None
        counts: dict[str, int] = {}
        offset = None
        scanned = 0
        while scanned < QDRANT_SCAN_CAP:
            points, offset = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=QDRANT_SCROLL_PAGE,
                offset=offset,
                with_payload=["runId"],
                with_vectors=False,
            )
            if not points:
                break
            for point in points:
                scanned += 1
                run_id = (point.payload or {}).get("runId")
                if run_id is None:
                    counts["<no runId in payload>"] = (
                        counts.get("<no runId in payload>", 0) + 1
                    )
                elif str(run_id) not in known:
                    counts[str(run_id)] = counts.get(str(run_id), 0) + 1
            if offset is None:
                break
        if scanned >= QDRANT_SCAN_CAP:
            print(
                f"  WARNING scan stopped at the {QDRANT_SCAN_CAP:,}-point cap; "
                "results are partial",
                file=sys.stderr,
            )
        return counts
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report crawl data whose run document does not exist."
    )
    parser.add_argument(
        "--grace-minutes",
        type=int,
        default=DEFAULT_GRACE_MINUTES,
        help=(
            "Ignore Mongo rows newer than this, so a crawl still writing is not reported "
            f"as an orphan (default {DEFAULT_GRACE_MINUTES})."
        ),
    )
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Mongo only. Use when Qdrant is unreachable; it does not make the check pass.",
    )
    args = parser.parse_args()
    if args.grace_minutes < 0:
        print("--grace-minutes cannot be negative", file=sys.stderr)
        return 2

    try:
        settings = Settings()
    except Exception as exc:  # configuration is the one thing that must be loud
        print(f"Could not load settings: {exc}", file=sys.stderr)
        return 2

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=args.grace_minutes)
    print(f"Orphan check at {now.isoformat()}  grace={args.grace_minutes}m")

    try:
        client = MongoClient(settings.mongo_crawler_url, serverSelectionTimeoutMS=10_000)
        db = client[settings.mongo_db_name]
        known = known_run_ids(db)
    except Exception as exc:
        print(f"Could not read crawl_runs: {exc}", file=sys.stderr)
        return 2

    print(f"crawl_runs holds {len(known)} run id(s)\n")

    found = False
    checks = (
        ("crawl_pages", "RunId", "CrawledAtUtc"),
        ("crawl_links", "RunId", "DiscoveredAtUtc"),
        (INDEX_JOBS_COLLECTION, "runId", "claimedAtUtc"),
    )
    for collection, run_field, time_field in checks:
        try:
            groups = orphans_in(
                db,
                collection=collection,
                run_field=run_field,
                time_field=time_field,
                known=known,
                cutoff=cutoff,
            )
        except Exception as exc:
            print(f"{collection}: could not be read: {exc}", file=sys.stderr)
            return 2
        if not groups:
            print(f"{collection}: no orphans")
            continue
        found = True
        total = sum(entry["count"] for entry in groups.values())
        print(f"{collection}: ORPHANED -- {total} row(s) across {len(groups)} run(s)")
        for run_id, entry in sorted(
            groups.items(), key=lambda kv: kv[1]["count"], reverse=True
        ):
            newest = entry["newest"].isoformat() if entry["newest"] else "no timestamp"
            undated = f", {entry['undated']} undated" if entry["undated"] else ""
            print(f"    {run_id}  rows={entry['count']}  newest={newest}{undated}")

    if args.skip_qdrant:
        print("\nqdrant: SKIPPED by --skip-qdrant (not a pass)")
    else:
        try:
            counts = qdrant_orphans(settings, known)
        except Exception as exc:
            print(f"\nqdrant: could not be read: {exc}", file=sys.stderr)
            return 2
        if counts is None:
            print(f"\nqdrant: collection {settings.qdrant_collection} is absent")
        elif not counts:
            print("\nqdrant: no orphans")
        else:
            found = True
            total = sum(counts.values())
            print(
                f"\nqdrant: ORPHANED -- {total} point(s) across {len(counts)} run(s). "
                "These are citable: retrieval filters on one runId and /v1/index/hosts "
                "resolves a host's runId from Qdrant, not from the published run."
            )
            for run_id, count in sorted(
                counts.items(), key=lambda kv: kv[1], reverse=True
            ):
                print(f"    {run_id}  points={count}")

    if not found:
        print("\nNo orphaned crawl data.")
        return 0

    print(
        "\nOrphans found. Nothing was deleted -- this check never writes.\n"
        "Every cleanup path in the system starts from a run document, so these rows cannot\n"
        "be reached by DeleteRunAsync, by the commit path's retirement, or by\n"
        "purge_runs_except.py. Removing them is a deliberate, targeted delete against the\n"
        "runIds above, made by someone who has decided they are not wanted.\n"
        "Delete vectors for a run with DELETE /v1/index/runs/{runId} before its rows, so a\n"
        "surviving point can never outlive the page it cites."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
