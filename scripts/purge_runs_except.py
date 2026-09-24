#!/usr/bin/env python3
"""Delete every crawl run outside an explicit keep-set, and everything hanging off it.

Takes the runIds to KEEP, not the ones to delete. A delete-list that is one id short leaves junk
behind; a keep-list that is one id short is caught by the guard below, which refuses to run unless
every id named is actually present in `crawl_runs`. That asymmetry is the whole reason the
argument is inverted.

Removes, for each run outside the keep-set, in this order:
  rag_index_jobs (runId)  ->  crawl_links (RunId)  ->  crawl_pages (RunId)  ->  crawl_runs (Id)

Children first so an interruption leaves the run document still naming what remains, rather than
orphans nothing can find. Field names are PascalCase on the crawl collections and camelCase on
rag_index_jobs; they are not interchangeable and a wrong one deletes nothing while reporting
success.

Refuses to delete a run whose index job is `running` or `pending` -- that one is being worked on.

Dry run by default. Nothing is deleted without --apply.

Usage:
  uv run python scripts/purge_runs_except.py <keepId> [<keepId> ...]
  uv run python scripts/purge_runs_except.py <keepId> [...] --apply
"""

from __future__ import annotations

import argparse
import sys

from pymongo import MongoClient

from geek_crawler_rag.config import Settings
from geek_crawler_rag.status_store import COLLECTION as INDEX_JOBS_COLLECTION

IN_FLIGHT_STATES = ("running", "pending")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keep", nargs="+", help="runIds to keep; everything else is deleted")
    parser.add_argument("--apply", action="store_true", help="delete; omit for a dry run")
    args = parser.parse_args(argv[1:])

    keep = sorted(set(args.keep))
    settings = Settings()
    client: MongoClient | None = None
    try:
        client = MongoClient(settings.mongo_crawler_url, serverSelectionTimeoutMS=10_000)
        db = client[settings.mongo_db_name]
        runs, pages, links = db["crawl_runs"], db["crawl_pages"], db["crawl_links"]
        jobs = db[INDEX_JOBS_COLLECTION]

        present = {r["Id"] for r in runs.find({"Id": {"$in": keep}}, {"Id": 1, "_id": 0})}
        missing = [run_id for run_id in keep if run_id not in present]
        if missing:
            print(
                "REFUSED: these keep-ids are not in crawl_runs, so the keep-set is wrong and "
                "deleting by exclusion would remove more than intended:",
                file=sys.stderr,
            )
            for run_id in missing:
                print(f"  {run_id}", file=sys.stderr)
            return 2

        doomed = [
            r["Id"]
            for r in runs.find({"Id": {"$nin": keep}}, {"Id": 1, "_id": 0})
            if r.get("Id")
        ]
        if not doomed:
            print(f"Nothing to do: all {len(present)} runs are in the keep-set.")
            return 0

        blocked = list(
            jobs.find(
                {"runId": {"$in": doomed}, "state": {"$in": list(IN_FLIGHT_STATES)}},
                {"runId": 1, "state": 1, "_id": 0},
            )
        )
        if blocked:
            print("REFUSED: runs outside the keep-set have index jobs in flight:", file=sys.stderr)
            for job in blocked:
                print(f"  {job.get('runId')}  {job.get('state')}", file=sys.stderr)
            return 3

        n_jobs = jobs.count_documents({"runId": {"$in": doomed}})
        n_links = links.count_documents({"RunId": {"$in": doomed}})
        n_pages = pages.count_documents({"RunId": {"$in": doomed}})

        print(f"keeping:  {len(present)} runs")
        print(f"deleting: {len(doomed)} runs")
        print(f"          {n_jobs} rag_index_jobs")
        print(f"          {n_links} crawl_links")
        print(f"          {n_pages} crawl_pages")
        print("")

        if not args.apply:
            print("Dry run: nothing was deleted. Re-run with --apply.")
            return 0

        deleted_jobs = jobs.delete_many({"runId": {"$in": doomed}}).deleted_count
        print(f"deleted rag_index_jobs: {deleted_jobs}", flush=True)
        deleted_links = links.delete_many({"RunId": {"$in": doomed}}).deleted_count
        print(f"deleted crawl_links:    {deleted_links}", flush=True)
        deleted_pages = pages.delete_many({"RunId": {"$in": doomed}}).deleted_count
        print(f"deleted crawl_pages:    {deleted_pages}", flush=True)
        deleted_runs = runs.delete_many({"Id": {"$in": doomed}}).deleted_count
        print(f"deleted crawl_runs:     {deleted_runs}", flush=True)

        print("")
        print(f"remaining crawl_runs:      {runs.count_documents({})}")
        print(f"remaining rag_index_jobs:  {jobs.count_documents({})}")
        return 0 if deleted_runs == len(doomed) else 1
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
