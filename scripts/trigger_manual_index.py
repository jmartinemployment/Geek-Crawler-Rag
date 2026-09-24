#!/usr/bin/env python3
"""Enqueue a manual index pass for content-ready crawl runs.

`POST /v1/index` is the live indexing trigger (README "Indexing trigger and OpenAI rate limits"):
the scheduler is deprecated and `INDEX_SCHEDULER_ENABLED` is `false`, so nothing enqueues a run on
its own. This is the operator-facing way to do what the scheduler used to.

One run per request -- `IndexRunRequest` carries a single `runId` -- so a batch is a loop, sent
smallest-first to match the ordering the scheduler documented. Index concurrency on the service is
1, so these queue rather than run together; each POST returns as soon as the job is accepted.

`indexer.py` calls `ensure_collection()` at job start, so the first accepted job recreates
`geek_crawler_chunks` with the dense + sparse schema if it is missing.

Usage:
    python scripts/trigger_manual_index.py                 # every content-ready, unindexed run
    python scripts/trigger_manual_index.py <runId> [...]   # exactly these runs

Environment:
    RAG_API_URL   Base URL of the service. Defaults to http://127.0.0.1:<settings.port>, which is
                  correct inside the API container and wrong from anywhere else.
    API_KEY       Sent as X-Api-Key. Read from settings; the service rejects the call without it.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import httpx
from pymongo import MongoClient

from geek_crawler_rag.config import get_settings

REQUEST_TIMEOUT_SECONDS = 120


def _base_url(settings: Any) -> str:
    """Where the service is listening. Port comes from settings, not a literal."""
    configured = os.environ.get("RAG_API_URL", "").strip().rstrip("/")
    if configured:
        return configured
    return f"http://127.0.0.1:{settings.port}"


def _discover_run_ids(settings: Any) -> list[str]:
    """Content-ready runs with no recorded index, smallest first.

    The readiness filter mirrors `MongoCorpus.find_smallest_content_ready_run`: `Status` complete or
    external, and a non-empty `ContentReadyAt`, the marker GeekAPI stamps once every persisted page
    of the run carries extracted content. `RagIndexedAtUtc` is the "already done" marker written
    back when a run finishes indexing.
    """
    client: MongoClient = MongoClient(settings.mongo_crawler_url)
    try:
        db = client[settings.mongo_db_name]
        runs = list(
            db["crawl_runs"].find(
                {
                    "Status": {"$in": ["complete", "external"]},
                    "ContentReadyAt": {"$exists": True, "$nin": [None, ""]},
                    "RagIndexedAtUtc": None,
                    "Id": {"$type": "string", "$ne": ""},
                },
                {"Id": 1, "_id": 0},
            )
        )
        sized: list[tuple[int, str]] = []
        for run in runs:
            run_id = run.get("Id")
            if not run_id:
                continue
            sized.append((db["crawl_pages"].count_documents({"RunId": run_id}), run_id))
        sized.sort()
        return [run_id for _, run_id in sized]
    finally:
        client.close()


def _enqueue(client: httpx.Client, base_url: str, api_key: str, run_id: str) -> bool:
    """POST one run. Returns whether the service accepted it. Never raises."""
    try:
        response = client.post(
            f"{base_url}/v1/index",
            json={"runId": run_id},
            headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        print(f"  {run_id}  REQUEST FAILED  {type(exc).__name__}: {exc}", flush=True)
        return False

    body = response.text.strip()
    if len(body) > 400:
        body = body[:400] + "..."
    print(f"  {run_id}  HTTP {response.status_code}  {body}", flush=True)
    return response.status_code == 200


def main(argv: list[str]) -> int:
    settings = get_settings()
    api_key = settings.api_key or ""
    if not api_key:
        print("API_KEY is not configured; the service will reject every call.", file=sys.stderr)
        return 1

    base_url = _base_url(settings)
    run_ids = argv[1:] if len(argv) > 1 else _discover_run_ids(settings)
    if not run_ids:
        print("No content-ready runs are waiting to be indexed.")
        return 0

    print(f"Target: {base_url}/v1/index")
    print(f"Runs to enqueue ({len(run_ids)}), smallest first:")
    for run_id in run_ids:
        print(f"  {run_id}")
    print("")

    accepted = 0
    with httpx.Client() as client:
        for run_id in run_ids:
            if _enqueue(client, base_url, api_key, run_id):
                accepted += 1

    print("")
    print(f"Accepted {accepted} of {len(run_ids)} runs.")
    return 0 if accepted == len(run_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
