#!/usr/bin/env python3
"""Parallel per-run Markdown backfill driver."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def backfill_one(run_id: str, mongo_url: str, batch_size: int) -> tuple[str, int, str]:
    env = os.environ.copy()
    env["MONGO_CRAWLER_URL"] = mongo_url
    print(f"--- start {run_id} ---", flush=True)
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-u",
            "scripts/backfill_markdown.py",
            "--run-id",
            run_id,
            "--write",
            "--batch-size",
            str(batch_size),
        ],
        cwd=str(ROOT),
        env=env,
    )
    return run_id, proc.returncode, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", required=True, help="TSV: runId\\t...\\tmissing")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--mongo-url", default=os.environ.get("MONGO_CRAWLER_URL", ""))
    args = parser.parse_args()
    if not args.mongo_url:
        print("MONGO_CRAWLER_URL required", file=sys.stderr)
        return 1

    run_ids: list[str] = []
    for line in Path(args.runs_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rid = line.split("\t", 1)[0].strip()
        if len(rid) == 36:
            run_ids.append(rid)

    print(
        f"parallel_backfill workers={args.workers} batch_size={args.batch_size} runs={len(run_ids)}",
        flush=True,
    )
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(backfill_one, rid, args.mongo_url, args.batch_size): rid
            for rid in run_ids
        }
        for fut in as_completed(futs):
            rid, code, _tail = fut.result()
            status = "OK" if code == 0 else f"FAIL({code})"
            print(f"=== {status} {rid} ===", flush=True)
            if code != 0:
                failed += 1
    print(f"parallel_backfill done failed={failed}/{len(run_ids)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
