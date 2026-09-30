"""Capture retrieval behaviour so a rebuild can be judged against evidence rather than impression.

Run it **before** changing the embedding stack and again after. It writes the same JSON shape both
times, so the two files diff.

    # before (against the running service)
    python scripts/retrieval_baseline.py --build-queries --out /tmp/before.json
    # after
    python scripts/retrieval_baseline.py --queries /tmp/queries.json --out /tmp/after.json
    python scripts/retrieval_baseline.py --compare /tmp/before.json /tmp/after.json

**The query set is fixed on the first run and reused.** Sampling fresh queries for the "after" run
would compare two different questions and call the difference a result.

Two strata, because a uniform sample cannot see the risk. 71.4% of this corpus has a parent byte-
identical to its child, so most queries land where the parent-tier change provably costs nothing:

* ``literal`` -- product codes, SKUs, version strings scraped out of the corpus itself. These test
  term binding, which is the job the sparse channel exists for. Scored on whether the literal comes
  back in a retrieved passage, which is the same thing ``citation_verify.quote_in_text`` asks.
* ``wide_parent`` -- section titles from units whose parent is at least twice its child. These test
  the synthesis window that no 200-token child contains, and they are the queries most likely to
  regress when the parent tier stops being emitted for duplicates.

Read-only against Qdrant and the API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

# What actually needs binding in THIS corpus. A first pass looked only for hyphenated serial codes
# and found 2 across 60 samples -- accounting-software marketing pages do not carry part numbers. The
# hard terms here are product and vendor names, which is what
# test_hybrid_query_surfaces_a_proper_noun_the_dense_vector_misses exists for: a dense vector maps
# "SuiteBilling" near "billing software" and loses the identity, while BM25 keeps the token.
CODE = re.compile(
    r"\b(?:"
    r"[A-Z][A-Z0-9]{1,}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)*"   # XJ-4420-B
    r"|\d+\.\d+\.\d+"                                # 7.3.1
    r"|[A-Z][a-z]+[A-Z][a-zA-Z]{2,}"                     # SuiteBilling, NetSuite, AvaTax
    r")\b"
)
# Words that are CamelCase but generic, so finding them proves nothing about term binding.
STOP = {
    "HTTP-API", "B-2-B", "SOC-2", "ISO-27001",
    "JavaScript", "TypeScript", "PowerPoint", "SharePoint", "LinkedIn", "YouTube",
    "WordPress", "PayPal", "iPhone", "iPad", "eCommerce", "ePayments",
}


async def build_queries(base: str, headers: dict[str, str], limit: int) -> list[dict[str, Any]]:
    """Sample both strata straight out of the live collection."""
    qdrant = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    collection = os.environ.get("QDRANT_COLLECTION", "geek_crawler_chunks")
    queries: list[dict[str, Any]] = []
    seen_literals: set[str] = set()
    offset = None
    # Half each. The first run came back 58 wide_parent to 2 literal, which cannot detect a
    # term-binding regression -- the thing the sparse change is for.
    per_stratum = max(1, limit // 2)

    async with httpx.AsyncClient(timeout=120) as client:
        while len(queries) < limit:
            body: dict[str, Any] = {"limit": 400, "with_payload": True, "with_vectors": False}
            if offset:
                body["offset"] = offset
            r = await client.post(f"{qdrant}/collections/{collection}/points/scroll", json=body)
            r.raise_for_status()
            result = r.json()["result"]
            points, offset = result["points"], result.get("next_page_offset")
            if not points:
                break

            counts = {
                "literal": sum(1 for q in queries if q["stratum"] == "literal"),
                "wide_parent": sum(1 for q in queries if q["stratum"] == "wide_parent"),
            }
            for p in points:
                pay = p["payload"]
                run_id, host = pay.get("runId"), pay.get("host")
                child = (pay.get("childText") or "").strip()
                parent = (pay.get("parentText") or "").strip()
                if not run_id or not child:
                    continue

                # The corpus's own idea of the entity is a better term-binding probe than anything
                # regex can find, and it is an indexed payload field.
                entity = (pay.get("entityName") or "").strip()
                candidates = CODE.findall(child)
                if entity and len(entity) > 3 and entity.lower() in child.lower():
                    candidates = [entity, *candidates]

                for match in candidates:
                    if counts["literal"] >= per_stratum:
                        break
                    if match in seen_literals or match in STOP or len(match) < 5:
                        continue
                    seen_literals.add(match)
                    queries.append({
                        "stratum": "literal",
                        "need": match,
                        "runId": str(run_id),
                        "host": host,
                        "expect_literal": match,
                    })
                    counts["literal"] += 1
                    break

                if counts["literal"] >= per_stratum and counts["wide_parent"] >= per_stratum:
                    break

                title = (pay.get("sectionTitle") or "").strip()
                if (counts["wide_parent"] < per_stratum
                        and title and parent and len(parent) >= 2 * max(1, len(child))):
                    queries.append({
                        "stratum": "wide_parent",
                        "need": title,
                        "runId": str(run_id),
                        "host": host,
                        "expect_literal": None,
                        "parent_child_ratio": round(len(parent) / max(1, len(child)), 2),
                    })
                    counts["wide_parent"] += 1
                if len(queries) >= limit:
                    break
            if not offset:
                break
    return queries


async def run_queries(base: str, headers: dict[str, str], queries: list[dict]) -> dict[str, Any]:
    results = []
    async with httpx.AsyncClient(timeout=180) as client:
        for q in queries:
            row: dict[str, Any] = {k: q[k] for k in ("stratum", "need", "runId", "host")}
            row["parent_child_ratio"] = q.get("parent_child_ratio")
            try:
                r = await client.post(
                    f"{base}/v1/query",
                    headers=headers,
                    json={"need": q["need"], "runId": q["runId"], "topK": 8},
                )
                row["status"] = r.status_code
                if r.status_code != 200:
                    row["error"] = r.text[:200]
                    results.append(row)
                    continue
                payload = r.json()
                chunks = payload.get("chunks") or payload.get("results") or []
                row["returned"] = len(chunks)
                texts = [
                    str(c.get("text") or c.get("childText") or c.get("parentText") or "")
                    for c in chunks
                ]
                row["top_ids"] = [c.get("chunkId") or c.get("id") for c in chunks[:3]]
                lit = q.get("expect_literal")
                if lit:
                    hit = next((i for i, t in enumerate(texts) if lit.lower() in t.lower()), None)
                    row["literal_rank"] = hit
                    row["literal_found"] = hit is not None
            except Exception as exc:  # noqa: BLE001
                row["status"] = 0
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            results.append(row)

    lits = [r for r in results if r["stratum"] == "literal"]
    wides = [r for r in results if r["stratum"] == "wide_parent"]
    return {
        "summary": {
            "queries": len(results),
            "literal": len(lits),
            "literal_found": sum(1 for r in lits if r.get("literal_found")),
            "literal_rank1": sum(1 for r in lits if r.get("literal_rank") == 0),
            "wide_parent": len(wides),
            "wide_parent_returned": sum(1 for r in wides if (r.get("returned") or 0) > 0),
            "errors": sum(1 for r in results if r.get("status") != 200),
        },
        "results": results,
    }


def compare(before_path: str, after_path: str) -> int:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    b, a = before["summary"], after["summary"]

    print(f"{'metric':28} {'before':>8} {'after':>8}   verdict")
    regressed = False
    for key, higher_is_better in (
        ("queries", None), ("literal", None),
        ("literal_found", True), ("literal_rank1", True),
        ("wide_parent", None), ("wide_parent_returned", True),
        ("errors", False),
    ):
        bv, av = b.get(key, 0), a.get(key, 0)
        verdict = ""
        if higher_is_better is True and av < bv:
            verdict, regressed = "REGRESSED", True
        elif higher_is_better is False and av > bv:
            verdict, regressed = "REGRESSED", True
        elif higher_is_better is not None and av > bv:
            verdict = "better"
        print(f"{key:28} {bv:>8} {av:>8}   {verdict}")

    bl = {r["need"]: r for r in before["results"] if r["stratum"] == "literal"}
    lost = [n for n, r in bl.items()
            if r.get("literal_found")
            and not next((x for x in after["results"] if x["need"] == n), {}).get("literal_found")]
    if lost:
        print(f"\nliterals that stopped being found ({len(lost)}):")
        for n in lost[:20]:
            print(f"   {n}")
    print("\nGATE: " + ("FAIL — a measured regression" if regressed else "PASS"))
    return 1 if regressed else 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-queries", action="store_true")
    ap.add_argument("--queries", default="/tmp/queries.json")
    ap.add_argument("--out", default="/tmp/retrieval.json")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)

    key = os.environ.get("API_KEY", "")
    if not key:
        print("API_KEY is not set", file=sys.stderr)
        return 2
    base = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8080")
    headers = {"X-API-Key": key, "Content-Type": "application/json"}

    if args.build_queries:
        queries = await build_queries(base, headers, args.limit)
        Path(args.queries).write_text(json.dumps(queries, indent=2))
        print(f"built {len(queries)} queries -> {args.queries}")
        by = {}
        for q in queries:
            by[q["stratum"]] = by.get(q["stratum"], 0) + 1
        print("  strata:", by)
    else:
        queries = json.loads(Path(args.queries).read_text())
        print(f"reusing {len(queries)} queries from {args.queries}")

    out = await run_queries(base, headers, queries)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
