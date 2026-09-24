#!/usr/bin/env python3
"""Report the live collection's vector configuration. Read-only.

This exists because the code cannot tell us the answer. `ensure_collection`
(qdrant_store.py) creates the collection with an *unnamed* dense vector —
`vectors_config=qm.VectorParams(...)` — while the LlamaIndex Qdrant store writes a
*named* one, `{"text-dense": embedding}`, even with `enable_hybrid=False`
(llama_index/vector_stores/qdrant/base.py, `_build_points`). Those are
incompatible shapes, and yet 168,391 points exist, so one of the two is not what
is actually running. Which one decides whether enabling sparse vectors is an
additive change or a full collection rebuild.

Answer this before writing a migration. Guessing costs either a silent no-op or a
collection that stops accepting upserts.

What it prints, all from the collection itself:

  * whether the dense vector is named or unnamed, and under which name
  * its size, distance and on_disk flag
  * whether any sparse vector is configured, and under which name
  * the point count, and the vector keys actually present on a sample point —
    the config says what the collection accepts, the sample says what was written

Nothing is created, updated or deleted. Safe against production.

Usage:
  uv run python scripts/inspect_qdrant_vectors.py
  uv run python scripts/inspect_qdrant_vectors.py --url https://qdrant.example --api-key "$QDRANT_API_KEY"
  uv run python scripts/inspect_qdrant_vectors.py --collection geek_crawler_chunks --samples 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "geek_crawler_chunks")


def _request(url: str, api_key: str | None, payload: dict | None = None) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def describe_dense(vectors: object) -> list[str]:
    """Named vs unnamed is the whole question, so report the shape verbatim."""
    lines: list[str] = []
    if vectors is None:
        lines.append("  dense: NONE CONFIGURED")
        return lines

    if isinstance(vectors, dict) and {"size", "distance"} <= set(vectors):
        lines.append("  dense: UNNAMED (single default vector)")
        lines.append(f"    size={vectors.get('size')} distance={vectors.get('distance')}")
        lines.append(f"    on_disk={vectors.get('on_disk')}")
        lines.append("    => LlamaIndex writes {'text-dense': ...}; a named vector cannot")
        lines.append("       be added to an unnamed collection, so hybrid needs a rebuild.")
        return lines

    if isinstance(vectors, dict):
        lines.append(f"  dense: NAMED ({len(vectors)} vector(s))")
        for name, params in vectors.items():
            size = params.get("size") if isinstance(params, dict) else None
            distance = params.get("distance") if isinstance(params, dict) else None
            on_disk = params.get("on_disk") if isinstance(params, dict) else None
            lines.append(f"    '{name}': size={size} distance={distance} on_disk={on_disk}")
        if "text-dense" in vectors:
            lines.append("    => matches LlamaIndex's DEFAULT_DENSE_VECTOR_NAME. Adding a sparse")
            lines.append("       vector to this collection is additive; no dense rewrite needed.")
        return lines

    lines.append(f"  dense: UNRECOGNISED SHAPE {type(vectors).__name__}: {vectors!r}")
    return lines


def describe_sparse(sparse: object) -> list[str]:
    if not sparse:
        return [
            "  sparse: NONE CONFIGURED",
            "    => hybrid queries would target a vector this collection does not have.",
        ]
    lines = [f"  sparse: CONFIGURED ({len(sparse)} vector(s))"]
    for name, params in sparse.items():
        lines.append(f"    '{name}': {json.dumps(params)}")
    if "text-sparse-new" in sparse:
        lines.append("    => matches DEFAULT_SPARSE_VECTOR_NAME.")
    elif "text-sparse" in sparse:
        lines.append("    => matches DEFAULT_SPARSE_VECTOR_NAME_OLD; the store switches to the")
        lines.append("       old encoder when it sees this name.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    print(f"collection: {args.collection}")
    print(f"url:        {base}")
    print()

    try:
        info = _request(f"{base}/collections/{args.collection}", args.api_key)["result"]
    except urllib.error.HTTPError as err:
        print(f"FAILED: HTTP {err.code} reading the collection", file=sys.stderr)
        print(err.read().decode("utf-8", "replace")[:400], file=sys.stderr)
        return 2
    except Exception as err:  # noqa: BLE001 - a diagnostic reports, it does not raise
        print(f"FAILED: {type(err).__name__}: {err}", file=sys.stderr)
        return 2

    params = info.get("config", {}).get("params", {})
    print(f"points_count:  {info.get('points_count')}")
    print(f"vectors_count: {info.get('vectors_count')}")
    print(f"status:        {info.get('status')}")
    print()
    print("configured vectors:")
    for line in describe_dense(params.get("vectors")):
        print(line)
    for line in describe_sparse(params.get("sparse_vectors")):
        print(line)

    if args.samples <= 0:
        return 0

    print()
    print(f"sample points (what was actually written, n={args.samples}):")
    try:
        scrolled = _request(
            f"{base}/collections/{args.collection}/points/scroll",
            args.api_key,
            {"limit": args.samples, "with_vector": True, "with_payload": False},
        )["result"]
    except Exception as err:  # noqa: BLE001
        print(f"  could not scroll: {type(err).__name__}: {err}")
        return 0

    points = scrolled.get("points") or []
    if not points:
        print("  none returned — the collection is empty.")
        return 0

    for point in points:
        vector = point.get("vector")
        if isinstance(vector, dict):
            summary = ", ".join(
                f"'{name}'({'sparse' if isinstance(value, dict) else len(value)})"
                for name, value in vector.items()
            )
            print(f"  {point.get('id')}: named vectors -> {summary}")
        elif isinstance(vector, list):
            print(f"  {point.get('id')}: unnamed vector, {len(vector)} dims")
        else:
            print(f"  {point.get('id')}: vector absent or unrecognised ({type(vector).__name__})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
