#!/usr/bin/env python3
"""Backfill SPLADE sparse vectors onto points that predate hybrid retrieval.

The collection gained a named sparse vector (`text-sparse`, added by
`QdrantStore.ensure_collection`), but the points already in it carry no sparse
values. A hybrid query scores the sparse branch against whatever is there, so
until this has run, that branch contributes nothing for older points.

Writes with `update_vectors`, never `upsert`. This is the whole safety story of
the script: `upsert` replaces a point, so upserting a PointStruct carrying only
`{"text-sparse": ...}` would delete the dense embedding beside it -- 168k of them,
each one an OpenAI call to regenerate. `update_vectors` (PUT
/collections/{c}/points/vectors) writes only the named vectors handed to it and
leaves the rest of the point untouched.

Safe to interrupt and safe to re-run. Each batch requests only the sparse vector
back (`with_vectors=[SPARSE_VECTOR_NAME]`, not the dense one), so a point that
already has values is skipped rather than re-embedded, and the scroll cursor is
printed on exit so a run can be resumed from where it stopped.

Dry run by default: it reads, embeds and reports, and writes nothing without
`--apply`. The first run should be a dry run against production -- it costs only
time and proves the text extraction finds bodies where you expect them.

Text comes from the payload in the order the chunker wrote it: `text` is the
string that was embedded, with `childText` then `parentText` as fallbacks for
points whose `text` key is absent. A point with no text in any of the three is
counted and skipped -- it has nothing to encode, and inventing a body for it would
put a vector in the index that matches nothing on the page.

Usage:
  uv run python scripts/migrate_sparse_vectors.py                          # dry run, whole collection
  uv run python scripts/migrate_sparse_vectors.py --limit 2000             # dry run, bounded
  uv run python scripts/migrate_sparse_vectors.py --apply                  # write
  uv run python scripts/migrate_sparse_vectors.py --apply --resume-from <id>
  uv run python scripts/migrate_sparse_vectors.py --url https://qdrant.example --api-key "$QDRANT_API_KEY" --apply
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from geek_crawler_rag.config import Settings
from geek_crawler_rag.qdrant_store import SPARSE_VECTOR_NAME

#: Qdrant's own scroll ceiling is high, but 500 points of payload text plus a SPLADE forward pass is
#: the part that has to fit in the VPS's RAM, so the batch is sized for the embedder, not the wire.
DEFAULT_BATCH_SIZE = 500

#: Payload keys holding the chunk body, in the order the chunker wrote them.
TEXT_KEYS = ("text", "childText", "parentText")


def payload_text(payload: dict | None) -> str:
    """The chunk's body, or "" when the point carries none."""
    if not payload:
        return ""
    for key in TEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def has_sparse(vectors: object) -> bool:
    """Whether this point already carries sparse values, so a re-run can skip it."""
    if not isinstance(vectors, dict):
        return False
    existing = vectors.get(SPARSE_VECTOR_NAME)
    if existing is None:
        return False
    indices = getattr(existing, "indices", None)
    if indices is None and isinstance(existing, dict):
        indices = existing.get("indices")
    return bool(indices)


def preflight(client: QdrantClient, collection: str) -> int:
    """Refuse to run against a collection that cannot hold the vectors. Returns point count."""
    try:
        info = client.get_collection(collection)
    except Exception as err:  # noqa: BLE001 - report the cause, do not trace out of a CLI
        print(
            f"FAILED: could not read collection '{collection}': {type(err).__name__}: {err}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    configured = info.config.params.sparse_vectors or {}
    if SPARSE_VECTOR_NAME not in configured:
        print(
            f"FAILED: collection '{collection}' has no sparse vector named "
            f"'{SPARSE_VECTOR_NAME}'.\n"
            "        Start the service once so QdrantStore.ensure_collection adds it, or add it "
            "with update_collection, then re-run.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return int(info.points_count or 0)


def main() -> int:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", settings.qdrant_url))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY", settings.qdrant_api_key))
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--model", default=settings.sparse_model)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many points (0 = all)")
    parser.add_argument("--resume-from", default=None, help="scroll offset (point id) to resume at")
    parser.add_argument("--apply", action="store_true", help="write; omit for a dry run")
    args = parser.parse_args()

    if args.batch_size < 1:
        print("FAILED: --batch-size must be at least 1", file=sys.stderr)
        return 2

    client = QdrantClient(url=args.url, api_key=args.api_key, timeout=120)
    total_points = preflight(client, args.collection)

    mode = "APPLY (writing)" if args.apply else "DRY RUN (no writes)"
    print(f"collection: {args.collection} ({total_points} points)")
    print(f"vector:     {SPARSE_VECTOR_NAME}")
    print(f"model:      {args.model}")
    print(f"batch:      {args.batch_size}")
    print(f"mode:       {mode}")
    print()

    print(f"loading {args.model} (first run downloads the ONNX weights)...", flush=True)
    embedder = SparseTextEmbedding(model_name=args.model)
    print("model ready", flush=True)
    print()

    offset = args.resume_from
    seen = written = skipped_no_text = skipped_existing = 0
    started = time.monotonic()

    while True:
        remaining = args.limit - seen if args.limit else args.batch_size
        if args.limit and remaining <= 0:
            break

        points, offset = client.scroll(
            collection_name=args.collection,
            limit=min(args.batch_size, remaining) if args.limit else args.batch_size,
            offset=offset,
            with_payload=TEXT_KEYS,
            with_vectors=[SPARSE_VECTOR_NAME],
        )
        if not points:
            break

        seen += len(points)

        pending_ids: list[object] = []
        pending_texts: list[str] = []
        for point in points:
            if has_sparse(point.vector):
                skipped_existing += 1
                continue
            text = payload_text(point.payload)
            if not text:
                skipped_no_text += 1
                continue
            pending_ids.append(point.id)
            pending_texts.append(text)

        if pending_texts:
            # One forward pass for the batch rather than per point: the model is the cost here.
            embeddings = list(embedder.embed(pending_texts))
            vectors = [
                qm.PointVectors(
                    id=point_id,
                    vector={
                        SPARSE_VECTOR_NAME: qm.SparseVector(
                            indices=embedding.indices.tolist(),
                            values=embedding.values.tolist(),
                        ),
                    },
                )
                for point_id, embedding in zip(pending_ids, embeddings)
            ]

            if args.apply:
                # update_vectors, not upsert: upsert would replace the point and drop its dense
                # vector. See the module docstring.
                client.update_vectors(
                    collection_name=args.collection,
                    points=vectors,
                    wait=True,
                )
            written += len(vectors)

        elapsed = max(time.monotonic() - started, 1e-6)
        print(
            f"  seen {seen}"
            + (f"/{total_points}" if not args.limit else f"/{args.limit}")
            + f"  {'written' if args.apply else 'would write'} {written}"
            f"  skipped(existing {skipped_existing}, no-text {skipped_no_text})"
            f"  {seen / elapsed:.0f} pts/s",
            flush=True,
        )

        if offset is None:
            break

    print()
    print(f"seen:             {seen}")
    print(f"{'written' if args.apply else 'would write'}:{'          ' if args.apply else '       '}{written}")
    print(f"skipped existing: {skipped_existing}")
    print(f"skipped no text:  {skipped_no_text}")
    print(f"elapsed:          {time.monotonic() - started:.1f}s")
    if offset is not None:
        print(f"next offset:      {offset}   (pass --resume-from to continue)")
    else:
        print("next offset:      exhausted")
    if not args.apply:
        print()
        print("Dry run: nothing was written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
