"""Reuse embeddings across runs, keyed on the text that produced them.

A re-crawl of a site produces chunks byte-identical to the previous crawl's, and
every one of them was embedded again from scratch. Nothing was wrong with the
existing skip: point IDs are deterministic (``qdrant_store.point_id``) and
``find_existing_point_ids`` skips chunks already committed - but the id is
``uuid5(f"{run_id}:{page_id}:{chunk_key}")``, so a new runId misses every time.

**This caches the vector, not the point.** That distinction is the whole design.
Reusing a point id from an earlier run would hand back a point whose payload
carries the old runId, and retrieval filters on runId (``qdrant_store``), so
those chunks would be invisible to the new run's queries - embedding saved,
corpus lost. Points stay per-run and cheap; the embedding call is what is expensive
and what is reused.

The key includes the model. Two embedding models index different spaces, so a
vector cached under one and served to the other scores nonsense without erroring
- the same failure mode the sparse model comment warns about.

Every failure here is a miss. A cache that cannot be read must not stop
indexing, and a cache that returns something it is unsure of is worse than no
cache at all, so the only value ever returned is one that was found, matched on
model, and is the right length.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, UpdateOne

logger = logging.getLogger(__name__)

COLLECTION = "rag_vector_cache"


def cache_key(text: str, *, model: str) -> str:
    """Identity of a vector: the exact text, under one model."""
    digest = hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()
    return digest


class VectorCache:
    """Text -> embedding, shared across runs. Misses are free; errors are misses."""

    def __init__(self, db: AsyncIOMotorDatabase, *, model: str, dimensions: int) -> None:
        self._collection = db[COLLECTION]
        self._model = model
        self._dimensions = dimensions
        self.hits = 0
        self.misses = 0

    async def ensure_indexes(self) -> None:
        try:
            await self._collection.create_index(
                [("key", ASCENDING)], unique=True, name="ix_rag_vector_cache_key"
            )
        except Exception:
            # An index that will not build is a performance problem, not a
            # correctness one - lookups still work, so indexing continues.
            logger.warning("Vector cache index not created; lookups will scan", exc_info=True)

    async def get_many(self, texts: list[str]) -> dict[str, list[float]]:
        """Cached vectors for whichever of `texts` are present."""
        if not texts:
            return {}
        by_key = {cache_key(t, model=self._model): t for t in dict.fromkeys(texts)}
        found: dict[str, list[float]] = {}
        try:
            cursor = self._collection.find(
                {"key": {"$in": list(by_key)}},
                projection={"key": 1, "vector": 1, "_id": 0},
            )
            async for doc in cursor:
                vector = doc.get("vector")
                key = doc.get("key")
                text = by_key.get(key or "")
                if text is None or not isinstance(vector, list):
                    continue
                # A stored vector of the wrong width belongs to another model or
                # a changed dimension setting. Dropping it is the only safe read.
                if len(vector) != self._dimensions:
                    logger.warning(
                        "Vector cache entry has %s dims, expected %s; ignoring",
                        len(vector),
                        self._dimensions,
                    )
                    continue
                found[text] = [float(v) for v in vector]
        except Exception:
            logger.warning("Vector cache read failed; embedding uncached", exc_info=True)
            return {}
        self.hits += len(found)
        self.misses += len(by_key) - len(found)
        return found

    async def put_many(self, vectors: dict[str, list[float]]) -> None:
        """Store newly embedded vectors. Never overwrites an existing entry."""
        if not vectors:
            return
        operations = [
            UpdateOne(
                {"key": cache_key(text, model=self._model)},
                {
                    "$setOnInsert": {
                        "key": cache_key(text, model=self._model),
                        "model": self._model,
                        "dimensions": len(vector),
                        "vector": vector,
                    }
                },
                upsert=True,
            )
            for text, vector in vectors.items()
            if len(vector) == self._dimensions
        ]
        if not operations:
            return
        try:
            await self._collection.bulk_write(operations, ordered=False)
        except Exception:
            # A write that does not land costs a re-embed next time. It must not
            # cost this run, which already has the vectors it needs.
            logger.warning("Vector cache write failed", exc_info=True)


def cache_stats(cache: VectorCache | None) -> dict[str, int]:
    if cache is None:
        return {"vectorCacheHits": 0, "vectorCacheMisses": 0}
    return {"vectorCacheHits": cache.hits, "vectorCacheMisses": cache.misses}
