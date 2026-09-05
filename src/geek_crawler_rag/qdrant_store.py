"""Qdrant collection ops for geek_crawler_chunks."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

logger = logging.getLogger(__name__)

# Stable namespace for deterministic point IDs.
_POINT_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def point_id(run_id: str, page_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_NS, f"{run_id}:{page_id}:{chunk_index}"))


class QdrantStore:
    def __init__(
        self,
        url: str,
        *,
        collection: str = "geek_crawler_chunks",
        api_key: str | None = None,
        vector_size: int = 1536,
    ) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key or None)
        self._collection = collection
        self._vector_size = vector_size

    async def close(self) -> None:
        await self._client.close()

    async def ping(self) -> bool:
        await self._client.get_collections()
        return True

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=self._vector_size,
                    distance=qm.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %s", self._collection)
        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        for field in ("runId", "crawlType", "host", "language"):
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # Index already exists — Qdrant raises; treat as idempotent.
                logger.debug("Payload index %s already present or create skipped", field)

    async def delete_by_run_id(self, run_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="runId",
                            match=qm.MatchValue(value=run_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        logger.info("Deleted Qdrant points for runId=%s", run_id)

    async def upsert(
        self,
        *,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids, vectors, and payloads must be the same length")
        points = [
            qm.PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in zip(ids, vectors, payloads, strict=True)
        ]
        await self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )

    async def search(
        self,
        vector: list[float],
        *,
        run_id: str,
        crawl_type: str | None = None,
        host: str | None = None,
        top_k: int = 8,
    ) -> list[qm.ScoredPoint]:
        must: list[qm.Condition] = [
            qm.FieldCondition(key="runId", match=qm.MatchValue(value=run_id)),
            qm.FieldCondition(key="language", match=qm.MatchValue(value="en")),
        ]
        if crawl_type:
            must.append(
                qm.FieldCondition(
                    key="crawlType",
                    match=qm.MatchValue(value=crawl_type),
                )
            )
        if host:
            must.append(
                qm.FieldCondition(
                    key="host",
                    match=qm.MatchValue(value=host.lower().strip()),
                )
            )
        result = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=qm.Filter(must=must),
            limit=top_k,
            with_payload=True,
        )
        return list(result.points)
