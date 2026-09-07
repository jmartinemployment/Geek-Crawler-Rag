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


def point_id(run_id: str, page_id: str, chunk_key: int | str) -> str:
    return str(uuid.uuid5(_POINT_NS, f"{run_id}:{page_id}:{chunk_key}"))


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
        keyword_fields = (
            "runId",
            "crawlType",
            "host",
            "language",
            "chunkRole",
            "sourceType",
            "entityName",
            "category",
            "pageId",
        )
        for field in keyword_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug("Payload index %s already present or create skipped", field)

        for field in ("childText", "text", "parentText"):
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.TEXT,
                )
            except Exception:
                logger.debug("Text index %s already present or create skipped", field)

        try:
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name="qualityScore",
                field_schema=qm.PayloadSchemaType.FLOAT,
            )
        except Exception:
            logger.debug("Payload index qualityScore already present or create skipped")

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

    async def delete_by_page_id(self, page_id: str) -> None:
        if not page_id:
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="pageId",
                            match=qm.MatchValue(value=page_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

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

    def build_filter(
        self,
        *,
        run_id: str,
        crawl_type: str | None = None,
        host: str | None = None,
        chunk_role: str | None = None,
        source_types: list[str] | None = None,
        entity_names: list[str] | None = None,
        categories: list[str] | None = None,
        min_quality: float | None = None,
    ) -> qm.Filter:
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
        if chunk_role:
            must.append(
                qm.FieldCondition(
                    key="chunkRole",
                    match=qm.MatchValue(value=chunk_role),
                )
            )
        if source_types:
            must.append(
                qm.FieldCondition(
                    key="sourceType",
                    match=qm.MatchAny(any=[s.strip().lower() for s in source_types if s]),
                )
            )
        if entity_names:
            must.append(
                qm.FieldCondition(
                    key="entityName",
                    match=qm.MatchAny(any=[n.strip() for n in entity_names if n]),
                )
            )
        if categories:
            must.append(
                qm.FieldCondition(
                    key="category",
                    match=qm.MatchAny(any=[c.strip().lower() for c in categories if c]),
                )
            )
        if min_quality is not None:
            must.append(
                qm.FieldCondition(
                    key="qualityScore",
                    range=qm.Range(gte=float(min_quality)),
                )
            )
        return qm.Filter(must=must)

    async def search(
        self,
        vector: list[float],
        *,
        run_id: str,
        crawl_type: str | None = None,
        host: str | None = None,
        top_k: int = 8,
        chunk_role: str | None = None,
        source_types: list[str] | None = None,
        entity_names: list[str] | None = None,
        categories: list[str] | None = None,
        min_quality: float | None = None,
    ) -> list[qm.ScoredPoint]:
        query_filter = self.build_filter(
            run_id=run_id,
            crawl_type=crawl_type,
            host=host,
            chunk_role=chunk_role,
            source_types=source_types,
            entity_names=entity_names,
            categories=categories,
            min_quality=min_quality,
        )
        result = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return list(result.points)

    async def search_text(
        self,
        text: str,
        *,
        run_id: str,
        crawl_type: str | None = None,
        host: str | None = None,
        top_k: int = 8,
        chunk_role: str | None = None,
        source_types: list[str] | None = None,
        entity_names: list[str] | None = None,
        categories: list[str] | None = None,
        min_quality: float | None = None,
        text_fields: tuple[str, ...] = ("childText", "text"),
    ) -> list[qm.ScoredPoint]:
        """Keyword/full-text style retrieval via payload TEXT indexes."""
        base = self.build_filter(
            run_id=run_id,
            crawl_type=crawl_type,
            host=host,
            chunk_role=chunk_role,
            source_types=source_types,
            entity_names=entity_names,
            categories=categories,
            min_quality=min_quality,
        )
        should = [
            qm.FieldCondition(key=field, match=qm.MatchText(text=text))
            for field in text_fields
        ]
        query_filter = qm.Filter(must=list(base.must or []), should=should)
        try:
            # Prefer scroll+filter when no sparse vector; use dummy dense if needed.
            # query_points without vector uses filter-only in recent clients via scroll.
            points, _ = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            scored: list[qm.ScoredPoint] = []
            for i, point in enumerate(points):
                scored.append(
                    qm.ScoredPoint(
                        id=point.id,
                        version=getattr(point, "version", 0) or 0,
                        score=float(top_k - i),
                        payload=point.payload,
                        vector=None,
                    )
                )
            return scored
        except Exception:
            logger.exception("Text search scroll failed for runId=%s", run_id)
            return []
