"""Query pipeline: embed need → filtered Qdrant search → chunk hits."""

from __future__ import annotations

import logging

from geek_crawler_rag.embed import Embedder
from geek_crawler_rag.extract import host_from_origin_or_url
from geek_crawler_rag.models import ChunkHit, QueryRequest, QueryResponse
from geek_crawler_rag.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class QueryService:
    def __init__(self, store: QdrantStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    async def query(self, request: QueryRequest) -> QueryResponse:
        host = None
        if request.host:
            host = host_from_origin_or_url(request.host, request.host) or request.host.lower().strip()

        try:
            vectors = await self._embedder.embed([request.need])
            hits = await self._store.search(
                vectors[0],
                run_id=request.run_id,
                crawl_type=request.crawl_type,
                host=host,
                top_k=request.top_k,
            )
        except Exception as ex:
            logger.exception("Query failed for runId=%s: %s", request.run_id, ex)
            return QueryResponse(
                run_id=request.run_id,
                chunks=[],
                warning=f"Query failed: {ex}",
            )

        chunks: list[ChunkHit] = []
        for hit in hits:
            payload = hit.payload or {}
            text = str(payload.get("text") or "")
            if not text:
                continue
            chunks.append(
                ChunkHit(
                    run_id=str(payload.get("runId") or request.run_id),
                    crawl_type=str(payload.get("crawlType") or ""),
                    host=str(payload.get("host") or ""),
                    url=str(payload.get("url") or ""),
                    final_url=str(payload.get("finalUrl") or payload.get("url") or ""),
                    title=payload.get("title"),
                    chunk_index=int(payload.get("chunkIndex") or 0),
                    language=str(payload.get("language") or "en"),
                    text=text,
                    score=float(hit.score or 0.0),
                )
            )

        warning = None
        if not chunks:
            warning = f"No chunks for runId={request.run_id}; notify-and-skip research"
            logger.warning(warning)

        return QueryResponse(run_id=request.run_id, chunks=chunks, warning=warning)
