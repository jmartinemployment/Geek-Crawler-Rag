"""LlamaIndex engine: OpenAI embeddings + Qdrant vector store under FastAPI."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from openai import RateLimitError
from qdrant_client import AsyncQdrantClient, QdrantClient

from geek_crawler_rag.config import Settings
from geek_crawler_rag.embedding_throttle import (
    EmbeddingRetryExhausted,
    EmbeddingThrottle,
    embedding_token_count,
    is_retryable_rate_limit,
    partition_embedding_batches,
    retry_after_seconds,
)

logger = logging.getLogger(__name__)


class LlamaIndexEngine:
    """Ingest/query via LlamaIndex; FastAPI keeps HTTP contracts."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for LlamaIndex embeddings")
        self._settings = settings
        self._embed_model = OpenAIEmbedding(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            dimensions=settings.embedding_dimensions,
            embed_batch_size=settings.embed_batch_size,
            max_retries=0,
        )
        self._embedding_throttle = EmbeddingThrottle(
            settings.openai_embedding_tokens_per_minute
        )
        self._embedding_call_lock = asyncio.Lock()
        client_kwargs: dict[str, Any] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            client_kwargs["api_key"] = settings.qdrant_api_key
        self._client = QdrantClient(**client_kwargs)
        self._aclient = AsyncQdrantClient(**client_kwargs)
        self._vector_store = QdrantVectorStore(
            client=self._client,
            aclient=self._aclient,
            collection_name=settings.qdrant_collection,
            batch_size=settings.embed_batch_size,
            enable_hybrid=False,
            text_key="text",
        )

    @property
    def vector_store(self) -> QdrantVectorStore:
        return self._vector_store

    @property
    def embed_model(self) -> OpenAIEmbedding:
        return self._embed_model

    def embedding_stats(self) -> dict[str, float | int]:
        return {
            "tokensInWindow": self._embedding_throttle.tokens_in_window,
            "rateLimitRetries": self._embedding_throttle.rate_limit_retries,
            "waitSeconds": round(self._embedding_throttle.total_wait_seconds, 3),
        }

    async def close(self) -> None:
        try:
            await self._aclient.close()
        except Exception:
            logger.debug("async qdrant client close failed", exc_info=True)
        try:
            self._client.close()
        except Exception:
            logger.debug("sync qdrant client close failed", exc_info=True)

    async def embed_and_upsert(self, nodes: list[TextNode]) -> int:
        if not nodes:
            return 0
        texts = [n.get_content() for n in nodes]
        embeddings = await self.embed_texts(texts)
        for node, emb in zip(nodes, embeddings, strict=True):
            node.embedding = emb
        await self._vector_store.async_add(nodes)
        return len(nodes)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        batches = partition_embedding_batches(
            texts,
            model=self._settings.openai_embedding_model,
            max_items=self._settings.embed_batch_size,
            max_tokens=self._settings.openai_embedding_max_batch_tokens,
        )
        for batch in batches:
            result = await self._call_with_retry(
                lambda: self._embed_model.aget_text_embedding_batch(batch.texts),
                token_count=batch.token_count,
            )
            embeddings.extend(result)
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        token_count = embedding_token_count(
            [text], self._settings.openai_embedding_model
        )
        return await self._call_with_retry(
            lambda: self._embed_model.aget_query_embedding(text),
            token_count=token_count,
        )

    async def _call_with_retry(self, operation: Any, *, token_count: int) -> Any:
        async with self._embedding_call_lock:
            max_retries = self._settings.openai_embedding_max_retries
            for attempt in range(max_retries + 1):
                await self._embedding_throttle.acquire(token_count)
                try:
                    return await operation()
                except asyncio.CancelledError:
                    raise
                except RateLimitError as exc:
                    if not is_retryable_rate_limit(exc):
                        raise
                    if attempt >= max_retries:
                        raise EmbeddingRetryExhausted(
                            f"OpenAI embedding rate limit persisted after "
                            f"{attempt + 1} attempts"
                        ) from exc
                    delay = retry_after_seconds(
                        exc,
                        attempt=attempt,
                        maximum=self._settings.openai_embedding_retry_max_seconds,
                    )
                    self._embedding_throttle.rate_limit_retries += 1
                    self._embedding_throttle.total_wait_seconds += delay
                    logger.warning(
                        "OpenAI embedding rate limited; retry=%s/%s wait=%.2fs",
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
        raise AssertionError("unreachable embedding retry loop")

    async def dense_query(
        self,
        need: str,
        *,
        run_id: str,
        top_k: int,
        crawl_type: str | None = None,
        host: str | None = None,
        chunk_role: str | None = None,
        source_types: list[str] | None = None,
        entity_names: list[str] | None = None,
        categories: list[str] | None = None,
        min_quality: float | None = None,
    ) -> list[NodeWithScore]:
        query_embedding = await self.embed_query(need)
        filters = build_metadata_filters(
            run_id=run_id,
            crawl_type=crawl_type,
            host=host,
            chunk_role=chunk_role,
            source_types=source_types,
            entity_names=entity_names,
            categories=categories,
            min_quality=min_quality,
        )
        result = await self._vector_store.aquery(
            VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                filters=filters,
                mode=VectorStoreQueryMode.DEFAULT,
            )
        )
        nodes = list(result.nodes or [])
        sims = list(result.similarities or [0.0] * len(nodes))
        return [NodeWithScore(node=n, score=s) for n, s in zip(nodes, sims, strict=False)]


def build_metadata_filters(
    *,
    run_id: str,
    crawl_type: str | None = None,
    host: str | None = None,
    chunk_role: str | None = None,
    source_types: list[str] | None = None,
    entity_names: list[str] | None = None,
    categories: list[str] | None = None,
    min_quality: float | None = None,
) -> MetadataFilters:
    filters: list[MetadataFilter] = [
        MetadataFilter(key="runId", value=run_id, operator=FilterOperator.EQ),
        MetadataFilter(key="language", value="en", operator=FilterOperator.EQ),
    ]
    if crawl_type:
        filters.append(
            MetadataFilter(key="crawlType", value=crawl_type, operator=FilterOperator.EQ)
        )
    if host:
        filters.append(
            MetadataFilter(
                key="host", value=host.lower().strip(), operator=FilterOperator.EQ
            )
        )
    if chunk_role:
        filters.append(
            MetadataFilter(key="chunkRole", value=chunk_role, operator=FilterOperator.EQ)
        )
    if source_types:
        filters.append(
            MetadataFilter(
                key="sourceType",
                value=[s.strip().lower() for s in source_types if s],
                operator=FilterOperator.IN,
            )
        )
    if entity_names:
        filters.append(
            MetadataFilter(
                key="entityName",
                value=[n.strip() for n in entity_names if n],
                operator=FilterOperator.IN,
            )
        )
    if categories:
        filters.append(
            MetadataFilter(
                key="category",
                value=[c.strip().lower() for c in categories if c],
                operator=FilterOperator.IN,
            )
        )
    if min_quality is not None:
        filters.append(
            MetadataFilter(
                key="qualityScore",
                value=float(min_quality),
                operator=FilterOperator.GTE,
            )
        )
    return MetadataFilters(filters=filters, condition=FilterCondition.AND)
