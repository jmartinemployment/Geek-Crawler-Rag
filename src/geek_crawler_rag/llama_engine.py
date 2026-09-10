"""LlamaIndex engine: OpenAI embeddings + Qdrant vector store under FastAPI."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from llama_index.core import Settings as LlamaSettings
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
from openai import InternalServerError
from qdrant_client import AsyncQdrantClient, QdrantClient

from geek_crawler_rag.config import Settings
from geek_crawler_rag.qdrant_store import find_existing_point_ids
from geek_crawler_rag.embedding_circuit import (
    EmbeddingCircuitOpen,
    open_embedding_circuit,
    should_quarantine_embedding_error,
)
from geek_crawler_rag.embedding_sanitize import (
    sanitize_embedding_text,
    sanitize_embedding_texts,
)
from geek_crawler_rag.embedding_throttle import (
    EmbeddingThrottle,
    embedding_token_count,
    partition_embedding_batches,
)

logger = logging.getLogger(__name__)


def _node_meta(node: TextNode) -> dict[str, Any]:
    meta = dict(node.metadata or {})
    return {
        "runId": meta.get("runId"),
        "pageId": meta.get("pageId"),
        "chunkId": meta.get("chunkId"),
        "chunkRole": meta.get("chunkRole"),
        "host": meta.get("host"),
        "crawlType": meta.get("crawlType"),
    }


class LlamaIndexEngine:
    """Ingest/query via LlamaIndex; FastAPI keeps HTTP contracts."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for LlamaIndex embeddings")
        self._settings = settings
        # Native LlamaIndex + OpenAI HTTP client: retries off at the source.
        # https://developers.llamaindex.ai/python/framework/module_guides/models/embeddings/
        embed_timeout = httpx.Timeout(60.0, connect=10.0)
        self._embed_http_client = httpx.Client(timeout=embed_timeout)
        self._embed_async_http_client = httpx.AsyncClient(timeout=embed_timeout)
        self._embed_model = OpenAIEmbedding(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            dimensions=settings.embedding_dimensions,
            embed_batch_size=settings.embed_batch_size,
            max_retries=settings.openai_embedding_max_retries,
            timeout=60.0,
            http_client=self._embed_http_client,
            async_http_client=self._embed_async_http_client,
        )
        LlamaSettings.embed_model = self._embed_model
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
        try:
            await self._embed_async_http_client.aclose()
        except Exception:
            logger.debug("async embed httpx client close failed", exc_info=True)
        try:
            self._embed_http_client.close()
        except Exception:
            logger.debug("sync embed httpx client close failed", exc_info=True)

    async def embed_and_upsert(self, nodes: list[TextNode]) -> int:
        if not nodes:
            return 0
        keep: list[TextNode] = []
        skipped_empty = 0
        for node in nodes:
            raw = node.get_content()
            safe = sanitize_embedding_text(raw)
            if safe != raw:
                node.set_content(safe)
            if not safe.strip():
                skipped_empty += 1
                continue
            keep.append(node)
        if skipped_empty:
            logger.info(
                "skipped_empty_embed_texts=%s kept=%s",
                skipped_empty,
                len(keep),
            )
        if not keep:
            return 0

        # Resume support: point IDs are deterministic (qdrant_store.point_id)
        # and already assigned as node.id_ before embedding, so a re-run can
        # skip chunks it already committed. Without this a retried job
        # re-embeds every chunk from zero, which is why large runs never
        # converge past a transient upstream failure.
        already = await find_existing_point_ids(
            self._aclient,
            self._settings.qdrant_collection,
            [str(n.id_) for n in keep],
        )
        if already:
            before = len(keep)
            keep = [n for n in keep if str(n.id_) not in already]
            logger.info(
                "resume_skipped_existing_points=%s remaining=%s of=%s",
                before - len(keep),
                len(keep),
                before,
            )
            if not keep:
                return 0

        texts = [n.get_content() for n in keep]
        embeddings = await self.embed_texts(
            texts,
            metadata_list=[_node_meta(n) for n in keep],
        )
        for node, emb in zip(keep, embeddings, strict=True):
            node.embedding = emb
        await self._vector_store.async_add(keep)
        return len(keep)

    async def embed_texts(
        self,
        texts: list[str],
        *,
        metadata_list: list[dict[str, Any] | None] | None = None,
    ) -> list[list[float]]:
        cleaned, mutated = sanitize_embedding_texts(texts)
        if mutated:
            logger.info(
                "Sanitized %s/%s embedding texts before OpenAI batch",
                mutated,
                len(cleaned),
            )
        # Defense in depth: never send empty strings (OpenAI 400).
        if metadata_list is not None and len(metadata_list) != len(cleaned):
            metadata_list = list(metadata_list)[: len(cleaned)]
        filtered: list[str] = []
        filtered_meta: list[dict[str, Any] | None] | None = (
            [] if metadata_list is not None else None
        )
        dropped = 0
        for i, text in enumerate(cleaned):
            if not text.strip():
                dropped += 1
                continue
            filtered.append(text)
            if filtered_meta is not None and metadata_list is not None:
                filtered_meta.append(
                    metadata_list[i] if i < len(metadata_list) else None
                )
        if dropped:
            logger.info(
                "skipped_empty_embed_texts=%s kept=%s (embed_texts)",
                dropped,
                len(filtered),
            )
        if not filtered:
            return []
        cleaned = filtered
        metadata_list = filtered_meta
        embeddings: list[list[float]] = []
        batches = partition_embedding_batches(
            cleaned,
            model=self._settings.openai_embedding_model,
            max_items=self._settings.embed_batch_size,
            max_tokens=self._settings.openai_embedding_max_batch_tokens,
        )
        offset = 0
        for batch in batches:
            batch_meta = None
            if metadata_list is not None:
                batch_meta = list(metadata_list[offset : offset + len(batch.texts)])
            offset += len(batch.texts)
            async with self._embedding_call_lock:
                await self._embedding_throttle.acquire(batch.token_count)
                try:
                    result = await self._embed_model.aget_text_embedding_batch(
                        batch.texts
                    )
                except EmbeddingCircuitOpen:
                    raise
                except Exception as exc:
                    code = should_quarantine_embedding_error(exc)
                    if code is None and isinstance(exc, InternalServerError):
                        code = int(getattr(exc, "status_code", 0) or 500)
                    if code is not None:
                        raise open_embedding_circuit(
                            texts=batch.texts,
                            metadata_list=batch_meta,
                            quarantine_dir=self._settings.embedding_quarantine_dir,
                            model=self._settings.openai_embedding_model,
                            token_count=batch.token_count,
                            status_code=code,
                            exc_type=type(exc).__name__,
                            exc=exc,
                        ) from exc
                    raise
            embeddings.extend(result)
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        text = sanitize_embedding_text(text)
        token_count = embedding_token_count(
            [text], self._settings.openai_embedding_model
        )
        async with self._embedding_call_lock:
            await self._embedding_throttle.acquire(token_count)
            try:
                return await self._embed_model.aget_query_embedding(text)
            except EmbeddingCircuitOpen:
                raise
            except Exception as exc:
                code = should_quarantine_embedding_error(exc)
                if code is None and isinstance(exc, InternalServerError):
                    code = int(getattr(exc, "status_code", 0) or 500)
                if code is not None:
                    raise open_embedding_circuit(
                        texts=[text],
                        quarantine_dir=self._settings.embedding_quarantine_dir,
                        model=self._settings.openai_embedding_model,
                        token_count=token_count,
                        status_code=code,
                        exc_type=type(exc).__name__,
                        exc=exc,
                    ) from exc
                raise

    async def dense_query(
        self,
        need: str,
        *,
        run_id: str,
        top_k: int,
        owner_id: str = "system:crawler",
        visibility: str = "service",
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
            owner_id=owner_id,
            visibility=visibility,
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
        return [
            NodeWithScore(node=n, score=s) for n, s in zip(nodes, sims, strict=False)
        ]


def build_metadata_filters(
    *,
    run_id: str,
    owner_id: str = "system:crawler",
    visibility: str = "service",
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
        MetadataFilter(key="ownerId", value=owner_id, operator=FilterOperator.EQ),
        MetadataFilter(key="visibility", value=visibility, operator=FilterOperator.EQ),
        MetadataFilter(key="language", value="en", operator=FilterOperator.EQ),
    ]
    if crawl_type:
        filters.append(
            MetadataFilter(
                key="crawlType", value=crawl_type, operator=FilterOperator.EQ
            )
        )
    if host:
        filters.append(
            MetadataFilter(
                key="host", value=host.lower().strip(), operator=FilterOperator.EQ
            )
        )
    if chunk_role:
        filters.append(
            MetadataFilter(
                key="chunkRole", value=chunk_role, operator=FilterOperator.EQ
            )
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
