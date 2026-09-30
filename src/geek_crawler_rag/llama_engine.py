"""LlamaIndex engine: local dense embeddings + Qdrant vector store under FastAPI."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

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
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.qdrant.utils import fastembed_sparse_encoder
from qdrant_client import AsyncQdrantClient, QdrantClient

from geek_crawler_rag.config import Settings
from geek_crawler_rag.qdrant_store import SPARSE_VECTOR_NAME, find_existing_point_ids
from geek_crawler_rag.embedding_circuit import (
    EmbeddingCircuitOpen,
    open_embedding_circuit,
    classify_embedding_failure,
)
from geek_crawler_rag.chunk_tokenizer import WordPieceChunkTokenizer
from geek_crawler_rag.local_embedding import LocalDenseEmbedding
from geek_crawler_rag.embedding_sanitize import (
    sanitize_embedding_text,
    sanitize_embedding_texts,
)
from geek_crawler_rag.embedding_batching import (
    EmbeddingBatch,
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

    def __init__(self, settings: Settings, *, vector_cache: Any = None) -> None:
        # Optional: set by app startup once Mongo is up. None means no
        # cross-run reuse, which is exactly the behaviour before it existed.
        self._vector_cache = vector_cache
        self._settings = settings
        # Local dense embeddings. No API key to require, no HTTP client, no rate limit: there is no
        # external service in the indexing path. Constructing this loads the ONNX model, so a missing
        # or unloadable model fails startup rather than the first page of the first index job.
        self._embed_model = LocalDenseEmbedding(
            model_name=settings.embedding_model,
            embed_batch_size=settings.embed_batch_size,
            threads=settings.embedding_threads,
        )
        if self._embed_model.dimensions != settings.embedding_dimensions:
            # Fail closed at startup rather than on every upsert. Qdrant rejects a vector whose width
            # does not match the collection's declared `size`, and the collection is created from
            # embedding_dimensions -- so a mismatch here means every write fails later, for a reason
            # that reads as a Qdrant problem.
            raise ValueError(
                f"embedding_dimensions={settings.embedding_dimensions} does not match "
                f"{settings.embedding_model}, which produces "
                f"{self._embed_model.dimensions}"
            )
        LlamaSettings.embed_model = self._embed_model
        # No throttle and no call lock. Both existed to pace a metered remote embedding API; there
        # is no meter to pace against now, and inference already runs in a worker thread.
        client_kwargs: dict[str, Any] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            client_kwargs["api_key"] = settings.qdrant_api_key
        self._client = QdrantClient(**client_kwargs)
        self._aclient = AsyncQdrantClient(**client_kwargs)
        self._hybrid_enabled = settings.hybrid_retrieval_enabled
        # Both sparse encoders are passed explicitly, and that is load-bearing rather than tidy.
        # QdrantVectorStore only picks its own encoder when these are None, and its pick is decided
        # by use_old_sparse_encoder(), which returns True for a collection carrying a vector named
        # "text-sparse" -- ours -- and then encodes with naver/efficient-splade-VI-BT-large-doc.
        # The backfill writes prithivida/Splade_PP_en_v1 weights. Two SPLADE models index different
        # vocabularies, so that mismatch raises nothing: it scores query terms against an index
        # built from other terms and returns plausible, wrong passages. Passing both functions keeps
        # index time, query time and the migration on the one model in settings.sparse_model.
        sparse_encoder = (
            fastembed_sparse_encoder(model_name=settings.sparse_model)
            if self._hybrid_enabled
            else None
        )
        self._vector_store = QdrantVectorStore(
            client=self._client,
            aclient=self._aclient,
            collection_name=settings.qdrant_collection,
            batch_size=settings.embed_batch_size,
            enable_hybrid=self._hybrid_enabled,
            sparse_vector_name=SPARSE_VECTOR_NAME,
            sparse_doc_fn=sparse_encoder,
            sparse_query_fn=sparse_encoder,
            text_key="text",
        )

    @property
    def vector_store(self) -> QdrantVectorStore:
        return self._vector_store

    @property
    def embed_model(self) -> LocalDenseEmbedding:
        return self._embed_model

    def embedding_stats(self) -> dict[str, float | int]:
        """Zeroed, and the keys kept deliberately.

        There is no throttle to report on: a local model has no tokens-per-minute ceiling to wait for
        and nothing to retry against a rate limit. But ``rateLimitRetries`` and ``waitSeconds`` reach
        the index-status webhook as ``embeddingRateLimitRetries`` and ``embeddingWaitSeconds``, which
        are pinned in ``contracts/rag-index-status/webhook.v1.json`` with a byte-matching GeekBackend
        copy that CI diffs. Dropping either from here reds that comparison and silently removes a
        bound field from the receiver.

        So they stay at zero until the coordinated two-repo change retires them --
        ``plans/go-local-embeddings.md``, "Follow-on". ``truncatedInputs`` is the one number here that
        now carries information: inputs that hit the model's sequence limit and were truncated.
        """
        return {
            "tokensInWindow": 0,
            "rateLimitRetries": 0,
            "waitSeconds": 0.0,
            "truncatedInputs": self._embed_model.truncated_inputs,
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

        # Embedding cache: a heading section shorter than child_chunk_size_tokens
        # cannot be sliced, so parent_child_units emits a child byte-identical to
        # its parent (chunk.py). Both points are still written, but the vector is
        # the same, so embed each distinct string once. Measured ~29% of calls on real
        # marketing pages -- and it still matters with a local model: the saving is CPU
        # rather than spend, and inference is now the dominant cost of indexing.
        uniq_texts = list(dict.fromkeys(texts))
        embeddings: list[list[float]] | None = None
        if len(uniq_texts) < len(texts):
            first_meta: dict[str, dict[str, Any] | None] = {}
            for node, text in zip(keep, texts, strict=True):
                first_meta.setdefault(text, _node_meta(node))
            uniq_vectors = await self.embed_texts(
                uniq_texts,
                metadata_list=[first_meta[t] for t in uniq_texts],
            )
            # embed_texts sanitizes and may drop empties, so its result is aligned
            # to its filtered input. On any length mismatch fall back rather than
            # mis-map vectors onto the wrong nodes.
            if len(uniq_vectors) == len(uniq_texts):
                by_text = dict(zip(uniq_texts, uniq_vectors, strict=True))
                embeddings = [by_text[t] for t in texts]
                logger.info(
                    "embed_cache_saved_calls=%s unique=%s of=%s",
                    len(texts) - len(uniq_texts),
                    len(uniq_texts),
                    len(texts),
                )
            else:
                logger.warning(
                    "embed_cache_length_mismatch got=%s want=%s; embedding uncached",
                    len(uniq_vectors),
                    len(uniq_texts),
                )

        if embeddings is None:
            # The branch is here rather than inside the helper so the no-cache
            # path is the same call it always was.
            if self._vector_cache is None:
                embeddings = await self.embed_texts(
                    texts, metadata_list=[_node_meta(n) for n in keep]
                )
            else:
                embeddings = await self._embed_with_cache(texts, keep)
        for node, emb in zip(keep, embeddings, strict=True):
            node.embedding = emb
        await self._vector_store.async_add(keep)
        return len(keep)

    def set_vector_cache(self, cache: Any) -> None:
        """Attach the cross-run cache. Separate from the constructor because the
        engine is built before Mongo is known to be reachable."""
        self._vector_cache = cache

    async def _embed_with_cache(
        self, texts: list[str], keep: list[TextNode]
    ) -> list[list[float]]:
        """Embed only what is not already known, from any previous run.

        The per-flush dedupe above collapses repeats inside one batch. This
        reaches across batches, pages and runs: a re-crawl of a site produces
        chunks byte-identical to the last crawl's, and those were embedded again
        every time because the point id carries the runId.

        Order is preserved by construction - cached and freshly embedded vectors
        are recombined against the original `texts` list, so a node never
        receives another node's vector.
        """
        cache = self._vector_cache
        cached = await cache.get_many(texts)
        missing = [t for t in dict.fromkeys(texts) if t not in cached]

        fresh: dict[str, list[float]] = {}
        if missing:
            meta_by_text: dict[str, dict[str, Any] | None] = {}
            for node, text in zip(keep, texts, strict=True):
                meta_by_text.setdefault(text, _node_meta(node))
            vectors = await self.embed_texts(
                missing, metadata_list=[meta_by_text.get(t) for t in missing]
            )
            # embed_texts sanitizes and may drop empties, so a length mismatch
            # means the mapping cannot be trusted. Fall back rather than risk
            # pairing a vector with the wrong node.
            if len(vectors) != len(missing):
                logger.warning(
                    "vector_cache_length_mismatch got=%s want=%s; embedding uncached",
                    len(vectors),
                    len(missing),
                )
                return await self.embed_texts(
                    texts, metadata_list=[_node_meta(n) for n in keep]
                )
            fresh = dict(zip(missing, vectors, strict=True))
            await cache.put_many(fresh)

        if cached:
            logger.info(
                "vector_cache_reused=%s embedded=%s of=%s",
                len(cached),
                len(missing),
                len(dict.fromkeys(texts)),
            )

        combined = {**cached, **fresh}
        return [combined[t] for t in texts]

    @property
    def chunk_tokenizer(self) -> WordPieceChunkTokenizer:
        """Satisfies ``asset_context.AssetEmbedder``; the chunker sizes with the embedder's own
        tokenizer so the two can never disagree about how long a chunk is."""
        return self._embed_model.chunk_tokenizer

    async def embed_texts(
        self,
        texts: list[str],
        *,
        metadata_list: list[dict[str, Any] | None] | None = None,
    ) -> list[list[float]]:
        cleaned, mutated = sanitize_embedding_texts(texts)
        if mutated:
            logger.info(
                "Sanitized %s/%s embedding texts before the embedding batch",
                mutated,
                len(cleaned),
            )
        # Defense in depth: never embed an empty string. fastembed returns a vector for "" without
        # raising, so an empty chunk would be stored as a real, retrievable point rather than
        # rejected -- this guard is the only thing that catches it.
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
            tokenizer=self._embed_model.chunk_tokenizer,
            max_items=self._settings.embed_batch_size,
            max_tokens=self._settings.embedding_max_batch_tokens,
        )
        offset = 0
        for batch in batches:
            batch_meta = None
            if metadata_list is not None:
                batch_meta = list(metadata_list[offset : offset + len(batch.texts)])
            offset += len(batch.texts)
            embeddings.extend(await self._embed_batch(batch, batch_meta))
        return embeddings

    async def _embed_batch(
        self, batch: EmbeddingBatch, batch_meta: list[Any] | None
    ) -> list[list[float]]:
        """One embed call. One attempt. It fails or it does not.

        There was a bounded retry here on transport and 5xx failures, removed 2026-09-28.
        Three things were wrong with it.

        It was a fallback, and its own docstring argued that it was not -- which is the
        tell. A retry that succeeds on its second attempt makes the first failure
        invisible; the run then looks clean and nothing says the network wobbled.

        Its justification was already obsolete when it was written. The premise was
        "'fix the root cause' presumes there is one", but the root cause of the
        embedding failures had been found on 2026-09-10 and fixed: the throttle was set
        to exactly the account's token ceiling, and lowering it to 40% produced a
        96-minute run with zero errors. The retry was added sixteen days later on the
        strength of a single APIConnectionError.

        And it cost money unauditably. A retry re-sent the same tokens, and the client could not
        tell a connection that dropped before the provider processed the batch from one that dropped
        after -- so a retry might pay twice for one batch with no way to detect which happened.

        That history is kept because the conclusion outlived its cause. Embeddings are local since
        2026-09-30: there is no rate limit, no billing, and no transient network failure to ride out,
        so the argument for a retry is weaker now than when it was rejected. A model that will not
        load is a startup failure, not something to retry per batch.

        What replaces it is what was always here: the failure is real, it is reported, and recovery is
        a deliberate re-post (`scripts/trigger_manual_index.py`, `scripts/list_unindexed_runs.py`).
        """
        try:
            return await self._embed_model.aget_text_embedding_batch(batch.texts)
        except EmbeddingCircuitOpen:
            raise
        except Exception as exc:
            # Unconditional. Every embed failure quarantines and fails the job closed -- there is no
            # status code to classify on and no "carry on" answer. See embedding_circuit.
            raise open_embedding_circuit(
                texts=batch.texts,
                metadata_list=batch_meta,
                quarantine_dir=self._settings.embedding_quarantine_dir,
                model=self._settings.embedding_model,
                token_count=batch.token_count,
                reason=classify_embedding_failure(exc),
                exc_type=type(exc).__name__,
                exc=exc,
            ) from exc

    async def embed_query(self, text: str) -> list[float]:
        text = sanitize_embedding_text(text)
        token_count = embedding_token_count(
            [text], tokenizer=self._embed_model.chunk_tokenizer
        )
        try:
            return await self._embed_model.aget_query_embedding(text)
        except EmbeddingCircuitOpen:
            raise
        except Exception as exc:
            raise open_embedding_circuit(
                texts=[text],
                quarantine_dir=self._settings.embedding_quarantine_dir,
                model=self._settings.embedding_model,
                token_count=token_count,
                reason=classify_embedding_failure(exc),
                exc_type=type(exc).__name__,
                exc=exc,
            ) from exc

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
        # HYBRID only when the sparse vector is actually populated. VectorStoreQueryMode.HYBRID
        # against a collection whose points carry no sparse values is not a no-op -- it is an error
        # the caller sees as an empty corpus.
        result = await self._vector_store.aquery(
            VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                filters=filters,
                mode=(
                    VectorStoreQueryMode.HYBRID
                    if self._hybrid_enabled
                    else VectorStoreQueryMode.DEFAULT
                ),
                sparse_top_k=top_k if self._hybrid_enabled else None,
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
