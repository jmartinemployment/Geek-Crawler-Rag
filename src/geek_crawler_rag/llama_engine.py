"""LlamaIndex engine: local dense embeddings + Qdrant vector store under FastAPI."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
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
    VectorStoreQueryResult,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.qdrant.utils import (
    fastembed_sparse_encoder,
    relative_score_fusion,
)
from qdrant_client import AsyncQdrantClient, QdrantClient

from geek_crawler_rag.config import Settings
from geek_crawler_rag.qdrant_store import SPARSE_VECTOR_NAME
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

# The runId of the hybrid query in flight, for the fusion log below. The fusion callback is called by
# QdrantVectorStore with the two result sets only, so the run reaches it through the async context.
_HYBRID_RUN_ID: ContextVar[str] = ContextVar("hybrid_run_id", default="")


def logged_relative_score_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,
    top_k: int = 2,
) -> VectorStoreQueryResult:
    """LlamaIndex's own `relative_score_fusion`, unchanged, with what it did to each half logged.

    A hybrid query runs a dense (meaning) search and a sparse BM25 (keyword) search and fuses them
    inside the vector store. If one half comes back empty the fusion returns the other half alone,
    and the answer looks the same as a healthy one: a silent drop to one signal. Logged before
    anything is ever built on top of it, so a failure can be understood from the log alone (Jeff,
    2026-10-06). Ranking is not touched; this only records what happened.

    What the fusion does, mechanically. Each half is min-max normalised over its own list, a node
    absent from a half scores 0 there, the two are summed at `alpha`, and the list is cut at
    `top_k` -- which `dense_query` leaves at the per-half fetch size, so up to half the union is
    dropped here, before `query.py` sees a candidate. Which half loses is decided by the shape of
    its score curve for that query, not by relevance: the steeper tail is cut harder. Measured on
    the live box on 2026-10-08, six partner runs, the bare keyword at topK 32: the halves shared
    2-15 of 64 chunks, the cut dropped 17-34 keyword-only and 15-44 meaning-only chunks per query,
    and the final 32 held between 4 (melio, a heavy-tailed keyword curve) and 21 (tipalti)
    keyword-only chunks -- while this line read `dense=64 sparse=64 fused=64` on every one of
    them. Three sizes cannot show that, so the line now reports the composition: how many of the
    kept are in both halves, in the meaning half only, in the keyword half only; how many of each
    half's own hits were dropped and the best rank among them; and the fused score of the last
    kept node.
    """
    dense_ids = _ranked_ids(dense_result)
    sparse_ids = _ranked_ids(sparse_result)
    dense_n = len(dense_ids)
    sparse_n = len(sparse_ids)
    fused = relative_score_fusion(dense_result, sparse_result, alpha=alpha, top_k=top_k)
    fused_ids = [n.node_id for n in (fused.nodes or [])]
    run_id = _HYBRID_RUN_ID.get()
    if dense_n == 0 or sparse_n == 0:
        logger.warning(
            "hybrid_half_empty runId=%s dense=%s sparse=%s fused=%s -- this answer used %s",
            run_id,
            dense_n,
            sparse_n,
            len(fused_ids),
            "neither half" if dense_n == sparse_n == 0
            else ("the keyword half only" if dense_n == 0 else "the meaning half only"),
        )
        return fused

    dense_set, sparse_set, fused_set = set(dense_ids), set(sparse_ids), set(fused_ids)
    both = dense_set & sparse_set
    kept_both = sum(1 for i in fused_ids if i in both)
    kept_dense_only = sum(1 for i in fused_ids if i in dense_set and i not in sparse_set)
    kept_sparse_only = sum(1 for i in fused_ids if i in sparse_set and i not in dense_set)
    dropped_dense = [r for r, i in enumerate(dense_ids, 1) if i not in sparse_set and i not in fused_set]
    dropped_sparse = [r for r, i in enumerate(sparse_ids, 1) if i not in dense_set and i not in fused_set]
    last_kept = (fused.similarities or [None])[-1]
    logger.info(
        "hybrid_halves runId=%s dense=%s sparse=%s both=%s union=%s cut=%s fused=%s "
        "kept=%s/%s/%s(both/denseOnly/sparseOnly) dropped=%s/%s(denseOnly/sparseOnly) "
        "firstDropped=%s/%s(denseRank/sparseRank) lastKept=%s",
        run_id,
        dense_n,
        sparse_n,
        len(both),
        len(dense_set | sparse_set),
        top_k,
        len(fused_ids),
        kept_both,
        kept_dense_only,
        kept_sparse_only,
        len(dropped_dense),
        len(dropped_sparse),
        dropped_dense[0] if dropped_dense else "-",
        dropped_sparse[0] if dropped_sparse else "-",
        "-" if last_kept is None else f"{float(last_kept):.3f}",
    )
    return fused


def _ranked_ids(result: VectorStoreQueryResult) -> list[str]:
    """Node ids of one half, best score first -- the rank each half gave its own hits."""
    nodes = list(result.nodes or [])
    sims = list(result.similarities or [])
    if len(sims) != len(nodes):
        return [n.node_id for n in nodes]
    ordered = sorted(zip(sims, nodes, strict=True), key=lambda t: float(t[0]), reverse=True)
    return [n.node_id for _, n in ordered]


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
        client_kwargs: dict[str, Any] = {
            "url": settings.qdrant_url,
            # Without this the client deadline is 5s, which a CPU-starved event loop misses.
            "timeout": settings.qdrant_timeout_seconds,
        }
        if settings.qdrant_api_key:
            client_kwargs["api_key"] = settings.qdrant_api_key
        self._client = QdrantClient(**client_kwargs)
        self._aclient = AsyncQdrantClient(**client_kwargs)
        # Both sparse encoders are passed explicitly, and that is load-bearing rather than tidy.
        # QdrantVectorStore only picks its own encoder when these are None, and its pick is decided
        # by use_old_sparse_encoder(), which returns True for a collection carrying a vector named
        # "text-sparse" -- ours -- and then encodes with naver/efficient-splade-VI-BT-large-doc.
        # The backfill writes prithivida/Splade_PP_en_v1 weights. Two SPLADE models index different
        # vocabularies, so that mismatch raises nothing: it scores query terms against an index
        # built from other terms and returns plausible, wrong passages. Passing both functions keeps
        # index time, query time and the migration on the one model in settings.sparse_model.
        try:
            sparse_encoder = fastembed_sparse_encoder(model_name=settings.sparse_model)
        except Exception:
            # Hybrid has no dense-only mode, so without this encoder the service cannot answer a
            # query. Named here so the log says which half failed, before startup stops.
            logger.exception(
                "Sparse (BM25 keyword) encoder failed to load model=%s vector=%s; hybrid retrieval "
                "cannot run and the service will not start",
                settings.sparse_model,
                SPARSE_VECTOR_NAME,
            )
            raise
        logger.info(
            "Hybrid retrieval: dense model=%s, sparse (BM25) model=%s on vector '%s'",
            settings.embedding_model,
            settings.sparse_model,
            SPARSE_VECTOR_NAME,
        )
        self._vector_store = QdrantVectorStore(
            client=self._client,
            aclient=self._aclient,
            collection_name=settings.qdrant_collection,
            batch_size=settings.embed_batch_size,
            enable_hybrid=True,
            sparse_vector_name=SPARSE_VECTOR_NAME,
            sparse_doc_fn=sparse_encoder,
            sparse_query_fn=sparse_encoder,
            hybrid_fusion_fn=logged_relative_score_fusion,
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

        texts = [n.get_content() for n in keep]

        # One embedding path, not two. Until 2026-10-01 there were two and they were mutually
        # exclusive: an in-flush dedupe ran `if len(uniq_texts) < len(texts)` and assigned
        # `embeddings` directly, so the `if embeddings is None` guard below skipped the persistent
        # cache entirely -- never reading it AND never writing it. A single repeated string anywhere
        # in a flush was enough to take that branch, so rag_vector_cache ended up holding 9,321 of
        # the corpus's 23,207 distinct texts: only the flushes that happened to be wholly unique
        # contributed. The weaker cache won whenever both applied.
        #
        # Deduplicating by text is now step one of the cached path rather than an alternative to it.
        embeddings = await self._embed_deduplicated(texts, keep)

        for node, emb in zip(keep, embeddings, strict=True):
            node.embedding = emb
        await self._vector_store.async_add(keep)
        return len(keep)

    def set_vector_cache(self, cache: Any) -> None:
        """Attach the cross-run cache. Separate from the constructor because the
        engine is built before Mongo is known to be reachable."""
        self._vector_cache = cache

    async def _embed_deduplicated(
        self, texts: list[str], keep: list[TextNode]
    ) -> list[list[float]]:
        """Embed each distinct string once, reusing anything embedded in any previous run.

        Three savings, in one pass:

        * **Within the flush.** A heading section shorter than ``child_chunk_size_tokens`` cannot be
          sliced, so ``parent_child_units`` emits a child byte-identical to its parent. Both points
          are written and both carry the same vector.
        * **Across the run.** Site chrome repeats on every page. The indexer now writes one point
          per distinct text per run (`indexer._admit_page_nodes`), so this case reaches here only
          on a retry whose repeat set lacked a text, and the cache covers it.
        * **Across runs.** A re-crawl produces chunks byte-identical to the last crawl's, and the
          point id carries the runId, so nothing else would collapse them.

        The first is free; the other two need ``rag_vector_cache``, which is why this must not be an
        either/or with it.

        Order is preserved by construction: vectors are recombined against the original ``texts``
        list, so a node can never receive another node's vector.
        """
        unique = list(dict.fromkeys(texts))
        meta_by_text: dict[str, dict[str, Any] | None] = {}
        for node, text in zip(keep, texts, strict=True):
            meta_by_text.setdefault(text, _node_meta(node))

        cached: dict[str, list[float]] = {}
        cache = self._vector_cache
        if cache is not None:
            cached = await cache.get_many(unique)

        missing = [t for t in unique if t not in cached]
        fresh: dict[str, list[float]] = {}
        if missing:
            vectors = await self.embed_texts(
                missing, metadata_list=[meta_by_text.get(t) for t in missing]
            )
            # embed_texts sanitizes and may drop empties, so its result is aligned to its filtered
            # input. On a length mismatch the mapping cannot be trusted, and pairing a vector with
            # the wrong node is silently wrong forever -- so embed positionally instead.
            if len(vectors) != len(missing):
                logger.warning(
                    "embed_dedupe_length_mismatch got=%s want=%s; embedding positionally",
                    len(vectors),
                    len(missing),
                )
                return await self.embed_texts(
                    texts, metadata_list=[_node_meta(n) for n in keep]
                )
            fresh = dict(zip(missing, vectors, strict=True))
            if cache is not None:
                await cache.put_many(fresh)

        logger.info(
            "embed_dedupe texts=%s unique=%s cache_hits=%s embedded=%s",
            len(texts),
            len(unique),
            len(cached),
            len(missing),
        )
        by_text = {**cached, **fresh}
        return [by_text[t] for t in texts]

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
        keyword: str | None = None,
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
        # `query_str` is what makes QdrantVectorStore run the keyword (BM25 sparse) half: its hybrid
        # branch is taken only when `mode == HYBRID and query_str is not None`; without the text it
        # falls through to a dense-only search and raises nothing. Absent until 2026-10-08, so every
        # query before that ran on the meaning half alone. `test_dense_query_is_hybrid.py` pins it.
        #
        # The two halves take different text when the caller says so: `need` is embedded for the
        # meaning half; `keyword`, when given, is the keyword half's text. A paragraph describing
        # the reader's situation is what the meaning half is good at, and on the keyword half every
        # common word in it is a term ("cost", "manual" matched every ERP page on 2026-10-08).
        _HYBRID_RUN_ID.set(run_id)
        keyword_text = (keyword or "").strip() or need
        result = await self._vector_store.aquery(
            VectorStoreQuery(
                query_str=keyword_text,
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                filters=filters,
                mode=VectorStoreQueryMode.HYBRID,
                sparse_top_k=top_k,
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
