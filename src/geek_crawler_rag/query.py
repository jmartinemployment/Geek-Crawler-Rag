"""Query pipeline: LlamaIndex dense + BM25/text RRF + optional Cohere rerank."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from llama_index.core.schema import NodeWithScore

from geek_crawler_rag.bm25_rank import bm25_rank_indices
from geek_crawler_rag.config import Settings
from geek_crawler_rag.extract import host_from_origin_or_url
from geek_crawler_rag.graph_retrieve import (
    build_theme_hits,
    graph_warning_if_empty,
    prefer_parent_for_graph,
)
from geek_crawler_rag.models import ChunkHit, QueryRequest, QueryResponse
from geek_crawler_rag.qdrant_store import QdrantStore
from geek_crawler_rag.rerank import Reranker
from geek_crawler_rag.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class DenseRetriever(Protocol):
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
    ) -> list[NodeWithScore]: ...


class QueryService:
    def __init__(
        self,
        store: QdrantStore,
        settings: Settings,
        llama: DenseRetriever,
        reranker: Reranker | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._llama = llama
        self._reranker = reranker or Reranker(None, enabled=False)

    async def query(self, request: QueryRequest) -> QueryResponse:
        mode = (request.retrieval_mode or "hybrid").strip().lower()
        if mode == "graph":
            return await self._query_graph(request)
        return await self._query_hybrid(request)

    async def _query_graph(self, request: QueryRequest) -> QueryResponse:
        prefer_parent, prefer_child = prefer_parent_for_graph(
            request.prefer_parent, request.prefer_child
        )
        hybrid_req = request.model_copy(
            update={
                "prefer_parent": prefer_parent,
                "prefer_child": prefer_child,
                "retrieval_mode": "hybrid",
                "top_k": max(request.top_k, 12),
            }
        )
        hybrid = await self._query_hybrid(hybrid_req)
        themes = build_theme_hits(hybrid.chunks, max_themes=12)
        warnings: list[str] = []
        if hybrid.warning:
            warnings.append(hybrid.warning)
        empty_warn = graph_warning_if_empty(themes, hybrid.chunks)
        if empty_warn:
            warnings.append(empty_warn)

        retrieval = "graph+llamaindex-hybrid"
        if hybrid.retrieval:
            retrieval = f"graph+{hybrid.retrieval}"

        return QueryResponse(
            run_id=request.run_id,
            chunks=hybrid.chunks[: request.top_k],
            themes=themes or None,
            warning="; ".join(warnings) if warnings else None,
            retrieval=retrieval,
        )

    async def _query_hybrid(self, request: QueryRequest) -> QueryResponse:
        host = None
        if request.host:
            host = host_from_origin_or_url(request.host, request.host) or request.host.lower().strip()

        chunk_role = (request.chunk_role or "").strip().lower() or None
        search_role = chunk_role
        if search_role is None and (request.prefer_parent or request.prefer_child):
            search_role = "child"

        try:
            dense_limit = max(request.top_k, self._settings.hybrid_dense_limit)
            dense_nodes = await self._llama.dense_query(
                request.need,
                run_id=request.run_id,
                top_k=dense_limit,
                crawl_type=request.crawl_type,
                host=host,
                chunk_role=search_role,
                source_types=request.source_types,
                entity_names=request.entity_names,
                categories=request.categories,
                min_quality=request.min_quality,
            )
            lexical_hits = await self._store.search_text(
                request.need,
                run_id=request.run_id,
                crawl_type=request.crawl_type,
                host=host,
                top_k=self._settings.hybrid_lexical_limit,
                chunk_role=search_role,
                source_types=request.source_types,
                entity_names=request.entity_names,
                categories=request.categories,
                min_quality=request.min_quality,
            )
        except Exception as ex:
            logger.exception("Query failed for runId=%s: %s", request.run_id, ex)
            return QueryResponse(
                run_id=request.run_id,
                chunks=[],
                warning=f"Query failed: {ex}",
                retrieval="error",
            )

        candidates = _merge_candidates(dense_nodes, lexical_hits)
        if not candidates:
            warning = f"No chunks for runId={request.run_id}; notify-and-skip research"
            logger.warning(warning)
            return QueryResponse(
                run_id=request.run_id,
                chunks=[],
                warning=warning,
                retrieval="empty",
            )

        dense_ids = [_node_id(n) for n in dense_nodes]
        docs_for_bm25 = [_lexical_doc(c["payload"]) for c in candidates]
        bm25_order = bm25_rank_indices(request.need, docs_for_bm25)
        bm25_ids = [candidates[i]["id"] for i in bm25_order]
        lexical_ids = [str(h.id) for h in lexical_hits]

        fused = reciprocal_rank_fusion([dense_ids, bm25_ids, lexical_ids])
        id_to_cand = {c["id"]: c for c in candidates}
        fused_candidates = [id_to_cand[i] for i, _ in fused if i in id_to_cand]

        pool_n = min(len(fused_candidates), max(request.top_k, self._settings.rerank_pool_size))
        pool = fused_candidates[:pool_n]
        rerank_docs = [_return_text(c["payload"], request) for c in pool]
        ranked = await self._reranker.rerank(
            request.need, rerank_docs, top_n=min(request.top_k, len(pool))
        )

        chunks: list[ChunkHit] = []
        for orig_idx, rerank_score in ranked:
            cand = pool[orig_idx]
            payload = cand["payload"]
            text = _return_text(payload, request)
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
                    score=float(rerank_score),
                    entity_name=payload.get("entityName"),
                    entity_id=str(payload["entityId"])
                    if payload.get("entityId") is not None
                    else None,
                    source_type=payload.get("sourceType"),
                    category=payload.get("category"),
                    content_intent=payload.get("contentIntent"),
                    chunk_role=payload.get("chunkRole"),
                    section_title=payload.get("sectionTitle"),
                    quality_score=_as_float(payload.get("qualityScore")),
                    dense_score=_as_float(cand.get("dense_score")),
                    rerank_score=float(rerank_score) if self._reranker.enabled else None,
                )
            )

        retrieval = "llamaindex-hybrid+rerank" if self._reranker.enabled else "llamaindex-hybrid"
        warning = None
        if not chunks:
            warning = f"No chunks for runId={request.run_id}; notify-and-skip research"
            logger.warning(warning)

        return QueryResponse(
            run_id=request.run_id,
            chunks=chunks,
            warning=warning,
            retrieval=retrieval,
        )


def _node_id(node: NodeWithScore) -> str:
    return str(node.node.node_id)


def _merge_candidates(
    dense_nodes: list[NodeWithScore], lexical_hits: list[Any]
) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for hit in dense_nodes:
        pid = _node_id(hit)
        meta = dict(hit.node.metadata or {})
        # Ensure embed text available for prefer* fallbacks.
        if "text" not in meta:
            meta["text"] = hit.node.get_content()
        out[pid] = {
            "id": pid,
            "payload": meta,
            "dense_score": float(hit.score or 0.0),
        }
    for hit in lexical_hits:
        pid = str(hit.id)
        if pid in out:
            continue
        out[pid] = {
            "id": pid,
            "payload": hit.payload or {},
            "dense_score": None,
        }
    return list(out.values())


def _lexical_doc(payload: dict[str, Any]) -> str:
    parts = [
        str(payload.get("childText") or ""),
        str(payload.get("parentText") or ""),
        str(payload.get("text") or ""),
        str(payload.get("title") or ""),
        str(payload.get("sectionTitle") or ""),
    ]
    return "\n".join(p for p in parts if p)


def _return_text(payload: dict[str, Any], request: QueryRequest) -> str:
    parent = str(payload.get("parentText") or "")
    child = str(payload.get("childText") or "")
    plain = str(payload.get("text") or "")

    prefer_parent = bool(request.prefer_parent)
    prefer_child = bool(request.prefer_child)
    if prefer_parent and not prefer_child:
        return parent or plain or child
    if prefer_child and not prefer_parent:
        return child or plain or parent
    return plain or child or parent


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
