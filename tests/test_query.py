"""Query hybrid selection via LlamaIndex dense + BM25."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from geek_crawler_rag.config import Settings
from geek_crawler_rag.models import QueryRequest
from geek_crawler_rag.query import QueryService, _return_text
from geek_crawler_rag.rerank import Reranker


def test_return_text_prefer_parent():
    payload = {"parentText": "PARENT", "childText": "CHILD", "text": "CHILD"}
    req = QueryRequest(need="n", runId="r", preferParent=True)
    assert _return_text(payload, req) == "PARENT"
    req2 = QueryRequest(need="n", runId="r", preferChild=True)
    assert _return_text(payload, req2) == "CHILD"
    req3 = QueryRequest(need="n", runId="r")
    assert _return_text(payload, req3) == "CHILD"


@pytest.mark.asyncio
async def test_query_hybrid_maps_hits():
    store = MagicMock()
    store.search_text = AsyncMock(return_value=[])

    node = TextNode(
        id_="p1",
        text="child text about pricing",
        metadata={
            "runId": "r1",
            "crawlType": "partner",
            "host": "acme.com",
            "url": "https://acme.com/pricing",
            "finalUrl": "https://acme.com/pricing",
            "title": "Pricing",
            "chunkIndex": 0,
            "language": "en",
            "parentText": "parent section about pricing plans",
            "childText": "child text about pricing",
            "chunkRole": "child",
            "entityName": "acme.com",
            "sourceType": "partner",
            "category": "pricing",
            "qualityScore": 0.8,
        },
    )
    llama = MagicMock()
    llama.dense_query = AsyncMock(return_value=[NodeWithScore(node=node, score=0.9)])

    settings = Settings(openai_api_key="test", hybrid_dense_limit=5, rerank_pool_size=5)
    svc = QueryService(store, settings, llama=llama, reranker=Reranker(None, enabled=False))

    resp = await svc.query(
        QueryRequest(need="pricing plans", runId="r1", topK=3, preferParent=True)
    )
    assert resp.warning is None
    assert len(resp.chunks) == 1
    assert resp.chunks[0].text == "parent section about pricing plans"
    assert resp.chunks[0].entity_name == "acme.com"
    assert resp.chunks[0].category == "pricing"
    assert resp.retrieval == "llamaindex-hybrid"


@pytest.mark.asyncio
async def test_query_graph_mode_returns_themes():
    store = MagicMock()
    store.search_text = AsyncMock(return_value=[])

    node = TextNode(
        id_="p1",
        text="child text about pricing",
        metadata={
            "runId": "r1",
            "crawlType": "partner",
            "host": "acme.com",
            "url": "https://acme.com/pricing",
            "finalUrl": "https://acme.com/pricing",
            "title": "Pricing",
            "chunkIndex": 0,
            "language": "en",
            "parentText": "parent section about pricing plans",
            "childText": "child text about pricing",
            "chunkRole": "child",
            "entityName": "Acme",
            "sourceType": "partner",
            "category": "pricing",
            "qualityScore": 0.8,
        },
    )
    llama = MagicMock()
    llama.dense_query = AsyncMock(return_value=[NodeWithScore(node=node, score=0.9)])

    settings = Settings(openai_api_key="test", hybrid_dense_limit=5, rerank_pool_size=5)
    svc = QueryService(store, settings, llama=llama, reranker=Reranker(None, enabled=False))

    resp = await svc.query(
        QueryRequest(need="pricing plans", runId="r1", topK=3, retrievalMode="graph")
    )
    assert resp.themes is not None
    assert len(resp.themes) >= 1
    assert resp.retrieval and resp.retrieval.startswith("graph+")
