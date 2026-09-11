"""Query hybrid selection via LlamaIndex dense + BM25."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from geek_crawler_rag.config import Settings
from geek_crawler_rag.models import QueryRequest
from geek_crawler_rag.query import (
    QueryService,
    _parent_lineage_key,
    _return_text,
    _select_ranked_candidates,
    _should_collapse_parents,
)
from geek_crawler_rag.rerank import Reranker


def _child_node(
    *,
    node_id: str,
    parent_text: str,
    child_text: str,
    page_id: str = "page-1",
    section_title: str = "Pricing",
) -> TextNode:
    return TextNode(
        id_=node_id,
        text=child_text,
        metadata={
            "runId": "r1",
            "crawlType": "partner",
            "host": "acme.com",
            "url": f"https://acme.com/{node_id}",
            "finalUrl": f"https://acme.com/{node_id}",
            "title": "Pricing",
            "chunkIndex": 0,
            "language": "en",
            "pageId": page_id,
            "sectionTitle": section_title,
            "parentText": parent_text,
            "childText": child_text,
            "chunkRole": "child",
            "entityName": "acme.com",
            "sourceType": "partner",
            "category": "pricing",
            "qualityScore": 0.8,
        },
    )


def test_return_text_prefer_parent():
    payload = {"parentText": "PARENT", "childText": "CHILD", "text": "CHILD"}
    req = QueryRequest(need="n", runId="r", preferParent=True)
    assert _return_text(payload, req) == "PARENT"
    req2 = QueryRequest(need="n", runId="r", preferChild=True)
    assert _return_text(payload, req2) == "CHILD"
    req3 = QueryRequest(need="n", runId="r")
    assert _return_text(payload, req3) == "CHILD"


def test_parent_lineage_key_stable_for_siblings():
    payload_a = {
        "pageId": "p1",
        "sectionTitle": "Intro",
        "parentText": "same parent block",
    }
    payload_b = {
        "pageId": "p1",
        "sectionTitle": "Intro",
        "parentText": "same parent block",
    }
    payload_c = {
        "pageId": "p1",
        "sectionTitle": "Intro",
        "parentText": "different parent block",
    }
    assert _parent_lineage_key(payload_a) == _parent_lineage_key(payload_b)
    assert _parent_lineage_key(payload_a) != _parent_lineage_key(payload_c)


def test_select_ranked_candidates_collapses_and_backfills():
    pool = [
        {"id": "a1", "payload": {"pageId": "p", "sectionTitle": "A", "parentText": "PA"}},
        {"id": "a2", "payload": {"pageId": "p", "sectionTitle": "A", "parentText": "PA"}},
        {"id": "b1", "payload": {"pageId": "p", "sectionTitle": "B", "parentText": "PB"}},
        {"id": "c1", "payload": {"pageId": "p", "sectionTitle": "C", "parentText": "PC"}},
        {"id": "c2", "payload": {"pageId": "p", "sectionTitle": "C", "parentText": "PC"}},
        {"id": "d1", "payload": {"pageId": "p", "sectionTitle": "D", "parentText": "PD"}},
    ]
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5), (5, 0.4)]
    selected = _select_ranked_candidates(
        pool,
        ranked,
        request=QueryRequest(need="n", runId="r"),
        target_top_k=4,
        collapse_parents=True,
    )
    assert [c["id"] for c, _ in selected] == ["a1", "b1", "c1", "d1"]


def test_select_ranked_candidates_insufficient_unique_parents():
    pool = [
        {"id": "a1", "payload": {"pageId": "p", "sectionTitle": "A", "parentText": "PA"}},
        {"id": "a2", "payload": {"pageId": "p", "sectionTitle": "A", "parentText": "PA"}},
    ]
    ranked = [(0, 0.9), (1, 0.8)]
    selected = _select_ranked_candidates(
        pool,
        ranked,
        request=QueryRequest(need="n", runId="r"),
        target_top_k=3,
        collapse_parents=True,
    )
    assert len(selected) == 1
    assert selected[0][0]["id"] == "a1"


def test_should_collapse_parents_only_for_prefer_parent():
    assert _should_collapse_parents(
        QueryRequest(need="n", runId="r", preferParent=True)
    )
    assert not _should_collapse_parents(
        QueryRequest(need="n", runId="r", preferChild=True)
    )
    assert not _should_collapse_parents(QueryRequest(need="n", runId="r"))


@pytest.mark.asyncio
async def test_query_hybrid_maps_hits():
    store = MagicMock()
    store.search_text = AsyncMock(return_value=[])

    node = _child_node(
        node_id="p1",
        parent_text="parent section about pricing plans",
        child_text="child text about pricing",
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
async def test_query_hybrid_collapses_sibling_parents_and_backfills():
    store = MagicMock()
    store.search_text = AsyncMock(return_value=[])

    nodes = [
        _child_node(
            node_id="a1",
            parent_text="PARENT A BLOCK",
            child_text="child a1",
            section_title="A",
        ),
        _child_node(
            node_id="a2",
            parent_text="PARENT A BLOCK",
            child_text="child a2",
            section_title="A",
        ),
        _child_node(
            node_id="b1",
            parent_text="PARENT B BLOCK",
            child_text="child b1",
            section_title="B",
        ),
        _child_node(
            node_id="c1",
            parent_text="PARENT C BLOCK",
            child_text="child c1",
            section_title="C",
        ),
        _child_node(
            node_id="c2",
            parent_text="PARENT C BLOCK",
            child_text="child c2",
            section_title="C",
        ),
        _child_node(
            node_id="d1",
            parent_text="PARENT D BLOCK",
            child_text="child d1",
            section_title="D",
        ),
    ]
    llama = MagicMock()
    llama.dense_query = AsyncMock(
        return_value=[
            NodeWithScore(node=n, score=0.9 - i * 0.01) for i, n in enumerate(nodes)
        ]
    )

    reranker = MagicMock()
    reranker.enabled = True
    reranker.rerank = AsyncMock(
        side_effect=lambda _need, documents, top_n: [
            (i, float(len(documents) - i)) for i in range(top_n)
        ]
    )

    settings = Settings(openai_api_key="test", hybrid_dense_limit=10, rerank_pool_size=10)
    svc = QueryService(store, settings, llama=llama, reranker=reranker)

    resp = await svc.query(
        QueryRequest(need="pricing", runId="r1", topK=4, preferParent=True)
    )
    assert [c.text for c in resp.chunks] == [
        "PARENT A BLOCK",
        "PARENT B BLOCK",
        "PARENT C BLOCK",
        "PARENT D BLOCK",
    ]
    reranker.rerank.assert_awaited_once()
    assert reranker.rerank.await_args.kwargs["top_n"] == 6
    docs = reranker.rerank.await_args.args[1]
    assert docs == [
        "child a1",
        "child a2",
        "child b1",
        "child c1",
        "child c2",
        "child d1",
    ]


@pytest.mark.asyncio
async def test_query_hybrid_prefer_child_keeps_sibling_children():
    store = MagicMock()
    store.search_text = AsyncMock(return_value=[])

    nodes = [
        _child_node(
            node_id="a1",
            parent_text="PARENT A BLOCK",
            child_text="child a1 unique",
            section_title="A",
        ),
        _child_node(
            node_id="a2",
            parent_text="PARENT A BLOCK",
            child_text="child a2 unique",
            section_title="A",
        ),
    ]
    llama = MagicMock()
    llama.dense_query = AsyncMock(
        return_value=[
            NodeWithScore(node=n, score=0.9 - i * 0.01) for i, n in enumerate(nodes)
        ]
    )

    settings = Settings(openai_api_key="test", hybrid_dense_limit=5, rerank_pool_size=5)
    svc = QueryService(store, settings, llama=llama, reranker=Reranker(None, enabled=False))

    resp = await svc.query(
        QueryRequest(need="pricing", runId="r1", topK=2, preferChild=True)
    )
    assert [c.text for c in resp.chunks] == ["child a1 unique", "child a2 unique"]


@pytest.mark.asyncio
async def test_query_hybrid_reranks_full_pool_even_when_topk_smaller():
    store = MagicMock()
    store.search_text = AsyncMock(return_value=[])

    nodes = [
        _child_node(
            node_id=f"n{i}",
            parent_text=f"PARENT {i}",
            child_text=f"child {i}",
            section_title=f"S{i}",
        )
        for i in range(5)
    ]
    llama = MagicMock()
    llama.dense_query = AsyncMock(
        return_value=[
            NodeWithScore(node=n, score=0.9 - i * 0.01) for i, n in enumerate(nodes)
        ]
    )

    reranker = MagicMock()
    reranker.enabled = True
    reranker.rerank = AsyncMock(
        side_effect=lambda _need, documents, top_n: [
            (i, float(len(documents) - i)) for i in range(top_n)
        ]
    )

    settings = Settings(openai_api_key="test", hybrid_dense_limit=10, rerank_pool_size=10)
    svc = QueryService(store, settings, llama=llama, reranker=reranker)

    resp = await svc.query(
        QueryRequest(need="pricing", runId="r1", topK=2, preferParent=True)
    )
    assert len(resp.chunks) == 2
    assert reranker.rerank.await_args.kwargs["top_n"] == 5


@pytest.mark.asyncio
async def test_query_graph_mode_returns_themes():
    store = MagicMock()
    store.search_text = AsyncMock(return_value=[])

    node = _child_node(
        node_id="p1",
        parent_text="parent section about pricing plans",
        child_text="child text about pricing",
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


def test_select_ranked_candidates_drops_duplicate_text_without_collapse():
    """Identical text must be dropped even when collapse_parents is off."""
    pool = [
        {"id": "child", "payload": {"pageId": "p", "sectionTitle": "A",
                                    "parentText": "SAME", "childText": "SAME",
                                    "text": "SAME"}},
        {"id": "parent", "payload": {"pageId": "p", "sectionTitle": "A",
                                     "parentText": "SAME", "childText": "",
                                     "text": "SAME"}},
        {"id": "other", "payload": {"pageId": "p", "sectionTitle": "B",
                                    "parentText": "DIFF", "childText": "DIFF",
                                    "text": "DIFF"}},
    ]
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7)]
    selected = _select_ranked_candidates(
        pool,
        ranked,
        request=QueryRequest(need="n", runId="r"),
        target_top_k=2,
        collapse_parents=False,
    )
    # duplicate "SAME" collapses; freed slot is filled by the distinct candidate
    assert [c["id"] for c, _ in selected] == ["child", "other"]


def test_select_ranked_candidates_fills_top_k_with_distinct_text():
    """Starvation regression: dedup must not shrink the result below top_k."""
    pool = []
    ranked = []
    # 8 distinct texts, each duplicated once -> 16 candidates, 8 unique
    for i in range(8):
        for role in ("child", "parent"):
            pool.append({
                "id": f"{role}{i}",
                "payload": {"pageId": "p", "sectionTitle": f"S{i}",
                            "parentText": f"T{i}", "childText": f"T{i}",
                            "text": f"T{i}"},
            })
    ranked = [(i, 1.0 - i * 0.01) for i in range(len(pool))]
    selected = _select_ranked_candidates(
        pool,
        ranked,
        request=QueryRequest(need="n", runId="r"),
        target_top_k=8,
        collapse_parents=False,
    )
    texts = [_return_text(c["payload"], QueryRequest(need="n", runId="r"))
             for c, _ in selected]
    assert len(selected) == 8, "dedup starved the result below top_k"
    assert len(set(texts)) == 8, "returned duplicate text"
