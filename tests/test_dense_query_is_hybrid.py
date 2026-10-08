"""`dense_query` must make LlamaIndex run the keyword half, not merely ask for it.

`QdrantVectorStore.aquery` takes its hybrid branch only when `query.mode == HYBRID` *and*
`query.query_str is not None`. Without the text it falls through to a dense-only search and
raises nothing. From 2026-09-24 (`448b25c`) to 2026-10-08 `dense_query` passed `mode=HYBRID` and
no `query_str`, so every production query ran on the meaning half alone while the response said
`retrieval="llamaindex-hybrid"`. `test_hybrid_sparse_retrieval.py` did not catch it because it
calls the store directly with `query_str="Dext"`; it proves LlamaIndex, not our call.

Both tests here go through `dense_query`, the function production calls. The first pins the
argument; the second drives a real in-process store and asserts the fusion actually ran, which is
the one observable that was missing from the live log.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode, VectorStoreQueryResult
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient

from geek_crawler_rag.config import Settings
from geek_crawler_rag.llama_engine import LlamaIndexEngine, logged_rank_fusion
from geek_crawler_rag.qdrant_store import SPARSE_VECTOR_NAME


def _sparse_by_token(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    """One sparse index per distinct lowercased word: identical terms collide, others do not."""
    indices: list[list[int]] = []
    values: list[list[float]] = []
    for text in texts:
        tokens = sorted({abs(hash(word.lower())) % 100_000 for word in text.split()})
        indices.append(tokens)
        values.append([1.0] * len(tokens))
    return indices, values


@pytest.fixture
def engine(monkeypatch) -> LlamaIndexEngine:
    """The real constructor with the network clients replaced, as test_engine_constructs does."""
    monkeypatch.setattr("geek_crawler_rag.llama_engine.QdrantClient", MagicMock())
    monkeypatch.setattr("geek_crawler_rag.llama_engine.AsyncQdrantClient", MagicMock())
    monkeypatch.setattr("geek_crawler_rag.llama_engine.QdrantVectorStore", MagicMock())
    return LlamaIndexEngine(Settings(qdrant_url="http://localhost:6333", api_key="x"))


async def test_dense_query_passes_the_question_text_so_the_keyword_half_runs(engine) -> None:
    """The argument that was missing for two weeks."""
    engine.embed_query = AsyncMock(return_value=[0.0, 0.0, 0.0, 1.0])
    engine._vector_store.aquery = AsyncMock(
        return_value=VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    )

    await engine.dense_query("Dext receipt capture", run_id="run-1", top_k=5)

    query = engine._vector_store.aquery.await_args.args[0]
    assert query.query_str == "Dext receipt capture"
    assert query.mode == VectorStoreQueryMode.HYBRID
    assert query.sparse_top_k == 5
    assert query.similarity_top_k == 5


async def test_the_keyword_half_gets_the_keyword_and_the_meaning_half_gets_the_need(engine) -> None:
    """plans/retrieval-from-the-brief.md P1: a paragraph for the meaning half, a few terms for the
    keyword half. Without `keyword`, both halves get `need`, as before."""
    engine.embed_query = AsyncMock(return_value=[0.0, 0.0, 0.0, 1.0])
    engine._vector_store.aquery = AsyncMock(
        return_value=VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    )

    await engine.dense_query(
        "The company cannot reconcile global payments to the right entity",
        run_id="run-1",
        top_k=8,
        keyword="payment reconciliation multi-entity",
    )
    query = engine._vector_store.aquery.await_args.args[0]
    assert query.query_str == "payment reconciliation multi-entity"
    engine.embed_query.assert_awaited_with(
        "The company cannot reconcile global payments to the right entity"
    )

    await engine.dense_query("Automated Payment Execution", run_id="run-1", top_k=8, keyword="   ")
    query = engine._vector_store.aquery.await_args.args[0]
    assert query.query_str == "Automated Payment Execution", "a blank keyword means the need"


async def test_the_service_forwards_the_request_keyword_to_dense_query() -> None:
    from conftest import FakeChunkTokenizer
    from geek_crawler_rag.models import QueryRequest
    from geek_crawler_rag.query import QueryService
    from geek_crawler_rag.rerank import Reranker

    llama = MagicMock()
    llama.chunk_tokenizer = FakeChunkTokenizer()
    llama.dense_query = AsyncMock(return_value=[])
    svc = QueryService(
        MagicMock(),
        Settings(openai_api_key="test"),
        llama=llama,
        reranker=Reranker(None, enabled=False),
    )

    await svc.query(QueryRequest(need="a paragraph", runId="r1", topK=8, keyword="W-9 W-8"))

    assert llama.dense_query.await_args.kwargs["keyword"] == "W-9 W-8"
    assert llama.dense_query.await_args.args[0] == "a paragraph"


async def test_dense_query_runs_both_halves_against_a_real_store(engine, caplog) -> None:
    """Through `dense_query`, a term the dense vector cannot rank is found by the keyword half,
    and the fusion logs that both halves answered -- the line that never appeared in production."""
    aclient = AsyncQdrantClient(location=":memory:")
    store = QdrantVectorStore(
        aclient=aclient,
        collection_name="hybrid",
        enable_hybrid=True,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        sparse_doc_fn=_sparse_by_token,
        sparse_query_fn=_sparse_by_token,
        hybrid_fusion_fn=logged_rank_fusion,
        text_key="text",
    )
    common = {"runId": "run-1", "ownerId": "system:crawler", "visibility": "service", "language": "en"}
    await store.async_add(
        [
            TextNode(
                text="Dext receipt capture for accountants",
                embedding=[1.0, 0.0, 0.0, 0.0],
                metadata=dict(common),
            ),
            TextNode(
                text="Zone and Co NetSuite billing",
                embedding=[0.0, 1.0, 0.0, 0.0],
                metadata=dict(common),
            ),
            TextNode(
                text="generic accounting software overview",
                embedding=[0.0, 0.0, 1.0, 0.0],
                metadata=dict(common),
            ),
        ]
    )
    engine._vector_store = store
    # Orthogonal to every stored vector: the meaning half can rank nothing, so a hit is the keyword half.
    engine.embed_query = AsyncMock(return_value=[0.0, 0.0, 0.0, 1.0])

    with caplog.at_level(logging.INFO, logger="geek_crawler_rag.llama_engine"):
        hits = await engine.dense_query("Dext", run_id="run-1", top_k=3)

    assert hits, "dense_query returned nothing: the keyword half did not run"
    assert "Dext" in hits[0].node.get_content()
    assert hits[0].score > 0.0

    fusion_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("hybrid_")]
    assert fusion_lines, "no hybrid_* log line: the fusion was never called, so the query ran dense-only"
    assert "runId=run-1" in fusion_lines[0]
