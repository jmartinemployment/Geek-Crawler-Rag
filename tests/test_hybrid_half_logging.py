"""Each half of a hybrid query is logged, so a silent drop to one signal shows in the log.

QdrantVectorStore runs the dense (meaning) and sparse BM25 (keyword) searches and fuses them
itself. When one half returns nothing, the fusion returns the other alone and the answer looks like
a healthy one. Jeff, 2026-10-06: log before any fallback, so a failure can be understood.

The fusion is reciprocal rank fusion since 2026-10-08. The test below that used to pin "the fusion
must be LlamaIndex's, unchanged" is reversed on purpose: min-max relative score dropped half the
union by the shape of each half's score curve (HANDOFF 9b), and rank fusion makes rank r in one
half worth rank r in the other and rewards a node both halves return.

Three sizes were not enough (2026-10-08). The fusion cuts the union at `top_k`, which is the
per-half fetch size, so up to half of what the halves returned is dropped before `query.py` sees
it, and `dense=64 sparse=64 fused=64` read the same on six live queries whose final answers held
between 4 and 21 keyword-only chunks. The line now says what was kept from each half, what was
dropped, and the best rank dropped on each side.
"""

from __future__ import annotations

import logging

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from geek_crawler_rag import llama_engine
from geek_crawler_rag.llama_engine import (
    _HYBRID_QUESTION,
    _HYBRID_RUN_ID,
    logged_rank_fusion,
)


def _result(*ids: str, host: str | None = None) -> VectorStoreQueryResult:
    return VectorStoreQueryResult(
        nodes=[TextNode(id_=i, text=f"text {i}", metadata={"host": host} if host else {}) for i in ids],
        similarities=[1.0 - n * 0.1 for n in range(len(ids))],
        ids=list(ids),
    )


def test_both_halves_are_counted(caplog):
    _HYBRID_RUN_ID.set("run-1")
    _HYBRID_QUESTION.set(("Automated payment reconciliation syncs results with the ERP", "payment reconciliation"))
    with caplog.at_level(logging.INFO, logger=llama_engine.__name__):
        out = logged_rank_fusion(
            _result("a", "b", host="tipalti.com"), _result("b", "c", host="tipalti.com"), top_k=5
        )

    [record] = [r for r in caplog.records if "hybrid_" in r.getMessage()]
    assert record.levelno == logging.INFO
    assert record.getMessage() == (
        'hybrid_halves runId=run-1 host=tipalti.com '
        'need="Automated payment reconciliation syncs results with the ERP" '
        'keyword="payment reconciliation" dense=2 sparse=2 overlap=1 union=3 cut=5 fused=3 '
        "survivors=1/1/1(overlap/denseOnly/sparseOnly) dropped=0/0(denseOnly/sparseOnly) "
        "firstDropped=-/-(denseRank/sparseRank) lastKept=0.0161 "
        "denseRaw=1.000..0.900 sparseRaw=1.000..0.900"
    )
    # b is rank 2 in the meaning half and rank 1 in the keyword half: 1/62 + 1/61. a is rank 1 in
    # one half: 1/61. c is rank 2 in one half: 1/62. Under min-max, a and b tied and a led.
    assert [n.node_id for n in out.nodes] == ["b", "a", "c"], "the node both halves return leads"


def test_the_cut_is_reported_per_half_with_the_best_rank_dropped(caplog):
    """Dense a,b,c,d and sparse c,e,f,g at 1.0/0.9/0.8/0.7, cut at 4.

    Rank fusion: c = 1/63 + 1/61 (both halves), a = 1/61, b = e = 1/62 (b first: the meaning half
    is scored first and the sort is stable), f = 1/63, d = g = 1/64. The cut keeps c, a, b, e: one
    in both halves, two meaning-only, one keyword-only. Dropped: d (meaning rank 4), f and g
    (keyword ranks 3, 4). In production `dense_query` sets the cut to the union, so dropped reads
    0/0 there; this pins what the fields say when a cut does happen.
    """
    _HYBRID_RUN_ID.set("run-4")
    _HYBRID_QUESTION.set(("x" * 80, ""))
    with caplog.at_level(logging.INFO, logger=llama_engine.__name__):
        out = logged_rank_fusion(_result("a", "b", "c", "d"), _result("c", "e", "f", "g"), top_k=4)

    [record] = [r for r in caplog.records if "hybrid_" in r.getMessage()]
    assert record.getMessage() == (
        'hybrid_halves runId=run-4 host=- need="' + "x" * 71 + '…" keyword=- '
        "dense=4 sparse=4 overlap=1 union=7 cut=4 fused=4 "
        "survivors=1/2/1(overlap/denseOnly/sparseOnly) dropped=1/2(denseOnly/sparseOnly) "
        "firstDropped=4/3(denseRank/sparseRank) lastKept=0.0161 "
        "denseRaw=1.000..0.700 sparseRaw=1.000..0.700"
    ), "a need over 72 characters is cut with an ellipsis; no keyword and no host print as -"
    assert [n.node_id for n in out.nodes] == ["c", "a", "b", "e"]


def test_nothing_is_cut_when_the_cut_is_the_union(caplog):
    """`dense_query` sends hybrid_top_k = both fetches: every node of both halves survives."""
    _HYBRID_RUN_ID.set("run-5")
    _HYBRID_QUESTION.set(("q", ""))
    with caplog.at_level(logging.INFO, logger=llama_engine.__name__):
        out = logged_rank_fusion(_result("a", "b", "c", "d"), _result("c", "e", "f", "g"), top_k=8)

    [record] = [r for r in caplog.records if "hybrid_" in r.getMessage()]
    assert "cut=8 fused=7 survivors=1/3/3(overlap/denseOnly/sparseOnly) dropped=0/0" in record.getMessage()
    assert [n.node_id for n in out.nodes] == ["c", "a", "b", "e", "f", "d", "g"]


def test_an_empty_keyword_half_is_a_warning_naming_what_the_answer_used(caplog):
    _HYBRID_RUN_ID.set("run-2")
    _HYBRID_QUESTION.set(("Automated Payment Execution", "Automated Payment Execution"))
    with caplog.at_level(logging.INFO, logger=llama_engine.__name__):
        logged_rank_fusion(_result("a", "b", host="melio.com"), _result(), top_k=5)

    [record] = [r for r in caplog.records if "hybrid_" in r.getMessage()]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert (
        'hybrid_half_empty runId=run-2 host=melio.com need="Automated Payment Execution" '
        'keyword="Automated Payment Execution" dense=2 sparse=0'
    ) in message
    assert "the meaning half only" in message


def test_an_empty_meaning_half_is_a_warning_too(caplog):
    _HYBRID_RUN_ID.set("run-3")
    with caplog.at_level(logging.INFO, logger=llama_engine.__name__):
        logged_rank_fusion(_result(), _result("c"), top_k=5)

    [record] = [r for r in caplog.records if "hybrid_" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert "the keyword half only" in record.getMessage()
