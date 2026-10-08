"""Each half of a hybrid query is logged, so a silent drop to one signal shows in the log.

QdrantVectorStore runs the dense (meaning) and sparse BM25 (keyword) searches and fuses them
itself. When one half returns nothing, the fusion returns the other alone and the answer looks like
a healthy one. Jeff, 2026-10-06: log before any fallback, so a failure can be understood. The
fusion itself is LlamaIndex's `relative_score_fusion`, unchanged.

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
from llama_index.vector_stores.qdrant.utils import relative_score_fusion

from geek_crawler_rag import llama_engine
from geek_crawler_rag.llama_engine import _HYBRID_RUN_ID, logged_relative_score_fusion


def _result(*ids: str) -> VectorStoreQueryResult:
    return VectorStoreQueryResult(
        nodes=[TextNode(id_=i, text=f"text {i}") for i in ids],
        similarities=[1.0 - n * 0.1 for n in range(len(ids))],
        ids=list(ids),
    )


def test_both_halves_are_counted(caplog):
    _HYBRID_RUN_ID.set("run-1")
    with caplog.at_level(logging.INFO, logger=llama_engine.__name__):
        out = logged_relative_score_fusion(_result("a", "b"), _result("b", "c"), top_k=5)

    [record] = [r for r in caplog.records if "hybrid_" in r.getMessage()]
    assert record.levelno == logging.INFO
    assert record.getMessage() == (
        "hybrid_halves runId=run-1 dense=2 sparse=2 both=1 union=3 cut=5 fused=3 "
        "kept=1/1/1(both/denseOnly/sparseOnly) dropped=0/0(denseOnly/sparseOnly) "
        "firstDropped=-/-(denseRank/sparseRank) lastKept=0.000"
    )
    assert [n.node_id for n in out.nodes] == [
        n.node_id for n in relative_score_fusion(_result("a", "b"), _result("b", "c"), top_k=5).nodes
    ], "the fusion must be LlamaIndex's, unchanged"


def test_the_cut_is_reported_per_half_with_the_best_rank_dropped(caplog):
    """Dense a,b,c,d and sparse c,e,f,g at 1.0/0.9/0.8/0.7, cut at 4.

    Min-max per half: a=1 b=.667 c=.333 d=0; c=1 e=.667 f=.333 g=0. Fused at alpha .5:
    c=.667 a=.5 b=.333 e=.333 f=.167 d=0 g=0. The cut keeps c, a, b, e: one in both halves, two
    meaning-only, one keyword-only. Dropped: d (meaning rank 4), f and g (keyword ranks 3, 4).
    """
    _HYBRID_RUN_ID.set("run-4")
    with caplog.at_level(logging.INFO, logger=llama_engine.__name__):
        out = logged_relative_score_fusion(
            _result("a", "b", "c", "d"), _result("c", "e", "f", "g"), top_k=4
        )

    [record] = [r for r in caplog.records if "hybrid_" in r.getMessage()]
    assert record.getMessage() == (
        "hybrid_halves runId=run-4 dense=4 sparse=4 both=1 union=7 cut=4 fused=4 "
        "kept=1/2/1(both/denseOnly/sparseOnly) dropped=1/2(denseOnly/sparseOnly) "
        "firstDropped=4/3(denseRank/sparseRank) lastKept=0.333"
    )
    assert [n.node_id for n in out.nodes] == ["c", "a", "b", "e"]


def test_an_empty_keyword_half_is_a_warning_naming_what_the_answer_used(caplog):
    _HYBRID_RUN_ID.set("run-2")
    with caplog.at_level(logging.INFO, logger=llama_engine.__name__):
        logged_relative_score_fusion(_result("a", "b"), _result(), top_k=5)

    [record] = [r for r in caplog.records if "hybrid_" in r.getMessage()]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "hybrid_half_empty runId=run-2 dense=2 sparse=0" in message
    assert "the meaning half only" in message


def test_an_empty_meaning_half_is_a_warning_too(caplog):
    _HYBRID_RUN_ID.set("run-3")
    with caplog.at_level(logging.INFO, logger=llama_engine.__name__):
        logged_relative_score_fusion(_result(), _result("c"), top_k=5)

    [record] = [r for r in caplog.records if "hybrid_" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert "the keyword half only" in record.getMessage()
