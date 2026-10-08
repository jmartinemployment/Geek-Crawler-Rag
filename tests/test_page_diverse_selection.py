"""Rank passages, then select pages.

Until 2026-10-08 `_select_ranked_candidates` took the top ``top_k`` straight off the ranked list.
Nothing said "enough from this page, give the next page a turn", so a page that repeated the
question's words filled slot after slot: on Stampli the bare keyword "Automated Approval Workflows"
put 13 of 32 passages on one blog post, and GeekAPI's extractor -- which reads the pages behind the
passages -- saw 14 pages where the pool held more. The rule now: every page's best passage first,
in rank order of those best passages, then every page's second-best, and so on until ``top_k`` is
filled. The R6 "flooded pool" test is the last one here, through the service.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from llama_index.core.schema import NodeWithScore, TextNode

from conftest import FakeChunkTokenizer
from geek_crawler_rag.config import Settings
from geek_crawler_rag.models import QueryRequest
from geek_crawler_rag.query import QueryService, _page_key, _select_ranked_candidates
from geek_crawler_rag.rerank import Reranker

REQ = QueryRequest(need="n", runId="r")


def _cand(cid: str, page: str, text: str | None = None, **extra: str) -> dict:
    payload = {"pageId": page, "sectionTitle": cid, "parentText": f"P-{cid}", "text": text or cid}
    payload.update(extra)
    return {"id": cid, "payload": payload}


def _ranked(pool: list[dict]) -> list[tuple[int, float]]:
    """Positional ranks, best first, as the disabled reranker produces them."""
    return [(i, float(len(pool) - i)) for i in range(len(pool))]


def _ids(selected: list[tuple[dict, float]]) -> list[str]:
    return [c["id"] for c, _ in selected]


def test_every_page_gets_its_first_slot_before_any_page_gets_a_second() -> None:
    """The Stampli shape: one page holds the 13 best-ranked passages, fifteen pages hold one each."""
    pool = [_cand(f"blog{i}", "blog") for i in range(13)]
    pool += [_cand(f"p{i}", f"page{i}") for i in range(15)]

    selected = _select_ranked_candidates(
        pool, _ranked(pool), request=REQ, target_top_k=16, collapse_parents=False
    )

    assert len(selected) == 16
    assert _ids(selected)[0] == "blog0", "the best passage overall still comes first"
    assert _ids(selected)[1:] == [f"p{i}" for i in range(15)]
    pages = [c["payload"]["pageId"] for c, _ in selected]
    assert len(set(pages)) == 16, "sixteen slots, sixteen pages"


def test_a_page_contributes_again_only_after_every_page_has_had_its_turn() -> None:
    # Rank order: A1, A2, B1, A3, C1, B2. Pages by their best passage: A, B, C.
    pool = [
        _cand("A1", "A"),
        _cand("A2", "A"),
        _cand("B1", "B"),
        _cand("A3", "A"),
        _cand("C1", "C"),
        _cand("B2", "B"),
    ]

    selected = _select_ranked_candidates(
        pool, _ranked(pool), request=REQ, target_top_k=6, collapse_parents=False
    )

    assert _ids(selected) == ["A1", "B1", "C1", "A2", "B2", "A3"]


def test_the_returned_order_is_the_selection_order_so_a_truncation_keeps_page_diversity() -> None:
    pool = [_cand("A1", "A"), _cand("A2", "A"), _cand("B1", "B"), _cand("C1", "C")]

    selected = _select_ranked_candidates(
        pool, _ranked(pool), request=REQ, target_top_k=4, collapse_parents=False
    )

    assert _ids(selected)[:3] == ["A1", "B1", "C1"]
    assert _ids(selected)[3] == "A2"


def test_fewer_pages_than_top_k_fills_from_each_page_in_turn() -> None:
    pool = [_cand(f"A{i}", "A") for i in range(5)] + [_cand(f"B{i}", "B") for i in range(5)]

    selected = _select_ranked_candidates(
        pool, _ranked(pool), request=REQ, target_top_k=6, collapse_parents=False
    )

    assert _ids(selected) == ["A0", "B0", "A1", "B1", "A2", "B2"]


def test_exact_repeated_text_is_still_dropped_and_does_not_use_up_the_page_s_turn() -> None:
    # B's best passage is the same text as A's best; B's turn goes to its next passage instead.
    pool = [
        _cand("A1", "A", text="same words"),
        _cand("B1", "B", text="same words"),
        _cand("B2", "B", text="other words"),
        _cand("C1", "C", text="third words"),
    ]

    selected = _select_ranked_candidates(
        pool, _ranked(pool), request=REQ, target_top_k=3, collapse_parents=False
    )

    assert _ids(selected) == ["A1", "B2", "C1"]


def test_sibling_collapse_still_applies_and_skips_to_the_next_parent_on_that_page() -> None:
    pool = [
        {"id": "a1", "payload": {"pageId": "A", "sectionTitle": "S", "parentText": "PA", "text": "a1"}},
        {"id": "a2", "payload": {"pageId": "A", "sectionTitle": "S", "parentText": "PA", "text": "a2"}},
        {"id": "a3", "payload": {"pageId": "A", "sectionTitle": "T", "parentText": "PT", "text": "a3"}},
        {"id": "b1", "payload": {"pageId": "B", "sectionTitle": "S", "parentText": "PB", "text": "b1"}},
    ]

    selected = _select_ranked_candidates(
        pool,
        _ranked(pool),
        request=QueryRequest(need="n", runId="r", preferParent=True),
        target_top_k=3,
        collapse_parents=True,
    )

    # Round 1: A's best (a1), B's best (b1). Round 2: A's next admissible is a3 -- a2 shares a1's parent.
    assert _ids(selected) == ["a1", "b1", "a3"]


def test_a_candidate_with_no_page_identity_is_its_own_page() -> None:
    pool = [
        {"id": "x1", "payload": {"text": "one"}},
        {"id": "x2", "payload": {"text": "two"}},
        _cand("A1", "A"),
    ]

    selected = _select_ranked_candidates(
        pool, _ranked(pool), request=REQ, target_top_k=3, collapse_parents=False
    )

    assert _ids(selected) == ["x1", "x2", "A1"]


def test_page_identity_falls_back_from_page_id_to_final_url_to_url() -> None:
    assert _page_key({"pageId": "p1", "url": "https://x.test/a"}) == "id:p1"
    assert _page_key({"finalUrl": "https://x.test/A", "url": "https://x.test/b"}) == "url:https://x.test/a"
    assert _page_key({"url": "https://x.test/b"}) == "url:https://x.test/b"
    assert _page_key({}) == ""


def test_a_top_k_of_zero_selects_nothing() -> None:
    pool = [_cand("A1", "A")]
    assert _select_ranked_candidates(
        pool, _ranked(pool), request=REQ, target_top_k=0, collapse_parents=False
    ) == []


def _node(node_id: str, page_id: str, text: str) -> TextNode:
    return TextNode(
        id_=node_id,
        text=text,
        metadata={
            "runId": "r1",
            "crawlType": "partner",
            "host": "acme.com",
            "url": f"https://acme.com/{page_id}",
            "finalUrl": f"https://acme.com/{page_id}",
            "title": page_id,
            "chunkIndex": 0,
            "language": "en",
            "pageId": page_id,
            "sectionTitle": node_id,
            "parentText": f"parent {node_id}",
            "childText": text,
            "chunkRole": "child",
        },
    )


async def test_the_flooded_pool_regression_through_the_service() -> None:
    """R6's flooded-pool test. Thirteen passages of one page outrank five other pages' single
    passages; a top-6 answer is six pages, the flooded page first, not six copies of its page."""
    nodes = [_node(f"blog{i}", "blog", f"blog passage {i}") for i in range(13)]
    nodes += [_node(f"p{i}", f"page{i}", f"page {i} passage") for i in range(5)]
    llama = MagicMock()
    llama.chunk_tokenizer = FakeChunkTokenizer()
    llama.dense_query = AsyncMock(
        return_value=[NodeWithScore(node=n, score=0.9 - i * 0.01) for i, n in enumerate(nodes)]
    )
    settings = Settings(openai_api_key="test", hybrid_dense_limit=30, rerank_pool_size=40)
    svc = QueryService(MagicMock(), settings, llama=llama, reranker=Reranker(None, enabled=False))

    resp = await svc.query(QueryRequest(need="approval workflows", runId="r1", topK=6))

    pages = [c.page_id for c in resp.chunks]
    assert len(resp.chunks) == 6
    assert len(set(pages)) == 6, f"six slots went to {len(set(pages))} page(s): {pages}"
    assert pages[0] == "blog"
    assert [c.rank for c in resp.chunks] == [1, 2, 3, 4, 5, 6]
