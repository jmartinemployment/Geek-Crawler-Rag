"""One point per distinct embedded text per run.

Ramp's run held 7,046 byte-identical copies among 33,728 points: one `/charge-finder` block on 194
pages, one call-to-action sentence 1,088 times. Identical text is an identical vector, so the copies
filled the dense candidate list and the query path's exact-text dedup collapsed them to one, with
nothing left behind them. A keyword question to that run returned one passage.

These pin the index-time collapse: a later page does not re-emit a text an earlier page of the run
already emitted, a retry rebuilds that set from the points it already wrote, and a failure to
rebuild it fails the job rather than writing copies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import TextNode

from conftest import FakeChunkTokenizer
from geek_crawler_rag.config import Settings
from geek_crawler_rag.indexer import IndexService, _admit_page_nodes
from geek_crawler_rag.llama_nodes import text_digest
from geek_crawler_rag.metadata import EntityRef
from geek_crawler_rag.models import IndexState, IndexStatusResponse
from geek_crawler_rag.mongo import CrawlPage, CrawlRun

FOOTER = "Start saving with Ramp today and close your books eight times faster than before."


def _page(page_id: str, heading: str, own: str) -> CrawlPage:
    blocks = [
        {"kind": "heading", "level": 2, "text": heading, "anchors": []},
        {"kind": "paragraph", "text": own, "anchors": []},
        {"kind": "heading", "level": 2, "text": "Get started", "anchors": []},
        {"kind": "paragraph", "text": FOOTER, "anchors": []},
    ]
    return CrawlPage(
        id=page_id,
        run_id="r-rep",
        origin="https://ramp.com",
        url=f"https://ramp.com/product/{page_id}",
        final_url=f"https://ramp.com/product/{page_id}",
        html="<html></html>",
        content_html="<main></main>",
        blocks=blocks,
        title=heading,
    )


PAGE_A = _page(
    "pa",
    "Bill pay",
    "Ramp bill pay sends vendor payments by ACH, card or check from one approval queue.",
)
PAGE_B = _page(
    "pb",
    "Accounting sync",
    "Ramp syncs every transaction to QuickBooks, NetSuite and Xero as it posts each day.",
)


def _service(pages: list[CrawlPage]) -> tuple[IndexService, MagicMock, MagicMock]:
    mongo = MagicMock()
    mongo.get_run = AsyncMock(
        return_value=CrawlRun(id="r-rep", crawl_type="partner", status="complete")
    )
    mongo.count_pages = AsyncMock(return_value=len(pages))
    mongo.resolve_entity = AsyncMock(
        return_value=EntityRef(
            entity_id=None,
            entity_name="ramp.com",
            source_type="partner",
            domains=("ramp.com",),
        )
    )

    async def iter_pages(_run_id, batch_size=25):
        yield pages

    mongo.iter_pages = iter_pages
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.ensure_collection = AsyncMock()
    store.run_text_owners = AsyncMock(return_value=({}, 0))

    llama = MagicMock()
    llama.chunk_tokenizer = FakeChunkTokenizer()

    async def upsert(nodes: list[TextNode]) -> int:
        return len(nodes)

    llama.embed_and_upsert = AsyncMock(side_effect=upsert)
    svc = IndexService(
        mongo, store, Settings(openai_api_key="test", embed_batch_size=1000), llama=llama
    )
    return svc, store, llama


def _upserted(llama: MagicMock) -> list[TextNode]:
    return [n for call in llama.embed_and_upsert.await_args_list for n in call.args[0]]


def _texts_by_page(nodes: list[TextNode]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n in nodes:
        out.setdefault(n.metadata["pageId"], []).append(n.get_content())
    return out


@pytest.mark.asyncio
async def test_a_text_on_two_pages_of_one_run_is_written_once():
    svc, _, llama = _service([PAGE_A, PAGE_B])
    await svc._index_run("r-rep")

    status = await svc.get_status("r-rep")
    assert status is not None and status.state == IndexState.COMPLETE

    by_page = _texts_by_page(_upserted(llama))
    footer_owners = [pid for pid, texts in by_page.items() if any(FOOTER in t for t in texts)]
    assert footer_owners == ["pa"], "the footer belongs to the first page that carried it"
    # Page B's own text still lands: only the repeat is dropped, not the page.
    assert any("QuickBooks" in t for t in by_page["pb"])
    assert status.chunks_skipped_repeat >= 1
    assert status.chunks_upserted == len(_upserted(llama))


@pytest.mark.asyncio
async def test_every_point_carries_the_digest_of_its_embedded_text():
    svc, _, llama = _service([PAGE_A])
    await svc._index_run("r-rep")

    nodes = _upserted(llama)
    assert nodes
    for n in nodes:
        assert n.metadata["textDigest"] == text_digest(n.get_content())


def test_the_digest_is_of_the_sanitised_string_the_embedder_receives():
    # A zero-width space is stripped before embedding, so both strings are one vector and must
    # be one key; a raw-string digest would let the copy through.
    assert text_digest("Close your books​ faster") == text_digest("Close your books faster")
    assert text_digest("Close your books faster") != text_digest("Close your books slower")


def test_repeats_within_one_page_are_kept():
    status = IndexStatusResponse(run_id="r", state=IndexState.RUNNING)
    owners: dict[str, str] = {}
    same = [
        TextNode(id_="1", text="Same text", metadata={"textDigest": "d1"}),
        TextNode(id_="2", text="Same text", metadata={"textDigest": "d1"}),
    ]
    assert len(_admit_page_nodes("p1", same, owners, status)) == 2
    assert status.chunks_skipped_repeat == 0
    assert owners == {"d1": "p1"}

    again = [TextNode(id_="3", text="Same text", metadata={"textDigest": "d1"})]
    assert _admit_page_nodes("p2", again, owners, status) == []
    assert status.chunks_skipped_repeat == 1


@pytest.mark.asyncio
async def test_a_retry_rebuilds_the_repeat_set_from_the_points_it_already_wrote():
    """Attempt 2 skips the delete, so the in-memory set starts empty. Without the rebuild, page B
    would write a second footer because page A's footer was committed by attempt 1."""
    svc, store, llama = _service([PAGE_B])
    svc._statuses["r-rep"] = IndexStatusResponse(
        run_id="r-rep", state=IndexState.PENDING, attempt=2
    )

    # The digest key must be what the chunker really emits for the footer section, or this
    # test passes for the wrong reason. Read it off a fresh attempt-1 run of page A.
    probe, _, probe_llama = _service([PAGE_A])
    await probe._index_run("r-rep")
    footer_digests = {
        n.metadata["textDigest"]
        for n in _upserted(probe_llama)
        if FOOTER in n.get_content()
    }
    assert footer_digests
    store.run_text_owners = AsyncMock(
        return_value=({d: "pa" for d in footer_digests}, 0)
    )

    await svc._index_run("r-rep")

    status = await svc.get_status("r-rep")
    assert status is not None and status.state == IndexState.COMPLETE
    store.run_text_owners.assert_awaited_once_with("r-rep")
    store.delete_by_run_id.assert_not_awaited()
    assert not any(FOOTER in n.get_content() for n in _upserted(llama))
    assert status.chunks_skipped_repeat >= 1


@pytest.mark.asyncio
async def test_a_retry_whose_repeat_set_cannot_be_read_fails_without_writing():
    """An empty set would read as "nothing written yet" and every repeat would be written again."""
    svc, store, llama = _service([PAGE_A, PAGE_B])
    store.run_text_owners = AsyncMock(side_effect=RuntimeError("qdrant unavailable"))
    svc._statuses["r-rep"] = IndexStatusResponse(
        run_id="r-rep", state=IndexState.PENDING, attempt=2
    )

    await svc._index_run("r-rep")

    status = await svc.get_status("r-rep")
    assert status is not None and status.state == IndexState.FAILED
    llama.embed_and_upsert.assert_not_awaited()
