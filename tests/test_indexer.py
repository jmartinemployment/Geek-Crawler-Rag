"""Indexer status transitions with mocked Mongo / Qdrant / LlamaIndex."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import TextNode

from geek_crawler_rag.config import Settings
from geek_crawler_rag.indexer import IndexService, LeaseLostError
from geek_crawler_rag.metadata import EntityRef
from geek_crawler_rag.models import IndexState
from geek_crawler_rag.mongo import CrawlPage, CrawlRun


def _entity_mock(mongo: MagicMock) -> None:
    mongo.resolve_entity = AsyncMock(
        return_value=EntityRef(
            entity_id=None,
            entity_name="example.com",
            source_type="partner",
            domains=("example.com",),
        )
    )


def _llama_mock() -> MagicMock:
    llama = MagicMock()

    async def upsert(nodes: list[TextNode]) -> int:
        return len(nodes)

    llama.embed_and_upsert = AsyncMock(side_effect=upsert)
    return llama


@pytest.mark.asyncio
async def test_index_deletes_locale_and_failure_pages():
    mongo = MagicMock()
    mongo.get_run = AsyncMock(
        return_value=CrawlRun(id="r-loc", crawl_type="partner", status="complete")
    )
    mongo.count_pages = AsyncMock(return_value=2)
    mongo.delete_page = AsyncMock(return_value=1)
    _entity_mock(mongo)

    async def pages(_run_id, batch_size=25):
        yield [
            CrawlPage(
                id="p-locale",
                run_id="r-loc",
                origin="https://speakai.co",
                url="https://speakai.co/es/approaches/",
                final_url="https://speakai.co/es/approaches/",
                html="<html><body>ok</body></html>",
            ),
            CrawlPage(
                id="p-fail",
                run_id="r-loc",
                origin="https://speakai.co",
                url="https://speakai.co/blocked",
                final_url="https://speakai.co/blocked",
                html="<html><body>challenge</body></html>",
                failure_reason="cloudflare_challenge",
            ),
        ]

    mongo.iter_pages = pages
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.delete_by_page_id = AsyncMock()
    store.ensure_collection = AsyncMock()
    llama = _llama_mock()

    svc = IndexService(mongo, store, Settings(openai_api_key="test"), llama=llama)
    await svc._index_run("r-loc")

    status = await svc.get_status("r-loc")
    assert status is not None
    assert status.state == IndexState.COMPLETE
    assert status.pages_skipped_unusable == 2
    # Indexing is read-only over the corpus: unusable pages are counted and left
    # where they are. Deleting them is what destroyed 5,274 pages on 2026-09-18.
    mongo.delete_page.assert_not_awaited()
    store.delete_by_page_id.assert_not_awaited()
    llama.embed_and_upsert.assert_not_called()


@pytest.mark.asyncio
async def test_index_never_deletes_a_page_it_cannot_use():
    """The regression that cost the corpus.

    A page this run cannot index may be perfectly usable to the next one — and
    on 2026-09-18 every page was unusable for one reason (the crawler had changed
    corpus format), so deleting them emptied the corpus and its Qdrant points in
    a single pass. Indexing must never mutate what it reads.
    """
    mongo = MagicMock()
    mongo.get_run = AsyncMock(
        return_value=CrawlRun(id="r-keep", crawl_type="partner", status="complete")
    )
    mongo.count_pages = AsyncMock(return_value=1)
    mongo.delete_page = AsyncMock()
    _entity_mock(mongo)

    async def pages(_run_id, batch_size=25):
        yield [
            CrawlPage(
                id="p-keep",
                run_id="r-keep",
                origin="https://speakai.co",
                url="https://speakai.co/es/approaches/",
                final_url="https://speakai.co/es/approaches/",
                html="<html><body>ok</body></html>",
            )
        ]

    mongo.iter_pages = pages
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.delete_by_page_id = AsyncMock()
    store.ensure_collection = AsyncMock()

    svc = IndexService(
        mongo, store, Settings(openai_api_key="test"), llama=_llama_mock()
    )
    await svc._index_run("r-keep")

    status = await svc.get_status("r-keep")
    assert status is not None
    # Nothing indexable, but that is a finished run, not a failed one.
    assert status.state == IndexState.COMPLETE
    assert status.pages_skipped_unusable >= 1
    mongo.delete_page.assert_not_awaited()
    store.delete_by_page_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_index_run_not_found():
    mongo = MagicMock()
    mongo.get_run = AsyncMock(return_value=None)
    _entity_mock(mongo)
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.ensure_collection = AsyncMock()
    settings = Settings(openai_api_key="test")

    svc = IndexService(mongo, store, settings, llama=_llama_mock())
    await svc.enqueue("missing-run")
    await svc._index_run("missing-run")

    status = await svc.get_status("missing-run")
    assert status is not None
    assert status.state == IndexState.FAILED
    assert status.error and "not found" in status.error.lower()


@pytest.mark.asyncio
async def test_index_skips_when_no_english():
    mongo = MagicMock()
    mongo.get_run = AsyncMock(
        return_value=CrawlRun(id="r1", crawl_type="partner", status="complete")
    )
    mongo.count_pages = AsyncMock(return_value=1)
    mongo.delete_page = AsyncMock(return_value=0)
    _entity_mock(mongo)

    spanish_blocks = [
        {"kind": "heading", "level": 1, "text": "Página", "anchors": []},
        {
            "kind": "paragraph",
            "text": "Esta es una página en español con contenido suficiente. " * 20,
            "anchors": [],
        },
    ]

    async def empty_english_pages(_run_id, batch_size=25):
        yield [
            CrawlPage(
                id="p1",
                run_id="r1",
                origin="https://ejemplo.es",
                url="https://ejemplo.es/",
                final_url="https://ejemplo.es/",
                html="<html><body>ignored</body></html>",
                blocks=spanish_blocks,
                title="Página",
            )
        ]

    mongo.iter_pages = empty_english_pages
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.delete_by_page_id = AsyncMock()
    store.ensure_collection = AsyncMock()
    llama = _llama_mock()
    settings = Settings(openai_api_key="test")

    svc = IndexService(mongo, store, settings, llama=llama)
    await svc._index_run("r1")

    status = await svc.get_status("r1")
    assert status is not None
    assert status.state == IndexState.COMPLETE
    assert status.mongo_page_count == 1
    assert status.pages_skipped_unusable >= 1
    assert status.pages_skipped_lang >= 1
    mongo.delete_page.assert_not_awaited()
    # The page's Qdrant points are left alone too — a non-English page this run
    # cannot embed is not a reason to destroy vectors another run may have built.
    store.delete_by_page_id.assert_not_awaited()
    llama.embed_and_upsert.assert_not_called()


@pytest.mark.asyncio
async def test_index_skips_html_only_pages_without_deleting_them():
    mongo = MagicMock()
    mongo.get_run = AsyncMock(
        return_value=CrawlRun(id="r-html", crawl_type="partner", status="complete")
    )
    mongo.count_pages = AsyncMock(return_value=1)
    mongo.delete_page = AsyncMock(return_value=0)
    _entity_mock(mongo)

    async def pages(_run_id, batch_size=25):
        yield [
            CrawlPage(
                id="p-html",
                run_id="r-html",
                origin="https://partner.com",
                url="https://partner.com/docs",
                final_url="https://partner.com/docs",
                html="<html><body>"
                + ("This English documentation explains the partner API thoroughly. " * 40)
                + "</body></html>",
                blocks=[],
            )
        ]

    mongo.iter_pages = pages
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.delete_by_page_id = AsyncMock()
    store.ensure_collection = AsyncMock()

    svc = IndexService(
        mongo, store, Settings(openai_api_key="test"), llama=_llama_mock()
    )
    await svc._index_run("r-html")

    status = await svc.get_status("r-html")
    assert status is not None
    assert status.state == IndexState.COMPLETE
    assert status.pages_skipped_empty >= 1
    mongo.delete_page.assert_not_awaited()


@pytest.mark.asyncio
async def test_index_upserts_english_chunks():
    blocks = [
        {"kind": "heading", "level": 1, "text": "Docs", "anchors": []},
        {
            "kind": "paragraph",
            "text": "This English documentation explains the partner API thoroughly. " * 40,
            "anchors": [],
        },
    ]
    mongo = MagicMock()
    mongo.get_run = AsyncMock(
        return_value=CrawlRun(id="r2", crawl_type="competitors", status="complete")
    )
    mongo.count_pages = AsyncMock(return_value=1)
    mongo.delete_page = AsyncMock(return_value=0)
    _entity_mock(mongo)

    async def pages(_run_id, batch_size=25):
        yield [
            CrawlPage(
                id="p2",
                run_id="r2",
                origin="https://rival.com",
                url="https://rival.com/docs",
                final_url="https://rival.com/docs",
                html="<html><body>ignored</body></html>",
                blocks=blocks,
                title="Docs",
            )
        ]

    mongo.iter_pages = pages
    mongo.delete_page = AsyncMock(return_value=0)
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.delete_by_page_id = AsyncMock()
    store.ensure_collection = AsyncMock()
    llama = _llama_mock()
    settings = Settings(openai_api_key="test", embed_batch_size=10)

    svc = IndexService(mongo, store, settings, llama=llama)
    await svc._index_run("r2")

    status = await svc.get_status("r2")
    assert status is not None
    assert status.state == IndexState.COMPLETE
    assert status.chunks_upserted >= 1
    assert status.crawl_type == "competitors"
    store.delete_by_run_id.assert_awaited_once_with(
        "r2",
        owner_id="system:crawler",
        visibility="service",
    )
    assert llama.embed_and_upsert.await_count >= 1
    nodes = llama.embed_and_upsert.await_args.args[0]
    assert any(n.metadata.get("chunkRole") == "child" for n in nodes)
    assert any(n.metadata.get("parentText") for n in nodes)


@pytest.mark.asyncio
async def test_index_flush_failure_preserves_partial_points():
    blocks = [
        {"kind": "heading", "level": 1, "text": "Docs", "anchors": []},
        {
            "kind": "paragraph",
            "text": "This English documentation explains the partner API thoroughly. " * 40,
            "anchors": [],
        },
    ]
    mongo = MagicMock()
    mongo.get_run = AsyncMock(
        return_value=CrawlRun(id="r3", crawl_type="partner", status="complete")
    )
    mongo.count_pages = AsyncMock(return_value=1)
    _entity_mock(mongo)

    async def pages(_run_id, batch_size=25):
        yield [
            CrawlPage(
                id="p3",
                run_id="r3",
                origin="https://partner.com",
                url="https://partner.com/docs",
                final_url="https://partner.com/docs",
                html="<html><body>ignored</body></html>",
                blocks=blocks,
            )
        ]

    mongo.iter_pages = pages
    mongo.delete_page = AsyncMock(return_value=0)
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.delete_by_page_id = AsyncMock()
    store.ensure_collection = AsyncMock()
    llama = MagicMock()
    llama.embed_and_upsert = AsyncMock(side_effect=RuntimeError("qdrant down"))
    settings = Settings(openai_api_key="test", embed_batch_size=10)

    svc = IndexService(mongo, store, settings, llama=llama)
    await svc._index_run("r3")

    status = await svc.get_status("r3")
    assert status is not None
    assert status.state == IndexState.FAILED
    assert store.delete_by_run_id.await_count == 1
    assert status.error and "qdrant down" in status.error.lower()


@pytest.mark.asyncio
async def test_enqueue_dedupes_pending():
    mongo = MagicMock()
    _entity_mock(mongo)
    store = MagicMock()
    settings = Settings(openai_api_key="test")
    svc = IndexService(mongo, store, settings, llama=_llama_mock())
    first = await svc.enqueue("same")
    second = await svc.enqueue("same")
    assert first is second
    assert svc._queue.qsize() == 1


@pytest.mark.asyncio
async def test_status_store_is_authoritative_over_local_cache():
    mongo = MagicMock()
    store = MagicMock()
    status_store = MagicMock()
    persisted = MagicMock()
    persisted.run_id = "shared"
    persisted.state = IndexState.COMPLETE
    status_store.get = AsyncMock(return_value=persisted)
    svc = IndexService(
        mongo,
        store,
        Settings(openai_api_key="test"),
        llama=_llama_mock(),
        status_store=status_store,
    )
    svc._statuses["shared"] = MagicMock(state=IndexState.RUNNING)

    loaded = await svc.get_status("shared")

    assert loaded is persisted
    status_store.get.assert_awaited_once_with("shared")


@pytest.mark.asyncio
async def test_claim_is_revalidated_before_index_execution():
    mongo = MagicMock()
    store = MagicMock()
    status_store = MagicMock()
    status_store.is_owned = AsyncMock(return_value=False)
    llama = _llama_mock()
    svc = IndexService(
        mongo,
        store,
        Settings(openai_api_key="test"),
        llama=llama,
        status_store=status_store,
    )

    with pytest.raises(LeaseLostError):
        await svc._run_claimed_job("expired")

    llama.embed_and_upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_does_not_auto_reclaim_stale_leases():
    mongo = MagicMock()
    store = MagicMock()
    status_store = MagicMock()
    status_store.ensure_indexes = AsyncMock()
    status_store.claim_recoverable = AsyncMock(
        return_value=[MagicMock(run_id="stale-run")]
    )
    svc = IndexService(
        mongo,
        store,
        Settings(openai_api_key="test"),
        llama=_llama_mock(),
        status_store=status_store,
    )

    await svc.start()
    try:
        status_store.ensure_indexes.assert_awaited_once()
        status_store.claim_recoverable.assert_not_awaited()
        assert svc._queue.empty()
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_shutdown_preserves_partial_qdrant_index():
    blocks = [
        {"kind": "heading", "level": 1, "text": "Docs", "anchors": []},
        {
            "kind": "paragraph",
            "text": "This English documentation explains the partner API thoroughly. " * 40,
            "anchors": [],
        },
    ]
    mongo = MagicMock()
    mongo.get_run = AsyncMock(
        return_value=CrawlRun(id="shutdown", crawl_type="partner", status="complete")
    )
    mongo.count_pages = AsyncMock(return_value=1)
    _entity_mock(mongo)

    async def pages(_run_id, batch_size=25):
        yield [
            CrawlPage(
                id="shutdown-page",
                run_id="shutdown",
                origin="https://partner.com",
                url="https://partner.com/docs",
                final_url="https://partner.com/docs",
                html="<html><body>ignored</body></html>",
                blocks=blocks,
            )
        ]

    mongo.iter_pages = pages
    mongo.delete_page = AsyncMock(return_value=0)
    store = MagicMock()
    store.ensure_collection = AsyncMock()
    store.delete_by_run_id = AsyncMock()
    store.delete_by_page_id = AsyncMock()
    started = asyncio.Event()
    never = asyncio.Event()
    llama = MagicMock()

    async def blocked_embed(_nodes):
        started.set()
        await never.wait()

    llama.embed_and_upsert = AsyncMock(side_effect=blocked_embed)
    svc = IndexService(
        mongo,
        store,
        Settings(openai_api_key="test", embed_batch_size=1),
        llama=llama,
    )
    await svc.start()
    await svc.enqueue("shutdown")
    await started.wait()

    await svc.stop()

    # Start may delete once for attempt 1; cancel/stop must not wipe again.
    assert store.delete_by_run_id.await_count == 1
    status = await svc.get_status("shutdown")
    assert status is not None
    assert status.state == IndexState.FAILED
    assert status.error
    assert (
        "shut down" in status.error.lower()
        or "cancelled" in status.error.lower()
    )
