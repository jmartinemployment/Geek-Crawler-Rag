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
    _entity_mock(mongo)

    async def empty_english_pages(_run_id, batch_size=25):
        yield [
            CrawlPage(
                id="p1",
                run_id="r1",
                origin="https://ejemplo.es",
                url="https://ejemplo.es/",
                final_url="https://ejemplo.es/",
                html="<html><body>"
                + ("Esta es una página en español con contenido suficiente. " * 20)
                + "</body></html>",
            )
        ]

    mongo.iter_pages = empty_english_pages
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.ensure_collection = AsyncMock()
    llama = _llama_mock()
    settings = Settings(openai_api_key="test")

    svc = IndexService(mongo, store, settings, llama=llama)
    await svc._index_run("r1")

    status = await svc.get_status("r1")
    assert status is not None
    assert status.state == IndexState.COMPLETE
    assert status.mongo_page_count == 1
    assert status.pages_deleted_non_english >= 1
    llama.embed_and_upsert.assert_not_called()


@pytest.mark.asyncio
async def test_index_upserts_english_chunks():
    html = (
        "<html><head><title>Docs</title></head><body>"
        + ("This English documentation explains the partner API thoroughly. " * 40)
        + "</body></html>"
    )
    mongo = MagicMock()
    mongo.get_run = AsyncMock(
        return_value=CrawlRun(id="r2", crawl_type="competitors", status="complete")
    )
    mongo.count_pages = AsyncMock(return_value=1)
    _entity_mock(mongo)

    async def pages(_run_id, batch_size=25):
        yield [
            CrawlPage(
                id="p2",
                run_id="r2",
                origin="https://rival.com",
                url="https://rival.com/docs",
                final_url="https://rival.com/docs",
                html=html,
            )
        ]

    mongo.iter_pages = pages
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
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
    store.delete_by_run_id.assert_awaited_once_with("r2")
    assert llama.embed_and_upsert.await_count >= 1
    nodes = llama.embed_and_upsert.await_args.args[0]
    assert any(n.metadata.get("chunkRole") == "child" for n in nodes)
    assert any(n.metadata.get("parentText") for n in nodes)


@pytest.mark.asyncio
async def test_index_flush_failure_cleans_partial_points():
    html = (
        "<html><body>"
        + ("This English documentation explains the partner API thoroughly. " * 40)
        + "</body></html>"
    )
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
                html=html,
            )
        ]

    mongo.iter_pages = pages
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.ensure_collection = AsyncMock()
    llama = MagicMock()
    llama.embed_and_upsert = AsyncMock(side_effect=RuntimeError("qdrant down"))
    settings = Settings(openai_api_key="test", embed_batch_size=10)

    svc = IndexService(mongo, store, settings, llama=llama)
    await svc._index_run("r3")

    status = await svc.get_status("r3")
    assert status is not None
    assert status.state == IndexState.FAILED
    assert store.delete_by_run_id.await_count >= 2


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
async def test_shutdown_cleans_partial_qdrant_index():
    html = (
        "<html><body>"
        + ("This English documentation explains the partner API thoroughly. " * 40)
        + "</body></html>"
    )
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
                html=html,
            )
        ]

    mongo.iter_pages = pages
    store = MagicMock()
    store.ensure_collection = AsyncMock()
    store.delete_by_run_id = AsyncMock()
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

    assert store.delete_by_run_id.await_count >= 2
