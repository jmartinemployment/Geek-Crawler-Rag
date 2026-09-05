"""Indexer status transitions with mocked Mongo / Qdrant / embedder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag.config import Settings
from geek_crawler_rag.indexer import IndexService
from geek_crawler_rag.models import IndexState
from geek_crawler_rag.mongo import CrawlPage, CrawlRun


@pytest.mark.asyncio
async def test_index_run_not_found():
    mongo = MagicMock()
    mongo.get_run = AsyncMock(return_value=None)
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.ensure_collection = AsyncMock()
    embedder = MagicMock()
    settings = Settings(openai_api_key="test")

    svc = IndexService(mongo, store, embedder, settings)
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
    store.upsert = AsyncMock()
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[])
    settings = Settings(openai_api_key="test")

    svc = IndexService(mongo, store, embedder, settings)
    await svc._index_run("r1")

    status = await svc.get_status("r1")
    assert status is not None
    assert status.state == IndexState.SKIPPED
    assert status.mongo_page_count == 1
    assert status.pages_skipped_lang >= 1
    store.upsert.assert_not_called()


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
    store.upsert = AsyncMock()

    async def fake_embed(texts):
        return [[0.1] * 8 for _ in texts]

    embedder = MagicMock()
    embedder.embed = AsyncMock(side_effect=fake_embed)
    settings = Settings(openai_api_key="test", embed_batch_size=10)

    svc = IndexService(mongo, store, embedder, settings)
    await svc._index_run("r2")

    status = await svc.get_status("r2")
    assert status is not None
    assert status.state == IndexState.COMPLETE
    assert status.chunks_upserted >= 1
    assert status.crawl_type == "competitors"
    store.delete_by_run_id.assert_awaited_once_with("r2")
    assert store.upsert.await_count >= 1


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
    store.upsert = AsyncMock(side_effect=RuntimeError("qdrant down"))
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[0.1] * 8])
    settings = Settings(openai_api_key="test", embed_batch_size=10)

    svc = IndexService(mongo, store, embedder, settings)
    await svc._index_run("r3")

    status = await svc.get_status("r3")
    assert status is not None
    assert status.state == IndexState.FAILED
    assert store.delete_by_run_id.await_count >= 2


@pytest.mark.asyncio
async def test_enqueue_dedupes_pending():
    mongo = MagicMock()
    store = MagicMock()
    embedder = MagicMock()
    settings = Settings(openai_api_key="test")
    svc = IndexService(mongo, store, embedder, settings)
    first = await svc.enqueue("same")
    second = await svc.enqueue("same")
    assert first is second
    assert svc._queue.qsize() == 1
