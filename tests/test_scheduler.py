from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag.config import Settings
from geek_crawler_rag.mongo import SchedulableRun
from geek_crawler_rag.scheduler import IndexScheduler


@pytest.mark.asyncio
async def test_due_scheduler_enqueues_smallest_ready_run():
    mongo = MagicMock()
    mongo.find_smallest_markdown_ready_run = AsyncMock(
        return_value=SchedulableRun("small-run", 12)
    )
    store = MagicMock()
    store.claim_scheduler_due = AsyncMock(return_value=True)
    store.has_active_job = AsyncMock(return_value=False)
    store.excluded_run_ids = AsyncMock(return_value={"done-run"})
    store.complete_scheduler_tick = AsyncMock()
    indexer = MagicMock()
    indexer.enqueue_scheduled = AsyncMock(return_value=True)
    settings = Settings(
        openai_api_key="test",
        index_scheduler_enabled=True,
        index_scheduler_interval_seconds=7200,
    )
    scheduler = IndexScheduler(
        mongo, store, indexer, settings, owner="scheduler-owner"
    )

    selected = await scheduler.tick()

    assert selected == "small-run"
    mongo.find_smallest_markdown_ready_run.assert_awaited_once_with(
        excluded_run_ids={"done-run"}
    )
    indexer.enqueue_scheduled.assert_awaited_once_with("small-run")
    store.complete_scheduler_tick.assert_awaited_once_with(
        owner="scheduler-owner",
        interval_seconds=7200,
        run_id="small-run",
        error=None,
    )


@pytest.mark.asyncio
async def test_not_due_scheduler_does_not_scan_corpus():
    mongo = MagicMock()
    store = MagicMock()
    store.claim_scheduler_due = AsyncMock(return_value=False)
    indexer = MagicMock()
    settings = Settings(openai_api_key="test", index_scheduler_enabled=True)
    scheduler = IndexScheduler(mongo, store, indexer, settings, owner="owner")

    assert await scheduler.tick() is None
    assert not mongo.find_smallest_markdown_ready_run.called
