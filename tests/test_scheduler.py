from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag.config import Settings
from geek_crawler_rag.mongo import SchedulableRun, SchedulableRunScan
from geek_crawler_rag.scheduler import IndexScheduler
from geek_crawler_rag.status_store import (
    _scheduler_from_doc,
    lease_expired_or_missing,
)


@pytest.mark.asyncio
async def test_due_scheduler_enqueues_smallest_ready_run():
    mongo = MagicMock()
    mongo.find_smallest_markdown_ready_run = AsyncMock(
        return_value=SchedulableRunScan(
            candidate=SchedulableRun("small-run", 12)
        )
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
        selection_reason="candidate runId=small-run pages=12",
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


@pytest.mark.asyncio
async def test_due_scheduler_reports_why_no_run_is_ready():
    mongo = MagicMock()
    mongo.find_smallest_markdown_ready_run = AsyncMock(
        return_value=SchedulableRunScan(
            candidate=None,
            missing_ready_marker=4,
            zero_pages=1,
            safety_cap=2,
            excluded=3,
        )
    )
    store = MagicMock()
    store.claim_scheduler_due = AsyncMock(return_value=True)
    store.has_active_job = AsyncMock(return_value=False)
    store.excluded_run_ids = AsyncMock(return_value=set())
    store.complete_scheduler_tick = AsyncMock()
    indexer = MagicMock()
    settings = Settings(openai_api_key="test", index_scheduler_enabled=True)
    scheduler = IndexScheduler(mongo, store, indexer, settings, owner="owner")

    assert await scheduler.tick() is None
    store.complete_scheduler_tick.assert_awaited_once_with(
        owner="owner",
        interval_seconds=settings.index_scheduler_interval_seconds,
        run_id=None,
        error=None,
        selection_reason=(
            "no_candidate missing_ready_marker=4 zero_pages=1 "
            "safety_cap=2 excluded=3"
        ),
    )


def test_scheduler_status_normalizes_mongo_naive_datetimes():
    status = _scheduler_from_doc(
        {"nextRunAtUtc": datetime(2026, 9, 8, 0, 0)},
        enabled=True,
        interval_seconds=7200,
    )

    assert status.next_run_at_utc is not None
    assert status.next_run_at_utc.tzinfo == timezone.utc


def test_lease_expired_or_missing_includes_null_lease():
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    clauses = lease_expired_or_missing(now)
    assert {"leaseUntil": {"$lte": now}} in clauses
    assert {"leaseUntil": {"$exists": False}} in clauses
    assert {"leaseUntil": None} in clauses


@pytest.mark.asyncio
async def test_claim_matches_pending_job_with_null_lease_until():
    from geek_crawler_rag.models import IndexState
    from geek_crawler_rag.status_store import IndexStatusStore

    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    captured: dict = {}

    async def fake_find_one_and_update(query, update, **kwargs):
        captured["query"] = query
        return {
            "runId": "null-lease-run",
            "state": IndexState.PENDING,
            "attempt": 1,
            "leaseOwner": "owner-1",
            "leaseUntil": now,
            "trigger": "manual",
        }

    col = MagicMock()
    col.find_one_and_update = AsyncMock(side_effect=fake_find_one_and_update)
    db = MagicMock()
    db.__getitem__ = MagicMock(side_effect=lambda name: col)
    store = IndexStatusStore(db)

    claimed = await store.claim(
        "null-lease-run",
        owner="owner-1",
        lease_seconds=900,
        trigger="manual",
        force=True,
    )

    assert claimed is not None
    assert claimed.run_id == "null-lease-run"
    or_clauses = captured["query"]["$or"]
    pending_branch = next(
        c
        for c in or_clauses
        if c.get("state") == {"$in": [IndexState.PENDING, IndexState.RUNNING]}
    )
    assert {"leaseUntil": None} in pending_branch["$or"]
