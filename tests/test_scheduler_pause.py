"""Pausing the index scheduler stops the feed, not the work already in flight.

There was no pause. The only lever was `INDEX_SCHEDULER_ENABLED`, which is read at
startup, so "stop scheduling" meant editing the environment and restarting -- and a
restart empties the in-process worker queue, which is the thing an operator pausing
mid-incident is trying not to do. So the lever that existed could only be pulled by
doing the damage it was being pulled to avoid.

Two properties are load-bearing and each has a way of quietly not holding:

* `enabled` is re-synced from config on every `scheduler_status` read
  (`status_store.scheduler_status`). A pause expressed by flipping `enabled` would be
  reverted by the next status call, which every tick makes.
* The read and the claim are two round trips, so checking `paused` before the claim is
  not enough on its own -- the filter has to carry it too.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag.config import Settings
from geek_crawler_rag.scheduler import IndexScheduler
from geek_crawler_rag.status_store import (
    SCHEDULER_ID,
    IndexStatusStore,
    _scheduler_from_doc,
)

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def _store_over(doc: dict | None) -> tuple[IndexStatusStore, MagicMock]:
    """A store whose scheduler collection is one mutable document."""
    held: dict = {"doc": doc}

    async def find_one(_query, *_a, **_k):
        return dict(held["doc"]) if held["doc"] is not None else None

    async def update_one(_query, update, **kwargs):
        if held["doc"] is None:
            if kwargs.get("upsert"):
                held["doc"] = dict(update.get("$setOnInsert") or {})
            return MagicMock(matched_count=0)
        held["doc"].update(update.get("$set") or {})
        return MagicMock(matched_count=1)

    col = MagicMock()
    col.find_one = AsyncMock(side_effect=find_one)
    col.update_one = AsyncMock(side_effect=update_one)
    col.find_one_and_update = AsyncMock(return_value=None)
    db = MagicMock()
    db.__getitem__ = MagicMock(side_effect=lambda _name: col)
    return IndexStatusStore(db), col


def test_pause_fields_are_read_off_the_document():
    status = _scheduler_from_doc(
        {
            "enabled": True,
            "intervalSeconds": 7200,
            "paused": True,
            "pausedAtUtc": datetime(2026, 9, 28, 12, 0),
            "pausedReason": "Qdrant collection being rebuilt",
        },
        enabled=True,
        interval_seconds=7200,
    )

    assert status.paused is True
    assert status.paused_reason == "Qdrant collection being rebuilt"
    # Naive out of Mongo, tz-aware out of here, same as nextRunAtUtc.
    assert status.paused_at_utc is not None
    assert status.paused_at_utc.tzinfo == timezone.utc


def test_an_unpaused_document_reports_not_paused():
    status = _scheduler_from_doc(
        {"enabled": True, "intervalSeconds": 7200}, enabled=True, interval_seconds=7200
    )

    assert status.paused is False
    assert status.paused_at_utc is None
    assert status.paused_reason is None


@pytest.mark.asyncio
async def test_pause_records_the_reason_and_resume_clears_it():
    store, _col = _store_over(
        {"_id": SCHEDULER_ID, "enabled": True, "intervalSeconds": 7200}
    )

    paused = await store.set_scheduler_paused(
        paused=True, reason="Qdrant rebuild", enabled=True, interval_seconds=7200
    )
    assert paused.paused is True
    assert paused.paused_reason == "Qdrant rebuild"
    assert paused.paused_at_utc is not None

    resumed = await store.set_scheduler_paused(
        paused=False, reason=None, enabled=True, interval_seconds=7200
    )
    assert resumed.paused is False
    assert resumed.paused_reason is None
    assert resumed.paused_at_utc is None


@pytest.mark.asyncio
async def test_a_config_resync_does_not_lift_a_pause():
    # The path that re-syncs `enabled` from the env var runs on every status read. If it
    # carried `paused`, the pause would last until the next tick and no longer.
    store, _col = _store_over(
        {"_id": SCHEDULER_ID, "enabled": False, "intervalSeconds": 7200, "paused": True,
         "pausedReason": "held"}
    )

    # enabled differs from the stored value, so this takes the re-sync branch.
    status = await store.scheduler_status(enabled=True, interval_seconds=3600)

    assert status.enabled is True
    assert status.interval_seconds == 3600
    assert status.paused is True
    assert status.paused_reason == "held"


@pytest.mark.asyncio
async def test_pausing_a_scheduler_that_never_ticked_creates_the_document():
    store, _col = _store_over(None)

    status = await store.set_scheduler_paused(
        paused=True, reason="pre-emptive", enabled=True, interval_seconds=7200
    )

    assert status.paused is True
    assert status.paused_reason == "pre-emptive"


@pytest.mark.asyncio
async def test_a_paused_scheduler_never_reaches_the_claim():
    store, col = _store_over(
        {
            "_id": SCHEDULER_ID,
            "enabled": True,
            "intervalSeconds": 7200,
            "nextRunAtUtc": NOW - timedelta(hours=3),
            "paused": True,
        }
    )

    claimed = await store.claim_scheduler_due(
        owner="owner", lease_seconds=900, enabled=True, interval_seconds=7200
    )

    assert claimed is False
    # Due, and it still did not try: no lease was taken.
    col.find_one_and_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_claim_filter_carries_paused_so_a_race_cannot_slip_through():
    store, col = _store_over(
        {
            "_id": SCHEDULER_ID,
            "enabled": True,
            "intervalSeconds": 7200,
            "nextRunAtUtc": NOW - timedelta(hours=3),
            "paused": False,
        }
    )
    captured: dict = {}

    async def capture(query, _update, **_kwargs):
        captured["query"] = query
        return None

    col.find_one_and_update = AsyncMock(side_effect=capture)

    await store.claim_scheduler_due(
        owner="owner", lease_seconds=900, enabled=True, interval_seconds=7200
    )

    assert captured["query"]["paused"] == {"$ne": True}


@pytest.mark.asyncio
async def test_an_unpaused_due_scheduler_still_claims():
    # The pause must not have made the ordinary path unreachable.
    store, col = _store_over(
        {
            "_id": SCHEDULER_ID,
            "enabled": True,
            "intervalSeconds": 7200,
            "nextRunAtUtc": NOW - timedelta(hours=3),
            "paused": False,
        }
    )
    col.find_one_and_update = AsyncMock(return_value={"_id": SCHEDULER_ID})

    claimed = await store.claim_scheduler_due(
        owner="owner", lease_seconds=900, enabled=True, interval_seconds=7200
    )

    assert claimed is True


@pytest.mark.asyncio
async def test_scheduler_pause_and_resume_pass_config_through():
    store = MagicMock()
    store.set_scheduler_paused = AsyncMock(return_value=MagicMock())
    settings = Settings(
        openai_api_key="test",
        index_scheduler_enabled=True,
        index_scheduler_interval_seconds=7200,
    )
    scheduler = IndexScheduler(
        MagicMock(), store, MagicMock(), settings, owner="owner"
    )

    await scheduler.pause("Qdrant rebuild")
    store.set_scheduler_paused.assert_awaited_once_with(
        paused=True,
        reason="Qdrant rebuild",
        enabled=True,
        interval_seconds=7200,
    )

    store.set_scheduler_paused.reset_mock()
    await scheduler.resume()
    store.set_scheduler_paused.assert_awaited_once_with(
        paused=False,
        reason=None,
        enabled=True,
        interval_seconds=7200,
    )


@pytest.mark.asyncio
async def test_a_paused_tick_scans_nothing_and_enqueues_nothing():
    mongo = MagicMock()
    mongo.find_oldest_content_ready_run = AsyncMock()
    store = MagicMock()
    store.claim_scheduler_due = AsyncMock(return_value=False)
    store.complete_scheduler_tick = AsyncMock()
    indexer = MagicMock()
    indexer.enqueue_scheduled = AsyncMock()
    settings = Settings(openai_api_key="test", index_scheduler_enabled=True)
    scheduler = IndexScheduler(mongo, store, indexer, settings, owner="owner")

    assert await scheduler.tick() is None
    mongo.find_oldest_content_ready_run.assert_not_awaited()
    indexer.enqueue_scheduled.assert_not_awaited()
    # No tick was consumed either, so nextRunAtUtc is not pushed out by the pause.
    store.complete_scheduler_tick.assert_not_awaited()
