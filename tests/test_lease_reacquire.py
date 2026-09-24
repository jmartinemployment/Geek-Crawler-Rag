"""A lease locks execution, not a place in the queue.

IndexStatusStore.reacquire takes the lease at the moment a queued job starts, which is the only
moment it means anything. It must not steal a lease another worker is actively holding, and it must
not spend an attempt on a job whose only offence was waiting.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag.models import IndexState
from geek_crawler_rag.status_store import IndexStatusStore


def _store_with(matched: int):
    captured: dict = {}

    async def fake_update_one(query, update, **kwargs):
        captured["query"] = query
        captured["update"] = update
        return MagicMock(matched_count=matched, modified_count=matched)

    col = MagicMock()
    col.update_one = AsyncMock(side_effect=fake_update_one)
    db = MagicMock()
    db.__getitem__ = MagicMock(side_effect=lambda name: col)
    return IndexStatusStore(db), captured


@pytest.mark.asyncio
async def test_a_lapsed_lease_is_retaken():
    store, _ = _store_with(matched=1)
    assert await store.reacquire("r1", owner="worker-1", lease_seconds=900) is True


@pytest.mark.asyncio
async def test_a_lease_held_by_another_live_owner_is_not_taken():
    # matched_count 0 is Mongo saying the filter excluded it -- a live foreign lease.
    store, _ = _store_with(matched=0)
    assert await store.reacquire("r1", owner="worker-1", lease_seconds=900) is False


@pytest.mark.asyncio
async def test_the_filter_accepts_expired_absent_null_or_our_own_lease():
    store, captured = _store_with(matched=1)
    await store.reacquire("r1", owner="worker-1", lease_seconds=900)

    branches = captured["query"]["$or"]
    assert {"leaseUntil": {"$exists": False}} in branches
    assert {"leaseUntil": None} in branches
    assert {"leaseOwner": "worker-1"} in branches
    assert any("$lte" in b.get("leaseUntil", {}) for b in branches if isinstance(b.get("leaseUntil"), dict))


@pytest.mark.asyncio
async def test_only_pending_or_running_jobs_are_eligible():
    # A completed or failed job must not be revived by a lease take.
    store, captured = _store_with(matched=1)
    await store.reacquire("r1", owner="worker-1", lease_seconds=900)

    assert captured["query"]["state"] == {"$in": [IndexState.PENDING, IndexState.RUNNING]}
    assert captured["query"]["runId"] == "r1"


@pytest.mark.asyncio
async def test_the_attempt_counter_is_untouched():
    # claim() increments attempt; waiting in the queue is not an attempt, and counting it would
    # spend the retry budget on jobs that never ran.
    store, captured = _store_with(matched=1)
    await store.reacquire("r1", owner="worker-1", lease_seconds=900)

    assert "$inc" not in captured["update"]
    assert set(captured["update"]) == {"$set"}


@pytest.mark.asyncio
async def test_only_the_lease_fields_are_written():
    # No state change, no trigger rewrite, no clearing of error or finishedAtUtc -- all of which
    # claim() does, and none of which belong to taking a lock.
    store, captured = _store_with(matched=1)
    before = datetime.now(timezone.utc)
    await store.reacquire("r1", owner="worker-1", lease_seconds=900)

    written = captured["update"]["$set"]
    assert set(written) == {"leaseOwner", "leaseUntil"}
    assert written["leaseOwner"] == "worker-1"
    assert written["leaseUntil"] >= before + timedelta(seconds=899)
