"""Pausing intake refuses new index jobs at both entrances, and leaves no job row.

The scheduler pause does not cover the traffic that actually fills the queue. Every one of
the fifteen jobs in flight on 2026-09-28 was `trigger=manual` -- GeekAPI posting a run as
each crawl committed -- and `POST /v1/index` is deliberately outside the scheduler pause's
scope. So a second, wider gate was needed, and it has to hold on both paths: a gate only
one of two entrances honours is not a gate.

The refusal must also create nothing. A job row for a run the service declined to accept is
a row that never reaches a terminal state, which is the "pending forever" outcome the job
rules exist to rule out.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag.config import Settings
from geek_crawler_rag.indexer import IndexService
from geek_crawler_rag.status_store import INTAKE_ID, IndexStatusStore


def _store_over(doc: dict | None) -> tuple[IndexStatusStore, MagicMock]:
    held: dict = {"doc": doc}

    async def find_one(_query, *_a, **_k):
        return dict(held["doc"]) if held["doc"] is not None else None

    async def update_one(_query, update, **kwargs):
        if held["doc"] is None:
            held["doc"] = {"_id": INTAKE_ID}
        held["doc"].update(update.get("$set") or {})
        return MagicMock(matched_count=1)

    col = MagicMock()
    col.find_one = AsyncMock(side_effect=find_one)
    col.update_one = AsyncMock(side_effect=update_one)
    db = MagicMock()
    db.__getitem__ = MagicMock(side_effect=lambda _name: col)
    return IndexStatusStore(db), col


def _service(status_store) -> IndexService:
    svc = IndexService(
        mongo=MagicMock(),
        store=MagicMock(),
        settings=Settings(openai_api_key="test"),
        llama=MagicMock(),
        status_store=status_store,
    )
    svc._persist = AsyncMock(return_value=True)  # type: ignore[method-assign]
    return svc


@pytest.mark.asyncio
async def test_no_document_means_intake_is_open():
    store, _ = _store_over(None)
    assert await store.intake_pause_reason() is None


@pytest.mark.asyncio
async def test_pause_then_resume_round_trip():
    store, _ = _store_over(None)

    paused, reason = await store.set_intake_paused(paused=True, reason="re-crawling")
    assert paused is True and reason == "re-crawling"
    assert await store.intake_pause_reason() == "re-crawling"

    paused, reason = await store.set_intake_paused(paused=False, reason=None)
    assert paused is False and reason is None
    assert await store.intake_pause_reason() is None


@pytest.mark.asyncio
async def test_a_pause_with_no_recorded_reason_still_reads_as_paused():
    # Never report "open" just because the reason went missing -- that would turn a lost
    # field into an open door.
    store, _ = _store_over({"_id": INTAKE_ID, "paused": True})
    assert await store.intake_pause_reason() == "no reason recorded"


@pytest.mark.asyncio
async def test_the_scheduled_path_is_refused_and_creates_no_row():
    store, _ = _store_over({"_id": INTAKE_ID, "paused": True, "pausedReason": "re-crawling"})
    store.claim = AsyncMock()
    store.get = AsyncMock(return_value=None)
    svc = _service(store)

    accepted = await svc.enqueue_scheduled("some-run")

    assert accepted is False
    # Nothing claimed, nothing queued: no row, and no entry the worker would later find.
    store.claim.assert_not_awaited()
    assert svc._queue.qsize() == 0


@pytest.mark.asyncio
async def test_the_scheduled_path_still_works_when_intake_is_open():
    store, _ = _store_over(None)
    svc = _service(store)
    svc._enqueue = AsyncMock(return_value=(MagicMock(), True))  # type: ignore[method-assign]

    assert await svc.enqueue_scheduled("some-run") is True
    svc._enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_service_reports_the_reason_for_the_route_to_refuse_with():
    store, _ = _store_over({"_id": INTAKE_ID, "paused": True, "pausedReason": "re-crawling"})
    svc = _service(store)

    assert await svc.intake_pause_reason() == "re-crawling"


@pytest.mark.asyncio
async def test_without_a_status_store_intake_is_open():
    # The no-persistence configuration used in tests must not be gated by state it has no
    # way to read.
    svc = _service(None)
    assert await svc.intake_pause_reason() is None
