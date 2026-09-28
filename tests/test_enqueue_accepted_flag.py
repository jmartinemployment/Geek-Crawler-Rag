"""A refused enqueue must not pass for an accepted one.

`claim(force=True)` correctly refuses a run that is already pending/running under a live
lease -- but `_enqueue` used to discard its accepted flag, so the refusal came back as
HTTP 200 carrying the stale row. On 2026-09-26 a re-post that read as successful
(200, state=pending) had queued nothing; the only tell was that `attempt` had not moved.
These pin the flag so nobody has to infer it again.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from geek_crawler_rag.app import start_index, state
from geek_crawler_rag.models import IndexRunRequest, IndexState, IndexStatusResponse


def _wire(
    monkeypatch,
    status: IndexStatusResponse,
    accepted: bool,
    *,
    intake_pause_reason: str | None = None,
) -> MagicMock:
    indexer = MagicMock()
    indexer.enqueue = AsyncMock(return_value=(status, accepted))
    # The route checks intake before claiming, so this has to be awaitable here; open by
    # default, because these cases are about the accepted flag and not about the gate.
    indexer.intake_pause_reason = AsyncMock(return_value=intake_pause_reason)
    monkeypatch.setattr(state, "indexer", indexer, raising=False)
    return indexer


@pytest.mark.asyncio
async def test_a_paused_intake_refuses_rather_than_reporting_not_accepted(monkeypatch):
    """503 and no claim -- not a 200 carrying accepted=false.

    A 200 would be read as "queued" by anything checking only the status code, and the run
    would go silently unindexed. GeekAPI's EnqueueIndexAsync fails closed on a non-2xx, so
    the refusal is what makes the loss visible.
    """
    fresh = IndexStatusResponse(runId="blocked-run", state=IndexState.PENDING, attempt=1)
    indexer = _wire(monkeypatch, fresh, accepted=True, intake_pause_reason="re-crawling")

    with pytest.raises(HTTPException) as caught:
        await start_index(IndexRunRequest(runId="blocked-run"))

    assert caught.value.status_code == 503
    assert "re-crawling" in str(caught.value.detail)
    # No row created for a run that was never accepted.
    indexer.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_refused_enqueue_reports_accepted_false(monkeypatch):
    existing = IndexStatusResponse(
        runId="held-run", state=IndexState.RUNNING, attempt=3, chunksUpserted=4387
    )
    _wire(monkeypatch, existing, accepted=False)

    response = await start_index(IndexRunRequest(runId="held-run"))

    assert response.accepted is False
    # The stale row still comes back -- the caller needs to see what holds the lease --
    # but it can no longer be mistaken for a fresh claim.
    assert response.state == IndexState.RUNNING
    assert response.attempt == 3
    assert response.chunks_upserted == 4387


@pytest.mark.asyncio
async def test_accepted_enqueue_reports_accepted_true(monkeypatch):
    fresh = IndexStatusResponse(runId="new-run", state=IndexState.PENDING, attempt=1)
    _wire(monkeypatch, fresh, accepted=True)

    response = await start_index(IndexRunRequest(runId="new-run"))

    assert response.accepted is True
    assert response.state == IndexState.PENDING


@pytest.mark.asyncio
async def test_accepted_is_serialized_for_the_wire(monkeypatch):
    """The flag has to survive aliasing, or a client still cannot see it."""
    fresh = IndexStatusResponse(runId="wire-run", state=IndexState.PENDING, attempt=1)
    _wire(monkeypatch, fresh, accepted=False)

    response = await start_index(IndexRunRequest(runId="wire-run"))
    payload = response.model_dump(by_alias=True)

    assert payload["accepted"] is False
    assert payload["runId"] == "wire-run"
