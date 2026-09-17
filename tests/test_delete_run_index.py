"""Deleting a run's index takes its job row with it.

The row outliving the vectors is what let a deleted run keep answering
GET /v1/index/{runId} with ``complete`` and the page and chunk counts of a
corpus that no longer existed.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from geek_crawler_rag.app import delete_run_index, state


@pytest.fixture
def wired_state(monkeypatch):
    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    status_store = MagicMock()
    status_store.delete = AsyncMock(return_value=True)
    settings = MagicMock()
    settings.crawler_owner_id = "owner"
    settings.crawler_visibility = "private"

    monkeypatch.setattr(state, "store", store, raising=False)
    monkeypatch.setattr(state, "status_store", status_store, raising=False)
    monkeypatch.setattr(state, "settings", settings, raising=False)
    return store, status_store


@pytest.mark.asyncio
async def test_delete_removes_vectors_and_the_job_row(wired_state):
    store, status_store = wired_state

    await delete_run_index("run-1")

    store.delete_by_run_id.assert_awaited_once_with(
        "run-1", owner_id="owner", visibility="private"
    )
    status_store.delete.assert_awaited_once_with("run-1")


@pytest.mark.asyncio
async def test_absent_job_row_is_not_an_error(wired_state):
    # Deleting an already-absent run is a no-op, not a failure.
    _, status_store = wired_state
    status_store.delete = AsyncMock(return_value=False)

    await delete_run_index("never-indexed")

    status_store.delete.assert_awaited_once_with("never-indexed")


@pytest.mark.asyncio
async def test_blank_run_id_touches_nothing(wired_state):
    store, status_store = wired_state

    with pytest.raises(HTTPException):
        await delete_run_index("   ")

    store.delete_by_run_id.assert_not_awaited()
    status_store.delete.assert_not_awaited()
