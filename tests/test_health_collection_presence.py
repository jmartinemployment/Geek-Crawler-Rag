"""A dropped collection has to be visible somewhere, and /health is that somewhere.

`ping()` calls `get_collections()` and returns True, so it answers "is Qdrant reachable" and
nothing else. With `geek_crawler_chunks` deleted, /health reported `qdrant: true, status: ok`
— the most damaging state in the system looking perfectly well. The collection was in fact
deleted out from under a running API on 2026-09-24.

This matters more after the purge fix, not less. Until 2026-09-28 a missing collection was at
least loud by accident: every `DELETE /v1/index/runs/{id}` raised a 500 that GeekAPI turned
into a 502. `delete_by_run_id` now treats it as a satisfied delete — correctly, because there
are no vectors for the run, which is what the caller asked — so that accidental alarm is gone.
Something had to replace it, or the trade would be a quieter system rather than a better one.

`status` deliberately does NOT flip to degraded: the container healthcheck urlopen()s this
endpoint, and the thing that recreates the collection is an index job served by this same
container, so failing health would block the only path that heals it.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag.app import health, state
from geek_crawler_rag.qdrant_store import QdrantStore


class _ExistsStub:
    def __init__(self, outcome):
        self._outcome = outcome

    async def collection_exists(self, _name):
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def _store_reporting(outcome) -> QdrantStore:
    store = QdrantStore.__new__(QdrantStore)
    store._client = _ExistsStub(outcome)
    store._collection = "geek_crawler_chunks"
    store._vector_size = 1536
    return store


@pytest.mark.asyncio
async def test_a_present_collection_reports_true():
    assert await _store_reporting(True).collection_present() is True


@pytest.mark.asyncio
async def test_a_missing_collection_reports_false():
    assert await _store_reporting(False).collection_present() is False


@pytest.mark.asyncio
async def test_an_unanswerable_check_reports_none_not_false():
    # "I could not tell" is not "it is missing". Reporting False on a transport blip would
    # raise a corpus-is-gone alarm for a dropped packet.
    assert await _store_reporting(RuntimeError("connection reset")).collection_present() is None


def _wire(monkeypatch, *, present, qdrant_ok=True, mongo_ok=True):
    mongo = MagicMock()
    mongo.ping = AsyncMock(return_value=mongo_ok)
    store = MagicMock()
    # The real ping() returns True or raises -- it has no falsy return -- so an unreachable
    # Qdrant has to be modelled as a raise, or the test proves something about a state the
    # code cannot produce.
    store.ping = (
        AsyncMock(return_value=True)
        if qdrant_ok
        else AsyncMock(side_effect=RuntimeError("connection refused"))
    )
    store.collection_present = AsyncMock(return_value=present)
    scheduler = MagicMock()
    scheduler.status = AsyncMock(return_value=MagicMock(
        model_dump=MagicMock(return_value={"enabled": False})
    ))
    llama = MagicMock()
    llama.embedding_stats = MagicMock(return_value={})
    settings = MagicMock()
    settings.qdrant_collection = "geek_crawler_chunks"

    monkeypatch.setattr(state, "mongo", mongo, raising=False)
    monkeypatch.setattr(state, "store", store, raising=False)
    monkeypatch.setattr(state, "scheduler", scheduler, raising=False)
    monkeypatch.setattr(state, "llama", llama, raising=False)
    monkeypatch.setattr(state, "settings", settings, raising=False)


async def _body(monkeypatch, **kwargs):
    _wire(monkeypatch, **kwargs)
    response = await health()
    return json.loads(response.body), response.status_code


@pytest.mark.asyncio
async def test_health_names_the_missing_collection_in_errors(monkeypatch):
    body, code = await _body(monkeypatch, present=False)

    assert body["qdrantCollection"] is False
    joined = " ".join(body["errors"])
    assert "geek_crawler_chunks" in joined
    assert "MISSING" in joined
    # Says what it means for the caller, not just that a flag is off.
    assert "every query will return nothing" in joined

    # Reachability is what `status` reports, and Qdrant is reachable. 503 here would mark the
    # container unhealthy and block the index job that recreates the collection.
    assert body["status"] == "ok"
    assert code == 200


@pytest.mark.asyncio
async def test_health_is_quiet_when_the_collection_is_there(monkeypatch):
    body, code = await _body(monkeypatch, present=True)

    assert body["qdrantCollection"] is True
    assert body["errors"] is None
    assert body["status"] == "ok"
    assert code == 200


@pytest.mark.asyncio
async def test_an_unreachable_qdrant_does_not_claim_the_collection_is_missing(monkeypatch):
    # Already degraded for the right reason; adding a corpus-is-gone alarm on top would send
    # an operator after the wrong problem.
    body, code = await _body(monkeypatch, present=None, qdrant_ok=False)

    assert body["status"] == "degraded"
    assert code == 503
    assert body["qdrantCollection"] is None
    assert "qdrant unavailable" in body["errors"]
    assert not any("MISSING" in e for e in body["errors"])
