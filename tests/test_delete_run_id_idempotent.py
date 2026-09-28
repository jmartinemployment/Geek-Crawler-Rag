"""A dropped collection is not a purge failure. A real failure still is.

On 2026-09-24 `geek_crawler_chunks` was deleted out from under a running API. Every
`DELETE /v1/index/runs/{id}` then raised out of the Qdrant call, FastAPI made it a 500, and
GeekAPI made that a 502 -- on the PATCH that had just published a *different* run. Eleven
crawls were recorded failed locally for a purge that had nothing to do with them.

The diagnosis at the time blamed "deleting points for a run whose points no longer
existed". That case was never the problem: a Qdrant filter delete matching nothing returns
success, verified against the live collection on 2026-09-28. Fixing only that would have
prevented none of the eleven. What raised was the *absent collection*, which is what this
covers.

The other half matters just as much: anything that is not a missing collection must still
fail. GeekAPI deletes the pages a run's vectors cite once this reports success, and
orphaned vectors are reachable -- `/v1/index/hosts` resolves a host to a runId by scrolling
Qdrant on host alone, and `GccGroundingResolver` feeds that runId straight into a query.
"""

from __future__ import annotations

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from geek_crawler_rag.qdrant_store import QdrantStore


def _missing_collection_error(collection: str = "geek_crawler_chunks") -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=404,
        reason_phrase="Not Found",
        content=(
            '{"status":{"error":"Not found: Collection `' + collection + "` doesn't exist!\"}}"
        ).encode("utf-8"),
        headers=None,
    )


class _DeleteStub:
    """Stands in for AsyncQdrantClient: one delete call, one predetermined outcome."""

    def __init__(self, outcome=None):
        self._outcome = outcome
        self.calls = 0
        self.last_kwargs: dict = {}

    async def delete(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def _store_with(outcome=None) -> QdrantStore:
    store = QdrantStore.__new__(QdrantStore)
    store._client = _DeleteStub(outcome)
    store._collection = "geek_crawler_chunks"
    store._vector_size = 1536
    return store


@pytest.mark.asyncio
async def test_a_missing_collection_is_a_successful_purge():
    store = _store_with(_missing_collection_error())

    # No raise: with no collection there are no vectors for this run, which is the state
    # the caller asked for.
    await store.delete_by_run_id("1be09b33-54fd-4a62-81de-462293db9314")
    assert store._client.calls == 1


@pytest.mark.asyncio
async def test_a_message_only_missing_collection_is_also_success():
    # A transport wrapping the error can keep the text and lose the 404.
    store = _store_with(RuntimeError("Collection `geek_crawler_chunks` doesn't exist!"))

    await store.delete_by_run_id("some-run")
    assert store._client.calls == 1


@pytest.mark.asyncio
async def test_an_unreachable_qdrant_still_fails():
    # The half that must not become idempotent. GeekAPI deletes the pages these vectors
    # cite once this returns success, so a purge that did not happen must not report that
    # it did.
    store = _store_with(RuntimeError("connection reset by peer"))

    with pytest.raises(RuntimeError, match="connection reset by peer"):
        await store.delete_by_run_id("some-run")


@pytest.mark.asyncio
async def test_a_qdrant_500_still_fails():
    store = _store_with(
        UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            content=b'{"status":{"error":"Service internal error"}}',
            headers=None,
        )
    )

    with pytest.raises(UnexpectedResponse):
        await store.delete_by_run_id("some-run")


@pytest.mark.asyncio
async def test_a_live_collection_deletes_and_filters_on_all_three_fields():
    store = _store_with(None)

    await store.delete_by_run_id("run-1", owner_id="system:crawler", visibility="service")

    assert store._client.calls == 1
    conditions = store._client.last_kwargs["points_selector"].filter.must
    got = {c.key: c.match.value for c in conditions}
    assert got == {
        "runId": "run-1",
        "ownerId": "system:crawler",
        "visibility": "service",
    }
    # wait=True, so a caller told the purge succeeded can rely on it having landed before
    # it deletes the pages those points cite.
    assert store._client.last_kwargs["wait"] is True
