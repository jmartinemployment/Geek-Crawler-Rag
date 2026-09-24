"""A dropped Qdrant collection reports hosts unindexed instead of raising.

On 2026-09-24 the `geek_crawler_chunks` collection was deleted out from under a running API. Every
`/v1/index/hosts` call then raised out of the scroll, FastAPI turned it into a 500, and GeekAPI
turned that into a 502 the frontend retried forever. With no index, no host is indexed -- which is
what this endpoint already has a way to say.
"""

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from geek_crawler_rag.qdrant_store import QdrantStore, _is_missing_collection


def _missing_collection_error(collection: str = "geek_crawler_chunks") -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=404,
        reason_phrase="Not Found",
        content=(
            '{"status":{"error":"Not found: Collection `' + collection + "` doesn't exist!\"}}"
        ).encode("utf-8"),
        headers=None,
    )


class _ScrollStub:
    """Stands in for AsyncQdrantClient: one scroll call, one predetermined outcome."""

    def __init__(self, outcome):
        self._outcome = outcome
        self.calls = 0

    async def scroll(self, **kwargs):
        self.calls += 1
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def _store_with(outcome) -> QdrantStore:
    store = QdrantStore.__new__(QdrantStore)
    store._client = _ScrollStub(outcome)
    store._collection = "geek_crawler_chunks"
    store._vector_size = 1536
    return store


def test_the_404_body_qdrant_actually_sent_is_recognised():
    assert _is_missing_collection(_missing_collection_error(), "geek_crawler_chunks")


def test_a_message_only_error_is_recognised_without_a_status():
    # A transport that wraps the error can keep the text and lose the status code.
    exc = RuntimeError("Collection `geek_crawler_chunks` doesn't exist!")
    assert _is_missing_collection(exc, "geek_crawler_chunks")


def test_an_unrelated_error_is_not_mistaken_for_a_missing_collection():
    assert not _is_missing_collection(RuntimeError("connection reset by peer"), "geek_crawler_chunks")


@pytest.mark.asyncio
async def test_a_missing_collection_reports_unindexed_rather_than_raising():
    store = _store_with(_missing_collection_error())
    assert await store.find_host_index_payload("medius.com") is None
    assert store._client.calls == 1


@pytest.mark.asyncio
async def test_any_other_failure_also_fails_closed():
    # No fallback path and no exception: the caller gets the same clean "not indexed" state.
    store = _store_with(RuntimeError("connection reset by peer"))
    assert await store.find_host_index_payload("medius.com") is None


@pytest.mark.asyncio
async def test_a_live_collection_still_returns_the_payload():
    class _Point:
        payload = {"runId": "r1", "host": "medius.com"}

    store = _store_with(([_Point()], None))
    assert await store.find_host_index_payload("medius.com") == {
        "runId": "r1",
        "host": "medius.com",
    }


@pytest.mark.asyncio
async def test_an_empty_collection_returns_none_without_touching_qdrant_twice():
    store = _store_with(([], None))
    assert await store.find_host_index_payload("medius.com") is None
    assert store._client.calls == 1


@pytest.mark.asyncio
async def test_an_empty_host_never_reaches_qdrant():
    store = _store_with(_missing_collection_error())
    assert await store.find_host_index_payload("") is None
    assert store._client.calls == 0
