"""Reranker outcomes: disabled, ranked, and failed are three different answers."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from geek_crawler_rag import rerank as rerank_module
from geek_crawler_rag.rerank import Reranker


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    """Stands in for httpx.AsyncClient so no test reaches the network."""

    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._payload = payload or {}
        self._raises = raises

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def post(self, *_args: object, **_kwargs: object) -> _FakeResponse:
        if self._raises is not None:
            raise self._raises
        return _FakeResponse(self._payload)


def _install(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    monkeypatch.setattr(
        rerank_module.httpx, "AsyncClient", lambda *a, **k: client, raising=True
    )


@pytest.mark.asyncio
async def test_disabled_keeps_order_and_says_it_is_not_ranked():
    # Soft-disable is a configured choice, not a failure. Positional order is
    # its documented behaviour, and the outcome says the scores are positional
    # so a caller never reports them as measured relevance.
    r = Reranker(None, enabled=True)
    assert r.enabled is False

    out = await r.rerank("q", ["a", "b", "c"], top_n=2)

    assert out.order == [(0, 3.0), (1, 2.0)]
    assert out.ranked is False
    assert out.failed is False


@pytest.mark.asyncio
async def test_no_documents_is_not_a_failure():
    r = Reranker("key", enabled=True)
    out = await r.rerank("q", [], top_n=5)
    assert out.order == []
    assert out.ranked is False
    assert out.failed is False


@pytest.mark.asyncio
async def test_a_successful_call_is_marked_ranked(monkeypatch: pytest.MonkeyPatch):
    _install(
        monkeypatch,
        _FakeClient(
            payload={
                "results": [
                    {"index": 2, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                ]
            }
        ),
    )
    r = Reranker("key", enabled=True)

    out = await r.rerank("q", ["a", "b", "c"], top_n=2)

    assert out.order == [(2, 0.91), (0, 0.42)]
    assert out.ranked is True
    assert out.failed is False


@pytest.mark.asyncio
async def test_transport_failure_returns_no_ranking(monkeypatch: pytest.MonkeyPatch):
    # The defect this replaces: a failure used to return positional scores that
    # were indistinguishable from measured ones, silently changing which chunks
    # reached the model as grounding evidence.
    _install(monkeypatch, _FakeClient(raises=httpx.ConnectError("no route")))
    r = Reranker("key", enabled=True)

    out = await r.rerank("q", ["a", "b", "c"], top_n=3)

    assert out.failed is True
    assert out.ranked is False
    assert out.order == []


@pytest.mark.asyncio
async def test_unparseable_results_are_a_failure_not_an_order(
    monkeypatch: pytest.MonkeyPatch,
):
    # The call succeeded and returned nothing usable. Asked, and no ranking came
    # back, which is the same class of outcome as the transport failing.
    _install(
        monkeypatch,
        _FakeClient(payload={"results": [{"no_index": True}, {"index": "abc"}]}),
    )
    r = Reranker("key", enabled=True)

    out = await r.rerank("q", ["a", "b", "c"], top_n=3)

    assert out.failed is True
    assert out.ranked is False
    assert out.order == []


@pytest.mark.asyncio
async def test_only_a_real_reranker_call_yields_ranked_true(
    monkeypatch: pytest.MonkeyPatch,
):
    # The property that matters downstream: ranked=True means the scores came
    # from the reranker, so rerank_score can be trusted to mean measured
    # relevance and never positional filler.
    ranked_orders = []
    for client in (
        _FakeClient(raises=httpx.ConnectError("down")),
        _FakeClient(payload={"results": []}),
        _FakeClient(payload={"results": [{"index": 1, "relevance_score": 0.7}]}),
    ):
        _install(monkeypatch, client)
        out = await Reranker("key", enabled=True).rerank("q", ["a", "b"], top_n=2)
        if out.ranked:
            ranked_orders.append(out.order)
        else:
            assert out.order == []

    assert ranked_orders == [[(1, 0.7)]]
