"""A host is not a run: `/v1/index/hosts` resolves `(host, crawlType)`, and refuses to guess.

tipalti.com, 2026-10-08. Two complete, indexed runs for one host: `39bbce59` crawled as a partner
on 10-05, `5cbbb85b` as a competitor on 10-06. The route looked up the host alone, answered the
competitor run, GeekAPI probed it with `crawlType: partner`, matched nothing, and excluded the
partner from the project with "its crawl finished, but a search of the index finds nothing from
it". The partner run answered the same probe with 32 passages; it was never asked.

These drive the route function with a fake store that holds exactly that shape.
"""

from __future__ import annotations

import pytest

from geek_crawler_rag.app import host_index_exists, state
from geek_crawler_rag.models import HostIndexRequest

PARTNER_RUN = "39bbce59-partner"
COMPETITOR_RUN = "5cbbb85b-competitors"


class _Store:
    """Points as (host, crawlType) -> runId. One host may carry several types."""

    def __init__(self, runs: dict[tuple[str, str], str]) -> None:
        self._runs = runs
        self.calls: list[tuple[str, str | None]] = []

    async def find_host_crawl_types(self, host: str) -> list[str] | None:
        return sorted({t for (h, t) in self._runs if h == host})

    async def find_host_index_payload(self, host: str, crawl_type: str | None = None):
        self.calls.append((host, crawl_type))
        for (h, t), run_id in self._runs.items():
            if h == host and (crawl_type is None or t == crawl_type):
                return {"runId": run_id, "host": h, "crawlType": t}
        return None


@pytest.fixture
def tipalti(monkeypatch) -> _Store:
    store = _Store(
        {
            ("tipalti.com", "partner"): PARTNER_RUN,
            ("tipalti.com", "competitors"): COMPETITOR_RUN,
            ("www.stampli.com", "partner"): "ab551881-partner",
        }
    )
    # `state` is Starlette's; `store` is set at startup, so it may not exist yet under pytest.
    monkeypatch.setattr(state, "store", store, raising=False)
    return store


async def test_a_typed_partner_request_gets_the_partner_run(tipalti) -> None:
    response = await host_index_exists(
        HostIndexRequest(urls=["https://tipalti.com/"], crawlType="partner")
    )
    [result] = response.results
    assert result.indexed is True
    assert result.run_id == PARTNER_RUN
    assert result.crawl_type == "partner"
    assert result.host == "tipalti.com"
    assert tipalti.calls == [("tipalti.com", "partner")]


async def test_a_typed_competitor_request_gets_the_competitor_run(tipalti) -> None:
    response = await host_index_exists(
        HostIndexRequest(urls=["https://tipalti.com/"], crawlType="competitors")
    )
    [result] = response.results
    assert result.indexed is True
    assert result.run_id == COMPETITOR_RUN
    assert result.crawl_type == "competitors"


async def test_an_untyped_request_on_a_two_type_host_is_refused_and_names_the_types(tipalti) -> None:
    """The coin flip is gone. The caller is told what to send, not handed one of the two runs."""
    response = await host_index_exists(HostIndexRequest(urls=["https://tipalti.com/"]))
    [result] = response.results
    assert result.indexed is False
    assert result.run_id is None
    assert result.host == "tipalti.com"
    assert result.reason is not None
    assert "competitors" in result.reason and "partner" in result.reason
    assert "crawlType" in result.reason
    assert tipalti.calls == [], "nothing was looked up: an ambiguous host is refused before any guess"


async def test_an_untyped_request_on_a_one_type_host_still_answers(tipalti) -> None:
    """Every host that answered before still answers: only ambiguity is new."""
    response = await host_index_exists(HostIndexRequest(urls=["https://stampli.com"]))
    [result] = response.results
    assert result.indexed is True
    assert result.run_id == "ab551881-partner"
    assert result.crawl_type == "partner"
    assert result.host == "www.stampli.com"


async def test_a_typed_request_for_a_type_the_host_was_not_crawled_as_is_not_indexed(tipalti) -> None:
    response = await host_index_exists(
        HostIndexRequest(urls=["https://stampli.com"], crawlType="competitors")
    )
    [result] = response.results
    assert result.indexed is False
    assert result.run_id is None
    assert result.reason is None


async def test_one_request_can_mix_answered_and_refused_hosts(tipalti) -> None:
    response = await host_index_exists(
        HostIndexRequest(urls=["https://stampli.com", "https://tipalti.com/", "https://nobody.example"])
    )
    stampli, tip, nobody = response.results
    assert stampli.indexed is True
    assert tip.indexed is False and tip.reason is not None
    assert nobody.indexed is False and nobody.host is None and nobody.reason is None


def test_the_request_model_reads_crawl_type_by_alias_and_by_name() -> None:
    assert HostIndexRequest(urls=["x.com"], crawlType="partner").crawl_type == "partner"
    assert HostIndexRequest(urls=["x.com"], crawl_type="partner").crawl_type == "partner"
    assert HostIndexRequest(urls=["x.com"]).crawl_type is None
