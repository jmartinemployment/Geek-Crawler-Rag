"""An unusable page must 404 on the citation read, not just be skipped at index time.

`/v1/pages/{pageId}?runId=` is the read side of citation verification: GeekAPI's
`GccV2PartnerExtractionVerify.VerifyAgainstLibraryAsync` fetches text here, matches a
model's quote against it, and stamps the citation verified.

Until 2026-09-29 the only checks were "does the page exist, does the run match, is the text
non-empty". A 4xx error page passes all three — its body ("Sorry, we could not find that
page. Try our blog.") clears every prose floor the pipeline has. So a page indexed before
the reject gate existed could be retrieved by `/v1/query`, quoted, confirmed against this
endpoint, and stamped `QuoteVerified=true` on a citation to a URL the server said it did not
serve. Two such pages were in the live corpus when this was found.

Rejecting at index time does not cover it, for two independent reasons: points already in
Qdrant carry no status in their payload, so retrieval cannot filter them; and this endpoint
reads Mongo directly, so it never consults the index at all.

`no_content` is deliberately NOT a refusal reason here — an empty body is already caught by
the `not text` check, and this service re-adjudicating "readable" is what destroyed 5,274
pages on 2026-09-18.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from geek_crawler_rag.app import get_page_text, get_page_text_by_url, state
from geek_crawler_rag.mongo import CrawlPage

PROSE = [{"kind": "paragraph", "text": "Sorry, we could not find that page. Try our blog."}]
GOOD = [{"kind": "paragraph", "text": "Pay vendors from one place with approval routing."}]


def _page(**over) -> CrawlPage:
    base = dict(
        id="page-1",
        run_id="run-1",
        origin="https://example.com",
        url="https://example.com/gone",
        final_url="https://example.com/gone",
        html=None,
        content_html="<p>x</p>",
        blocks=PROSE,
        title="Not found",
        crawled_at=None,
        failure_reason=None,
        robots_allowed=True,
        status_code=404,
    )
    base.update(over)
    return CrawlPage(**base)  # type: ignore[arg-type]


def _wire(monkeypatch, page: CrawlPage | None):
    mongo = MagicMock()
    mongo.get_page = AsyncMock(return_value=page)
    mongo.get_page_by_url = AsyncMock(return_value=page)
    monkeypatch.setattr(state, "mongo", mongo, raising=False)


@pytest.mark.parametrize("code", [400, 401, 403, 404, 410, 429, 500, 502, 503])
@pytest.mark.asyncio
async def test_an_error_page_is_not_citable_by_id(code, monkeypatch):
    """The case that let a 404's prose be verified as a citation."""
    _wire(monkeypatch, _page(status_code=code))
    with pytest.raises(HTTPException) as exc:
        await get_page_text("page-1", "run-1")
    assert exc.value.status_code == 404
    assert "not citable" in str(exc.value.detail)


@pytest.mark.parametrize("code", [400, 403, 404, 500])
@pytest.mark.asyncio
async def test_an_error_page_is_not_citable_by_url(code, monkeypatch):
    _wire(monkeypatch, _page(status_code=code))
    with pytest.raises(HTTPException) as exc:
        await get_page_text_by_url("run-1", "https://example.com/gone")
    assert exc.value.status_code == 404
    assert "not citable" in str(exc.value.detail)


@pytest.mark.parametrize("code", [200, 201, 204, 301, 302, 304])
@pytest.mark.asyncio
async def test_a_served_page_is_still_readable(code, monkeypatch):
    # The gate must not have broken the path it guards.
    _wire(monkeypatch, _page(status_code=code, blocks=GOOD, url="https://example.com/real"))
    out = await get_page_text("page-1", "run-1")
    assert out.page_id == "page-1"
    assert "Pay vendors" in out.text


@pytest.mark.asyncio
async def test_an_unknown_status_is_still_readable(monkeypatch):
    # status_code=None means "could not be read", not "was an error". Refusing on unknown
    # would make every row written before the field existed uncitable.
    _wire(monkeypatch, _page(status_code=None, blocks=GOOD))
    out = await get_page_text("page-1", "run-1")
    assert "Pay vendors" in out.text


@pytest.mark.asyncio
async def test_a_robots_denied_page_is_not_citable(monkeypatch):
    _wire(monkeypatch, _page(status_code=200, robots_allowed=False, blocks=GOOD))
    with pytest.raises(HTTPException) as exc:
        await get_page_text("page-1", "run-1")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_a_failed_fetch_is_not_citable(monkeypatch):
    _wire(monkeypatch, _page(status_code=200, failure_reason="request timed out", blocks=GOOD))
    with pytest.raises(HTTPException) as exc:
        await get_page_text("page-1", "run-1")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_a_wrong_run_still_404s_before_the_gate(monkeypatch):
    # Authorization first: a caller must not learn a page is "not citable" for a run it does
    # not own, because that leaks that the page exists.
    _wire(monkeypatch, _page(status_code=404))
    with pytest.raises(HTTPException) as exc:
        await get_page_text("page-1", "other-run")
    assert "No text for the authorized page" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_an_empty_body_is_refused_as_no_text_not_as_unusable(monkeypatch):
    # no_content is excluded from the gate on purpose; `not text` already covers it, and this
    # service re-adjudicating readability is what destroyed 5,274 pages.
    _wire(monkeypatch, _page(status_code=200, blocks=[]))
    with pytest.raises(HTTPException) as exc:
        await get_page_text("page-1", "run-1")
    assert "No text for the authorized page" in str(exc.value.detail)
    assert "not citable" not in str(exc.value.detail)
