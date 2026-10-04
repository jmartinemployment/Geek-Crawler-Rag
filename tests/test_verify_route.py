"""`POST /v1/verify` is the one answer to "is this quote on this page".

GeekAPI's verify pass fetched text from `GET /v1/pages/{id}` and compared with its own
`IndexOf(quote, OrdinalIgnoreCase)`, while this service compared with `quote_in_text`. Two rules for
one question disagree on whitespace: a quote spanning a line break in the page is a match here and a
miss there. The route moves the comparison behind the Library, so a caller has none of its own.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from geek_crawler_rag.app import state, verify_quotes
from geek_crawler_rag.llama_nodes import page_source_digest
from geek_crawler_rag.models import VerifyQuotesRequest
from geek_crawler_rag.mongo import CrawlPage

BLOCKS = [
    {"kind": "heading", "level": 2, "text": "Bill pay"},
    {
        "kind": "paragraph",
        "text": "Pay vendors by ACH, card or check.\nApprovers release each payment.",
    },
    {"kind": "row", "cells": ["Plus", "$15/month"]},
]


def _page(**over) -> CrawlPage:
    base = dict(
        id="page-1",
        run_id="run-1",
        origin="https://ramp.com",
        url="https://ramp.com/bill-pay",
        final_url="https://ramp.com/bill-pay",
        html=None,
        content_html="<main><h2>Bill pay</h2></main>",
        blocks=BLOCKS,
        title="Bill pay",
        crawled_at=None,
        failure_reason=None,
        robots_allowed=True,
        status_code=200,
    )
    base.update(over)
    return CrawlPage(**base)  # type: ignore[arg-type]


def _wire(monkeypatch, pages: dict[str, CrawlPage]) -> MagicMock:
    mongo = MagicMock()
    mongo.get_page = AsyncMock(side_effect=lambda page_id: pages.get(page_id))
    monkeypatch.setattr(state, "mongo", mongo, raising=False)
    return mongo


def _request(*quotes: tuple[str, str], run_id: str = "run-1") -> VerifyQuotesRequest:
    return VerifyQuotesRequest(
        runId=run_id, quotes=[{"pageId": p, "quote": q} for p, q in quotes]
    )


@pytest.mark.asyncio
async def test_a_quote_on_the_page_is_found_with_the_pages_source_digest(monkeypatch):
    page = _page()
    _wire(monkeypatch, {"page-1": page})

    out = await verify_quotes(_request(("page-1", "Pay vendors by ACH, card or check.")))

    [verdict] = out.results
    assert verdict.found is True
    assert verdict.reason is None
    # The same value stamped on every point cut from this page, so a verdict can be matched to
    # the passage the quote was taken from.
    assert verdict.source_digest == page_source_digest(page)


@pytest.mark.asyncio
async def test_whitespace_and_case_follow_quote_in_text(monkeypatch):
    """The page breaks the line between the two sentences; the quote does not. One rule."""
    _wire(monkeypatch, {"page-1": _page()})

    out = await verify_quotes(
        _request(("page-1", "card or CHECK.   approvers release each payment"))
    )

    assert out.results[0].found is True


@pytest.mark.asyncio
async def test_a_short_quote_matches_a_whole_table_cell_only(monkeypatch):
    _wire(monkeypatch, {"page-1": _page()})

    out = await verify_quotes(_request(("page-1", "$15/month"), ("page-1", "month")))

    assert [r.found for r in out.results] == [True, False]


@pytest.mark.asyncio
async def test_a_quote_not_on_the_page_is_not_found(monkeypatch):
    _wire(monkeypatch, {"page-1": _page()})

    out = await verify_quotes(_request(("page-1", "Ramp syncs with QuickBooks Desktop nightly")))

    assert out.results[0].found is False
    assert out.results[0].reason == "not_on_page"


@pytest.mark.asyncio
async def test_a_page_from_another_run_is_not_found(monkeypatch):
    _wire(monkeypatch, {"page-1": _page(run_id="run-2")})

    out = await verify_quotes(_request(("page-1", "Pay vendors by ACH, card or check.")))

    assert out.results[0].found is False
    assert out.results[0].reason == "page_not_found"
    assert out.results[0].source_digest is None


@pytest.mark.asyncio
async def test_a_missing_page_is_not_found(monkeypatch):
    _wire(monkeypatch, {})

    out = await verify_quotes(_request(("nope", "Pay vendors by ACH, card or check.")))

    assert out.results[0].found is False
    assert out.results[0].reason == "page_not_found"


@pytest.mark.asyncio
async def test_an_error_page_is_not_citable_even_when_the_quote_is_on_it(monkeypatch):
    """The quote is literally in the body, and the body is the server's 404 page."""
    _wire(monkeypatch, {"page-1": _page(status_code=404)})

    out = await verify_quotes(_request(("page-1", "Pay vendors by ACH, card or check.")))

    assert out.results[0].found is False
    assert out.results[0].reason == "page_not_citable:http_error"


@pytest.mark.asyncio
async def test_each_page_is_read_once_however_many_quotes_cite_it(monkeypatch):
    mongo = _wire(monkeypatch, {"page-1": _page()})

    out = await verify_quotes(
        _request(
            ("page-1", "Pay vendors by ACH, card or check."),
            ("page-1", "Approvers release each payment."),
            ("page-1", "invented"),
        )
    )

    assert [r.found for r in out.results] == [True, True, False]
    mongo.get_page.assert_awaited_once_with("page-1")


def test_an_empty_request_is_refused():
    with pytest.raises(ValidationError):
        VerifyQuotesRequest(runId="run-1", quotes=[])
    with pytest.raises(ValidationError):
        VerifyQuotesRequest(runId="run-1", quotes=[{"pageId": "p", "quote": ""}])


@pytest.mark.asyncio
async def test_a_quote_cut_by_the_csharp_projection_from_a_row_with_empty_cells_is_found(
    monkeypatch,
):
    """F-R10 through the route: C# drops empty cells (`A | B`), Python keeps them (`A |  | B`)."""
    page = _page(blocks=[{"kind": "row", "cells": ["AI invoice capture", "", "", "Included on Plus"]}])
    _wire(monkeypatch, {"page-1": page})

    out = await verify_quotes(_request(("page-1", "AI invoice capture | Included on Plus")))

    assert out.results[0].found is True


@pytest.mark.asyncio
async def test_a_page_without_content_html_has_no_source_digest(monkeypatch):
    """One definition, sha256(contentHtml). There is no fallback to the raw html."""
    _wire(monkeypatch, {"page-1": _page(content_html=None, html="<html>raw</html>")})

    out = await verify_quotes(_request(("page-1", "Pay vendors by ACH, card or check.")))

    assert out.results[0].found is True
    assert out.results[0].source_digest is None
