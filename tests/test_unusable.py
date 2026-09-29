"""Tests for unusable page classification."""

from __future__ import annotations

from geek_crawler_rag.unusable import (
    classify_from_mongo_doc,
    classify_unusable_page,
    should_exclude_locale_path,
)


def test_locale_keeps_us_and_bare() -> None:
    assert should_exclude_locale_path("https://example.com/us/docs") is False
    assert should_exclude_locale_path("https://example.com/docs") is False
    assert should_exclude_locale_path("https://example.com/en/docs") is False


def test_locale_drops_region_and_language() -> None:
    assert should_exclude_locale_path("https://example.com/gb/docs") is True
    assert should_exclude_locale_path("https://example.com/fr/docs") is True


def test_classify_failure_and_locale() -> None:
    assert (
        classify_unusable_page(
            url="https://adzooma.com/",
            failure_reason="Cloudflare challenge page detected",
            blocks=[{"kind": "paragraph", "text": "x"}],
        )
        == "failure"
    )
    assert (
        classify_unusable_page(
            url="https://speakai.co/es/approaches/",
            blocks=[{"kind": "paragraph", "text": "x"}],
        )
        == "locale"
    )
    assert (
        classify_unusable_page(url="https://speakai.co/blog/ok", blocks=[{"kind": "paragraph", "text": "ok"}])
        is None
    )
    assert classify_unusable_page(url="https://speakai.co/blog/ok") == "no_content"


def test_classify_failure_signals() -> None:
    assert classify_unusable_page(failure_reason="challenge") == "failure"
    assert classify_unusable_page(robots_allowed=False) == "failure"


def test_an_error_response_is_not_corpus():
    # A branded 404 carries nav, an apology and suggested links, which clears
    # every prose floor the pipeline has. Before the crawler gated on status,
    # pages like this were stored, chunked and embedded under a URL that does
    # not exist, and nothing downstream ever looked at the status.
    for code in (400, 401, 403, 404, 410, 451, 500, 503):
        assert (
            classify_unusable_page(
                url="https://example.com/gone",
                status_code=code,
                blocks=[{"kind": "paragraph", "text": "Sorry, we could not find that page."}],
            )
            == "http_error"
        ), code


def test_a_served_page_is_kept():
    for code in (200, 201, 204, 301, 302, 304):
        assert (
            classify_unusable_page(
                url="https://example.com/ok",
                status_code=code,
                blocks=[{"kind": "paragraph", "text": "real prose"}],
            )
            is None
        ), code


def test_a_missing_status_is_not_an_error():
    # 0 is what a row written before the field carries, and None is what a
    # projection without it yields. Absence of a status is not evidence of an
    # error, and reading it as one would delete corpus that is perfectly good.
    for code in (0, None):
        assert (
            classify_unusable_page(
                url="https://example.com/ok",
                status_code=code,
                blocks=[{"kind": "paragraph", "text": "real prose"}],
            )
            is None
        ), code


def test_status_is_read_off_a_mongo_document():
    doc = {
        "Url": "https://example.com/gone",
        "StatusCode": 404,
        "Blocks": [{"kind": "paragraph", "text": "Page not found. Try our blog."}],
    }
    assert classify_from_mongo_doc(doc) == "http_error"
