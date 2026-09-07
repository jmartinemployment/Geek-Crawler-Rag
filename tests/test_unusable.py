"""Tests for unusable page classification."""

from __future__ import annotations

from geek_crawler_rag.unusable import classify_unusable_page, should_exclude_locale_path


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
        )
        == "failure"
    )
    assert (
        classify_unusable_page(url="https://speakai.co/es/approaches/")
        == "locale"
    )
    assert classify_unusable_page(url="https://speakai.co/blog/ok") is None


def test_classify_backfill_skip_maps() -> None:
    assert classify_unusable_page(markdown_backfill_skip="extract_empty") == "extract_empty"
    assert classify_unusable_page(markdown_backfill_skip="failure") == "failure"
    assert classify_unusable_page(robots_allowed=False) == "failure"
