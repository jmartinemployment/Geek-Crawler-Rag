"""Unit tests for markdown backfill helpers (no Mongo)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backfill_markdown import (  # noqa: E402
    _missing_markdown_query,
    extract_clean_content,
    should_exclude_locale_path,
)


def test_locale_keeps_us_and_bare() -> None:
    assert should_exclude_locale_path("https://example.com/us/docs") is False
    assert should_exclude_locale_path("https://example.com/docs") is False
    assert should_exclude_locale_path("https://example.com/en/docs") is False


def test_locale_drops_region_and_language() -> None:
    assert should_exclude_locale_path("https://example.com/gb/docs") is True
    assert should_exclude_locale_path("https://example.com/fr/docs") is True


def test_extract_readability_article() -> None:
    html = """
    <html><head><title>Vendor Tool</title></head>
    <body>
      <nav>Home Pricing</nav>
      <article>
        <h1>Vendor Tool</h1>
        <p>Vendor Tool helps teams automate partner workflows with clear metrics.</p>
        <p>Integration APIs support webhooks and OAuth for enterprise customers.</p>
      </article>
      <footer>Cookie banner</footer>
    </body></html>
    """
    title, markdown, _excerpt = extract_clean_content(html, "https://example.com/us/tool")
    assert title
    assert markdown
    assert "Vendor Tool" in markdown or "automate" in markdown.lower()
    assert "Cookie banner" not in markdown


def test_missing_markdown_query_requires_both_supported_fields_empty() -> None:
    query = _missing_markdown_query("run-1")
    assert query["RunId"] == "run-1"
    assert len(query["$and"]) == 2
