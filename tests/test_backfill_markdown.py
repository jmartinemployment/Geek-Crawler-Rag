"""Unit tests for markdown backfill helpers (no Mongo)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import backfill_markdown as backfill_module
from backfill_markdown import (
    _missing_markdown_query,
    backfill,
    extract_clean_content,
    should_exclude_locale_path,
)


class _Cursor:
    def __init__(self, documents: list[dict[str, object]]) -> None:
        self.documents = documents

    def sort(self, *_args: object) -> _Cursor:
        return self

    def limit(self, value: int) -> _Cursor:
        self.documents = self.documents[:value]
        return self

    def __iter__(self):
        return iter(self.documents)


def _mongo(
    monkeypatch: pytest.MonkeyPatch,
    *,
    documents: list[dict[str, object]],
    total_pages: int,
    missing_pages: int,
    run_status: str = "complete",
) -> tuple[MagicMock, MagicMock, MagicMock]:
    pages = MagicMock()
    pages.find.side_effect = [_Cursor(documents), _Cursor([])]
    pages.count_documents.side_effect = [total_pages, missing_pages]
    links = MagicMock()
    links.delete_many.return_value = SimpleNamespace(deleted_count=0)
    runs = MagicMock()
    runs.find_one.return_value = {"Status": run_status}
    runs.update_one.return_value = SimpleNamespace(modified_count=1)
    database = {
        "crawl_pages": pages,
        "crawl_links": links,
        "crawl_runs": runs,
    }
    client = MagicMock()
    client.__getitem__.return_value = database
    monkeypatch.setattr(backfill_module, "MongoClient", MagicMock(return_value=client))
    return pages, links, runs


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
    title, markdown, _excerpt = extract_clean_content(
        html, "https://example.com/us/tool"
    )
    assert title
    assert markdown
    assert "Vendor Tool" in markdown or "automate" in markdown.lower()
    assert "Cookie banner" not in markdown


def test_missing_markdown_query_requires_both_supported_fields_empty() -> None:
    query = _missing_markdown_query("run-1")
    assert query["RunId"] == "run-1"
    expressions = query["$expr"]["$and"]
    assert len(expressions) == 2
    assert "$trim" in expressions[0]["$eq"][0]


def test_external_run_is_marked_ready_after_complete_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pages, _links, runs = _mongo(
        monkeypatch,
        documents=[],
        total_pages=3,
        missing_pages=0,
        run_status="external",
    )

    counts = backfill(
        mongo_url="mongodb://test",
        db_name="test",
        run_id="run-1",
        write=True,
        limit=None,
        batch_size=25,
    )

    assert counts.complete_pass == 1
    assert counts.runs_marked_ready == 1
    assert runs.update_one.call_args_list[-1].args[0]["Status"] == {
        "$in": ["complete", "external"]
    }


def test_missing_html_page_is_deleted_instead_of_being_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages, _links, _runs = _mongo(
        monkeypatch,
        documents=[
            {
                "_id": 1,
                "Id": "page-1",
                "RunId": "run-1",
                "Url": "https://example.com/page",
                "Html": None,
            }
        ],
        total_pages=1,
        missing_pages=0,
    )

    counts = backfill(
        mongo_url="mongodb://test",
        db_name="test",
        run_id="run-1",
        write=True,
        limit=None,
        batch_size=25,
    )

    assert counts.deleted_extract_empty == 1
    pages.delete_one.assert_called_once_with({"Id": "page-1"})


def test_limited_pass_never_marks_run_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _pages, _links, runs = _mongo(
        monkeypatch,
        documents=[
            {
                "_id": 1,
                "Id": "page-1",
                "RunId": "run-1",
                "Url": "https://example.com/page",
                "Html": None,
            }
        ],
        total_pages=0,
        missing_pages=0,
    )

    counts = backfill(
        mongo_url="mongodb://test",
        db_name="test",
        run_id="run-1",
        write=True,
        limit=1,
        batch_size=25,
    )

    assert counts.complete_pass == 0
    assert counts.runs_marked_ready == 0
    assert len(runs.update_one.call_args_list) == 1


def test_bulk_write_error_blocks_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    pages, _links, runs = _mongo(
        monkeypatch,
        documents=[
            {
                "_id": 1,
                "Id": "page-1",
                "RunId": "run-1",
                "Url": "https://example.com/page",
                "Html": (
                    "<html><body><article><h1>Useful page</h1>"
                    "<p>This is sufficiently long source material for extraction.</p>"
                    "</article></body></html>"
                ),
            }
        ],
        total_pages=1,
        missing_pages=0,
    )
    pages.bulk_write.side_effect = RuntimeError("write failed")

    counts = backfill(
        mongo_url="mongodb://test",
        db_name="test",
        run_id="run-1",
        write=True,
        limit=None,
        batch_size=25,
    )

    assert counts.errors == 1
    assert counts.runs_marked_ready == 0
    assert len(runs.update_one.call_args_list) == 1


def test_zero_page_run_is_never_marked_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _pages, _links, runs = _mongo(
        monkeypatch,
        documents=[],
        total_pages=0,
        missing_pages=0,
    )

    counts = backfill(
        mongo_url="mongodb://test",
        db_name="test",
        run_id="run-1",
        write=True,
        limit=None,
        batch_size=25,
    )

    assert counts.complete_pass == 1
    assert counts.runs_marked_ready == 0
    assert len(runs.update_one.call_args_list) == 1


def test_incomplete_external_run_with_missing_markdown_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pages, _links, runs = _mongo(
        monkeypatch,
        documents=[],
        total_pages=10,
        missing_pages=1,
        run_status="external",
    )

    counts = backfill(
        mongo_url="mongodb://test",
        db_name="test",
        run_id="run-1",
        write=True,
        limit=None,
        batch_size=25,
    )

    assert counts.complete_pass == 1
    assert counts.runs_marked_ready == 0
    assert len(runs.update_one.call_args_list) == 1


def test_cli_defaults_match_remediation_load_profile() -> None:
    args = backfill_module.build_arg_parser().parse_args([])
    assert args.batch_size == 25
    assert args.delay_seconds == 2.0
    assert args.write is False
