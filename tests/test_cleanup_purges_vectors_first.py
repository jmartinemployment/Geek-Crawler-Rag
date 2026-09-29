"""Vectors go before rows, and a failed purge stops the row delete.

`cleanup_unusable_pages.py` deleted `crawl_pages` and `crawl_links` and touched Qdrant not
at all. That was survivable only while its filters matched nothing: the `http_error` step
queried `{"StatusCode": {"$gte": 400}}` against a field GeekAPI stores as a string, so it
returned 0 on a store that held them. That filter is fixed now, so the step deletes for real
and the missing purge became live.

The ordering rule is stated in this repo, in `scripts/check_orphaned_crawl_data.py`:

    Delete vectors for a run with DELETE /v1/index/runs/{runId} before its rows, so a
    surviving point can never outlive the page it cites.

An orphaned point is worse than an undeleted page, which is why the order is this way round
and not the other. Retrieval reads chunk text straight from the Qdrant payload and filters
on `runId`, which a surviving point still carries — so the prose stays retrievable and
quotable. Meanwhile `/v1/pages/{id}` 404s for the deleted row, so `citation_verify` drops
every quote taken from it. The generator gets fed the text and can cite none of it.

So the failure that matters is not "the purge errored" — it is "the purge errored and the
rows went anyway".
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "cleanup_unusable_pages.py"
_spec = importlib.util.spec_from_file_location("cleanup_unusable_pages_vectors", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


class _Collection:
    """Records delete_many calls so ordering and non-calls are both observable."""

    def __init__(self, name: str, log: list[str]):
        self.name = name
        self.log = log
        self.calls: list[dict] = []

    def delete_many(self, query):
        self.calls.append(query)
        self.log.append(self.name)

        class _Result:
            deleted_count = 1

        return _Result()


def _docs(n: int = 2) -> list[dict]:
    return [{"_id": f"oid-{i}", "Id": f"page-{i}"} for i in range(n)]


def test_vectors_are_purged_before_links_and_pages(monkeypatch):
    log: list[str] = []
    pages = _Collection("pages", log)
    links = _Collection("links", log)
    counts = _module.CleanupCounts()
    purged: list[list[str]] = []

    def fake_purge(page_ids, *, counts):
        log.append("qdrant")
        purged.append(list(page_ids))
        counts.purged_vector_pages += len(page_ids)

    monkeypatch.setattr(_module, "_purge_vectors", fake_purge)

    _module._delete_ids(
        pages, links, _docs(), write=True, counts=counts, delete_links=True
    )

    # The order is the assertion. Qdrant first, then Mongo.
    assert log == ["qdrant", "links", "pages"]
    assert purged == [["page-0", "page-1"]]
    assert counts.purged_vector_pages == 2
    assert counts.deleted_pages == 1
    assert counts.deleted_links == 1


def test_a_failed_purge_leaves_every_row_alone(monkeypatch):
    """The load-bearing case: no row may be deleted after the purge fails."""
    log: list[str] = []
    pages = _Collection("pages", log)
    links = _Collection("links", log)
    counts = _module.CleanupCounts()

    def exploding_purge(page_ids, *, counts):
        log.append("qdrant")
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(_module, "_purge_vectors", exploding_purge)

    with pytest.raises(RuntimeError, match="qdrant unreachable"):
        _module._delete_ids(
            pages, links, _docs(), write=True, counts=counts, delete_links=True
        )

    assert log == ["qdrant"]  # and nothing after it
    assert pages.calls == []
    assert links.calls == []
    assert counts.deleted_pages == 0
    assert counts.deleted_links == 0


def test_dry_run_purges_nothing(monkeypatch):
    # A dry run must not touch Qdrant either -- it is a report, and a purge is not reportable.
    log: list[str] = []
    pages = _Collection("pages", log)
    links = _Collection("links", log)
    counts = _module.CleanupCounts()
    monkeypatch.setattr(
        _module, "_purge_vectors", lambda ids, *, counts: log.append("qdrant")
    )

    _module._delete_ids(
        pages, links, _docs(3), write=False, counts=counts, delete_links=True
    )

    assert log == []
    assert counts.purged_vector_pages == 0
    # would_delete, not deleted_pages. A dry run removes nothing, so the key that means
    # "removed" must stay at 0 -- one label carrying both meanings is what an operator had to
    # interpret before a mass delete.
    assert counts.would_delete == 3
    assert counts.deleted_pages == 0


def test_purged_count_is_reported(monkeypatch):
    # A run that purged nothing must be distinguishable from one that purged, in the output.
    counts = _module.CleanupCounts()
    assert "purged_vector_pages" in counts.as_dict()
    counts.purged_vector_pages = 7
    assert counts.as_dict()["purged_vector_pages"] == 7


def test_purge_with_no_page_ids_is_a_noop():
    # Real function, no stub: an empty list must not construct a Qdrant client at all.
    counts = _module.CleanupCounts()
    _module._purge_vectors([], counts=counts)
    assert counts.purged_vector_pages == 0
