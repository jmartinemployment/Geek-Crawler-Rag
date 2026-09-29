"""A dry run reports the real magnitude, in its own counter, and no step is skipped.

Defects in one loop, all of which made the script lie to the operator sizing a mass delete
against `crawl_pages`.

**The dry run walked one batch and reported it as the total.** It ran the same loop as
`--write`, broke after the first batch (`if not write: break`), and `_delete_ids` had
already added that batch to `deleted_pages`. So at `batch_size=500` the report read
`deleted_pages=500` per step whether the true figure was 500 or 50,000 — capped at the
batch size, with no indication it was a cap. It also could not reach the last step: the
outer loop breaks once `deleted_pages >= limit`, and the docstring's own example is
`--dry-run --limit 500` with `batch_size` defaulting to 500, so the first step filled the
budget and `http_error` never ran — printing no `reason_http_error` line at all, which
reads as "there are none".

**The hint killed the run, and the first fix caught it instead of removing it.**
`cursor.hint()` only sets an option on a local pymongo Cursor and never contacts the server,
so the original try/except around *it* caught nothing — `OperationFailure` is raised on first
iteration, `list(cursor)`, which sat outside the try. Neither `FailureReason_1` nor
`RobotsAllowed_1` is created by any code in any repo, so this killed the whole run on step
one. The fix for that moved the try around `list(cursor)` and retried the query unhinted,
which is a fallback — forbidden by CLAUDE.md §2 in as many words, "never write backup paths,
secondary loops, or default to unverified data to salvage the operation" — and under `--write`
it went on to purge vectors and `delete_many` rows gathered by an unplanned scan. The hint is
gone now, so there is no exception to catch.

**`deleted_pages` meant two things.** A dry run added its matches to the same counter a write
adds its deletions to, so one key read "this many rows are gone" in one mode and "this many
rows matched, nothing was touched" in the other — and `--limit` inherited the ambiguity,
bounding deletions under `--write` and matches under `--dry-run`. `would_delete` and
`budget_used` split the three ideas apart.

**`--write` could not reach the last step.** The `--dry-run` half of that was fixed; the
`--write` half was left, so `--write --limit 500` spent the budget on step one, broke out of
the loop, and never ran the `http_error` step that deletes 4xx error pages — printing no
`reason_http_error` line, which reads as "there were none".
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
from pymongo.errors import OperationFailure

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "cleanup_unusable_pages.py"
_spec = importlib.util.spec_from_file_location("cleanup_dry_run", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


class _Cursor:
    def __init__(self, docs, *, raise_on_iter=None):
        self._docs = docs
        self._raise = raise_on_iter
        self.hinted = None

    def limit(self, _n):
        return self

    def hint(self, name):
        self.hinted = name
        return self

    def batch_size(self, _n):
        return self

    def __iter__(self):
        if self._raise is not None:
            raise self._raise
        return iter(self._docs)


class _Pages:
    """Counts count_documents separately from find, so we can prove which path ran."""

    def __init__(self, *, counts_by_query=None, find_docs=None, find_raises=None):
        self._counts = counts_by_query or {}
        self._find_docs = find_docs if find_docs is not None else []
        self._find_raises = find_raises
        self.count_calls = 0
        self.find_calls = 0
        self.cursors: list[_Cursor] = []

    def count_documents(self, query, **_kw):
        self.count_calls += 1
        # $or is checked FIRST and by top-level key, not substring: the distinct query
        # embeds every step's query inside it, so repr() contains "FailureReason" as well
        # and a substring scan returns the wrong step's number.
        if "$or" in query:
            return self._counts["$or"]
        for key, n in self._counts.items():
            if key == "$or":
                continue
            if key in repr(query):
                return n
        return self._counts.get("*", 0)

    def find(self, _query, _projection=None, **_kw):
        self.find_calls += 1
        # Only the first find in a step yields docs; later ones are empty so the loop ends.
        docs = self._find_docs if self.find_calls == 1 else []
        raise_on = self._find_raises if self.find_calls == 1 else None
        c = _Cursor(docs, raise_on_iter=raise_on)
        self.cursors.append(c)
        return c

    def delete_many(self, _q):
        class _R:
            deleted_count = 0

        return _R()


class _Links(_Pages):
    pass


def _fast_steps():
    return [
        ("failure", {"FailureReason": {"$gt": ""}}),
        ("failure", {"RobotsAllowed": {"$in": ["f", "false", False]}}),
        ("http_error", {"$expr": {"status": "ish"}}),
    ]


def test_dry_run_reports_the_real_total_not_one_batch(monkeypatch, capsys):
    # 50,000 matches with batch_size 500: the old shape reported 500.
    pages = _Pages(counts_by_query={"FailureReason": 50_000, "RobotsAllowed": 12, "$expr": 7, "$or": 50_015})
    counts = _module.CleanupCounts()

    _module._run_fast_steps(
        pages, _Links(), _fast_steps(), write=False, limit=None,
        batch_size=500, counts=counts, delete_links=True,
    )

    assert counts.by_reason["failure"] == 50_012  # 50,000 + 12, both steps counted
    assert counts.by_reason["http_error"] == 7
    # The headline figure is the UNION, not the sum of the labels. The labels sum to 50,019
    # because the steps overlap; 50,015 rows exist. A --write deletes each row once, so the
    # number an operator reads before a mass delete has to be the row count.
    assert counts.would_delete == 50_015
    assert counts.scanned == 50_015
    assert counts.deleted_pages == 0  # a dry run deletes nothing, and says so
    # It counted; it never walked a cursor.
    assert pages.find_calls == 0
    assert pages.count_calls == 4  # three steps plus the distinct $or

    out = capsys.readouterr().out
    assert "matched=50000" in out
    assert "DISTINCT rows across all steps=50015" in out


def test_dry_run_reaches_the_last_step_even_with_a_small_limit(capsys):
    # --dry-run --limit 500 used to let step one fill the budget so http_error never ran,
    # and its absent line read as "there are none".
    pages = _Pages(counts_by_query={"FailureReason": 9_999, "RobotsAllowed": 0, "$expr": 3, "$or": 10_002})
    counts = _module.CleanupCounts()

    _module._run_fast_steps(
        pages, _Links(), _fast_steps(), write=False, limit=500,
        batch_size=500, counts=counts, delete_links=True,
    )

    assert counts.by_reason["http_error"] == 3
    assert "label=http_error matched=3" in capsys.readouterr().out


def test_overlap_is_reported_rather_than_left_to_be_discovered(capsys):
    # A row with both a FailureReason and a 403 is in two steps. The per-label numbers are
    # exact; the distinct line is what an operator needs for the true row count.
    pages = _Pages(counts_by_query={"FailureReason": 10, "RobotsAllowed": 10, "$expr": 10, "$or": 12})
    counts = _module.CleanupCounts()

    _module._run_fast_steps(
        pages, _Links(), _fast_steps(), write=False, limit=None,
        batch_size=500, counts=counts, delete_links=True,
    )

    out = capsys.readouterr().out
    assert "DISTINCT rows across all steps=12" in out
    assert "overlap" in out
    # 30 across the three labels, 12 rows. would_delete reports 12.
    assert sum(counts.by_reason.values()) == 30
    assert counts.would_delete == 12


def test_no_hint_is_ever_set_on_a_cursor(monkeypatch):
    """The hint is gone, not caught.

    `FailureReason_1` and `RobotsAllowed_1` are created by no code in any repo, and Mongo
    errors on a hint naming an index that does not exist -- so the hint WAS the failure. The
    first fix caught that OperationFailure and re-ran the query unhinted, which is a fallback
    CLAUDE.md section 2 forbids in as many words, and worse than it looked: under --write it
    went on to purge vectors and delete_many rows gathered by an unplanned scan. Dropping the
    hint removes the exception instead of recovering from it.
    """
    docs = [{"_id": "o1", "Id": "p1"}]
    pages = _Pages(find_docs=docs)
    counts = _module.CleanupCounts()
    monkeypatch.setattr(_module, "_purge_vectors", lambda ids, *, counts: None)

    _module._run_fast_steps(
        pages, _Links(), _fast_steps(),
        write=True, limit=None, batch_size=500, counts=counts, delete_links=False,
    )

    assert pages.find_calls > 0
    assert [c.hinted for c in pages.cursors] == [None] * len(pages.cursors)


def test_a_query_failure_is_not_retried(monkeypatch):
    """No second attempt, no unhinted salvage: it raises and the run ends."""
    pages = _Pages(find_docs=[{"_id": "o1", "Id": "p1"}], find_raises=OperationFailure("boom"))
    counts = _module.CleanupCounts()
    purged: list[object] = []
    monkeypatch.setattr(_module, "_purge_vectors", lambda ids, *, counts: purged.append(ids))

    with pytest.raises(OperationFailure, match="boom"):
        _module._run_fast_steps(
            pages, _Links(), [("failure", {"FailureReason": {"$gt": ""}})],
            write=True, limit=None, batch_size=500, counts=counts, delete_links=True,
        )

    assert pages.find_calls == 1  # one attempt
    assert purged == []  # and nothing was deleted on the way out
    assert counts.deleted_pages == 0


def test_the_write_path_reaches_every_step_under_a_small_limit(monkeypatch):
    """The half of the fix that was missed.

    `--dry-run --limit 500` was fixed to reach the last step; `--write --limit 500` was not.
    The budget is spent by step one, the loop broke out, and the http_error step -- the one
    that deletes 4xx error pages -- never ran and printed no line, which reads as "there were
    none" rather than "never checked".
    """
    pages = _Pages(find_docs=[{"_id": f"o{i}", "Id": f"p{i}"} for i in range(500)])
    counts = _module.CleanupCounts()
    monkeypatch.setattr(_module, "_purge_vectors", lambda ids, *, counts: None)

    _module._run_fast_steps(
        pages, _Links(), _fast_steps(),
        write=True, limit=500, batch_size=500, counts=counts, delete_links=False,
    )

    # Every label is present, including the ones the budget left no room for. A key at 0 is
    # "checked, found nothing"; an absent key is indistinguishable from "never ran".
    assert set(counts.by_reason) == {"failure", "http_error"}
    assert counts.by_reason["http_error"] == 0


def test_limit_is_charged_against_the_budget_in_both_modes():
    # --limit meant "rows deleted" under --write and "rows matched" under --dry-run, because
    # both were counted into deleted_pages. budget_used is the one number either mode spends.
    counts = _module.CleanupCounts()
    assert counts.budget_used == 0
    counts.would_delete = 7
    assert counts.budget_used == 7
    counts.deleted_pages = 3
    assert counts.budget_used == 10
    assert counts.as_dict()["would_delete"] == 7
    assert counts.as_dict()["deleted_pages"] == 3
