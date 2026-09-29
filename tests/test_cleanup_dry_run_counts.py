"""A dry run has to report the real magnitude, and a bad index hint must not kill the run.

Two defects in the same loop, both of which made the script lie to the operator sizing a
mass delete against `crawl_pages`.

**The dry run walked one batch and reported it as the total.** It ran the same loop as
`--write`, broke after the first batch (`if not write: break`), and `_delete_ids` had
already added that batch to `deleted_pages`. So at `batch_size=500` the report read
`deleted_pages=500` per step whether the true figure was 500 or 50,000 — capped at the
batch size, with no indication it was a cap. It also could not reach the last step: the
outer loop breaks once `deleted_pages >= limit`, and the docstring's own example is
`--dry-run --limit 500` with `batch_size` defaulting to 500, so the first step filled the
budget and `http_error` never ran — printing no `reason_http_error` line at all, which
reads as "there are none".

**The hint guard could not fire.** `cursor.hint()` only sets an option on a local pymongo
Cursor and never contacts the server, so wrapping *it* in try/except caught nothing.
`OperationFailure` is raised on first iteration — `list(cursor)` — which sat outside the
try. Neither `FailureReason_1` nor `RobotsAllowed_1` is created by any code in any repo, so
on a database where they were not made by hand this killed the whole run on step one while
the author's `WARN ... skipped` message was unreachable.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

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
        ("failure", {"FailureReason": {"$gt": ""}}, "FailureReason_1"),
        ("failure", {"RobotsAllowed": False}, "RobotsAllowed_1"),
        ("http_error", {"$expr": {"status": "ish"}}, None),
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
    assert counts.deleted_pages == 50_019
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


def test_a_missing_index_hint_falls_back_instead_of_killing_the_run(capsys, monkeypatch):
    # The WARN that was unreachable. OperationFailure fires on list(cursor), not on hint().
    docs = [{"_id": "o1", "Id": "p1"}]
    pages = _Pages(find_docs=docs, find_raises=OperationFailure("bad hint: index not found"))
    counts = _module.CleanupCounts()
    monkeypatch.setattr(_module, "_purge_vectors", lambda ids, *, counts: None)

    _module._run_fast_steps(
        pages, _Links(), [("failure", {"FailureReason": {"$gt": ""}}, "FailureReason_1")],
        write=True, limit=None, batch_size=500, counts=counts, delete_links=True,
    )

    out = capsys.readouterr().out
    assert "unusable" in out and "FailureReason_1" in out
    assert "retrying failure unhinted" in out
    # It recovered: the retry cursor was consulted, so the step did work.
    assert pages.find_calls >= 2
