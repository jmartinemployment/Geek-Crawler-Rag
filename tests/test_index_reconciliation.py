"""A crawl that is content-ready with no index job is a lost enqueue.

GeekAPI's trigger is fire-and-forget and EnqueueIndexAsync fails closed by returning null, so the
loss leaves no trace against the run. The only way to find it afterwards is to compare crawl_runs
against rag_index_jobs, which is what these rules do.
"""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_index_reconciliation",
    Path(__file__).resolve().parents[1] / "scripts" / "check_index_reconciliation.py",
)
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)

NOW = datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc)
GRACE = timedelta(minutes=30)


def _run(ready_at):
    return {"Id": "r1", "ContentReadyAt": ready_at}


def test_the_postgres_export_string_geekapi_actually_writes_parses():
    # Space separator and a two-digit offset; fromisoformat rejects both untouched.
    parsed = check.parse_ready_at("2026-09-24 12:32:44.213000+00")
    assert parsed == datetime(2026, 9, 24, 12, 32, 44, 213000, tzinfo=timezone.utc)


def test_a_real_datetime_is_accepted_and_a_naive_one_is_read_as_utc():
    aware = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    assert check.parse_ready_at(aware) == aware
    assert check.parse_ready_at(datetime(2026, 9, 24, 12, 0)) == aware


def test_a_trailing_z_parses():
    assert check.parse_ready_at("2026-09-24T12:00:00Z") == datetime(
        2026, 9, 24, 12, 0, tzinfo=timezone.utc
    )


def test_unreadable_values_yield_none_rather_than_raising():
    for raw in [None, "", "   ", "not a date", 12345, {}]:
        assert check.parse_ready_at(raw) is None


def test_a_run_past_grace_with_no_job_is_lost():
    run = _run("2026-09-24 11:00:00+00")
    assert check.classify(run, [], NOW, GRACE) == "no index job was ever created"


def test_a_run_inside_the_grace_period_is_not_yet_lost():
    # The enqueue is asynchronous; a job document appears a moment after the crawl closes.
    run = _run("2026-09-24 13:45:00+00")
    assert check.classify(run, [], NOW, GRACE) is None


def test_a_run_with_a_pending_or_running_job_is_fine():
    run = _run("2026-09-24 11:00:00+00")
    assert check.classify(run, [{"state": "pending"}], NOW, GRACE) is None
    assert check.classify(run, [{"state": "running"}], NOW, GRACE) is None


def test_a_run_whose_only_job_failed_is_lost():
    run = _run("2026-09-24 11:00:00+00")
    assert check.classify(run, [{"state": "failed"}], NOW, GRACE) == "its only index job failed"


def test_a_failed_job_followed_by_a_retry_is_not_lost():
    # A later attempt is the recovery; reporting it would send an operator to re-enqueue work
    # that is already running.
    run = _run("2026-09-24 11:00:00+00")
    assert check.classify(run, [{"state": "failed"}, {"state": "running"}], NOW, GRACE) is None


def test_an_unreadable_ready_at_is_left_alone_rather_than_guessed():
    assert check.classify(_run("not a date"), [], NOW, GRACE) is None
