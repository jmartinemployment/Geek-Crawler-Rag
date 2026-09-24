"""Purging a stale index job must not reach a job that is actually working.

`pending` with an expired lease is abandoned; `running` under a live lease is an indexer mid-pass,
and deleting its document while it works loses the only record of the run.
"""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "purge_stale_job",
    Path(__file__).resolve().parents[1] / "scripts" / "purge_stale_job.py",
)
purge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(purge)

NOW = datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc)


def test_running_under_a_live_lease_is_protected():
    job = {"state": "running", "leaseUntil": NOW + timedelta(minutes=5)}
    assert purge.is_live_job(job, NOW)


def test_running_with_an_expired_lease_is_stale():
    # The container holding it died; the lease lapsed and nothing renewed it.
    job = {"state": "running", "leaseUntil": NOW - timedelta(hours=23)}
    assert not purge.is_live_job(job, NOW)


def test_a_naive_lease_timestamp_is_read_as_utc():
    # Mongo hands back naive datetimes; comparing one to an aware `now` raises unless it is
    # localised first, and a raise here would abort a purge that should simply have proceeded.
    job = {"state": "running", "leaseUntil": (NOW + timedelta(minutes=5)).replace(tzinfo=None)}
    assert purge.is_live_job(job, NOW)


def test_pending_is_never_live_however_fresh_its_lease():
    job = {"state": "pending", "leaseUntil": NOW + timedelta(hours=1)}
    assert not purge.is_live_job(job, NOW)


def test_a_missing_or_null_lease_is_not_live():
    assert not purge.is_live_job({"state": "running"}, NOW)
    assert not purge.is_live_job({"state": "running", "leaseUntil": None}, NOW)


def test_terminal_states_are_not_live():
    for state in ("complete", "failed", None):
        assert not purge.is_live_job({"state": state, "leaseUntil": NOW + timedelta(hours=1)}, NOW)


def test_describe_reads_absent_counters_as_zero():
    line = purge.describe({"state": "pending", "leaseOwner": None})
    assert "state=pending" in line
    assert "pagesSeen=0" in line
    assert "chunksUpserted=0" in line
