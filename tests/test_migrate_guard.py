"""The sparse-vector migration refuses to run beside an active index pass.

Two writers on one Qdrant collection with no coordination, where one of them calls
ensure_collection() at job start, is how a migration gets rebuilt out from under itself. The policy
is a pure function so it can be proved here without a Mongo or a Qdrant.
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "migrate_sparse_vectors",
    Path(__file__).resolve().parents[1] / "scripts" / "migrate_sparse_vectors.py",
)
migrate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migrate)


def _job(state: str, run_id: str = "r1", **extra):
    return {"state": state, "runId": run_id, **extra}


def test_an_idle_queue_proceeds():
    ok, message = migrate.guard_verdict(check_succeeded=True, jobs=[], override=False)
    assert ok
    assert message == "index queue: idle"


@pytest.mark.parametrize("state", ["running", "pending"])
def test_either_in_flight_state_blocks(state):
    # pending counts as in flight: index concurrency is 1, so a queued job starts the moment the
    # current one ends -- which can be in the middle of a migration.
    ok, message = migrate.guard_verdict(
        check_succeeded=True, jobs=[_job(state)], override=False
    )
    assert not ok
    assert "BLOCKED" in message
    assert "ingestion pass is active" in message


def test_the_block_message_counts_the_jobs():
    jobs = [_job("running", "r1"), _job("pending", "r2"), _job("pending", "r3")]
    _, message = migrate.guard_verdict(check_succeeded=True, jobs=jobs, override=False)
    assert "3 job(s) running or pending" in message


def test_an_unreadable_queue_blocks_rather_than_assuming_idle():
    # Silence is not permission. A Mongo that did not answer has not said the queue is empty.
    ok, message = migrate.guard_verdict(check_succeeded=False, jobs=[], override=False)
    assert not ok
    assert "could not read the index queue" in message


def test_the_override_clears_an_in_flight_queue():
    ok, message = migrate.guard_verdict(
        check_succeeded=True, jobs=[_job("running")], override=True
    )
    assert ok
    assert "--ignore-running-jobs" in message


def test_the_override_clears_an_unreadable_queue():
    ok, message = migrate.guard_verdict(check_succeeded=False, jobs=[], override=True)
    assert ok
    assert "--ignore-running-jobs" in message


def test_in_flight_states_are_exactly_running_and_pending():
    # complete and failed hold nothing; if they ever blocked, the queue would never look idle.
    assert set(migrate.IN_FLIGHT_STATES) == {"running", "pending"}


def test_jobs_are_described_one_aligned_line_each():
    lines = migrate.describe_jobs(
        [
            _job("running", "r1", pagesSeen=12, chunksUpserted=340),
            _job("pending", "r2"),
        ]
    )
    assert len(lines) == 2
    assert any("r1" in line and "pagesSeen=12" in line and "chunksUpserted=340" in line for line in lines)
    # Absent counters read as zero rather than None.
    assert any("r2" in line and "pagesSeen=0" in line for line in lines)


def test_describing_no_jobs_is_empty_not_an_error():
    assert migrate.describe_jobs([]) == []
