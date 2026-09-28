"""Killing one index job kills one index job.

Before this there was no way to stop a single job. The only lever was stopping the
process, and `Indexer.stop()` takes everything in the queue with it: on 2026-09-28,
stopping one run marked fourteen jobs failed, thirteen of which had never started a page.

The trap this has to avoid is the obvious implementation. `_worker_loop` catches
`asyncio.CancelledError`, fails the job and then *re-raises* -- which ends the worker task.
That is right for shutdown, where everything is stopping, and ruinous for a per-job kill,
where it would strand every job queued behind the one being killed. So the kill is
cooperative: a flag, a checkpoint, and a `JobKilled` that is deliberately not a
`CancelledError`.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag.config import Settings
from geek_crawler_rag.indexer import IndexService, JobKilled
from geek_crawler_rag.models import IndexState, IndexStatusResponse


def _service() -> IndexService:
    settings = Settings(openai_api_key="test", embed_batch_size=2, page_batch_size=10)
    svc = IndexService(
        mongo=MagicMock(),
        store=MagicMock(),
        settings=settings,
        llama=MagicMock(),
        status_store=None,
    )
    svc._persist = AsyncMock(return_value=True)  # type: ignore[method-assign]
    return svc


def _status(run_id: str, state: IndexState, *, chunks: int = 0) -> IndexStatusResponse:
    return IndexStatusResponse(
        run_id=run_id, state=state, attempt=1, chunks_upserted=chunks
    )


@pytest.mark.asyncio
async def test_killing_a_queued_job_marks_it_failed_immediately():
    # It never started, so nothing is mid-flight to unwind. Leaving it `pending` until the
    # worker happens to reach it is the "pending forever" state the job rules rule out.
    svc = _service()
    svc._statuses["queued-run"] = _status("queued-run", IndexState.PENDING)

    status, outcome = await svc.kill("queued-run", reason="wrong seed")

    assert outcome == "dequeued"
    assert status is not None
    assert status.state == IndexState.FAILED
    assert "wrong seed" in (status.error or "")
    assert "before it started" in (status.error or "")


@pytest.mark.asyncio
async def test_killing_a_running_job_reports_stopping_not_killed():
    # The run stops at its next checkpoint, so it is still RUNNING when kill() returns.
    # Reporting it dead here would have the operator believe a live job had stopped.
    svc = _service()
    svc._statuses["live-run"] = _status("live-run", IndexState.RUNNING, chunks=900)

    status, outcome = await svc.kill("live-run", reason="wrong corpus")

    assert outcome == "stopping"
    assert status is not None
    assert status.state == IndexState.RUNNING
    assert svc._killed["live-run"] == "wrong corpus"


@pytest.mark.asyncio
async def test_killing_an_absent_job_is_not_reported_as_a_kill():
    svc = _service()

    status, outcome = await svc.kill("never-existed", reason="x")

    assert outcome == "absent"
    assert status is None


@pytest.mark.parametrize(
    "state", [IndexState.COMPLETE, IndexState.FAILED, IndexState.SKIPPED]
)
@pytest.mark.asyncio
async def test_a_finished_job_cannot_be_killed(state):
    # Especially COMPLETE: answering "killed" would tell an operator they stopped an index
    # that is in fact live and citable.
    svc = _service()
    svc._statuses["done-run"] = _status("done-run", state)

    status, outcome = await svc.kill("done-run", reason="too late")

    assert outcome == "already_terminal"
    assert status is not None
    assert status.state == state
    assert "done-run" not in svc._killed


@pytest.mark.asyncio
async def test_a_blank_run_id_is_refused():
    svc = _service()
    with pytest.raises(ValueError):
        await svc.kill("   ", reason="x")


def test_the_checkpoint_raises_only_for_a_killed_run():
    svc = _service()
    svc._killed["doomed"] = "operator said so"

    with pytest.raises(JobKilled) as caught:
        svc._raise_if_killed("doomed")
    assert caught.value.reason == "operator said so"

    svc._raise_if_killed("healthy")  # must not raise


def test_job_killed_is_not_a_cancelled_error():
    # If it were, _worker_loop's CancelledError handler would re-raise and end the worker,
    # which is the entire failure this design exists to avoid.
    assert not issubclass(JobKilled, asyncio.CancelledError)
    assert not isinstance(JobKilled("r"), asyncio.CancelledError)


@pytest.mark.asyncio
async def test_a_killed_job_does_not_take_the_worker_down_with_it():
    """The load-bearing test: job 1 is killed, job 2 still runs."""
    svc = _service()
    ran: list[str] = []

    async def fake_run_claimed(run_id: str) -> None:
        svc._raise_if_killed(run_id)
        ran.append(run_id)

    svc._run_claimed_job = fake_run_claimed  # type: ignore[method-assign]
    svc._statuses["doomed"] = _status("doomed", IndexState.RUNNING, chunks=42)
    svc._statuses["survivor"] = _status("survivor", IndexState.PENDING)
    svc._killed["doomed"] = "operator said so"

    await svc._queue.put("doomed")
    await svc._queue.put("survivor")

    worker = asyncio.create_task(svc._worker_loop())
    await svc._queue.join()
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass

    # The killed one never ran its body; the one behind it did.
    assert ran == ["survivor"]
    assert not worker.done() or worker.cancelled()


@pytest.mark.asyncio
async def test_a_job_killed_mid_run_is_failed_with_its_partial_count_kept():
    svc = _service()
    svc._statuses["doomed"] = _status("doomed", IndexState.RUNNING, chunks=752)

    async def raise_killed(run_id: str) -> None:
        raise JobKilled("operator said so")

    svc._run_claimed_job = raise_killed  # type: ignore[method-assign]
    await svc._queue.put("doomed")

    worker = asyncio.create_task(svc._worker_loop())
    await svc._queue.join()
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass

    status = svc._statuses["doomed"]
    assert status.state == IndexState.FAILED
    assert "operator said so" in (status.error or "")
    # The partial index is reported, not silently discarded.
    assert "752 already upserted points preserved" in (status.error or "")


@pytest.mark.asyncio
async def test_the_kill_flag_is_cleared_when_the_job_leaves_the_queue():
    # A flag outliving its job would kill an unrelated later run of the same id.
    svc = _service()
    svc._statuses["doomed"] = _status("doomed", IndexState.PENDING)
    svc._killed["doomed"] = "operator said so"
    await svc._queue.put("doomed")

    worker = asyncio.create_task(svc._worker_loop())
    await svc._queue.join()
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass

    assert "doomed" not in svc._killed


@pytest.mark.asyncio
async def test_a_re_post_clears_a_stale_kill_flag():
    # Otherwise killing a run once would silently kill every re-post of it.
    svc = _service()
    svc._killed["retry-me"] = "killed earlier"

    status, accepted = await svc.enqueue("retry-me")

    assert accepted is True
    assert status.state == IndexState.PENDING
    assert "retry-me" not in svc._killed
