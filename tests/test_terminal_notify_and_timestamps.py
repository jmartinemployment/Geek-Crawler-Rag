"""A quarantined failure notifies once, and a rehydrated status carries aware timestamps.

Two defects in the webhook path, both found by code review on 2026-09-29.

`_fail_quarantined` called `_persist` -- which notifies internally -- and then notified again.
Every quarantined failure delivered the terminal webhook twice, and the second call fired even
when `_persist` returned False because the status store had REJECTED the write (lease lost). The
mid-run flush site had the identical bug and its fix comment already states the rule: a status the
store rejected is not one to push onward.

`_from_doc` read `startedAtUtc`/`finishedAtUtc` straight from Mongo while `_scheduler_from_doc`
routed them through `_as_utc`. Motor returns naive datetimes, so a status rehydrated by `claim()`
or `get()` carried tz-naive timestamps. `finishedAtUtc` is the key GeekAPI's out-of-order guard
compares (`body.FinishedAtUtc > run.RagIndexedAtUtc`), and a naive value serialises with no offset
for C# to read in the server's local zone -- shifting the ordering key so a stale terminal status
can overwrite a newer one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag.config import Settings
from geek_crawler_rag.indexer import IndexService
from geek_crawler_rag.models import IndexState, IndexStatusResponse
from geek_crawler_rag.status_store import _from_doc


def _service(*, save_returns: bool) -> tuple[IndexService, MagicMock]:
    webhook = MagicMock()
    webhook.notify = AsyncMock(return_value=None)
    store = MagicMock()
    store.save = AsyncMock(return_value=save_returns)
    svc = IndexService(
        settings=Settings(),
        mongo=MagicMock(),
        store=MagicMock(),
        llama=MagicMock(),
        webhook=webhook,
        status_store=store,
    )
    return svc, webhook


def _status() -> IndexStatusResponse:
    return IndexStatusResponse(run_id="r-1", state=IndexState.RUNNING)


@pytest.mark.asyncio
async def test_a_quarantined_failure_notifies_exactly_once():
    svc, webhook = _service(save_returns=True)
    await svc._fail_quarantined(_status(), error="boom")

    assert webhook.notify.await_count == 1
    pushed = webhook.notify.await_args.args[0]
    assert pushed.state is IndexState.FAILED
    assert pushed.error == "boom"


@pytest.mark.asyncio
async def test_a_rejected_write_is_not_pushed_onward():
    """The case where the second notify was not merely a duplicate."""
    svc, webhook = _service(save_returns=False)
    await svc._fail_quarantined(_status(), error="boom")

    assert webhook.notify.await_count == 0


@pytest.mark.asyncio
async def test_a_terminal_status_still_reaches_geekapi():
    # The guard must not have silenced the path it guards: FAILED is the state GeekAPI needs most.
    svc, webhook = _service(save_returns=True)
    await svc._fail_quarantined(_status(), error="quarantined")
    assert webhook.notify.await_count == 1


def test_a_rehydrated_status_has_aware_timestamps():
    naive = datetime(2026, 9, 29, 17, 58, 39)
    status = _from_doc(
        {
            "runId": "r-1",
            "state": "complete",
            "startedAtUtc": naive,
            "finishedAtUtc": naive,
        }
    )

    assert status.started_at_utc is not None
    assert status.finished_at_utc is not None
    assert status.started_at_utc.tzinfo is not None
    assert status.finished_at_utc.tzinfo is not None
    assert status.finished_at_utc == naive.replace(tzinfo=timezone.utc)


def test_an_already_aware_timestamp_is_unchanged():
    aware = datetime(2026, 9, 29, 17, 58, 39, tzinfo=timezone.utc)
    status = _from_doc({"runId": "r-1", "state": "complete", "finishedAtUtc": aware})
    assert status.finished_at_utc == aware


def test_a_missing_timestamp_stays_none():
    status = _from_doc({"runId": "r-1", "state": "pending"})
    assert status.started_at_utc is None
    assert status.finished_at_utc is None


def test_the_serialised_ordering_key_carries_an_offset():
    # What GeekAPI actually parses. Without an offset C# reads it in the server's local zone.
    status = _from_doc(
        {"runId": "r-1", "state": "complete", "finishedAtUtc": datetime(2026, 9, 29, 17, 58, 39)}
    )
    wire = status.model_dump(by_alias=True, mode="json")["finishedAtUtc"]
    assert wire.endswith("+00:00") or wire.endswith("Z"), wire
