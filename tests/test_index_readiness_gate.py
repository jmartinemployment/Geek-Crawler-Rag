"""`POST /v1/index` refuses a run with no `ContentReadyAt`, like the scheduler always has.

Readiness is the marker GeekAPI stamps in the same PATCH that marks a run complete. The scheduler's
candidate query filtered on it; the manual route did not, and indexed whatever runId it was handed,
logging a non-terminal status as a warning and carrying on. A gate one entrance honours is not a
gate. Both entrances now ask `IndexService.readiness_refusal`, before the claim, so a refused run
leaves no job row behind.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from geek_crawler_rag.app import start_index, state
from geek_crawler_rag.config import Settings
from geek_crawler_rag.indexer import IndexService
from geek_crawler_rag.models import IndexRunRequest
from geek_crawler_rag.mongo import CrawlRun, MongoCorpus

READY = datetime(2026, 10, 4, tzinfo=timezone.utc)


def _service(run: CrawlRun | None) -> IndexService:
    mongo = MagicMock()
    mongo.get_run = AsyncMock(return_value=run)
    svc = IndexService(
        mongo=mongo,
        store=MagicMock(),
        settings=Settings(openai_api_key="test"),
        llama=MagicMock(),
    )
    svc.intake_pause_reason = AsyncMock(return_value=None)  # type: ignore[method-assign]
    svc._enqueue = AsyncMock(return_value=(MagicMock(), True))  # type: ignore[method-assign]
    return svc


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", [None, ""])
async def test_a_run_without_content_ready_at_is_refused(marker):
    svc = _service(CrawlRun(id="r", crawl_type="partner", status="external", content_ready_at=marker))

    reason = await svc.readiness_refusal("r")

    assert reason is not None and "ContentReadyAt" in reason


@pytest.mark.asyncio
async def test_a_missing_run_is_refused():
    assert await _service(None).readiness_refusal("gone") == "Run not found: gone"


@pytest.mark.asyncio
async def test_a_content_ready_run_is_allowed():
    run = CrawlRun(id="r", crawl_type="partner", status="complete", content_ready_at=READY)
    assert await _service(run).readiness_refusal("r") is None


@pytest.mark.asyncio
async def test_the_route_answers_409_and_claims_nothing(monkeypatch):
    indexer = MagicMock()
    indexer.intake_pause_reason = AsyncMock(return_value=None)
    indexer.readiness_refusal = AsyncMock(return_value="runId=r has no ContentReadyAt")
    indexer.enqueue = AsyncMock()
    monkeypatch.setattr(state, "indexer", indexer, raising=False)

    with pytest.raises(HTTPException) as exc:
        await start_index(IndexRunRequest(runId="r"))

    assert exc.value.status_code == 409
    assert "Nothing was queued" in str(exc.value.detail)
    indexer.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_scheduled_entrance_is_gated_too():
    svc = _service(CrawlRun(id="r", crawl_type="partner", status="external"))

    assert await svc.enqueue_scheduled("r") is False
    svc._enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_run_carries_the_marker_from_mongo():
    class Runs:
        async def find_one(self, _query):
            return {"Id": "r", "CrawlType": "partner", "Status": "complete", "ContentReadyAt": READY}

    corpus = MongoCorpus.__new__(MongoCorpus)
    corpus._db = {"crawl_runs": Runs()}

    run = await corpus.get_run("r")

    assert run is not None and run.content_ready_at == READY
