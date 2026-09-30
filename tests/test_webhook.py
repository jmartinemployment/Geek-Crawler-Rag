"""Index status webhook posts camelCase payload with eventType."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from geek_crawler_rag.models import IndexState, IndexStatusResponse
from geek_crawler_rag.webhook import IndexStatusWebhook


@pytest.mark.asyncio
async def test_webhook_disabled_when_no_url() -> None:
    wh = IndexStatusWebhook(None, "key")
    assert wh.enabled is False
    await wh.notify(
        IndexStatusResponse(run_id="r1", state=IndexState.RUNNING, pages_seen=1)
    )


@pytest.mark.asyncio
async def test_webhook_posts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: dict = {}

    class FakeResponse:
        status_code = 202
        text = ""

    class FakeClient:
        async def post(self, url, json=None, headers=None):
            posted["url"] = url
            posted["json"] = json
            posted["headers"] = headers
            return FakeResponse()

        async def aclose(self):
            return None

    wh = IndexStatusWebhook(
        "https://api.example/webhook",
        "secret-key",
    )
    wh._client = FakeClient()  # type: ignore[assignment]

    status = IndexStatusResponse(
        run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        state=IndexState.COMPLETE,
        pages_seen=10,
        pages_english=8,
        chunks_upserted=40,
        started_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        finished_at_utc=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
    )
    await wh.notify(status)

    assert posted["url"] == "https://api.example/webhook"
    assert posted["headers"]["X-API-Key"] == "secret-key"
    body = posted["json"]
    assert body["eventType"] == "rag_index"
    assert body["runId"] == status.run_id
    assert body["state"] == "complete"
    assert body["chunksUpserted"] == 40


@pytest.mark.asyncio
async def test_webhook_swallows_errors() -> None:
    class BoomClient:
        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("down")

        async def aclose(self):
            return None

    wh = IndexStatusWebhook("https://api.example/webhook", "k")
    wh._client = BoomClient()  # type: ignore[assignment]
    await wh.notify(IndexStatusResponse(run_id="r1", state=IndexState.FAILED, error="x"))


def _client_returning(status_code: int, body: str = ""):
    """A client whose POST answers with one status, so the log branch can be asserted."""

    class FakeResponse:
        pass

    FakeResponse.status_code = status_code
    FakeResponse.text = body

    class FakeClient:
        async def post(self, url, json=None, headers=None):
            return FakeResponse()

        async def aclose(self):
            return None

    return FakeClient()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [404, 500, 502, 503])
async def test_a_frame_the_receiver_did_not_record_logs_at_error(
    status_code: int, caplog: pytest.LogCaptureFixture
) -> None:
    """The one failure mode that loses data must not share a log level with the others.

    Until GeekBackend stopped swallowing the persist failure, this case answered ``202 Accepted``:
    ``chunksUpserted``, ``pagesEnglish`` and ``pagesSkippedUnusable`` were dropped and the sender was
    told they had landed. 404 means the run is gone from GeekAPI so there is nothing to record onto;
    5xx means the hop failed. Both leave ``crawl_runs`` carrying stale numbers, which the
    declared-URL evidence gate then reads.
    """
    wh = IndexStatusWebhook("https://api.example/webhook", "secret-key")
    wh._client = _client_returning(status_code, "nothing was written")  # type: ignore[assignment]

    with caplog.at_level("WARNING"):
        await wh.notify(
            IndexStatusResponse(run_id="r-404", state=IndexState.COMPLETE, pages_seen=12)
        )

    records = [r for r in caplog.records if "NOT RECORDED" in r.getMessage()]
    assert records, f"a {status_code} must be reported as not recorded"
    assert records[0].levelname == "ERROR"


@pytest.mark.asyncio
async def test_an_ordinary_4xx_stays_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """409 and friends are not this failure mode, and must not be promoted to ERROR.

    401/403 and 400 keep their own dedicated ERROR branches above; this pins that the catch-all
    below them did not widen to swallow every remaining 4xx into the same level.
    """
    wh = IndexStatusWebhook("https://api.example/webhook", "secret-key")
    wh._client = _client_returning(409, "conflict")  # type: ignore[assignment]

    with caplog.at_level("WARNING"):
        await wh.notify(
            IndexStatusResponse(run_id="r-409", state=IndexState.COMPLETE, pages_seen=1)
        )

    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert not any("NOT RECORDED" in r.getMessage() for r in caplog.records)
