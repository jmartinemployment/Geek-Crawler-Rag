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
