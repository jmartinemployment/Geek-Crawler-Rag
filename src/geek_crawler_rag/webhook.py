"""Fire-and-forget index status webhooks to GeekAPI (SignalR bridge)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from geek_crawler_rag.models import IndexStatusResponse

logger = logging.getLogger(__name__)


class IndexStatusWebhook:
    def __init__(self, url: str | None, api_key: str | None) -> None:
        self._url = (url or "").strip() or None
        self._api_key = (api_key or "").strip() or None
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return self._url is not None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def notify(self, status: IndexStatusResponse) -> None:
        if not self._url:
            return
        payload = status.model_dump(by_alias=True, mode="json")
        payload["eventType"] = "rag_index"
        try:
            client = await self._ensure_client()
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["X-API-Key"] = self._api_key
            response = await client.post(self._url, json=payload, headers=headers)
            if response.status_code >= 400:
                logger.warning(
                    "Index status webhook failed runId=%s status=%s body=%s",
                    status.run_id,
                    response.status_code,
                    response.text[:300],
                )
        except Exception:
            # Never fail the indexer because GeekAPI is down.
            logger.exception("Index status webhook error for runId=%s", status.run_id)

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client
