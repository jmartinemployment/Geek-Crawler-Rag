"""Optional Cohere rerank (soft-disabled when API key missing)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self,
        api_key: str | None,
        *,
        model: str = "rerank-english-v3.0",
        enabled: bool = True,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._model = model
        self._enabled = bool(enabled and self._api_key)
        self._timeout = timeout_seconds

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        """Return (original_index, relevance_score) sorted best-first.

        On disable/failure, returns identity order with descending synthetic scores.
        """
        if not documents:
            return []
        top_n = max(1, min(top_n, len(documents)))
        if not self._enabled:
            return [(i, float(len(documents) - i)) for i in range(top_n)]

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    "https://api.cohere.com/v2/rerank",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "query": query,
                        "documents": documents,
                        "top_n": top_n,
                    },
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
        except Exception:
            logger.exception("Cohere rerank failed; using dense/hybrid order")
            return [(i, float(len(documents) - i)) for i in range(top_n)]

        results = data.get("results") or []
        out: list[tuple[int, float]] = []
        for row in results:
            try:
                idx = int(row["index"])
                score = float(row.get("relevance_score") or row.get("score") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            out.append((idx, score))
        if not out:
            return [(i, float(len(documents) - i)) for i in range(top_n)]
        return out[:top_n]
