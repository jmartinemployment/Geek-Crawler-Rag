"""Optional Cohere rerank (soft-disabled when API key missing)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankOutcome:
    """What one rerank call produced, and whether a reranker produced it.

    order is best-first (original_index, score).

    ranked separates two things the caller previously could not tell apart.
    Scores from Cohere and positional scores standing in for them are both
    floats in the same shape, so a caller reading order alone cannot know
    whether relevance was measured or assumed.

    failed says the reranker was asked and could not answer. That is distinct
    from being switched off: a disabled reranker is a configured choice and
    positional order is its documented behaviour, while a failed one broke a
    promise the pipeline made, and rule 2 does not permit papering over that
    with invented scores.
    """

    order: list[tuple[int, float]]
    ranked: bool
    failed: bool


def _positional(documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """Input order with descending scores. Only legitimate when disabled."""
    return [(i, float(len(documents) - i)) for i in range(top_n)]


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
    ) -> RerankOutcome:
        """Rank documents against the query, saying what actually happened.

        Disabled returns positional order marked unranked. A failure returns no
        order at all and is marked failed, because the alternative - returning
        positional scores that look like measured ones - changes which chunks
        reach the model as grounding evidence with nothing in the answer saying
        the ranking was never performed.
        """
        if not documents:
            return RerankOutcome(order=[], ranked=False, failed=False)
        top_n = max(1, min(top_n, len(documents)))
        if not self._enabled:
            return RerankOutcome(
                order=_positional(documents, top_n), ranked=False, failed=False
            )

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
            logger.exception("Cohere rerank failed for query of %d documents", len(documents))
            return RerankOutcome(order=[], ranked=False, failed=True)

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
            # The call succeeded and returned nothing usable. Same class of
            # outcome as a transport failure: asked, and no ranking came back.
            logger.error(
                "Cohere rerank returned %d result row(s), none parseable", len(results)
            )
            return RerankOutcome(order=[], ranked=False, failed=True)
        return RerankOutcome(order=out[:top_n], ranked=True, failed=False)
