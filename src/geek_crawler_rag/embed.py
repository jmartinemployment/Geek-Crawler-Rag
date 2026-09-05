"""OpenAI embeddings (text-embedding-3-small)."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 64,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embedding")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            # Empty strings break the API; callers should not send them.
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            # API returns data sorted by index.
            ordered = sorted(response.data, key=lambda d: d.index)
            out.extend(list(d.embedding) for d in ordered)
            logger.debug("Embedded batch of %s texts (total %s)", len(batch), len(out))
        if len(out) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(out)} for {len(texts)} inputs"
            )
        return out
