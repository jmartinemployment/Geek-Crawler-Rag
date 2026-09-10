"""Token-aware pacing and retry helpers for OpenAI embeddings."""

from __future__ import annotations

import asyncio
import email.utils
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import tiktoken


@dataclass(frozen=True)
class EmbeddingBatch:
    texts: list[str]
    token_count: int


class EmbeddingRetryExhausted(RuntimeError):
    """Legacy name; embedding calls no longer retry in-process."""


class EmbeddingThrottle:
    """Strict rolling-window token limiter shared by all embedding calls."""

    def __init__(
        self,
        tokens_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")
        self.tokens_per_minute = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()
        self.total_wait_seconds = 0.0
        self.rate_limit_retries = 0

    def _expire(self, now: float) -> None:
        cutoff = now - 60.0
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()

    @property
    def tokens_in_window(self) -> int:
        now = self._clock()
        self._expire(now)
        return sum(tokens for _, tokens in self._events)

    async def acquire(self, token_count: int) -> float:
        if token_count <= 0:
            token_count = 1
        if token_count > self.tokens_per_minute:
            raise ValueError(
                f"embedding request tokens={token_count} exceeds TPM="
                f"{self.tokens_per_minute}"
            )

        waited = 0.0
        while True:
            wait_for = 0.0
            async with self._lock:
                now = self._clock()
                self._expire(now)
                used = sum(tokens for _, tokens in self._events)
                if used + token_count <= self.tokens_per_minute:
                    self._events.append((now, token_count))
                    self.total_wait_seconds += waited
                    return waited
                oldest_at = self._events[0][0]
                wait_for = max(0.01, 60.0 - (now - oldest_at))
            await self._sleep(wait_for)
            waited += wait_for


def embedding_token_count(texts: Sequence[str], model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return max(1, sum(len(encoding.encode(text)) for text in texts))


def partition_embedding_batches(
    texts: Sequence[str],
    *,
    model: str,
    max_items: int,
    max_tokens: int,
) -> list[EmbeddingBatch]:
    if max_items <= 0 or max_tokens <= 0:
        raise ValueError("embedding batch limits must be positive")

    batches: list[EmbeddingBatch] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = embedding_token_count([text], model)
        if tokens > max_tokens:
            raise ValueError(
                f"single embedding input tokens={tokens} exceeds max={max_tokens}"
            )
        if current and (
            len(current) >= max_items or current_tokens + tokens > max_tokens
        ):
            batches.append(EmbeddingBatch(current, current_tokens))
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append(EmbeddingBatch(current, current_tokens))
    return batches


def _error_code(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            return str(error.get("code") or error.get("type") or "").lower()
    return str(getattr(exc, "code", "") or "").lower()


def is_retryable_rate_limit(exc: BaseException) -> bool:
    if int(getattr(exc, "status_code", 0) or 0) != 429:
        return False
    code = _error_code(exc)
    return "insufficient_quota" not in code


def retry_after_seconds(
    exc: BaseException,
    *,
    attempt: int,
    maximum: float,
    random_value: Callable[[], float] = random.random,
) -> float:
    headers: Any = getattr(getattr(exc, "response", None), "headers", {}) or {}
    retry_ms = headers.get("retry-after-ms")
    if retry_ms:
        try:
            return min(maximum, max(0.0, float(retry_ms) / 1000.0))
        except (TypeError, ValueError):
            pass
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return min(maximum, max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            try:
                parsed = email.utils.parsedate_to_datetime(str(retry_after))
                now = datetime.now(timezone.utc)
                return min(maximum, max(0.0, (parsed - now).total_seconds()))
            except (TypeError, ValueError):
                pass
    ceiling = min(maximum, 2.0 ** max(0, attempt))
    return max(0.01, ceiling * random_value())
