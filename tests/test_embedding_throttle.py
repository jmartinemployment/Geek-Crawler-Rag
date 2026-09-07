from __future__ import annotations

import pytest

from geek_crawler_rag.embedding_throttle import (
    EmbeddingThrottle,
    is_retryable_rate_limit,
    partition_embedding_batches,
    retry_after_seconds,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.asyncio
async def test_throttle_waits_for_rolling_window_capacity():
    clock = FakeClock()
    throttle = EmbeddingThrottle(100, clock=clock, sleep=clock.sleep)

    assert await throttle.acquire(70) == 0
    waited = await throttle.acquire(40)

    assert waited == pytest.approx(60.0)
    assert throttle.total_wait_seconds == pytest.approx(60.0)
    assert throttle.tokens_in_window == 40


def test_partition_respects_item_and_token_limits():
    batches = partition_embedding_batches(
        ["one two", "three four", "five six"],
        model="text-embedding-3-small",
        max_items=2,
        max_tokens=100,
    )

    assert [len(batch.texts) for batch in batches] == [2, 1]
    assert all(batch.token_count > 0 for batch in batches)


class FakeRateLimit:
    status_code = 429
    body = {"error": {"code": "rate_limit_exceeded"}}

    class response:
        headers = {"retry-after": "12"}


class FakeQuotaLimit(FakeRateLimit):
    body = {"error": {"code": "insufficient_quota"}}


def test_retry_classification_and_retry_after():
    transient = FakeRateLimit()

    assert is_retryable_rate_limit(transient)
    assert not is_retryable_rate_limit(FakeQuotaLimit())
    assert retry_after_seconds(transient, attempt=2, maximum=60) == 12
