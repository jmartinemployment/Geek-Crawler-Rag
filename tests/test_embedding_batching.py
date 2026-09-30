from __future__ import annotations

import pytest

from geek_crawler_rag.embedding_batching import (
        partition_embedding_batches,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_partition_respects_item_and_token_limits():
    batches = partition_embedding_batches(
        ["one two", "three four", "five six"],
        model="BAAI/bge-small-en-v1.5",
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


