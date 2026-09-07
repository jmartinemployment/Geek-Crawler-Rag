"""Reranker soft-disable behavior."""

from __future__ import annotations

import pytest

from geek_crawler_rag.rerank import Reranker


@pytest.mark.asyncio
async def test_reranker_disabled_keeps_order():
    r = Reranker(None, enabled=True)
    assert r.enabled is False
    out = await r.rerank("q", ["a", "b", "c"], top_n=2)
    assert out == [(0, 3.0), (1, 2.0)]
