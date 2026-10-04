"""Embedding dedupe within a flush, and the deterministic point id."""

from __future__ import annotations

import functools

from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag.qdrant_store import point_id


def test_point_id_is_deterministic():
    a = point_id("run1", "page1", "chunk1")
    b = point_id("run1", "page1", "chunk1")
    c = point_id("run1", "page1", "chunk2")
    assert a == b
    assert a != c


@pytest.mark.asyncio
async def test_embed_cache_sends_each_distinct_string_once():
    """Identical parent/child text must cost one OpenAI call, not two."""
    from llama_index.core.schema import TextNode
    from geek_crawler_rag import llama_engine as le

    nodes = [
        TextNode(id_="p0", text="SAME"),   # parent
        TextNode(id_="c0", text="SAME"),   # child, identical text
        TextNode(id_="c1", text="OTHER"),
    ]
    sent: list[list[str]] = []

    async def fake_embed_texts(texts, *, metadata_list=None):
        sent.append(list(texts))
        return [[float(i)] for i in range(len(texts))]

    engine = MagicMock()
    engine._settings = MagicMock(qdrant_collection="coll")
    engine._aclient = MagicMock()
    engine._vector_store = MagicMock()
    # No cross-run cache in these tests: they assert the per-flush dedupe, and a
    # MagicMock here would be awaited as if it were one.
    engine._vector_cache = None
    # embed_and_upsert delegates to _embed_deduplicated; on a MagicMock engine that is a mock,
    # not a coroutine. Bind the real one so these tests still exercise the real path.
    engine._embed_deduplicated = functools.partial(
        le.LlamaIndexEngine._embed_deduplicated, engine
    )
    engine._vector_store.async_add = AsyncMock()
    engine.embed_texts = fake_embed_texts

    n = await le.LlamaIndexEngine.embed_and_upsert(engine, nodes)

    assert n == 3
    assert sent == [["SAME", "OTHER"]], "duplicate string was sent twice"
    added = engine._vector_store.async_add.await_args.args[0]
    assert [x.id_ for x in added] == ["p0", "c0", "c1"]
    # both nodes sharing text get the same vector
    assert added[0].embedding == added[1].embedding
    assert added[2].embedding != added[0].embedding


@pytest.mark.asyncio
async def test_embed_cache_falls_back_on_length_mismatch():
    """A short return from embed_texts must not mis-map vectors onto nodes."""
    from llama_index.core.schema import TextNode
    from geek_crawler_rag import llama_engine as le

    nodes = [TextNode(id_="p0", text="SAME"), TextNode(id_="c0", text="SAME")]
    calls: list[list[str]] = []

    async def flaky_embed_texts(texts, *, metadata_list=None):
        calls.append(list(texts))
        if len(calls) == 1:
            return []                      # mismatch on the cached attempt
        return [[1.0]] * len(texts)        # fallback path

    engine = MagicMock()
    engine._settings = MagicMock(qdrant_collection="coll")
    engine._aclient = MagicMock()
    engine._vector_store = MagicMock()
    # No cross-run cache in these tests: they assert the per-flush dedupe, and a
    # MagicMock here would be awaited as if it were one.
    engine._vector_cache = None
    # embed_and_upsert delegates to _embed_deduplicated; on a MagicMock engine that is a mock,
    # not a coroutine. Bind the real one so these tests still exercise the real path.
    engine._embed_deduplicated = functools.partial(
        le.LlamaIndexEngine._embed_deduplicated, engine
    )
    engine._vector_store.async_add = AsyncMock()
    engine.embed_texts = flaky_embed_texts

    n = await le.LlamaIndexEngine.embed_and_upsert(engine, nodes)

    assert n == 2
    assert calls == [["SAME"], ["SAME", "SAME"]], "did not fall back uncached"
