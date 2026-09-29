"""Cross-run embedding reuse: what it returns, and what it refuses to return."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag import llama_engine as le
from geek_crawler_rag.vector_cache import VectorCache, cache_key


class _FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    def __aiter__(self):
        async def gen():
            for d in self._docs:
                yield d

        return gen()


def _cache(docs, *, dimensions: int = 3, raises: Exception | None = None):
    collection = MagicMock()
    if raises is not None:
        collection.find = MagicMock(side_effect=raises)
    else:
        collection.find = MagicMock(return_value=_FakeCursor(docs))
    collection.bulk_write = AsyncMock()
    db = {"rag_vector_cache": collection}
    cache = VectorCache(db, model="text-embedding-3-small", dimensions=dimensions)
    return cache, collection


def test_key_includes_the_model():
    # Two models index different spaces. A vector cached under one and served to
    # the other scores nonsense without erroring, so they must not share a key.
    a = cache_key("same text", model="text-embedding-3-small")
    b = cache_key("same text", model="text-embedding-3-large")
    assert a != b
    assert cache_key("same text", model="m") == cache_key("same text", model="m")


@pytest.mark.asyncio
async def test_a_hit_returns_the_stored_vector():
    key = cache_key("hello", model="text-embedding-3-small")
    cache, _ = _cache([{"key": key, "vector": [0.1, 0.2, 0.3]}])

    found = await cache.get_many(["hello"])

    assert found == {"hello": [0.1, 0.2, 0.3]}
    assert cache.hits == 1
    assert cache.misses == 0


@pytest.mark.asyncio
async def test_a_wrong_width_vector_is_refused():
    # Belongs to another model or a changed dimension setting. Returning it
    # would poison the collection silently.
    key = cache_key("hello", model="text-embedding-3-small")
    cache, _ = _cache([{"key": key, "vector": [0.1, 0.2]}], dimensions=3)

    assert await cache.get_many(["hello"]) == {}


@pytest.mark.asyncio
async def test_a_read_failure_is_a_miss_not_an_error():
    # A cache that cannot be read must not stop indexing.
    cache, _ = _cache([], raises=RuntimeError("mongo down"))

    assert await cache.get_many(["hello"]) == {}


@pytest.mark.asyncio
async def test_a_write_failure_does_not_fail_the_run():
    cache, collection = _cache([])
    collection.bulk_write = AsyncMock(side_effect=RuntimeError("mongo down"))

    await cache.put_many({"hello": [0.1, 0.2, 0.3]})  # must not raise


@pytest.mark.asyncio
async def test_put_skips_vectors_of_the_wrong_width():
    cache, collection = _cache([], dimensions=3)

    await cache.put_many({"bad": [0.1, 0.2]})

    collection.bulk_write.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_embeds_only_the_misses_and_keeps_order():
    """The property the whole feature rests on: a node never gets another
    node's vector, however the cached and fresh vectors interleave."""
    engine = MagicMock()
    engine._settings = MagicMock(qdrant_collection="coll")
    engine._aclient = MagicMock()

    cached_vec = [9.0, 9.0, 9.0]
    cache = MagicMock()
    cache.get_many = AsyncMock(return_value={"b": cached_vec})
    cache.put_many = AsyncMock()
    engine._vector_cache = cache

    embedded: list[list[str]] = []

    async def fake_embed(texts, metadata_list=None):
        embedded.append(list(texts))
        return [[float(len(t))] * 3 for t in texts]

    engine.embed_texts = fake_embed

    out = await le.LlamaIndexEngine._embed_with_cache(
        engine, ["a", "b", "c", "b"], [MagicMock() for _ in range(4)]
    )

    # "b" was cached, so only a and c reached the model, once each.
    assert embedded == [["a", "c"]]
    # Order preserved, and both "b" positions carry the cached vector.
    assert out[1] == cached_vec
    assert out[3] == cached_vec
    assert out[0] == [1.0, 1.0, 1.0]
    assert out[2] == [1.0, 1.0, 1.0]
    cache.put_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_length_mismatch_falls_back_rather_than_mis_mapping():
    engine = MagicMock()
    engine._vector_cache = MagicMock()
    engine._vector_cache.get_many = AsyncMock(return_value={})
    engine._vector_cache.put_many = AsyncMock()

    calls: list[list[str]] = []

    async def short_embed(texts, metadata_list=None):
        calls.append(list(texts))
        # One short on the first call, correct on the fallback.
        return [[1.0]] * (len(texts) - 1) if len(calls) == 1 else [[1.0]] * len(texts)

    engine.embed_texts = short_embed

    out = await le.LlamaIndexEngine._embed_with_cache(
        engine, ["a", "b"], [MagicMock(), MagicMock()]
    )

    assert len(out) == 2, "fell back to embedding everything rather than mis-pairing"
    engine._vector_cache.put_many.assert_not_awaited()
