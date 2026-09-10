from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from geek_crawler_rag.qdrant_store import find_existing_point_ids, point_id


class _Rec:
    def __init__(self, rid: str) -> None:
        self.id = rid


@pytest.mark.asyncio
async def test_find_existing_point_ids_returns_present_subset():
    client = MagicMock()
    client.retrieve = AsyncMock(return_value=[_Rec("a"), _Rec("c")])
    found = await find_existing_point_ids(client, "coll", ["a", "b", "c"])
    assert found == {"a", "c"}
    client.retrieve.assert_awaited_once()
    kwargs = client.retrieve.await_args.kwargs
    assert kwargs["with_payload"] is False
    assert kwargs["with_vectors"] is False


@pytest.mark.asyncio
async def test_find_existing_point_ids_batches():
    client = MagicMock()
    client.retrieve = AsyncMock(return_value=[])
    await find_existing_point_ids(client, "coll", [str(i) for i in range(600)], batch=256)
    # 600 ids at 256/batch -> 3 calls
    assert client.retrieve.await_count == 3


@pytest.mark.asyncio
async def test_find_existing_point_ids_empty_makes_no_calls():
    client = MagicMock()
    client.retrieve = AsyncMock(return_value=[])
    found = await find_existing_point_ids(client, "coll", [])
    assert found == set()
    client.retrieve.assert_not_awaited()


def test_point_id_is_deterministic():
    a = point_id("run1", "page1", "chunk1")
    b = point_id("run1", "page1", "chunk1")
    c = point_id("run1", "page1", "chunk2")
    assert a == b
    assert a != c


@pytest.mark.asyncio
async def test_embed_and_upsert_skips_already_committed(monkeypatch):
    """A re-run must not re-embed chunks already in Qdrant."""
    from llama_index.core.schema import TextNode

    from geek_crawler_rag import llama_engine as le

    nodes = [TextNode(id_=f"id{i}", text=f"body {i}") for i in range(3)]

    engine = MagicMock()
    engine._settings = MagicMock(qdrant_collection="coll")
    engine._aclient = MagicMock()
    engine._vector_store = MagicMock()
    engine._vector_store.async_add = AsyncMock()
    embedded: list[list[str]] = []

    async def fake_embed_texts(texts, *, metadata_list=None):
        embedded.append(list(texts))
        return [[0.0]] * len(texts)

    engine.embed_texts = fake_embed_texts

    # id0 and id2 already present -> only id1 should be embedded
    async def fake_find(client, collection, ids, *, batch=256):
        assert collection == "coll"
        return {"id0", "id2"}

    monkeypatch.setattr(le, "find_existing_point_ids", fake_find)

    n = await le.LlamaIndexEngine.embed_and_upsert(engine, nodes)

    assert n == 1
    assert embedded == [["body 1"]]
    added = engine._vector_store.async_add.await_args.args[0]
    assert [x.id_ for x in added] == ["id1"]


@pytest.mark.asyncio
async def test_embed_and_upsert_no_openai_call_when_all_present(monkeypatch):
    """Fully-committed batch must short-circuit without touching OpenAI."""
    from llama_index.core.schema import TextNode

    from geek_crawler_rag import llama_engine as le

    nodes = [TextNode(id_="id0", text="body")]
    engine = MagicMock()
    engine._settings = MagicMock(qdrant_collection="coll")
    engine._aclient = MagicMock()
    engine._vector_store = MagicMock()
    engine._vector_store.async_add = AsyncMock()
    engine.embed_texts = AsyncMock(side_effect=AssertionError("must not embed"))

    async def fake_find(client, collection, ids, *, batch=256):
        return {"id0"}

    monkeypatch.setattr(le, "find_existing_point_ids", fake_find)

    assert await le.LlamaIndexEngine.embed_and_upsert(engine, nodes) == 0
    engine._vector_store.async_add.assert_not_awaited()
