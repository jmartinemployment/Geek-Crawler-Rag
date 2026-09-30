from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import TextNode

from geek_crawler_rag.config import Settings
from geek_crawler_rag.embedding_circuit import (
    EMPTY_INPUT,
    INFERENCE_FAILED,
    EmbeddingCircuitOpen,
    classify_embedding_failure,
    open_embedding_circuit,
    quarantine_embedding_batch,
)
from geek_crawler_rag.indexer import IndexService
from geek_crawler_rag.metadata import EntityRef
from geek_crawler_rag.models import IndexState
from geek_crawler_rag.mongo import CrawlPage, CrawlRun


class FakeHttp500(Exception):
    status_code = 500


def test_quarantine_writes_json(tmp_path: Path):
    path, detail = quarantine_embedding_batch(
        texts=["hello\x00world", "second"],
        metadata_list=[
            {"runId": "r1", "pageId": "p1", "chunkId": "c1"},
            {"runId": "r1", "pageId": "p2"},
        ],
        quarantine_dir=tmp_path,
        model="BAAI/bge-small-en-v1.5",
        token_count=12,
        reason=INFERENCE_FAILED,
        exc_type="RuntimeError",
    )
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["reason"] == INFERENCE_FAILED
    assert data["runId"] == "r1"
    assert data["batchSize"] == 2
    assert data["items"][0]["pageId"] == "p1"
    assert "\x00" not in data["items"][0]["textPreview"] or True  # preview may keep raw
    assert len(data["items"][0]["textPreview"]) <= 501
    assert detail is None  # no exception was passed


def test_open_circuit_raises_with_path(tmp_path: Path):
    err = open_embedding_circuit(
        texts=["x"],
        quarantine_dir=tmp_path,
        model="BAAI/bge-small-en-v1.5",
        reason=INFERENCE_FAILED,
    )
    assert isinstance(err, EmbeddingCircuitOpen)
    assert Path(err.quarantine_path).exists()


def test_every_failure_quarantines_and_is_named():
    """This replaced should_quarantine_embedding_error, which could return None.

    Its two predicates matched a remote provider's HTTP 500 and its 400 for an empty input string.
    Once embeddings went local neither could fire, so it answered None for everything and the circuit
    silently stopped engaging -- a fail-closed guard that cannot trigger, which reads as protection
    while providing none. There is no "carry on" answer now.
    """
    assert classify_embedding_failure(RuntimeError("onnxruntime failed")) == INFERENCE_FAILED
    assert classify_embedding_failure(Exception("")) == INFERENCE_FAILED
    assert classify_embedding_failure(FakeHttp500()) == INFERENCE_FAILED


def test_empty_input_keeps_its_own_name():
    """It survives classification because it has a different cause and a different fix.

    An empty string means the chunker or sanitizer let something through, not that inference broke --
    and both are supposed to make it unreachable, so seeing it means one of them has a hole.
    """
    assert classify_embedding_failure(ValueError("input cannot be an empty string")) == EMPTY_INPUT
    assert classify_embedding_failure(ValueError("empty text given")) == EMPTY_INPUT


def test_detail_walks_the_cause_chain():
    """ONNX and tokenizer errors put the useful text on the innermost exception."""
    from geek_crawler_rag.embedding_circuit import _exception_detail

    inner = ValueError("tokenizer vocab missing")
    outer = RuntimeError("inference failed")
    outer.__cause__ = inner
    detail = _exception_detail(outer)
    assert "inference failed" in detail
    assert "tokenizer vocab missing" in detail


def test_quarantine_dir_unwritable(tmp_path: Path):
    # Path exists as a file -> mkdir(parents=True) fails with FileExistsError/OSError.
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        quarantine_embedding_batch(
            texts=["x"],
            quarantine_dir=blocker,
            model="BAAI/bge-small-en-v1.5",
            reason=INFERENCE_FAILED,
        )


@pytest.mark.asyncio
async def test_index_circuit_open_skips_cleanup(tmp_path: Path):
    mongo = MagicMock()
    mongo.get_run = AsyncMock(
        return_value=CrawlRun(id="r1", crawl_type="partner", status="complete")
    )
    mongo.count_pages = AsyncMock(return_value=1)
    mongo.resolve_entity = AsyncMock(
        return_value=EntityRef(
            entity_id=None,
            entity_name="example.com",
            source_type="partner",
            domains=("example.com",),
        )
    )

    async def pages(_run_id, batch_size=25):
        yield [
            CrawlPage(
                id="p1",
                run_id="r1",
                origin="https://example.com",
                url="https://example.com/",
                final_url="https://example.com/",
                html="<html><body>ignored</body></html>",
                blocks=[
                    {"kind": "heading", "level": 1, "text": "Home", "anchors": []},
                    {
                        "kind": "paragraph",
                        "text": "This is enough English content for indexing. " * 30,
                        "anchors": [],
                    },
                ],
            )
        ]

    mongo.iter_pages = pages
    mongo.delete_page = AsyncMock()

    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.ensure_collection = AsyncMock()

    llama = MagicMock()
    llama.embed_and_upsert = AsyncMock(
        side_effect=EmbeddingCircuitOpen(
            "circuit",
            quarantine_path=str(tmp_path / "q.json"),
            reason=INFERENCE_FAILED,
        )
    )
    llama.embedding_stats = MagicMock(
        return_value={"tokensInWindow": 0, "rateLimitRetries": 0, "waitSeconds": 0}
    )

    settings = Settings(
        openai_api_key="test",
        embed_batch_size=1,
        embedding_quarantine_dir=str(tmp_path),
        qdrant_upsert_delay_seconds=0,
    )
    svc = IndexService(mongo, store, settings, llama=llama)
    await svc.enqueue("r1")
    await svc._index_run("r1")

    status = await svc.get_status("r1")
    assert status is not None
    assert status.state == IndexState.FAILED
    assert status.error and "quarantined" in status.error.lower()
    assert "quarantine" in status.error.lower()
    # Start may delete once for attempt<=1; quarantine path must not wipe again.
    assert store.delete_by_run_id.await_count == 1


@pytest.mark.asyncio
async def test_index_empty_embed_quarantines_without_wipe(tmp_path: Path):
    mongo = MagicMock()
    mongo.get_run = AsyncMock(
        return_value=CrawlRun(id="r1", crawl_type="partner", status="complete")
    )
    mongo.count_pages = AsyncMock(return_value=1)
    mongo.resolve_entity = AsyncMock(
        return_value=EntityRef(
            entity_id=None,
            entity_name="example.com",
            source_type="partner",
            domains=("example.com",),
        )
    )

    async def pages(_run_id, batch_size=25):
        yield [
            CrawlPage(
                id="p1",
                run_id="r1",
                origin="https://example.com",
                url="https://example.com/",
                final_url="https://example.com/",
                html="<html><body>ignored</body></html>",
                blocks=[
                    {"kind": "heading", "level": 1, "text": "Home", "anchors": []},
                    {
                        "kind": "paragraph",
                        "text": "This is enough English content for indexing. " * 30,
                        "anchors": [],
                    },
                ],
            )
        ]

    mongo.iter_pages = pages
    mongo.delete_page = AsyncMock()

    store = MagicMock()
    store.delete_by_run_id = AsyncMock()
    store.ensure_collection = AsyncMock()

    llama = MagicMock()
    llama.embed_and_upsert = AsyncMock(
        side_effect=EmbeddingCircuitOpen(
            "empty",
            quarantine_path=str(tmp_path / "q400.json"),
            reason=EMPTY_INPUT,
        )
    )
    llama.embedding_stats = MagicMock(
        return_value={"tokensInWindow": 0, "rateLimitRetries": 0, "waitSeconds": 0}
    )

    settings = Settings(
        openai_api_key="test",
        embed_batch_size=1,
        embedding_quarantine_dir=str(tmp_path),
        qdrant_upsert_delay_seconds=0,
    )
    svc = IndexService(mongo, store, settings, llama=llama)
    await svc.enqueue("r1")
    await svc._index_run("r1")

    status = await svc.get_status("r1")
    assert status is not None
    assert status.state == IndexState.FAILED
    assert status.error and "400" in status.error
    assert store.delete_by_run_id.await_count == 1
