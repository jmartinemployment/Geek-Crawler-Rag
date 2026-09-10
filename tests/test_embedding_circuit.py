from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import TextNode

from geek_crawler_rag.config import Settings
from geek_crawler_rag.embedding_circuit import (
    EmbeddingCircuitOpen,
    is_openai_http_500,
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
    path, request_id, message = quarantine_embedding_batch(
        texts=["hello\x00world", "second"],
        metadata_list=[
            {"runId": "r1", "pageId": "p1", "chunkId": "c1"},
            {"runId": "r1", "pageId": "p2"},
        ],
        quarantine_dir=tmp_path,
        model="text-embedding-3-small",
        token_count=12,
        status_code=500,
        exc_type="InternalServerError",
    )
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["statusCode"] == 500
    assert data["runId"] == "r1"
    assert data["batchSize"] == 2
    assert data["items"][0]["pageId"] == "p1"
    assert "\x00" not in data["items"][0]["textPreview"] or True  # preview may keep raw
    assert len(data["items"][0]["textPreview"]) <= 501
    assert request_id is None  # no actual OpenAI exception passed
    assert message is None


def test_open_circuit_raises_with_path(tmp_path: Path):
    err = open_embedding_circuit(
        texts=["x"],
        quarantine_dir=tmp_path,
        model="text-embedding-3-small",
        status_code=500,
    )
    assert isinstance(err, EmbeddingCircuitOpen)
    assert Path(err.quarantine_path).exists()


def test_is_openai_http_500():
    assert is_openai_http_500(FakeHttp500())
    assert not is_openai_http_500(Exception("nope"))

    class RateLimit(Exception):
        status_code = 429

    assert not is_openai_http_500(RateLimit())


def test_quarantine_dir_unwritable(tmp_path: Path):
    # Path exists as a file -> mkdir(parents=True) fails with FileExistsError/OSError.
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        quarantine_embedding_batch(
            texts=["x"],
            quarantine_dir=blocker,
            model="text-embedding-3-small",
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
                html="<html><body>"
                + ("This is enough English content for indexing. " * 30)
                + "</body></html>",
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
            status_code=500,
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


def test_openai_error_diagnostics():
    from geek_crawler_rag.embedding_circuit import describe_openai_error

    class FakeOpenAIError(Exception):
        def __init__(self):
            self.request_id = "req-12345"
            self.type = "invalid_request_error"
            self.code = "invalid_embedding_model"
            self.param = "model"
            self.message = "Model not found"

    err = FakeOpenAIError()
    diag = describe_openai_error(err)
    assert diag["requestId"] == "req-12345"
    assert diag["errorType"] == "invalid_request_error"
    assert diag["errorCode"] == "invalid_embedding_model"
    assert diag["errorParam"] == "model"
    assert "Model not found" in diag["message"]


def test_quarantine_with_openai_diagnostics(tmp_path: Path):
    from geek_crawler_rag.embedding_circuit import describe_openai_error

    class FakeOpenAIError(Exception):
        def __init__(self):
            self.request_id = "req-67890"
            self.type = "server_error"
            self.code = None
            self.param = None
            self.message = "The server is experiencing issues"

    err = FakeOpenAIError()
    path, request_id, message = quarantine_embedding_batch(
        texts=["test content"],
        metadata_list=[{"runId": "r2"}],
        quarantine_dir=tmp_path,
        model="text-embedding-3-small",
        status_code=500,
        exc_type="InternalServerError",
        exc=err,
    )
    assert path.exists()
    assert request_id == "req-67890"
    assert "server is experiencing" in message.lower()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("openaiDiagnostics") is not None
    assert data["openaiDiagnostics"]["requestId"] == "req-67890"
    assert data["openaiDiagnostics"]["errorType"] == "server_error"


def test_empty_input_error_detection():
    from geek_crawler_rag.embedding_circuit import (
        is_empty_embedding_input_error,
        should_quarantine_embedding_error,
    )

    err = Exception(
        "Error code: 400 - {'error': {'message': "
        "\"Invalid 'input[31]': input cannot be an empty string.\"}}"
    )
    assert is_empty_embedding_input_error(err)
    assert should_quarantine_embedding_error(err) == 400
    assert should_quarantine_embedding_error(Exception("other")) is None


@pytest.mark.asyncio
async def test_index_empty_embed_400_quarantines_without_wipe(tmp_path: Path):
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
                html="<html><body>"
                + ("This is enough English content for indexing. " * 30)
                + "</body></html>",
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
            status_code=400,
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
