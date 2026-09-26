"""The §3a amendment: a bounded, logged retry for failures with no cause in this repo.

What these pin is as much about what does NOT retry as what does. The rule the amendment
narrows is still in force everywhere else: an empty-input 400 is a defect in what we
sent, a 429 is the throttle's job, and once attempts are spent the batch quarantines or
the job fails carrying the real error. Nothing here may turn a failure into apparent
success -- that would be the fallback the rule forbids, not the retry it now permits.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError

from geek_crawler_rag.config import Settings
from geek_crawler_rag.embedding_throttle import EmbeddingBatch
from geek_crawler_rag.llama_engine import LlamaIndexEngine

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status_code=status, request=REQUEST)


def _engine(monkeypatch: pytest.MonkeyPatch, retries: int) -> LlamaIndexEngine:
    """A LlamaIndexEngine with construction bypassed: these tests exercise _embed_batch only.

    Building a real one reaches OpenAI, Qdrant and a 500MB SPLADE download, none of
    which this behaviour depends on.
    """
    engine = object.__new__(LlamaIndexEngine)
    engine._settings = Settings(
        openai_api_key="test",
        openai_embedding_transient_retries=retries,
        openai_embedding_retry_max_seconds=0.0,  # keep the suite fast; 0 = no sleep
        embedding_quarantine_dir="/tmp/geek-crawler-rag-test-quarantine",
    )
    engine._embedding_call_lock = asyncio.Lock()

    class _NoThrottle:
        async def acquire(self, _tokens: int) -> None:
            return None

    engine._embedding_throttle = _NoThrottle()
    return engine


def _batch() -> EmbeddingBatch:
    return EmbeddingBatch(texts=["dext receipt capture"], token_count=4)


@pytest.mark.asyncio
async def test_connection_error_is_retried_and_then_succeeds():
    """The failure that killed runId=73243bae at chunk 8,609 now costs a retry, not a run."""
    engine = _engine(pytest.MonkeyPatch(), retries=2)
    engine._embed_model = type("M", (), {})()
    engine._embed_model.aget_text_embedding_batch = AsyncMock(
        side_effect=[
            APIConnectionError(request=REQUEST),
            [[0.1, 0.2]],
        ]
    )

    result = await engine._embed_batch(_batch(), None)

    assert result == [[0.1, 0.2]]
    assert engine._embed_model.aget_text_embedding_batch.await_count == 2


@pytest.mark.asyncio
async def test_timeout_is_retried():
    engine = _engine(pytest.MonkeyPatch(), retries=1)
    engine._embed_model = type("M", (), {})()
    engine._embed_model.aget_text_embedding_batch = AsyncMock(
        side_effect=[APITimeoutError(request=REQUEST), [[0.3]]]
    )

    assert await engine._embed_batch(_batch(), None) == [[0.3]]


@pytest.mark.asyncio
async def test_attempts_are_bounded_and_the_real_error_survives(tmp_path):
    """Exhausting the budget must not invent success, and must not hide the cause.

    A 500 is retryable but also quarantinable, so when the budget is spent the batch
    takes the quarantine path exactly as it did before the amendment.
    """
    engine = _engine(pytest.MonkeyPatch(), retries=2)
    engine._settings = engine._settings.model_copy(
        update={"embedding_quarantine_dir": str(tmp_path)}
    )
    engine._embed_model = type("M", (), {})()
    engine._embed_model.aget_text_embedding_batch = AsyncMock(
        side_effect=InternalServerError(
            "boom", response=_response(500), body=None
        )
    )

    from geek_crawler_rag.embedding_circuit import EmbeddingCircuitOpen

    with pytest.raises(EmbeddingCircuitOpen):
        await engine._embed_batch(_batch(), None)

    # 2 retries = 3 attempts, then quarantine. Not 4, not forever.
    assert engine._embed_model.aget_text_embedding_batch.await_count == 3


@pytest.mark.asyncio
async def test_bad_request_is_not_retried_at_all():
    """A 400 is a defect in what we sent. Retrying it would hide our own bug."""
    engine = _engine(pytest.MonkeyPatch(), retries=3)
    engine._embed_model = type("M", (), {})()
    engine._embed_model.aget_text_embedding_batch = AsyncMock(
        side_effect=BadRequestError("bad", response=_response(400), body=None)
    )

    with pytest.raises(Exception):
        await engine._embed_batch(_batch(), None)

    assert engine._embed_model.aget_text_embedding_batch.await_count == 1


@pytest.mark.asyncio
async def test_zero_retries_restores_the_previous_behaviour_exactly():
    """The amendment is switchable: 0 means one attempt and an immediate failure."""
    engine = _engine(pytest.MonkeyPatch(), retries=0)
    engine._embed_model = type("M", (), {})()
    engine._embed_model.aget_text_embedding_batch = AsyncMock(
        side_effect=APIConnectionError(request=REQUEST)
    )

    with pytest.raises(APIConnectionError):
        await engine._embed_batch(_batch(), None)

    assert engine._embed_model.aget_text_embedding_batch.await_count == 1


def test_classifier_names_only_causes_outside_this_repo():
    assert LlamaIndexEngine._transient_embedding_error(APITimeoutError(request=REQUEST)) == "timeout"
    assert LlamaIndexEngine._transient_embedding_error(APIConnectionError(request=REQUEST)) == "connection"
    assert (
        LlamaIndexEngine._transient_embedding_error(
            InternalServerError("x", response=_response(503), body=None)
        )
        == "server_5xx"
    )
    # Ours to fix, or the throttle's to prevent: never retried.
    assert LlamaIndexEngine._transient_embedding_error(
        BadRequestError("x", response=_response(400), body=None)
    ) is None
    assert LlamaIndexEngine._transient_embedding_error(ValueError("nope")) is None
