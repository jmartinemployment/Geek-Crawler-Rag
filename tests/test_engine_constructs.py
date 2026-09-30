"""The engine's constructor must actually run.

This exists because 410 tests passed while the service could not start. `llama_engine` referenced
`LocalDenseEmbedding` without importing it -- a patch matched the relative form `.embedding_sanitize`
when the file uses the absolute `geek_crawler_rag.embedding_sanitize`, so the import line was never
added. Nothing in the suite constructed `LlamaIndexEngine`, so a `NameError` on line 66 of the hot
path survived a green run and surfaced as a failed deploy: twelve health checks, then
`Application startup failed`.

Unit tests that only exercise unbound methods and mocked engines cannot see that class of error. This
one drives the real constructor with the Qdrant clients stubbed, which is the cheapest thing that
would have failed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from geek_crawler_rag.config import Settings
from geek_crawler_rag.llama_engine import LlamaIndexEngine


@pytest.fixture
def engine(monkeypatch) -> LlamaIndexEngine:
    """Construct for real, with only the network clients replaced."""
    monkeypatch.setattr("geek_crawler_rag.llama_engine.QdrantClient", MagicMock())
    monkeypatch.setattr("geek_crawler_rag.llama_engine.AsyncQdrantClient", MagicMock())
    monkeypatch.setattr("geek_crawler_rag.llama_engine.QdrantVectorStore", MagicMock())
    return LlamaIndexEngine(Settings(qdrant_url="http://localhost:6333", api_key="x"))


def test_the_constructor_runs(engine) -> None:
    """The whole point: every name the constructor references must resolve."""
    assert engine.embed_model is not None


def test_it_needs_no_api_key(monkeypatch) -> None:
    """The OPENAI_API_KEY guard made a missing key a hard startup failure for the whole service.

    Embeddings are local; there is no key to require. A settings object with nothing but a Qdrant URL
    and the service's own API key must be enough.
    """
    monkeypatch.setattr("geek_crawler_rag.llama_engine.QdrantClient", MagicMock())
    monkeypatch.setattr("geek_crawler_rag.llama_engine.AsyncQdrantClient", MagicMock())
    monkeypatch.setattr("geek_crawler_rag.llama_engine.QdrantVectorStore", MagicMock())
    LlamaIndexEngine(Settings(qdrant_url="http://localhost:6333", api_key="x"))


def test_the_model_width_matches_the_configured_dimensions(engine) -> None:
    """Qdrant rejects a vector that does not match its collection's declared size.

    The collection is created from `embedding_dimensions`, so a disagreement between the setting and
    the model means every upsert fails later with an error that reads as a Qdrant fault.
    """
    assert engine.embed_model.dimensions == 384


def test_a_dimension_mismatch_fails_at_construction(monkeypatch) -> None:
    monkeypatch.setattr("geek_crawler_rag.llama_engine.QdrantClient", MagicMock())
    monkeypatch.setattr("geek_crawler_rag.llama_engine.AsyncQdrantClient", MagicMock())
    monkeypatch.setattr("geek_crawler_rag.llama_engine.QdrantVectorStore", MagicMock())
    with pytest.raises(ValueError, match="does not match"):
        LlamaIndexEngine(
            Settings(qdrant_url="http://localhost:6333", api_key="x", embedding_dimensions=1536)
        )


def test_embedding_stats_keeps_the_contract_keys(engine) -> None:
    """rateLimitRetries and waitSeconds reach the webhook as fields pinned in a cross-repo contract
    with a byte-matching GeekBackend copy that CI diffs. They are zero now -- there is no throttle --
    but dropping them here reds that comparison and silently unbinds a receiver field."""
    stats = engine.embedding_stats()
    assert stats["rateLimitRetries"] == 0
    assert stats["waitSeconds"] == 0.0
    assert "truncatedInputs" in stats, "the one value here that now carries information"
