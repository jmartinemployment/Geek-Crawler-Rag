"""Shared fixtures.

The chunker takes a ``ChunkTokenizer`` rather than choosing one internally, so every chunking test
needs to supply one. Two options, and the fake is the default for a reason: constructing the real
one means loading bge-small's ONNX model, which is 69 MB and several hundred milliseconds -- per
test process, and pointless for asserting window arithmetic.

``real_chunk_tokenizer`` exists for the handful of assertions that are actually about the model's
tokenization rather than about windowing, and is skipped when the model is not cached locally so
the suite stays runnable offline.
"""

from __future__ import annotations

import re

import pytest

#: Words and single non-space characters. Close enough to WordPiece for window arithmetic, and
#: deliberately NOT a WordPiece implementation -- a fake that tries to be accurate invites tests
#: that assert on the fake's quirks instead of the chunker's behaviour.
_TOKEN = re.compile(r"\w+|[^\w\s]")


class FakeChunkTokenizer:
    """Deterministic ``ChunkTokenizer`` with no model behind it.

    Spans come from a regex over the input, so slicing is exercised exactly as it is in production:
    the chunker still cuts the original string at span boundaries and never reconstructs text.
    """

    def __init__(self, *, sequence_limit: int = 512, special_token_overhead: int = 2) -> None:
        self._sequence_limit = sequence_limit
        self._overhead = special_token_overhead

    @property
    def sequence_limit(self) -> int:
        return self._sequence_limit

    @property
    def special_token_overhead(self) -> int:
        return self._overhead

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _TOKEN.finditer(text or "")]


@pytest.fixture
def chunk_tokenizer() -> FakeChunkTokenizer:
    return FakeChunkTokenizer()


@pytest.fixture
def unlimited_chunk_tokenizer() -> FakeChunkTokenizer:
    """No ceiling, for tests that deliberately ask for a window wider than a real model accepts."""
    return FakeChunkTokenizer(sequence_limit=0, special_token_overhead=0)


@pytest.fixture(scope="session")
def real_chunk_tokenizer():
    """The actual bge-small tokenizer. Skipped rather than downloading a model mid-suite."""
    pytest.importorskip("fastembed")
    from fastembed import TextEmbedding

    from geek_crawler_rag.chunk_tokenizer import WordPieceChunkTokenizer

    try:
        model = TextEmbedding("BAAI/bge-small-en-v1.5", threads=2)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"bge-small not available locally: {exc}")
    return WordPieceChunkTokenizer(model.model.tokenizer)
