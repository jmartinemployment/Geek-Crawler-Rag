"""Token counting and batch partitioning for embedding calls.

Named ``embedding_throttle`` until 2026-09-30, when the throttle it was named for was deleted along
with the metered remote API it paced. What is left counts tokens and splits a list of texts into
batches -- no waiting, no rate limit, no retry classification.

**These counts are the model's own WordPiece, not tiktoken BPE.** They were tiktoken until
2026-09-30, and the mismatch was not a rounding difference: WordPiece runs far longer on code and
technical identifiers, so a parent inside its configured 480-token budget arrived at inference over
the model's 512 ceiling and was silently truncated -- 136 of 1,984 parent points, worst case 638
tokens. The chunker (``chunk``) and this module now measure with the same ``ChunkTokenizer`` the
embedder derives from its own model, so "how many tokens" has one answer everywhere.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from geek_crawler_rag.chunk_tokenizer import ChunkTokenizer


@dataclass(frozen=True)
class EmbeddingBatch:
    texts: list[str]
    token_count: int


def embedding_token_count(texts: Sequence[str], *, tokenizer: ChunkTokenizer) -> int:
    """Exact content-token count for these texts, as the embedding model counts them.

    Content tokens only -- the model's ``special_token_overhead`` is per input, not per batch, and
    this total exists to size batches rather than to decide whether one item fits.

    The ``model: str`` parameter this used to take is gone. It was accepted and never used, because
    tiktoken has no encoding for a local ONNX model; keeping a parameter that names the model while
    counting with a different tokenizer is precisely how the two drifted apart unnoticed.
    """
    return max(1, sum(len(tokenizer.token_spans(text)) for text in texts))


def partition_embedding_batches(
    texts: Sequence[str],
    *,
    tokenizer: ChunkTokenizer,
    max_items: int,
    max_tokens: int,
) -> list[EmbeddingBatch]:
    if max_items <= 0 or max_tokens <= 0:
        raise ValueError("embedding batch limits must be positive")

    batches: list[EmbeddingBatch] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = embedding_token_count([text], tokenizer=tokenizer)
        if tokens > max_tokens:
            raise ValueError(
                f"single embedding input tokens={tokens} exceeds max={max_tokens}"
            )
        if current and (
            len(current) >= max_items or current_tokens + tokens > max_tokens
        ):
            batches.append(EmbeddingBatch(current, current_tokens))
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append(EmbeddingBatch(current, current_tokens))
    return batches

