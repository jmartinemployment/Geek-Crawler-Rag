"""Token counting and batch partitioning for embedding calls.

Named ``embedding_throttle`` until 2026-09-30, when the throttle it was named for was deleted along
with the metered remote API it paced. What is left counts tokens and splits a list of texts into
batches -- no waiting, no rate limit, no retry classification.

**The token counts here are tiktoken BPE, and the embedder counts WordPiece.** That is a real
mismatch, not a rounding difference: WordPiece runs longer on technical text, so a chunk inside its
configured budget here can still cross the model's sequence limit and be silently truncated at
inference. ``LocalDenseEmbedding`` counts with the model's own tokenizer and reports every input that
hits the ceiling, which is the number to tune ``parent_chunk_size_tokens`` against. These counts remain
useful for batch *sizing*; they are not an authority on whether a single item fits.
"""

from __future__ import annotations

import asyncio
import email.utils
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import tiktoken


@dataclass(frozen=True)
class EmbeddingBatch:
    texts: list[str]
    token_count: int


def embedding_token_count(texts: Sequence[str], model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return max(1, sum(len(encoding.encode(text)) for text in texts))


def partition_embedding_batches(
    texts: Sequence[str],
    *,
    model: str,
    max_items: int,
    max_tokens: int,
) -> list[EmbeddingBatch]:
    if max_items <= 0 or max_tokens <= 0:
        raise ValueError("embedding batch limits must be positive")

    batches: list[EmbeddingBatch] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = embedding_token_count([text], model)
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

