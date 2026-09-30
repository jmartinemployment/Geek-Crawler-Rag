"""Dense embeddings, computed locally.

Replaces ``OpenAIEmbedding``. There is no API key, no rate limit, no per-chunk cost and no external
service in the indexing path -- see ``plans/go-fully-local-bge-small-bm25.md``.

**Why a local adapter instead of ``llama-index-embeddings-fastembed``.**

``embed_and_upsert`` assigns ``node.embedding`` itself and only then calls ``async_add``
(``llama_engine.py``), so LlamaIndex never embeds on the dense path. Exactly two methods are ever
reached: ``aget_text_embedding_batch`` and ``aget_query_embedding``. That makes the wrapper package a
dependency bought for nothing -- and worse, LlamaIndex's async embedding methods generally wrap the
sync call, which would run ONNX inference on the event loop.

That last point is not hypothetical. A py-spy profile on 2026-09-30 found 87% of indexing wall time
inside the sparse encoder on ``MainThread`` -- the asyncio event loop -- because
``QdrantVectorStore._build_points`` calls its encoder synchronously. Nothing else in the process
progressed during it. This adapter offloads every inference call with ``asyncio.to_thread`` so the
loop stays free for Mongo reads, Qdrant writes and the API's own requests.

Subclassing ``BaseEmbedding`` keeps ``LlamaSettings.embed_model`` valid for any LlamaIndex internal
that reads it, even though nothing in this repo does.

**Queries go through ``query_embed``, documents through ``embed``** -- the API that would apply a
model's query instruction prefix if it declared one. Measured for this model on 2026-09-30: it does
not. Same text down both paths gives ``cos = 1.0000``, so bge-small-en-v1.5 is symmetric under
fastembed 0.8.1, which matches BAAI's own guidance that the v1.5 models no longer need the
instruction. The split is kept because it is correct and because a future model swap may need it --
not because it is doing anything today. Do not "simplify" it into one path and assume nothing changed;
re-measure that cosine first.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastembed import TextEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class LocalDenseEmbedding(BaseEmbedding):
    """fastembed dense embeddings behind LlamaIndex's embedding interface."""

    _model: TextEmbedding = PrivateAttr()
    _threads: int = PrivateAttr()
    _max_tokens: int = PrivateAttr()
    _truncated: int = PrivateAttr()

    def __init__(
        self,
        *,
        model_name: str,
        embed_batch_size: int,
        threads: int,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name, embed_batch_size=embed_batch_size, **kwargs
        )
        self._threads = threads
        # Constructing this loads (and on a cold cache downloads) the ONNX model. Deliberately eager:
        # a missing model must fail at startup, not on the first page of the first index job.
        self._model = TextEmbedding(
            model_name=model_name, threads=threads, cache_dir=cache_dir
        )
        self._truncated = 0
        self._max_tokens = self._read_max_tokens()
        logger.info(
            "Local dense embedder ready model=%s threads=%s batch=%s max_tokens=%s",
            model_name,
            threads,
            embed_batch_size,
            self._max_tokens,
        )

    def _read_max_tokens(self) -> int:
        """The model's own sequence limit, read from its tokenizer rather than hardcoded.

        bge-small-en-v1.5 reports 512. Reading it means a model swap cannot leave a stale constant
        behind claiming the wrong ceiling.
        """
        try:
            tokenizer = self._model.model.tokenizer  # type: ignore[attr-defined]
            limit = int(tokenizer.truncation["max_length"])
            return limit if limit > 0 else 0
        except Exception:
            # Unknown rather than assumed: 0 disables the check instead of inventing a ceiling.
            logger.warning(
                "Could not read the embedding model's sequence limit; "
                "over-length inputs will not be reported"
            )
            return 0

    def _count_tokens(self, text: str) -> int:
        """Token count as the model sees it, **capped at the model's own limit.**

        The tokenizer has truncation enabled (that is where ``_max_tokens`` is read from), so this
        can never return more than the ceiling. The caller therefore tests ``>=``, not ``>``: an
        encoding sitting exactly on the limit is one that was truncated. The only false positive is
        text that happens to be exactly the limit long, which is worth a warning anyway.
        """
        tokenizer = self._model.model.tokenizer  # type: ignore[attr-defined]
        return len(tokenizer.encode(text).ids)

    @property
    def truncated_inputs(self) -> int:
        """How many inputs exceeded the model's limit and were silently truncated by it."""
        return self._truncated

    def _report_over_length(self, texts: list[str]) -> None:
        """Truncation is the model's default and it is silent. Make it visible and countable.

        ONNX truncates at ``max_length`` with no error, so a chunk at or over the limit yields a
        vector representing only its head. That is a fail-open, and the reason it can happen at all is a
        tokenizer mismatch: the chunker sizes parents and children with tiktoken's BPE
        (``embedding_throttle.partition_embedding_batches``) while the model counts WordPiece, which
        runs longer on technical text. So a parent inside its configured 480-token budget can still
        cross 512 here.

        Not raised. The chunk is legitimate and the run should continue -- but nobody should have to
        guess whether it happened. The count is the input for tuning
        ``parent_chunk_size_tokens`` empirically instead of by estimate.
        """
        if self._max_tokens <= 0:
            return
        for text in texts:
            n = self._count_tokens(text)
            if n >= self._max_tokens:
                self._truncated += 1
                logger.warning(
                    "Embedding input reached the model limit and was truncated: "
                    "tokens>=%s limit=%s chars=%s head=%r",
                    n,
                    self._max_tokens,
                    len(text),
                    text[:80],
                )

    @classmethod
    def class_name(cls) -> str:
        return "LocalDenseEmbedding"

    # -- documents -------------------------------------------------------------------------------

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One ONNX pass over the batch, in input order.

        fastembed yields numpy arrays; every caller here expects ``list[float]``, and Qdrant's client
        will not serialise an ndarray.
        """
        if not texts:
            return []
        self._report_over_length(texts)
        vectors = list(self._model.embed(texts, batch_size=self.embed_batch_size))
        if len(vectors) != len(texts):
            # Never mis-map: the caller zips these onto nodes by position.
            raise ValueError(
                f"embedder returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return [v.tolist() for v in vectors]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed_documents([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embed_documents(texts)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        # The hot path: aget_text_embedding_batch lands here. Off the event loop.
        return await asyncio.to_thread(self._get_text_embeddings, texts)

    # -- queries ---------------------------------------------------------------------------------

    def _get_query_embedding(self, query: str) -> list[float]:
        """``query_embed``, not ``embed`` -- see the asymmetry note in the module docstring."""
        vectors = list(self._model.query_embed(query))
        if not vectors:
            raise ValueError("embedder returned no vector for the query")
        return vectors[0].tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)
