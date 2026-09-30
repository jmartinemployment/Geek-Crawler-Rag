"""The local dense embedder's contract.

Loads the real ONNX model rather than a stub: the properties worth pinning here -- vector width,
determinism, and not blocking the event loop -- are properties of the model and the threading, and a
fake would assert nothing. The suite already pulls fastembed models for the sparse side.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from geek_crawler_rag.local_embedding import LocalDenseEmbedding

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


@pytest.fixture(scope="module")
def embedder() -> LocalDenseEmbedding:
    return LocalDenseEmbedding(
        model_name=MODEL, embed_batch_size=64, threads=2
    )


@pytest.mark.asyncio
async def test_documents_embed_to_the_declared_width(embedder) -> None:
    """384, and the collection is created with `size: 384`. A mismatch is a rejected upsert."""
    vectors = await embedder._aget_text_embeddings(
        ["Accounts payable automation extracts invoice line items.", "Second document."]
    )
    assert len(vectors) == 2
    assert all(len(v) == DIM for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])


@pytest.mark.asyncio
async def test_identical_text_gives_an_identical_vector(embedder) -> None:
    """The cross-run vector cache keys on (model, text), so this must hold or the cache lies."""
    text = "The Acme XJ-4420-B pump requires firmware 7.3.1."
    first, second = await embedder._aget_text_embeddings([text, text])
    assert first == second


@pytest.mark.asyncio
async def test_order_is_preserved(embedder) -> None:
    """embed_and_upsert zips vectors onto nodes by position; a reorder mis-maps every one."""
    a, b, c = await embedder._aget_text_embeddings(["alpha alpha", "beta beta", "alpha alpha"])
    assert a == c
    assert a != b


@pytest.mark.asyncio
async def test_a_length_mismatch_raises_rather_than_mis_mapping(embedder) -> None:
    """Silent truncation of the vector list would attach the wrong vector to the wrong node."""
    original = embedder._model.embed

    def short(texts, **kwargs):  # noqa: ANN001, ANN202
        out = list(original(list(texts), **kwargs))
        return iter(out[:-1])

    embedder._model.embed = short
    try:
        with pytest.raises(ValueError, match="vectors for"):
            await embedder._aget_text_embeddings(["one", "two"])
    finally:
        embedder._model.embed = original


@pytest.mark.asyncio
async def test_queries_use_query_embed_and_are_symmetric_for_this_model(embedder) -> None:
    """Measured 2026-09-30: bge-small-en-v1.5 applies no query instruction under fastembed 0.8.1.

    The query path is kept separate because it is the API that *would* apply a prefix, and a future
    model swap may need it. This pins the current fact so that a swap which breaks the symmetry is
    visible here rather than as a quiet retrieval regression.
    """
    text = "Which subscription tier includes multi-currency consolidation?"
    (doc_vector,) = await embedder._aget_text_embeddings([text])
    query_vector = await embedder._aget_query_embedding(text)

    assert len(query_vector) == DIM
    assert _cos(query_vector, doc_vector) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.asyncio
async def test_inference_does_not_block_the_event_loop(embedder) -> None:
    """The reason this adapter exists.

    A py-spy profile on 2026-09-30 found 87% of indexing wall time inside the sparse encoder on the
    asyncio event loop, so nothing else in the process progressed during it. Dense inference must not
    repeat that: it runs in a worker thread, and a concurrent coroutine must keep being scheduled.
    """
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await embedder.aget_text_embedding_batch(
            ["a moderately long sentence about invoice automation and matching"] * 96
        )
    finally:
        task.cancel()

    assert ticks > 0, "the event loop was blocked for the whole of inference"


@pytest.mark.asyncio
async def test_empty_input_is_empty_output_not_an_error(embedder) -> None:
    """Callers sanitize and may filter everything out; that is not a failure."""
    assert await embedder._aget_text_embeddings([]) == []
