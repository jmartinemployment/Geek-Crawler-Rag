from __future__ import annotations

from geek_crawler_rag.embedding_batching import (
    embedding_token_count,
    partition_embedding_batches,
)


def test_partition_respects_item_and_token_limits(chunk_tokenizer):
    batches = partition_embedding_batches(
        ["one two", "three four", "five six"],
        tokenizer=chunk_tokenizer,
        max_items=2,
        max_tokens=100,
    )

    assert [len(batch.texts) for batch in batches] == [2, 1]
    assert all(batch.token_count > 0 for batch in batches)


def test_partition_splits_on_the_token_budget_not_just_item_count(chunk_tokenizer):
    texts = ["alpha beta gamma delta", "epsilon zeta eta theta", "iota kappa"]
    batches = partition_embedding_batches(
        texts, tokenizer=chunk_tokenizer, max_items=99, max_tokens=5
    )
    assert len(batches) == 3, "each 4-token text alone exceeds a 5-token budget with a neighbour"
    for batch in batches:
        assert batch.token_count <= 5


def test_a_single_item_over_the_batch_budget_raises(chunk_tokenizer):
    try:
        partition_embedding_batches(
            ["one two three four five"],
            tokenizer=chunk_tokenizer,
            max_items=10,
            max_tokens=3,
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "exceeds max" in str(exc)


def test_token_count_is_the_tokenizers_own_count(chunk_tokenizer):
    """Counted with the embedder's tokenizer, not an approximation of it.

    This took a `model: str` it never used and counted with tiktoken regardless, so a call could
    name bge-small and measure cl100k_base. That is how the chunker and the embedder came to
    disagree by 158 tokens on a single chunk, silently.
    """
    text = "invoice = Invoice.model_validate_json(raw_json)"
    assert embedding_token_count([text], tokenizer=chunk_tokenizer) == len(
        chunk_tokenizer.token_spans(text)
    )
    # Never zero: a batch of empties still costs one request.
    assert embedding_token_count([""], tokenizer=chunk_tokenizer) == 1
    assert embedding_token_count([], tokenizer=chunk_tokenizer) == 1


def test_counts_are_additive_across_texts(chunk_tokenizer):
    a, b = "alpha beta", "gamma delta epsilon"
    assert embedding_token_count([a, b], tokenizer=chunk_tokenizer) == (
        embedding_token_count([a], tokenizer=chunk_tokenizer)
        + embedding_token_count([b], tokenizer=chunk_tokenizer)
    )
