"""The guarantee, asserted against the real model rather than a fake.

Truncation used to be a tuning problem: the chunker sized parents with tiktoken BPE, the model
counted WordPiece, and no value of ``parent_chunk_size_tokens`` could be correct because the two
sides were not measuring the same thing. Measured on the live corpus 2026-09-30 at the 480 setting:
136 of 1,984 parent points crossed the 512 ceiling, worst case 638 tokens, and the discarded tail
was Python source -- WordPiece expands code roughly threefold.

These tests use the model's OWN tokenizer, so they fail if that ever stops being true.
"""

from __future__ import annotations

from geek_crawler_rag.chunk import chunk_text, parent_child_units
from geek_crawler_rag.config import Settings

#: Deliberately code- and identifier-heavy. This is the content that broke the BPE estimate, so it
#: is the content the guarantee has to hold for.
CODE_HEAVY = (
    "Step 3: reconcile the invoice. "
    "invoice = Invoice.model_validate_json(raw_json) "
    "df = pd.DataFrame([item.model_dump() for item in invoice.lineItems]) "
    'df["amount"] = df["quantity"] * df["unitPrice"] '
    'if round(df["amount"].sum(), 2) != invoice.total: '
    'raise ValueError("line items do not add up") '
    "import great_expectations as gx; ctx = gx.get_context() "
    "Serial XJ-4420-B, firmware 7.3.1, SKU 88192-QA. "
) * 12


def _token_count(tokenizer, text: str) -> int:
    """Content tokens plus whatever the model prepends, i.e. what inference actually sees."""
    return len(tokenizer.token_spans(text)) + tokenizer.special_token_overhead


def test_the_configured_parent_budget_fits_the_model(real_chunk_tokenizer):
    """A config error caught here rather than as silent truncation in production."""
    settings = Settings()
    usable = (
        real_chunk_tokenizer.sequence_limit
        - real_chunk_tokenizer.special_token_overhead
    )
    assert settings.parent_chunk_size_tokens <= usable, (
        f"parent_chunk_size_tokens={settings.parent_chunk_size_tokens} exceeds the usable "
        f"budget {usable}; chunks would be truncated at inference"
    )
    assert settings.child_chunk_size_tokens <= usable


def test_no_chunk_of_code_heavy_text_can_exceed_the_model_limit(real_chunk_tokenizer):
    """The claim that replaced tuning: this is arithmetic, not an estimate."""
    settings = Settings()
    limit = real_chunk_tokenizer.sequence_limit

    chunks = chunk_text(
        CODE_HEAVY,
        tokenizer=real_chunk_tokenizer,
        size_tokens=settings.parent_chunk_size_tokens,
        overlap_tokens=settings.parent_chunk_overlap_tokens,
    )
    assert len(chunks) > 1, "the fixture must be long enough to actually window"
    for chunk in chunks:
        seen = _token_count(real_chunk_tokenizer, chunk)
        assert seen <= limit, f"chunk would be truncated: {seen} > {limit}"


def test_parent_and_child_points_both_fit(real_chunk_tokenizer):
    """Both tiers are embedded, so both have to fit. Parents are the tier that broke."""
    settings = Settings()
    limit = real_chunk_tokenizer.sequence_limit
    blocks = [
        {"kind": "heading", "level": 2, "text": "Worked Example", "anchors": []},
        {"kind": "paragraph", "text": CODE_HEAVY, "anchors": []},
        {"kind": "heading", "level": 2, "text": "Totals", "anchors": []},
        {"kind": "code", "text": 'df["amount"].sum()', "anchors": []},
    ]
    units = parent_child_units(
        blocks,
        tokenizer=real_chunk_tokenizer,
        child_size_tokens=settings.child_chunk_size_tokens,
        child_overlap_tokens=settings.child_chunk_overlap_tokens,
        parent_size_tokens=settings.parent_chunk_size_tokens,
        parent_overlap_tokens=settings.parent_chunk_overlap_tokens,
    )
    assert units
    for unit in units:
        for label, text in (("parent", unit.parent_text), ("child", unit.child_text)):
            seen = _token_count(real_chunk_tokenizer, text)
            assert seen <= limit, f"{label} would be truncated: {seen} > {limit}"


def test_chunks_are_verbatim_so_citation_verification_still_works(real_chunk_tokenizer):
    """Chunk text is the quote-verification target; a lossy chunk breaks verification.

    WordPiece decode is destructive -- uncased, and it splits on punctuation -- so
    ``Invoice.model_validate_json`` would round-trip as ``invoice. model _ validate _ json``.
    Slicing the original string at span boundaries is what keeps identifiers citable.
    """
    chunks = chunk_text(
        CODE_HEAVY, tokenizer=real_chunk_tokenizer, size_tokens=120, overlap_tokens=20
    )
    for chunk in chunks:
        assert chunk in CODE_HEAVY
    joined = " ".join(chunks)
    for literal in (
        "Invoice.model_validate_json",
        "pd.DataFrame",
        "XJ-4420-B",
        "88192-QA",
        "7.3.1",
        "great_expectations",
    ):
        assert literal in joined, f"{literal!r} did not survive chunking"


#: Long compound words, so most token boundaries are WordPiece ``##`` continuations and a window
#: starting mid-word is near-certain. Verified to reproduce a +2 naive drift; a text of ordinary
#: prose does NOT, which is how the first version of the test below passed without the fix.
MID_WORD_BOUNDARIES = (
    "unstructured internationalisation reconciliation subledgerIdentifier "
    "interoperability characterisation disintermediation "
) * 40


def test_a_window_starting_mid_word_still_measures_within_budget(real_chunk_tokenizer):
    """The defect the span arithmetic hid, reproduced from the mechanism that caused it.

    ``chunk_text`` slices the character range of exactly N tokens. Measured on its own, that slice
    can exceed N: WordPiece marks continuations with ``##``, so when a window begins mid-word the
    leading fragment becomes a standalone word and may split into more pieces than the single
    ``##piece`` it came from. Found on 27,309 live points -- a parent of
    https://parseur.com/blog/ measured 502 against a 500 budget, its slice beginning
    ``'ructured text from emails'``, i.e. inside ``unstructured``.

    At a 500 budget the +2 is invisible against a 512 ceiling. At the 510 the guard permits, it
    truncates -- the exact failure this module exists to rule out. So the budget is enforced on the
    produced string, and this asserts it at every window position rather than only the first.
    """
    limit = real_chunk_tokenizer.sequence_limit
    overhead = real_chunk_tokenizer.special_token_overhead
    ceiling = limit - overhead

    for budget, overlap in ((50, 7), (120, 20), (ceiling, 80)):
        chunks = chunk_text(
            MID_WORD_BOUNDARIES,
            tokenizer=real_chunk_tokenizer,
            size_tokens=budget,
            overlap_tokens=overlap,
        )
        assert len(chunks) > 1, f"budget {budget} must actually window this text"
        for chunk in chunks:
            content = len(real_chunk_tokenizer.token_spans(chunk))
            assert content <= budget, (
                f"slice measures {content} content tokens against a {budget} budget "
                f"(+{content - budget} drift); head={chunk[:40]!r}"
            )
            assert content + overhead <= limit, (
                f"chunk reaches inference as {content + overhead} against a {limit} limit"
            )
