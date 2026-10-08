from geek_crawler_rag.block_text import derive_plaintext_from_blocks
from geek_crawler_rag.chunk import (
    chunk_text,
    parent_child_units,
    split_blocks_into_sections,
)
from geek_crawler_rag.extract import page_text_and_title
from geek_crawler_rag.metadata import (
    entity_from_crawl,
    infer_category,
    infer_competitor_chunk_kind,
    infer_feature_tag,
    normalize_host,
    quality_score,
)


def test_chunk_empty(chunk_tokenizer):
    assert chunk_text("", tokenizer=chunk_tokenizer) == []
    assert chunk_text("   ", tokenizer=chunk_tokenizer) == []


def test_chunk_short_text_single_piece(chunk_tokenizer):
    text = "Hello world. " * 20
    chunks = chunk_text(text, tokenizer=chunk_tokenizer, size_tokens=200, overlap_tokens=20)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)


def test_chunk_overlap_covers_long_doc(chunk_tokenizer):
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, tokenizer=chunk_tokenizer, size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 3
    assert "word1999" in chunks[-1]
    assert "word0" in chunks[0]


def test_chunk_rejects_bad_overlap(chunk_tokenizer):
    try:
        chunk_text("abc", tokenizer=chunk_tokenizer, size_tokens=10, overlap_tokens=10)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_every_chunk_is_a_verbatim_slice_of_the_input(chunk_tokenizer):
    """The guarantee that makes WordPiece sizing safe at all.

    Chunks are cut from the original string at token boundaries, never rebuilt from tokens. The old
    implementation decoded a token slice, which is lossless for BPE and destructive for WordPiece --
    bge-small is uncased and splits on punctuation, so `Invoice.model_validate_json` would come back
    as `invoice. model _ validate _ json`. Chunk text is also the citation-verification target, so a
    lossy round-trip there breaks quote verification, not just readability.
    """
    text = (
        "Parse it: invoice = Invoice.model_validate_json(raw_json). "
        "Then df = pd.DataFrame([item.model_dump() for item in invoice.lineItems]). "
        "Check TOTALS against XJ-4420-B and 7.3.1 exactly. "
    ) * 6
    chunks = chunk_text(text, tokenizer=chunk_tokenizer, size_tokens=30, overlap_tokens=5)
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk in text, "chunk is not a substring of its source"
    # Casing, punctuation and identifiers survive intact.
    joined = " ".join(chunks)
    assert "Invoice.model_validate_json" in joined
    assert "XJ-4420-B" in joined
    assert "7.3.1" in joined
    assert "TOTALS" in joined


def test_a_window_wider_than_the_model_accepts_is_refused(chunk_tokenizer):
    """Configuration error at the boundary, not silent truncation at inference.

    The model adds special tokens per input, so the usable budget is sequence_limit minus that
    overhead -- 510 for a 512-token model. Asking for more is how 6.9% of parents got cut.
    """
    usable = chunk_tokenizer.sequence_limit - chunk_tokenizer.special_token_overhead
    assert chunk_text("word " * 50, tokenizer=chunk_tokenizer, size_tokens=usable)
    try:
        chunk_text("word " * 50, tokenizer=chunk_tokenizer, size_tokens=usable + 1)
        raise AssertionError("expected ValueError for a window above the usable budget")
    except ValueError as exc:
        assert "usable budget" in str(exc)


def test_parents_are_bounded_by_headings_and_carry_their_section(chunk_tokenizer):
    """Each chunk reports the heading it sits under, and its own section's anchors.

    Sections are cut at heading blocks, so a parent window never straddles two
    headings. Blocks before the first heading are a preamble with `section_title`
    of None — an absent heading, never an invented one.
    """
    blocks = [
        {"kind": "paragraph", "text": "Intro before any heading.", "anchors": []},
        {"kind": "heading", "level": 2, "text": "Pricing", "anchors": []},
        {
            "kind": "paragraph",
            "text": "Our plans start at nine dollars per month with usage based billing.",
            "anchors": [],
        },
        {"kind": "heading", "level": 2, "text": "Features", "anchors": []},
        {
            "kind": "paragraph",
            "text": "Feature alpha includes SSO and audit logs for regulated teams.",
            "anchors": [{"label": "Zapier", "href": "/tools/zapier"}],
        },
    ]
    units = parent_child_units(
        blocks,
        tokenizer=chunk_tokenizer,
        child_size_tokens=40,
        child_overlap_tokens=5,
        parent_size_tokens=120,
        parent_overlap_tokens=10,
    )
    assert units
    assert all(u.parent_text and u.child_text for u in units)

    titles = [u.section_title for u in units]
    assert None in titles, "the preamble keeps an absent title"
    assert "Pricing" in titles
    assert "Features" in titles

    # A parent never spans two headings.
    for unit in units:
        others = {"Pricing", "Features"} - {unit.section_title}
        for other in others:
            assert other not in unit.parent_text

    # The heading stays in its own section body, so its words are embedded.
    pricing = [u for u in units if u.section_title == "Pricing"]
    assert pricing and all("Pricing" in u.parent_text for u in pricing)

    # Anchors are attributed to the section they appear in, not to the whole page.
    features = [u for u in units if u.section_title == "Features"]
    assert features and all(
        u.section_anchors == (("Zapier", "/tools/zapier"),) for u in features
    )
    assert all(u.section_anchors == () for u in pricing)

    assert any(
        u.child_text in u.parent_text or u.child_text[:20] in u.parent_text
        for u in units
    )


def test_section_text_is_the_page_projection_restricted_to_those_blocks(chunk_tokenizer):
    """The invariant that keeps citations verifiable.

    A quote is taken from a retrieved chunk and matched against the whole-page
    projection. Section text must therefore be built by the same projection over a
    subset of blocks — identical rendering, identical join — so every section's
    text is a substring of the page's.
    """
    blocks = [
        {"kind": "heading", "level": 1, "text": "Plans", "anchors": []},
        {"kind": "paragraph", "text": "Starter is free for one seat.", "anchors": []},
        {"kind": "row", "header": False, "cells": ["Pro", "$15/month"], "anchors": []},
        {"kind": "heading", "level": 2, "text": "Limits", "anchors": []},
        {"kind": "paragraph", "text": "Uptime is 99.9% on every plan.", "anchors": []},
    ]
    page_text = derive_plaintext_from_blocks(blocks)
    sections = split_blocks_into_sections(blocks)

    assert [s.title for s in sections] == ["Plans", "Limits"]
    assert [s.level for s in sections] == [1, 2]
    for section in sections:
        assert section.text in page_text

    units = parent_child_units(
        blocks, tokenizer=chunk_tokenizer, parent_size_tokens=500, child_size_tokens=200
    )
    for unit in units:
        assert unit.parent_text in page_text


def test_no_blocks_yields_no_units(chunk_tokenizer):
    assert parent_child_units(None, tokenizer=chunk_tokenizer) == []
    assert parent_child_units([], tokenizer=chunk_tokenizer) == []
    assert split_blocks_into_sections(None) == []
    # Blocks that render to nothing are not a section.
    assert split_blocks_into_sections([{"kind": "paragraph", "text": "  "}]) == []


def test_text_comes_from_blocks_and_title_falls_back_to_the_first_heading():
    text, title, has_blocks = page_text_and_title(
        blocks=[
            {"kind": "heading", "level": 1, "text": "Hello", "anchors": []},
            {
                "kind": "paragraph",
                "text": "Block body with enough English words for indexing.",
                "anchors": [],
            },
        ],
        title=None,
    )
    assert has_blocks is True
    assert "Block body" in text
    assert title == "Hello"


def test_no_blocks_yields_no_text():
    """HTML is never synthesised into a corpus body.

    A page without typed blocks has no body this service can index, and
    guessing one from markup is the behaviour the block format exists to
    replace.
    """
    text, title, has_blocks = page_text_and_title(blocks=[], title=None)
    assert has_blocks is False
    assert text == ""
    assert title is None


def test_metadata_heuristics():
    assert infer_category("https://acme.com/pricing", "plans") == "pricing"
    assert normalize_host("www.Acme.com") == "acme.com"
    ent = entity_from_crawl("partner", "acme.com")
    assert ent.source_type == "partner"
    assert ent.entity_name == "acme.com"
    score = quality_score(text="x" * 600, title="T", has_blocks=True)
    assert 0.5 <= score <= 1.0


def test_competitor_chunk_kind_and_feature_tag():
    assert (
        infer_competitor_chunk_kind(
            url="https://rival.example/pricing",
            section_title="Plans",
            text="Per seat pricing starts at $49",
            category="pricing",
        )
        == "pricing"
    )
    assert (
        infer_competitor_chunk_kind(
            url="https://rival.example/compare/us-vs-them",
            section_title="Versus",
            text="Compared to alternatives",
            category="compare",
        )
        == "comparison"
    )
    assert (
        infer_competitor_chunk_kind(
            url="https://rival.example/security",
            section_title="Trust",
            text="We maintain SOC 2 Type II and ISO 27001",
            category="docs",
        )
        == "proof"
    )
    assert (
        infer_competitor_chunk_kind(
            url="https://rival.example/faq",
            section_title="FAQ",
            text="Does not support SSO for starter plans",
            category="docs",
        )
        == "gap"
    )
    assert (
        infer_competitor_chunk_kind(
            url="https://rival.example/features/analytics",
            section_title="Analytics",
            text="Capability to report funnel conversion",
            category="product",
        )
        == "feature"
    )
    assert infer_feature_tag("SSO & SCIM", ["docs"]) == "SSO  SCIM"
    assert infer_feature_tag(None, ["pricing", "analytics"]) == "analytics"
    assert infer_feature_tag(None, ["pricing", "docs"]) is None
