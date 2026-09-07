from geek_crawler_rag.chunk import chunk_text, parent_child_units
from geek_crawler_rag.extract import page_text_and_title
from geek_crawler_rag.metadata import (
    entity_from_crawl,
    infer_category,
    normalize_host,
    quality_score,
)
from geek_crawler_rag.rrf import reciprocal_rank_fusion
from geek_crawler_rag.bm25_rank import bm25_rank_indices, tokenize


def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_short_text_single_piece():
    text = "Hello world. " * 20
    chunks = chunk_text(text, size_tokens=200, overlap_tokens=20)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)


def test_chunk_overlap_covers_long_doc():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 3
    assert "word1999" in chunks[-1]
    assert "word0" in chunks[0]


def test_chunk_rejects_bad_overlap():
    try:
        chunk_text("abc", size_tokens=10, overlap_tokens=10)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_parent_child_from_headings():
    text = """# Pricing

Our plans start at nine dollars per month with usage based billing and enterprise support.

## Features

Feature alpha includes SSO and audit logs for regulated teams. Feature beta adds sandboxes.
"""
    units = parent_child_units(
        text,
        child_size_tokens=40,
        child_overlap_tokens=5,
        parent_size_tokens=120,
        parent_overlap_tokens=10,
    )
    assert units
    assert any(u.section_title and "Pricing" in u.section_title for u in units)
    assert all(u.parent_text and u.child_text for u in units)
    # Child is contained in / related to parent context.
    assert any(u.child_text in u.parent_text or u.child_text[:20] in u.parent_text for u in units)


def test_prefer_markdown_over_html():
    text, title, used_md = page_text_and_title(
        markdown="# Hello\n\nMarkdown body with enough English words for indexing.",
        title=None,
        html="<html><head><title>HTML Title</title></head><body>Ignored html</body></html>",
    )
    assert used_md is True
    assert "Markdown body" in text
    assert title == "Hello"


def test_html_fallback_when_no_markdown():
    text, title, used_md = page_text_and_title(
        markdown=None,
        title=None,
        html="<html><head><title>Docs</title></head><body>Plain HTML English content here.</body></html>",
    )
    assert used_md is False
    assert "Plain HTML" in text
    assert title == "Docs"


def test_metadata_heuristics():
    assert infer_category("https://acme.com/pricing", "plans") == "pricing"
    assert normalize_host("www.Acme.com") == "acme.com"
    ent = entity_from_crawl("partner", "acme.com")
    assert ent.source_type == "partner"
    assert ent.entity_name == "acme.com"
    score = quality_score(text="x" * 600, title="T", has_markdown=True)
    assert 0.5 <= score <= 1.0


def test_rrf_and_bm25():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]])
    assert fused[0][0] == "b"
    docs = [
        "alpha product pricing nine dollars",
        "unrelated cooking recipes",
        "alpha enterprise SSO audit",
    ]
    order = bm25_rank_indices("alpha pricing", docs)
    assert order[0] == 0
    assert tokenize("Hello World") == ["hello", "world"]
