"""Unit tests for the block to plaintext projection."""

from geek_crawler_rag.block_text import (
    BLOCK_SEPARATOR,
    ROW_CELL_SEPARATOR,
    derive_plaintext_from_blocks,
    render_block_text,
)


def para(text: str) -> dict:
    return {"kind": "paragraph", "text": text, "html": f"<p>{text}</p>", "anchors": []}


def row(*cells: str) -> dict:
    return {"kind": "row", "header": False, "cells": list(cells), "cellsHtml": [], "anchors": []}


def test_prose_kinds_project_to_their_text():
    for kind in ("heading", "paragraph", "listItem", "quote", "code", "term", "definition"):
        assert render_block_text({"kind": kind, "text": "  Invoicing  "}) == "Invoicing"


def test_row_projects_its_cells_not_a_missing_text_field():
    # The regression this module exists for: `row` is the only kind without
    # `text`, so mapping block["text"] drops every table row from the page.
    block = row("Plan", "Price", "Seats")
    assert render_block_text(block) == f"Plan{ROW_CELL_SEPARATOR}Price{ROW_CELL_SEPARATOR}Seats"
    assert render_block_text(block) != ""


def test_row_trims_cells_and_tolerates_none():
    assert render_block_text(row("  Lite  ", "  $15  ")) == "Lite | $15"
    assert render_block_text({"kind": "row", "cells": ["Lite", None, "$15"]}) == "Lite | $15"


def test_empty_and_malformed_blocks_project_to_empty_string():
    assert render_block_text(None) == ""
    assert render_block_text({}) == ""
    assert render_block_text({"kind": "paragraph"}) == ""
    assert render_block_text({"kind": "row"}) == ""


def test_page_projection_joins_and_drops_blank_blocks():
    blocks = [para("First."), para("   "), row("A", "B"), para("Last.")]
    assert derive_plaintext_from_blocks(blocks) == BLOCK_SEPARATOR.join(
        ["First.", "A | B", "Last."]
    )


def test_page_projection_handles_absent_input():
    assert derive_plaintext_from_blocks(None) == ""
    assert derive_plaintext_from_blocks([]) == ""


def test_projection_is_deterministic_for_the_same_blocks():
    # The invariant the module exists to guarantee: whatever the chunker and the
    # verification path each ask for, they get byte-identical strings.
    blocks = [para("Acme supports SSO."), row("Plan", "$15/month")]
    assert derive_plaintext_from_blocks(blocks) == derive_plaintext_from_blocks(blocks)
