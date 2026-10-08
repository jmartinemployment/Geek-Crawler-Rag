"""Stub chunks are not indexed, and the index status says how many were dropped.

plans/retrieval-from-the-brief.md P3. On 2026-10-08 the blockquote probe on Tipalti's run got eight
candidates; three were a bare ebook title ("The CFO's Guide to Payables Automation"), a heading
("Tech Companies Payment Automation FAQs") and a breadcrumb ("Home / AP Automation / Payment
Reconciliation"). Each had been indexed as a passage and each took a slot nothing could be quoted
from. `chunk.is_stub_text` names them; `page_to_nodes` drops them and counts them; the count
reaches the status as `chunksSkippedStub`.
"""

from __future__ import annotations

from conftest import FakeChunkTokenizer
from geek_crawler_rag.chunk import STUB_MAX_WORDS, is_stub_text
from geek_crawler_rag.config import Settings
from geek_crawler_rag.llama_nodes import page_to_nodes
from geek_crawler_rag.metadata import EntityRef
from geek_crawler_rag.mongo import CrawlPage

PROSE = (
    "Automated payment reconciliation syncs payment results with your ERP or accounting system, "
    "integrating payment status into the general ledger and sub-ledgers so every payment is "
    "matched back to the right vendor, entity and accounting period without manual work. "
)


def test_the_three_tipalti_stubs_are_stubs() -> None:
    assert is_stub_text("The CFO's Guide to Payables Automation", "The CFO's Guide to Payables Automation")
    assert is_stub_text("Tech Companies Payment Automation FAQs", "Tech Companies Payment Automation FAQs")
    assert is_stub_text("Home / AP Automation / Payment Reconciliation", None)


def test_a_heading_standing_alone_is_a_stub_even_when_long() -> None:
    title = "Global AP Automation and Payments Integration for Oracle NetSuite and SAP S/4HANA"
    assert is_stub_text(title, title)
    assert is_stub_text("  " + title.upper() + "  ", title), "case and whitespace do not matter"


def test_a_breadcrumb_is_a_stub_and_a_sentence_with_a_slash_is_not() -> None:
    assert is_stub_text("Home / Resources / Learn / Payment API Guide", None)
    assert not is_stub_text(
        "Our support team is available 24/7 for every customer on every plan, in every region.", None
    )


def test_fewer_than_eight_words_is_a_stub_and_eight_is_content() -> None:
    assert STUB_MAX_WORDS == 8
    assert is_stub_text("Read the customer story", "Customers")
    assert is_stub_text("", "Anything")
    assert not is_stub_text("Pay suppliers in 120 currencies across 200 countries", "Global payments")


def _page(blocks: list[dict], title: str) -> CrawlPage:
    return CrawlPage(
        id="p",
        run_id="r1",
        origin="https://tipalti.com",
        url="https://tipalti.com/x/",
        final_url="https://tipalti.com/x/",
        html="<html><body>ignored</body></html>",
        blocks=blocks,
        title=title,
    )


def _nodes(page: CrawlPage):
    return page_to_nodes(
        page=page,
        run_id="r1",
        crawl_type="partner",
        entity=EntityRef(None, "tipalti.com", "partner", ("tipalti.com",)),
        settings=Settings(openai_api_key="test"),
        tokenizer=FakeChunkTokenizer(),
    )


def test_a_heading_section_with_no_body_is_dropped_and_the_page_still_indexes() -> None:
    """The CFO-guide case: an ebook landing page whose h1 has no prose under it. The heading-only
    section emitted a child equal to the heading and it was indexed as a passage. (A page that is
    nothing but a six-word title never reaches chunking: language detection refuses it first, which
    is existing behaviour and not this rule's.)"""
    title = "The CFO's Guide to Payables Automation"
    nodes, skip, stubs = _nodes(
        _page(
            [
                {"kind": "heading", "level": 1, "text": title, "anchors": []},
                {"kind": "heading", "level": 2, "text": "Why payables automation pays for itself", "anchors": []},
                {"kind": "paragraph", "text": PROSE * 12, "anchors": []},
            ],
            title,
        )
    )
    assert skip == ""
    assert stubs == 1
    assert nodes, "the section with prose still indexes"
    assert all(n.get_content().strip() != title for n in nodes), "the bare title is not a passage"
    assert all(n.metadata.get("sectionTitle") != title for n in nodes)


def test_a_breadcrumb_preamble_is_dropped_and_the_real_section_is_kept() -> None:
    nodes, skip, stubs = _nodes(
        _page(
            [
                {"kind": "paragraph", "text": "Home / AP Automation / Payment Reconciliation", "anchors": []},
                {"kind": "heading", "level": 1, "text": "Automated Payment Reconciliation", "anchors": []},
                {"kind": "paragraph", "text": PROSE * 12, "anchors": []},
            ],
            "Automated Payment Reconciliation",
        )
    )
    assert skip == ""
    assert stubs == 1
    assert nodes, "the real section still indexes"
    assert all("Home / AP Automation" not in n.get_content() for n in nodes)
    assert all(n.metadata.get("sectionTitle") == "Automated Payment Reconciliation" for n in nodes)


def test_a_prose_page_drops_nothing() -> None:
    nodes, skip, stubs = _nodes(
        _page(
            [
                {"kind": "heading", "level": 1, "text": "Reconciliation", "anchors": []},
                {"kind": "paragraph", "text": PROSE * 30, "anchors": []},
            ],
            "Reconciliation",
        )
    )
    assert skip == "" and nodes and stubs == 0
