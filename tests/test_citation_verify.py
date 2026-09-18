"""Unit tests for quote verification against the block plaintext projection."""

from geek_crawler_rag.citation_verify import quote_in_text, verify_citations
from geek_crawler_rag.models import GenerateCitation, GenerateSource


def test_quote_in_text_exact():
    body = "Title\n\nAcme supports SSO and SCIM provisioning for enterprises."
    assert quote_in_text("Acme supports SSO and SCIM provisioning", body)


def test_quote_in_text_rejects_short_or_missing():
    body = "Hello world product page."
    assert not quote_in_text("nope", body)
    assert not quote_in_text("completely invented claim about pricing tiers", body)


def test_short_table_cell_quote_verifies_by_whole_cell_match():
    """The 12-character floor turns correct cell citations into failures.

    "$15/month" is 9 normalized characters. Under the floor alone a generator
    quoting a price off a pricing table would be reported as unverifiable.
    """
    blocks = [
        {"kind": "paragraph", "text": "Plans and pricing for every team size."},
        {"kind": "row", "cells": ["Lite", "$15/month"]},
    ]
    body = "Plans and pricing for every team size.\n\nLite | $15/month"

    assert quote_in_text("$15/month", body, blocks)
    # Still a whole-cell match, not any short substring of the page.
    assert not quote_in_text("month", body, blocks)
    assert not quote_in_text("$99", body, blocks)


def test_punctuation_only_quotes_never_verify():
    blocks = [{"kind": "row", "cells": ["—", "|"]}]
    assert not quote_in_text("—", "— | |", blocks)


def test_citation_integrity_keeps_only_loaded_verbatim_evidence():
    pages = [
        {
            "pageId": "page-1",
            "url": "https://example.test/crm",
            "text": "Acme synchronizes CRM records every five minutes for active accounts.",
        }
    ]
    sources = [
        GenerateSource(
            pageId="page-1",
            url="https://example.test/crm",
            title="CRM",
        )
    ]
    citations = [
        GenerateCitation(
            pageId="page-1",
            url="https://example.test/crm",
            quote="Acme synchronizes CRM records every five minutes",
        ),
        GenerateCitation(
            pageId="page-1",
            url="https://example.test/crm",
            quote="Acme guarantees a 300 percent revenue increase",
        ),
        GenerateCitation(
            pageId="page-2",
            url="https://invented.test/crm",
            quote="This citation points at an unrequested source",
        ),
    ]
    kept, dropped = verify_citations(citations, sources, pages)
    assert [citation.quote for citation in kept] == [
        "Acme synchronizes CRM records every five minutes"
    ]
    assert dropped == 2
