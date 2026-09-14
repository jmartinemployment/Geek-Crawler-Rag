"""Unit tests for Markdown citation verification helpers."""

from geek_crawler_rag.citation_verify import quote_in_markdown, verify_citations
from geek_crawler_rag.models import GenerateCitation, GenerateSource


def test_quote_in_markdown_exact():
    md = "# Title\n\nAcme supports SSO and SCIM provisioning for enterprises."
    assert quote_in_markdown("Acme supports SSO and SCIM provisioning", md)


def test_quote_in_markdown_rejects_short_or_missing():
    md = "Hello world product page."
    assert not quote_in_markdown("nope", md)
    assert not quote_in_markdown("completely invented claim about pricing tiers", md)


def test_citation_integrity_keeps_only_loaded_verbatim_evidence():
    pages = [
        {
            "pageId": "page-1",
            "url": "https://example.test/crm",
            "markdown": "Acme synchronizes CRM records every five minutes for active accounts.",
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
