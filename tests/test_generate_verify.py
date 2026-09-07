"""Unit tests for citeable generate helpers."""

from geek_crawler_rag.generate import quote_in_markdown


def test_quote_in_markdown_exact():
    md = "# Title\n\nAcme supports SSO and SCIM provisioning for enterprises."
    assert quote_in_markdown("Acme supports SSO and SCIM provisioning", md)


def test_quote_in_markdown_rejects_short_or_missing():
    md = "Hello world product page."
    assert not quote_in_markdown("nope", md)
    assert not quote_in_markdown("completely invented claim about pricing tiers", md)
