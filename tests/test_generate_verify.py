"""Unit tests for citeable generate helpers."""

from geek_crawler_rag.generate import _build_prompts, _stage, quote_in_markdown
from geek_crawler_rag.models import GenerateOutlineSection, GenerateRequest


def test_quote_in_markdown_exact():
    md = "# Title\n\nAcme supports SSO and SCIM provisioning for enterprises."
    assert quote_in_markdown("Acme supports SSO and SCIM provisioning", md)


def test_quote_in_markdown_rejects_short_or_missing():
    md = "Hello world product page."
    assert not quote_in_markdown("nope", md)
    assert not quote_in_markdown("completely invented claim about pricing tiers", md)


def test_outline_stage_prompt_requests_structured_outline():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="outline",
    )
    _, user = _build_prompts(req, "long", [])
    assert _stage(req) == "outline"
    assert "Generation stage: OUTLINE" in user
    assert '"outline"' in user
    assert "Do not draft the article yet" in user


def test_section_stage_prompt_includes_outline_and_completed_context():
    req = GenerateRequest(
        writingIntent="Case Study",
        topic="CRM synchronization",
        generationStage="section",
        sectionKey="results",
        sectionHeading="Results",
        sectionBrief="Quantify the operational impact.",
        outline=[
            GenerateOutlineSection(
                key="problem",
                heading="The problem",
                brief="Explain duplicate records.",
            ),
            GenerateOutlineSection(
                key="results",
                heading="Results",
                brief="Quantify the operational impact.",
            ),
        ],
        completedSectionSummaries=["The problem: duplicate records slowed campaigns."],
    )
    _, user = _build_prompts(req, "long", [])
    assert _stage(req) == "section"
    assert "Generation stage: SECTION" in user
    assert "Requested key: results" in user
    assert "The problem" in user
    assert "duplicate records slowed campaigns" in user
