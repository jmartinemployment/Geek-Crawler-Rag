"""Unit tests for citeable generate helpers."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from geek_crawler_rag.config import Settings
from geek_crawler_rag.generate import (
    CiteableGenerateWorkflow,
    _brief_retrieval_context,
    _build_prompts,
    _governed_output_violations,
    _parse_llm_json,
    _stage,
    quote_in_markdown,
    select_generation_model,
    verify_citations,
)
from geek_crawler_rag.models import (
    GenerateCitation,
    GenerateOutlineSection,
    GenerateRequest,
    GenerateResponse,
    GenerateSource,
    GovernedContextEntry,
)


def test_quote_in_markdown_exact():
    md = "# Title\n\nAcme supports SSO and SCIM provisioning for enterprises."
    assert quote_in_markdown("Acme supports SSO and SCIM provisioning", md)


def test_quote_in_markdown_rejects_short_or_missing():
    md = "Hello world product page."
    assert not quote_in_markdown("nope", md)
    assert not quote_in_markdown("completely invented claim about pricing tiers", md)


def test_quote_in_markdown_rejects_quote_with_invented_suffix():
    exact = (
        "Acme synchronizes customer records every five minutes and logs each "
        "successful update for audit review."
    )
    assert not quote_in_markdown(
        exact + " It also guarantees perfect data accuracy.",
        exact,
    )


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


def test_canonical_brief_propagates_to_retrieval_outline_and_section_prompts():
    brief = {
        "version": "gcc-v2-generation-brief.v1",
        "title": "The CRM integration guide",
        "targetKeyword": "CRM synchronization",
        "contentType": "guide",
        "primaryIntent": "commercial investigation",
        "audience": {"role": "RevOps leader"},
        "buyingStage": "consideration",
        "toneOfVoice": ["expert", "direct"],
        "brandKit": {"name": "Geek", "promise": "practical automation"},
        "paaQuestions": ["How does CRM synchronization work?"],
        "requiredTopics": ["conflict resolution"],
        "operatorInstructions": ["Prioritize operational detail"],
        "exclusions": ["Do not claim guaranteed ROI"],
        "hierarchy": {"parent": "/guides", "siblings": ["/guides/crm"]},
        "internalLinks": [{"url": "/services/crm", "anchor": "CRM services"}],
        "outputRequirements": {"wordCount": 1800},
        "channelRequirements": "website",
        "ctaRequirements": {"action": "Book a consultation"},
        "conversionObjective": "qualified consultation",
        "publishingDestination": "primary website",
    }
    outline_req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="outline",
        canonicalBrief=brief,
    )
    retrieval = _brief_retrieval_context(outline_req)
    _, outline_prompt = _build_prompts(outline_req, "long", [])
    assert "content type: guide" in retrieval
    assert "RevOps leader" in retrieval
    assert "conflict resolution" in retrieval
    assert "Prioritize operational detail" in retrieval
    assert "Book a consultation" in retrieval
    assert '"operatorInstructions"' in outline_prompt
    assert "Do not claim guaranteed ROI" in outline_prompt
    assert "Book a consultation" in outline_prompt
    assert "CRM services" in outline_prompt

    section_req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        sectionKey="conflicts",
        sectionHeading="Resolve synchronization conflicts",
        sectionBrief="Explain evidence-backed conflict handling.",
        canonicalBrief=brief,
    )
    _, section_prompt = _build_prompts(section_req, "long", [])
    assert '"buyingStage": "consideration"' in section_prompt
    assert '"toneOfVoice"' in section_prompt
    assert "Prioritize operational detail" in section_prompt
    assert "qualified consultation" in section_prompt


def test_stage_aware_model_policy_selection_and_legacy_compatibility():
    settings = Settings(
        openai_longform_model="legacy-long",
        openai_standard_model="legacy-fast",
    )
    legacy = GenerateRequest(writingIntent="Technical Article", topic="CRM sync")
    assert select_generation_model(legacy, settings) == "legacy-long"

    outline = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM sync",
        generationStage="outline",
        modelPolicyPreset="best-quality",
    )
    section = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM sync",
        generationStage="section",
        modelPolicyPreset="best-quality",
    )
    o3_only = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM sync",
        generationStage="outline",
        modelPolicyPreset="o3-only",
    )
    custom = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM sync",
        generationStage="section",
        modelPolicyPreset="custom",
        stageModelOverrides={"section": "o1-pro"},
    )
    assert select_generation_model(outline, settings) == "o1-pro"
    assert select_generation_model(section, settings) == "o3"
    assert select_generation_model(o3_only, settings) == "o3"
    assert select_generation_model(custom, settings) == "o1-pro"


@pytest.mark.parametrize(
    ("policy_fields", "message"),
    [
        ({"modelPolicyPreset": "cheap"}, "Unapproved modelPolicyPreset"),
        (
            {
                "modelPolicyPreset": "custom",
                "stageModelOverrides": {"section": "gpt-4o"},
            },
            "Unapproved stage model",
        ),
        (
            {
                "generationStage": "section",
                "modelPolicyPreset": "custom",
                "stageModelOverrides": {"outline": "o1-pro"},
            },
            "no override for generation stage 'section'",
        ),
        (
            {
                "modelPolicyPreset": "best-quality",
                "modelPolicyVersion": "content-model-policy.v0",
            },
            "Unsupported modelPolicyVersion",
        ),
    ],
)
def test_model_policy_rejects_unapproved_or_missing_selection(policy_fields, message):
    with pytest.raises(ValidationError, match=message):
        GenerateRequest(
            writingIntent="Technical Article",
            topic="CRM sync",
            **policy_fields,
        )


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


@pytest.mark.parametrize("model", ["o1-pro", "o3"])
async def test_reasoning_models_use_responses_api_without_storage(model):
    response_calls = []
    chat_calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            response_calls.append(kwargs)
            return SimpleNamespace(output_text='{"content":"draft"}', output=[])

    class FakeCompletions:
        async def create(self, **kwargs):
            chat_calls.append(kwargs)
            raise AssertionError("reasoning model must not use Chat Completions")

    workflow = object.__new__(CiteableGenerateWorkflow)
    workflow._settings = Settings(openai_reasoning_max_completion_tokens=12_345)
    workflow._openai = SimpleNamespace(
        responses=FakeResponses(),
        chat=SimpleNamespace(completions=FakeCompletions()),
    )

    raw = await workflow._chat(model, "system instructions", "user input")

    assert raw == '{"content":"draft"}'
    assert _parse_llm_json(raw, "long") == {
        "content": "draft",
    }
    assert chat_calls == []
    assert response_calls == [
        {
            "model": model,
            "instructions": "system instructions",
            "input": "user input",
            "reasoning": {"effort": "high"},
            "max_output_tokens": 12_345,
            "store": False,
        }
    ]


async def test_responses_api_extracts_typed_output_fallback():
    class FakeResponses:
        async def create(self, **kwargs):
            return SimpleNamespace(
                output_text="",
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[
                            SimpleNamespace(
                                type="output_text",
                                text='{"outline":[],"citations":[]}',
                            )
                        ],
                    )
                ],
            )

    workflow = object.__new__(CiteableGenerateWorkflow)
    workflow._settings = Settings()
    workflow._openai = SimpleNamespace(
        responses=FakeResponses(),
        chat=SimpleNamespace(completions=None),
    )

    raw = await workflow._chat("o1-pro", "system", "user")
    assert _parse_llm_json(raw, "long") == {"outline": [], "citations": []}


@pytest.mark.parametrize("model", ["o1-pro", "o3"])
async def test_responses_api_error_does_not_fall_back_to_chat_or_another_model(model):
    chat_calls = []

    class FailingResponses:
        async def create(self, **kwargs):
            raise RuntimeError("responses transport failed")

    class FakeCompletions:
        async def create(self, **kwargs):
            chat_calls.append(kwargs)

    workflow = object.__new__(CiteableGenerateWorkflow)
    workflow._settings = Settings()
    workflow._openai = SimpleNamespace(
        responses=FailingResponses(),
        chat=SimpleNamespace(completions=FakeCompletions()),
    )

    with pytest.raises(RuntimeError, match="responses transport failed"):
        await workflow._chat(model, "system", "user")
    assert chat_calls == []


async def test_legacy_non_reasoning_model_keeps_chat_completions_json_mode():
    response_calls = []
    chat_calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            response_calls.append(kwargs)

    class FakeCompletions:
        async def create(self, **kwargs):
            chat_calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"variations":["one"]}')
                    )
                ]
            )

    workflow = object.__new__(CiteableGenerateWorkflow)
    workflow._settings = Settings()
    workflow._openai = SimpleNamespace(
        responses=FakeResponses(),
        chat=SimpleNamespace(completions=FakeCompletions()),
    )

    raw = await workflow._chat("gpt-4o", "system", "user")
    assert _parse_llm_json(raw, "short") == {"variations": ["one"]}
    assert response_calls == []
    assert chat_calls[0]["temperature"] == 0.3
    assert chat_calls[0]["response_format"] == {"type": "json_object"}
    assert "max_completion_tokens" not in chat_calls[0]


def _digest(seed: str = "a") -> str:
    return seed * 64


def test_style_guide_prompt_includes_deterministic_constraints():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="style_guide",
                stableId="style-1",
                versionId="style-v1",
                versionNumber=1,
                digest=_digest(),
                payload={
                    "schemaVersion": 1,
                    "grammar": {"oxfordComma": True, "allowEmDash": False},
                    "termRules": [
                        {
                            "kind": "replace",
                            "match": "users",
                            "replacement": "customers",
                        },
                        {"kind": "prohibit", "match": "synergy"},
                    ],
                    "customInstructions": "Prefer concrete outcomes.",
                },
            )
        ],
    )
    system, _ = _build_prompts(req, "long", [])
    assert "Style Guide deterministic constraints" in system
    assert 'Replace "users" with "customers"' in system
    assert 'Never use the phrase "synergy"' in system
    assert "Use the Oxford comma" in system
    assert "Do not use em dashes" in system
    assert "Prefer concrete outcomes" in system


def test_style_guide_output_violations_cover_typed_rules():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="style_guide",
                stableId="style-1",
                versionId="style-v1",
                versionNumber=1,
                digest=_digest("b"),
                payload={
                    "schemaVersion": 1,
                    "termRules": [
                        {"kind": "prohibit", "match": "synergy"},
                        {
                            "kind": "replace",
                            "match": "users",
                            "replacement": "customers",
                        },
                        {
                            "kind": "capitalize",
                            "match": "Acme Cloud",
                            "caseSensitive": True,
                        },
                        {
                            "kind": "firstMention",
                            "match": "CRM",
                            "replacement": "customer relationship management (CRM)",
                        },
                    ],
                    "requiredPhrases": ["security review"],
                },
            )
        ],
    )
    result = GenerateResponse(
        intent="Technical Article",
        content=(
            "Our users love synergy on acme cloud. CRM helps teams collaborate "
            "without the mandated check."
        ),
    )
    violations = _governed_output_violations(req, result)
    joined = "\n".join(violations)
    assert "synergy" in joined
    assert "replace rule" in joined
    assert "capitalization" in joined
    assert "first-mention" in joined
    assert "required phrase" in joined


def test_style_guide_output_accepts_compliant_text():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="style_guide",
                stableId="style-1",
                versionId="style-v1",
                versionNumber=1,
                digest=_digest("c"),
                payload={
                    "termRules": [
                        {
                            "kind": "replace",
                            "match": "users",
                            "replacement": "customers",
                        },
                        {
                            "kind": "capitalize",
                            "match": "Acme Cloud",
                            "caseSensitive": True,
                        },
                        {
                            "kind": "firstMention",
                            "match": "CRM",
                            "replacement": "customer relationship management (CRM)",
                        },
                    ],
                    "requiredPhrases": ["security review"],
                },
            )
        ],
    )
    result = GenerateResponse(
        intent="Technical Article",
        content=(
            "Acme Cloud helps customers after a security review. "
            "customer relationship management (CRM) then keeps records aligned."
        ),
    )
    assert _governed_output_violations(req, result) == []


def test_visual_guideline_prompt_includes_deterministic_constraints():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="visual_guideline",
                stableId="visual-1",
                versionId="visual-v1",
                versionNumber=1,
                digest=_digest("d"),
                payload={
                    "schemaVersion": 1,
                    "palette": {"primary": "#0F172A", "accent": "#0F766E"},
                    "typography": {
                        "displayFont": "Source Serif 4",
                        "bodyFont": "IBM Plex Sans",
                        "minBodySizePx": 16,
                    },
                    "logoUsage": {
                        "clearSpaceRatio": 0.5,
                        "allowedBackgrounds": ["light"],
                        "prohibitedTreatments": ["stretch", "recolor"],
                    },
                    "layout": {"preferFullBleedHero": True, "maxContentWidthPx": 1200},
                    "imagery": {
                        "styleNotes": "Natural light documentary shots.",
                        "prohibitedMotifs": ["stock handshake"],
                    },
                    "customInstructions": "Prefer full-bleed photography over inset cards.",
                },
            )
        ],
    )
    system, _ = _build_prompts(req, "long", [])
    assert "Visual Guidelines deterministic constraints" in system
    assert "primary=#0F172A" in system
    assert 'Prefer display typography "Source Serif 4"' in system
    assert 'Never describe logo treatment "stretch"' in system
    assert "Prefer full-bleed hero imagery" in system
    assert 'Never use imagery motif "stock handshake"' in system
    assert "Prefer full-bleed photography over inset cards" in system


def test_visual_guideline_output_violations_cover_prohibited_motifs():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="visual_guideline",
                stableId="visual-1",
                versionId="visual-v1",
                versionNumber=1,
                digest=_digest("e"),
                payload={
                    "logoUsage": {"prohibitedTreatments": ["recolor"]},
                    "imagery": {"prohibitedMotifs": ["stock handshake"]},
                },
            )
        ],
    )
    result = GenerateResponse(
        intent="Technical Article",
        content=(
            "Hero art uses a stock handshake and a recolor of the logo "
            "against a busy background."
        ),
    )
    joined = "\n".join(_governed_output_violations(req, result))
    assert "prohibited imagery motif" in joined
    assert "stock handshake" in joined
    assert "prohibited logo treatment" in joined
    assert "recolor" in joined


def test_visual_guideline_output_accepts_compliant_text():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="visual_guideline",
                stableId="visual-1",
                versionId="visual-v1",
                versionNumber=1,
                digest=_digest("f"),
                payload={
                    "logoUsage": {"prohibitedTreatments": ["stretch"]},
                    "imagery": {"prohibitedMotifs": ["stock handshake"]},
                },
            )
        ],
    )
    result = GenerateResponse(
        intent="Technical Article",
        content="Use full-bleed documentary photography with clear logo spacing.",
    )
    assert _governed_output_violations(req, result) == []


def test_audience_prompt_includes_deterministic_constraints():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="audience",
                stableId="audience-1",
                versionId="audience-v1",
                versionNumber=1,
                digest=_digest("1"),
                payload={
                    "schemaVersion": 1,
                    "summary": "Technical buyers evaluating evidence systems.",
                    "locale": "en",
                    "roles": ["VP Engineering"],
                    "pains": ["ungrounded AI claims"],
                    "goals": ["citeable drafts"],
                    "readingLevel": "professional",
                    "preferredLanguage": ["specific", "operational"],
                    "bannedTopics": ["hype"],
                    "avoidPhrases": ["synergy"],
                    "positioningStatement": "Evidence first.",
                    "customInstructions": "Prefer concrete systems over slogans.",
                },
            )
        ],
    )
    system, _ = _build_prompts(req, "long", [])
    assert "Audience deterministic constraints" in system
    assert "Technical buyers evaluating evidence systems" in system
    assert 'Address roles such as "VP Engineering"' in system
    assert 'Never cover banned topic "hype"' in system
    assert 'Never use the audience avoid-phrase "synergy"' in system
    assert "Prefer concrete systems over slogans" in system


def test_audience_output_violations_cover_banned_topics_and_avoid_phrases():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="audience",
                stableId="audience-1",
                versionId="audience-v1",
                versionNumber=1,
                digest=_digest("2"),
                payload={
                    "bannedTopics": ["hype"],
                    "avoidPhrases": ["synergy"],
                },
            )
        ],
    )
    result = GenerateResponse(
        intent="Technical Article",
        content="This launch is pure hype and synergy for enterprise buyers.",
    )
    joined = "\n".join(_governed_output_violations(req, result))
    assert "banned topic" in joined
    assert "hype" in joined
    assert "avoid-phrase" in joined
    assert "synergy" in joined


def test_audience_output_accepts_compliant_text():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="audience",
                stableId="audience-1",
                versionId="audience-v1",
                versionNumber=1,
                digest=_digest("3"),
                payload={
                    "bannedTopics": ["hype"],
                    "avoidPhrases": ["synergy"],
                },
            )
        ],
    )
    result = GenerateResponse(
        intent="Technical Article",
        content="Evidence-first drafts cite every operational claim for technical buyers.",
    )
    assert _governed_output_violations(req, result) == []


def test_brand_voice_prompt_includes_deterministic_constraints():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="brand_kit",
                stableId="brand-1",
                versionId="brand-v1",
                versionNumber=1,
                digest=_digest("c"),
                payload={
                    "schemaVersion": 1,
                    "toneAttributes": ["precise"],
                    "preferredPhrases": ["evidence-first"],
                    "avoidPhrases": ["synergy"],
                    "bannedClaims": ["guaranteed ROI"],
                    "customInstructions": "Sound like a careful operator.",
                },
            )
        ],
    )
    system, _ = _build_prompts(req, "long", [])
    assert "Brand Voice deterministic constraints" in system
    assert 'Prefer Brand Voice tone attribute "precise"' in system
    assert 'Never use the Brand Voice avoid-phrase "synergy"' in system
    assert 'Never emit banned Brand Voice claim "guaranteed ROI"' in system
    assert "Sound like a careful operator" in system


def test_brand_voice_output_violations_cover_avoid_phrases_and_banned_claims():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="brand_kit",
                stableId="brand-1",
                versionId="brand-v1",
                versionNumber=1,
                digest=_digest("d"),
                payload={
                    "avoidPhrases": ["synergy"],
                    "bannedClaims": ["guaranteed ROI"],
                },
            )
        ],
    )
    result = GenerateResponse(
        intent="Technical Article",
        content="Our synergy delivers guaranteed ROI for every buyer.",
    )
    joined = "\n".join(_governed_output_violations(req, result))
    assert "Brand Voice avoid-phrase" in joined
    assert "synergy" in joined
    assert "banned claim" in joined
    assert "guaranteed ROI" in joined


def test_brand_voice_output_accepts_compliant_text():
    req = GenerateRequest(
        writingIntent="Technical Article",
        topic="CRM synchronization",
        generationStage="section",
        governedContext=[
            GovernedContextEntry(
                kind="brand_kit",
                stableId="brand-1",
                versionId="brand-v1",
                versionNumber=1,
                digest=_digest("e"),
                payload={
                    "avoidPhrases": ["synergy"],
                    "bannedClaims": ["guaranteed ROI"],
                },
            )
        ],
    )
    result = GenerateResponse(
        intent="Technical Article",
        content="Evidence-first drafts cite every operational claim for technical buyers.",
    )
    assert _governed_output_violations(req, result) == []


def test_product_output_blocks_missing_disclaimers_and_unsupported_claims():
    product = GovernedContextEntry(
        kind="product",
        stableId="product-1",
        versionId="product-v1",
        versionNumber=1,
        digest=_digest("a"),
        payload={"pricing": "Contact sales"},
        approvedClaims=["Evidence Engine cites every claim"],
        prohibitedClaims=["guarantees perfect accuracy"],
        mandatoryDisclaimers=["Results depend on source coverage."],
    )
    missing = GenerateRequest(
        writingIntent="Technical Article",
        topic="Evidence Engine",
        generationStage="complete",
        governedContext=[product],
    )
    missing_result = GenerateResponse(
        intent="Technical Article",
        content="Evidence Engine cites every claim and is ready for launch.",
    )
    missing_joined = "\n".join(_governed_output_violations(missing, missing_result))
    assert "required phrase is missing" in missing_joined

    unsupported = GenerateRequest(
        writingIntent="Technical Article",
        topic="Evidence Engine",
        generationStage="section",
        governedContext=[product],
    )
    unsupported_result = GenerateResponse(
        intent="Technical Article",
        content="Evidence Engine never fails and offers risk-free accuracy.",
    )
    unsupported_joined = "\n".join(
        _governed_output_violations(unsupported, unsupported_result)
    )
    assert "Unsupported product claim" in unsupported_joined

    prohibited = GenerateRequest(
        writingIntent="Technical Article",
        topic="Evidence Engine",
        generationStage="complete",
        governedContext=[product],
    )
    prohibited_result = GenerateResponse(
        intent="Technical Article",
        content=(
            "Evidence Engine cites every claim and guarantees perfect accuracy. "
            "Results depend on source coverage."
        ),
    )
    prohibited_joined = "\n".join(
        _governed_output_violations(prohibited, prohibited_result)
    )
    assert "prohibited phrase was emitted" in prohibited_joined

    ok = GenerateRequest(
        writingIntent="Technical Article",
        topic="Evidence Engine",
        generationStage="complete",
        governedContext=[product],
    )
    ok_result = GenerateResponse(
        intent="Technical Article",
        content=(
            "Evidence Engine cites every claim with source links. "
            "Results depend on source coverage."
        ),
    )
    assert _governed_output_violations(ok, ok_result) == []
