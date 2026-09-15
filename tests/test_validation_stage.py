"""Contract tests for evidence-aware editorial validation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from geek_crawler_rag.config import Settings
from geek_crawler_rag.generate import _build_prompts, select_generation_model
from geek_crawler_rag.models import (
    GenerateCitation,
    GenerateProvenance,
    GenerateRequest,
    GenerateResponse,
    GenerateSource,
    GenerateValidation,
)
from geek_crawler_rag.quality_eval import evaluate_quality_contract

FIXTURE = Path(__file__).parent / "fixtures" / "validation_quality.json"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text())["request"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"draftContent": ""}, "requires non-empty draftContent"),
        ({"canonicalBrief": None}, "requires canonicalBrief"),
        (
            {"partnerRunId": None, "sources": None},
            "requires sources, partnerRunId, or competitorRunId",
        ),
        ({"modelPolicyPreset": None}, "requires modelPolicyPreset"),
        ({"modelPolicyVersion": None}, "requires modelPolicyVersion"),
    ],
)
def test_validation_requires_full_quality_contract(override, message):
    payload = _payload()
    payload.update(override)
    with pytest.raises(ValidationError, match=message):
        GenerateRequest(**payload)


def test_validation_model_policy_routes_strictly():
    settings = Settings()
    payload = _payload()
    assert select_generation_model(GenerateRequest(**payload), settings) == "o3"

    payload["modelPolicyPreset"] = "o3-only"
    assert select_generation_model(GenerateRequest(**payload), settings) == "o3"

    payload["modelPolicyPreset"] = "custom"
    payload["stageModelOverrides"] = {"validation": "o1-pro"}
    assert select_generation_model(GenerateRequest(**payload), settings) == "o1-pro"

    payload["stageModelOverrides"] = {"section": "o3"}
    with pytest.raises(ValidationError, match="no override.*validation"):
        GenerateRequest(**payload)


def test_validation_prompt_reviews_full_draft_brief_and_full_page_evidence():
    fixture = json.loads(FIXTURE.read_text())
    request = GenerateRequest(**fixture["request"])
    system, user = _build_prompts(request, "long", fixture["pages"])

    assert request.generation_stage == "validation"
    assert "Generation stage: VALIDATION" in user
    assert request.draft_content in user
    assert '"targetKeyword": "CRM synchronization"' in user
    assert fixture["pages"][0]["markdown"] in user
    assert "Do not rewrite, edit, or return replacement content" in user
    for category in (
        "unsupportedClaim",
        "sourceConflict",
        "briefAlignment",
        "brandVoice",
        "originalityRepetition",
        "usefulness",
        "cta",
        "seoGeo",
        "contentTypeRequirements",
    ):
        assert category in user
    assert "Use ONLY the provided source markdown" in system


def test_validation_shape_is_strict_and_unsupported_claims_fail_approval():
    fixture = json.loads(FIXTURE.read_text())
    validation = GenerateValidation.model_validate(fixture["validation"])

    assert validation.approved is False
    assert validation.unsupported_claim_count == 1

    incomplete = dict(fixture["validation"])
    incomplete.pop("usefulnessScore")
    with pytest.raises(ValidationError):
        GenerateValidation.model_validate(incomplete)

    invalid_category = json.loads(json.dumps(fixture["validation"]))
    invalid_category["issues"][0]["category"] = "general"
    with pytest.raises(ValidationError):
        GenerateValidation.model_validate(invalid_category)


def test_validation_response_serializes_structured_result_and_provenance():
    fixture = json.loads(FIXTURE.read_text())
    response = GenerateResponse(
        intent="Technical Article",
        validation=GenerateValidation.model_validate(fixture["validation"]),
        citations=[GenerateCitation(**item) for item in fixture["citations"]],
        sources=[GenerateSource(**item) for item in fixture["request"]["sources"]],
        evidenceWarnings=[],
        modelUsed="o3",
        retrieval="hybrid",
        provenance=GenerateProvenance(**fixture["provenance"]),
    )

    wire = response.model_dump(by_alias=True, exclude_none=True)
    assert "content" not in wire
    assert wire["validation"]["approved"] is False
    assert wire["validation"]["issues"][0]["category"] == "unsupportedClaim"
    assert wire["provenance"]["generationStage"] == "validation"
    assert wire["provenance"]["modelUsed"] == "o3"
    assert wire["provenance"]["evidenceIds"] == ["page-monitoring"]
    assert wire["citations"][0]["quote"] == "Acme records every synchronization attempt."


def test_validation_quality_fixture_scores_completeness():
    fixture = json.loads(FIXTURE.read_text())
    request = GenerateRequest(**fixture["request"])
    system, user = _build_prompts(request, "long", fixture["pages"])
    validation = GenerateValidation.model_validate(fixture["validation"])
    result = evaluate_quality_contract(
        request=request,
        system_prompt=system,
        user_prompt=user,
        citations=[GenerateCitation(**item) for item in fixture["citations"]],
        sources=[GenerateSource(**item) for item in fixture["request"]["sources"]],
        pages=fixture["pages"],
        provenance=fixture["provenance"],
        validation=validation,
    )

    assert result["signals"] == fixture["expectedSignals"]
    assert result["score"] == 1.0
