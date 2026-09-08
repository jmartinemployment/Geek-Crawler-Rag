"""Offline contract and quality checks for final synthesis."""

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
    GenerateSource,
)
from geek_crawler_rag.quality_eval import evaluate_quality_contract

FIXTURE = Path(__file__).parent / "fixtures" / "final_synthesis_quality.json"


def _request(**overrides) -> GenerateRequest:
    payload = {
        "writingIntent": "Technical Article",
        "topic": "CRM synchronization",
        "partnerRunId": "partner-run",
        "generationStage": "finalSynthesis",
        "draftContent": "# CRM synchronization\n\n## Conflicts\n\nSupported draft.",
        "canonicalBrief": {
            "version": "gcc-v2-generation-brief.v1",
            "title": "CRM synchronization",
        },
        "sources": [
            {
                "pageId": "page-1",
                "url": "https://example.test/crm",
                "crawlType": "partner",
            }
        ],
        "modelPolicyPreset": "best-quality",
        "modelPolicyVersion": "content-model-policy.v1",
    }
    payload.update(overrides)
    return GenerateRequest(**payload)


@pytest.mark.parametrize(
    "wire_stage", ["finalSynthesis", "FinalSynthesis", "final_synthesis", "final-synthesis"]
)
def test_final_synthesis_stage_normalizes_to_camel_case_wire_value(wire_stage):
    request = _request(generationStage=wire_stage)
    assert request.generation_stage == "finalSynthesis"
    wire = request.model_dump(by_alias=True, exclude_none=True)
    assert wire["generationStage"] == "finalSynthesis"
    assert wire["draftContent"].startswith("# CRM")
    assert wire["sources"][0]["pageId"] == "page-1"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"draftContent": " "}, "requires non-empty draftContent"),
        ({"canonicalBrief": None}, "requires canonicalBrief"),
        (
            {"partnerRunId": None, "sources": None},
            "requires sources, partnerRunId, or competitorRunId",
        ),
        ({"modelPolicyPreset": None}, "requires modelPolicyPreset"),
        ({"modelPolicyVersion": None}, "requires modelPolicyVersion"),
    ],
)
def test_final_synthesis_rejects_incomplete_contract(override, message):
    with pytest.raises(ValidationError, match=message):
        _request(**override)


def test_final_synthesis_model_policy_is_strict_and_stage_aware():
    settings = Settings()
    assert select_generation_model(_request(), settings) == "o1-pro"
    assert (
        select_generation_model(
            _request(modelPolicyPreset="o3-only"),
            settings,
        )
        == "o3"
    )
    assert (
        select_generation_model(
            _request(
                modelPolicyPreset="custom",
                stageModelOverrides={"finalSynthesis": "o3"},
            ),
            settings,
        )
        == "o3"
    )
    with pytest.raises(ValidationError, match="no override.*finalSynthesis"):
        _request(
            modelPolicyPreset="custom",
            stageModelOverrides={"section": "o3"},
        )


def test_final_synthesis_prompt_contains_full_draft_and_quality_constraints():
    request = _request()
    system, user = _build_prompts(request, "long", [])
    assert "Generation stage: FINAL SYNTHESIS" in user
    assert request.draft_content in user
    assert "whole-document editorial coherence" in user
    assert "alignment with every applicable canonical-brief requirement" in user
    assert "originality" in user
    assert "remove repetition" in user
    assert "Preserve every Markdown heading exactly" in user
    assert "Preserve the exact meaning, numbers, product names" in user
    assert "Do not invent, infer, strengthen, or add facts" in user
    assert "Every factual claim" in system
    assert "citation quote must be copied verbatim" in user


def test_offline_quality_fixture_scores_all_contract_signals():
    fixture = json.loads(FIXTURE.read_text())
    request = GenerateRequest(**fixture["request"])
    system, user = _build_prompts(request, "long", fixture["pages"])
    citations = [GenerateCitation(**item) for item in fixture["citations"]]
    sources = [GenerateSource(**item) for item in fixture["request"]["sources"]]
    provenance = GenerateProvenance(**fixture["provenance"])

    result = evaluate_quality_contract(
        request=request,
        system_prompt=system,
        user_prompt=user,
        citations=citations,
        sources=sources,
        pages=fixture["pages"],
        provenance=provenance,
    )

    assert result["signals"] == fixture["expectedSignals"]
    assert result["score"] == 1.0


def test_offline_quality_evaluation_detects_unverified_and_uncovered_evidence():
    fixture = json.loads(FIXTURE.read_text())
    request = GenerateRequest(**fixture["request"])
    system, user = _build_prompts(request, "long", fixture["pages"])
    citations = [GenerateCitation(**fixture["citations"][0])]
    citations[0].quote = "Invented unsupported quote that does not occur in Markdown."

    result = evaluate_quality_contract(
        request=request,
        system_prompt=system,
        user_prompt=user,
        citations=citations,
        sources=[GenerateSource(**item) for item in fixture["request"]["sources"]],
        pages=fixture["pages"],
        provenance=fixture["provenance"],
    )

    assert result["signals"]["citationExactness"] == 0.0
    assert result["signals"]["evidenceCoverage"] == 0.0
    assert result["details"]["droppedCitationCount"] == 1


def test_offline_quality_evaluation_detects_prompt_and_provenance_gaps():
    fixture = json.loads(FIXTURE.read_text())
    request = GenerateRequest(**fixture["request"])
    system, user = _build_prompts(request, "long", fixture["pages"])
    user = user.replace('"targetKeyword"', '"omittedTargetKeyword"')
    user = user.replace("Preserve every Markdown heading exactly", "Keep headings")
    incomplete_provenance = dict(fixture["provenance"])
    incomplete_provenance.pop("promptVersion")

    result = evaluate_quality_contract(
        request=request,
        system_prompt=system,
        user_prompt=user,
        citations=[GenerateCitation(**item) for item in fixture["citations"]],
        sources=[GenerateSource(**item) for item in fixture["request"]["sources"]],
        pages=fixture["pages"],
        provenance=incomplete_provenance,
    )

    assert result["signals"]["briefPromptCoverage"] < 1.0
    assert result["signals"]["headingPreservationRequirements"] == 0.0
    assert result["signals"]["provenanceCompleteness"] < 1.0
