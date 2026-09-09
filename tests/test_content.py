from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from geek_crawler_rag.content import ContentService
from geek_crawler_rag.content_models import (
    CitableClaimsRequest,
    ClaimLedgerArtifact,
    ComparisonBriefArtifact,
    ComparisonBriefRequest,
    CompetitiveResponseArtifact,
    CompetitiveResponseRequest,
    FaqGeneratorRequest,
    FaqSetArtifact,
    PillarOutlineArtifact,
    PillarOutlineRequest,
)

FIXTURES = Path(__file__).parent / "fixtures"
FAQ_FIXTURE = FIXTURES / "faq-generator-golden.v1.json"
CLAIMS_FIXTURE = FIXTURES / "citable-claims-golden.v1.json"
COMPARISON_FIXTURE = FIXTURES / "comparison-brief-golden.v1.json"
RESPONSE_FIXTURE = FIXTURES / "competitive-response-golden.v1.json"
PILLAR_FIXTURE = FIXTURES / "pillar-outline-golden.v1.json"


def _faq_request() -> FaqGeneratorRequest:
    return FaqGeneratorRequest.model_validate(json.loads(FAQ_FIXTURE.read_text()))


def _claims_request() -> CitableClaimsRequest:
    return CitableClaimsRequest.model_validate(json.loads(CLAIMS_FIXTURE.read_text()))


def _comparison_request() -> ComparisonBriefRequest:
    return ComparisonBriefRequest.model_validate(
        json.loads(COMPARISON_FIXTURE.read_text())
    )


def _response_request() -> CompetitiveResponseRequest:
    return CompetitiveResponseRequest.model_validate(
        json.loads(RESPONSE_FIXTURE.read_text())
    )


def _pillar_request() -> PillarOutlineRequest:
    return PillarOutlineRequest.model_validate(json.loads(PILLAR_FIXTURE.read_text()))


def test_faq_set_golden_is_deterministic_and_evidence_linked() -> None:
    service = ContentService()
    first = service.faq_set(_faq_request())
    second = service.faq_set(_faq_request())

    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json", by_alias=True
    )
    assert first.artifact_type == "faqSet.v1"
    assert first.pairs
    supported = [pair for pair in first.pairs if pair.verification_status == "supported"]
    assert supported
    assert all(pair.citations for pair in supported)
    assert all(
        citation.quote in _faq_request().source_document.visible_content
        for pair in supported
        for citation in pair.citations
    )
    assert any("not verified against live search" in warning for warning in first.warnings)


def test_faq_generator_rejects_generated_hypothesis_queries() -> None:
    payload = json.loads(FAQ_FIXTURE.read_text())
    payload["queries"] = [{
        "query": "What is AI content readiness?",
        "origin": "generatedHypothesis",
        "sourceReference": "bad",
    }]
    with pytest.raises(ValidationError):
        FaqGeneratorRequest.model_validate(payload)


def test_faq_contract_round_trip_forbids_extras() -> None:
    artifact = ContentService().faq_set(_faq_request())
    payload = artifact.model_dump(mode="json", by_alias=True)
    assert FaqSetArtifact.model_validate(payload) == artifact
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        FaqSetArtifact.model_validate(payload)


def test_citable_claims_extracts_specifics_without_inventing_stats() -> None:
    service = ContentService()
    first = service.citable_claims(_claims_request())
    second = service.citable_claims(_claims_request())

    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json", by_alias=True
    )
    assert first.artifact_type == "claimLedger.v1"
    assert first.claims
    supported = [
        claim for claim in first.claims if claim.verification_status == "supported"
    ]
    assert supported
    assert all(claim.evidence_ids for claim in supported)
    assert all(
        evidence.quote in _claims_request().source_document.visible_content
        for evidence in first.provenance.evidence
    )
    unsupported = [
        claim for claim in first.claims if claim.verification_status == "unsupported"
    ]
    assert unsupported
    assert all("%" not in claim.claim_text or "37%" in claim.claim_text for claim in unsupported)
    assert any("no statistics were invented" in warning for warning in first.warnings)
    assert any(
        claim.source_statement is not None and claim.claim_type == "rewrittenSpecific"
        for claim in first.claims
    )


def test_citable_claims_partial_marks_unknown() -> None:
    payload = json.loads(CLAIMS_FIXTURE.read_text())
    payload["sourceDocument"]["contentCompleteness"] = "partial"
    payload["targetStatements"] = []
    artifact = ContentService().citable_claims(
        CitableClaimsRequest.model_validate(payload)
    )
    assert artifact.claims
    assert all(claim.confidence == "unknown" for claim in artifact.claims)
    assert any("partial" in warning.casefold() for warning in artifact.warnings)


def test_citable_claims_contract_round_trip_forbids_extras() -> None:
    artifact = ContentService().citable_claims(_claims_request())
    payload = artifact.model_dump(mode="json", by_alias=True)
    assert ClaimLedgerArtifact.model_validate(payload) == artifact
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ClaimLedgerArtifact.model_validate(payload)


def test_comparison_brief_is_deterministic_and_labeled() -> None:
    service = ContentService()
    first = service.comparison_brief(_comparison_request())
    second = service.comparison_brief(_comparison_request())

    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json", by_alias=True
    )
    assert first.artifact_type == "comparisonBrief.v1"
    assert first.subject_name == "Subject Analyzer"
    assert first.competitor_name == "Competitor Inc."
    assert len(first.criteria) == 4
    assert first.recommended_verdict.disclaimer
    assert first.differentiators or first.positioning_angles or first.proof_requirements
    assert all(
        item.origin == "generatedHypothesis"
        for item in [
            *first.differentiators,
            *first.positioning_angles,
            *first.proof_requirements,
        ]
    )
    assert any("not a full article" in warning for warning in first.warnings)
    assert first.provenance.sources


def test_comparison_brief_partial_keeps_unknown() -> None:
    payload = json.loads(COMPARISON_FIXTURE.read_text())
    payload["subjectPages"][0]["contentCompleteness"] = "partial"
    payload["subjectPages"][0]["visibleContent"] = "# Subject Analyzer\n\nPartial fragment."
    artifact = ContentService().comparison_brief(
        ComparisonBriefRequest.model_validate(payload)
    )
    assert any("unknown" in warning.casefold() or "partial" in warning.casefold() for warning in artifact.warnings)
    assert "unknown" in artifact.recommended_verdict.framing.casefold() or any(
        "unknown" in row.summary.casefold() for row in artifact.criteria
    )


def test_comparison_brief_contract_round_trip_forbids_extras() -> None:
    artifact = ContentService().comparison_brief(_comparison_request())
    payload = artifact.model_dump(mode="json", by_alias=True)
    assert ComparisonBriefArtifact.model_validate(payload) == artifact
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ComparisonBriefArtifact.model_validate(payload)


def test_competitive_response_selects_mode_without_copying() -> None:
    service = ContentService()
    first = service.competitive_response(_response_request())
    second = service.competitive_response(_response_request())

    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json", by_alias=True
    )
    assert first.artifact_type == "competitiveResponse.v1"
    assert first.selected_mode in {"newPage", "targetedUpdate", "counterNarrative"}
    assert first.content_angles
    assert all(angle.origin == "generatedHypothesis" for angle in first.content_angles)
    assert first.outline_sections
    assert any("Does not copy competitor prose" in warning for warning in first.warnings)
    assert any("generatedHypothesis" in warning for warning in first.warnings)
    assert all(
        "Trusted by 500 customer teams." not in angle.text
        for angle in first.content_angles
    )
    assert "Trusted by 500 customer teams." not in first.rationale


def test_competitive_response_contract_round_trip_forbids_extras() -> None:
    artifact = ContentService().competitive_response(_response_request())
    payload = artifact.model_dump(mode="json", by_alias=True)
    assert CompetitiveResponseArtifact.model_validate(payload) == artifact
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        CompetitiveResponseArtifact.model_validate(payload)


def test_pillar_outline_builds_cluster_sections() -> None:
    service = ContentService()
    first = service.pillar_outline(_pillar_request())
    second = service.pillar_outline(_pillar_request())

    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json", by_alias=True
    )
    assert first.artifact_type == "pillarOutline.v1"
    assert first.topic == "AI content readiness"
    assert first.sections
    assert any(section.evidence_ids for section in first.sections)
    assert first.supporting_content_plan
    assert all(
        item.origin == "generatedHypothesis" for item in first.supporting_content_plan
    )
    assert any("not a full article" in warning for warning in first.warnings)


def test_pillar_outline_rejects_generated_hypothesis_queries() -> None:
    payload = json.loads(PILLAR_FIXTURE.read_text())
    payload["queries"] = [{
        "query": "What is AI content readiness?",
        "origin": "generatedHypothesis",
        "sourceReference": "bad",
    }]
    with pytest.raises(ValidationError):
        PillarOutlineRequest.model_validate(payload)


def test_pillar_outline_contract_round_trip_forbids_extras() -> None:
    artifact = ContentService().pillar_outline(_pillar_request())
    payload = artifact.model_dump(mode="json", by_alias=True)
    assert PillarOutlineArtifact.model_validate(payload) == artifact
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        PillarOutlineArtifact.model_validate(payload)
