from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from geek_crawler_rag.app import app
from geek_crawler_rag.diagnostic_models import (
    EntityMapArtifact,
    EntityMapRequest,
    FactDensityArtifact,
    FactDensityRequest,
    ReadinessScoreArtifact,
    ReadinessScoreRequest,
    SchemaMarkupArtifact,
    SchemaMarkupRequest,
)
from geek_crawler_rag.diagnostics import DiagnosticService

FIXTURE = Path(__file__).parent / "fixtures" / "diagnostic-golden.v1.json"


def _readiness_request() -> ReadinessScoreRequest:
    return ReadinessScoreRequest.model_validate(json.loads(FIXTURE.read_text()))


def test_readiness_golden_is_deterministic_and_seven_dimensional() -> None:
    service = DiagnosticService()
    first = service.readiness_score(_readiness_request())
    second = service.readiness_score(_readiness_request())

    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json", by_alias=True
    )
    assert first.artifact_type == "readinessScore.v1"
    assert first.methodology == "heuristic"
    assert first.overall_score == 76.4
    assert [item.dimension for item in first.dimensions] == [
        "headingHierarchy",
        "answerFirst",
        "faq",
        "schema",
        "factDensity",
        "eeat",
        "technicalCrawlabilityPerformance",
    ]


def test_partial_readiness_omits_unknown_scores_instead_of_fabricating() -> None:
    request = _readiness_request()
    request.document.content_completeness = "partial"
    request.document.existing_json_ld = None
    request.document.technical = None
    request.document.visible_content = "A short supplied fragment."

    artifact = DiagnosticService().readiness_score(request)

    assert artifact.overall_score is None
    scores = {item.dimension: item.score for item in artifact.dimensions}
    assert scores["headingHierarchy"] is None
    assert scores["faq"] is None
    assert scores["schema"] is None
    assert scores["eeat"] is None
    assert scores["technicalCrawlabilityPerformance"] is None
    assert any("partial" in warning.lower() for warning in artifact.warnings)


def test_fact_density_uses_exact_evidence_and_marks_unsupported_claims() -> None:
    request = FactDensityRequest(document=_readiness_request().document)
    artifact = DiagnosticService().fact_density(request)

    assert artifact.artifact_type == "factDensityReport.v1"
    assert len(artifact.sections) > 1
    supported = [
        claim for claim in artifact.claims if claim.verification_status == "supported"
    ]
    assert supported
    assert supported[0].evidence_ids == ["ev-1"]
    assert artifact.unsupported_claim_ids
    assert artifact.provenance.evidence_ids == ["ev-1"]

    request.document.evidence[0].quote = "This quote is not in the supplied content."
    invalid = DiagnosticService().fact_density(request)
    assert invalid.provenance.evidence_ids == []
    assert any("failed exact" in warning for warning in invalid.warnings)


def test_entity_map_is_canonical_typed_and_evidence_linked() -> None:
    document = _readiness_request().document
    request = EntityMapRequest(
        document=document,
        seeds=[
            {
                "canonicalName": "Acme Analyzer",
                "entityType": "product",
                "aliases": ["Analyzer"],
            },
            {
                "canonicalName": "Acme Research",
                "entityType": "brand",
                "aliases": [],
            },
        ],
    )
    artifact = DiagnosticService().entity_map(request)

    assert artifact.artifact_type == "entityMap.v1"
    by_name = {entity.canonical_name: entity for entity in artifact.entities}
    assert by_name["Acme Analyzer"].entity_type == "product"
    assert by_name["Acme Analyzer"].evidence_ids
    assert by_name["Acme Research"].entity_type == "brand"
    assert any(
        relationship.relation == "coOccursWith"
        for relationship in artifact.relationships
    )
    assert artifact.coverage_comparisons == []
    assert artifact.recommendations == []
    assert set(artifact.provenance.evidence_ids) == {
        evidence.evidence_id for evidence in artifact.provenance.evidence
    }


def test_entity_map_competitor_coverage_delta() -> None:
    subject = _readiness_request().document
    competitor = subject.model_copy(
        update={
            "visible_content": (
                "# Rival overview\n\n"
                "Rival Labs positions Trust Layer next to Acme Analyzer on shortlists."
            ),
            "source": subject.source.model_copy(
                update={"source_id": "competitor:rival", "url": "https://rival.example/page"}
            ),
        }
    )
    artifact = DiagnosticService().entity_map(
        EntityMapRequest(
            document=subject,
            competitorDocument=competitor,
            seeds=[
                {
                    "canonicalName": "Acme Analyzer",
                    "entityType": "product",
                    "aliases": ["Analyzer"],
                },
                {
                    "canonicalName": "Trust Layer",
                    "entityType": "product",
                    "aliases": [],
                },
            ],
        )
    )

    by_name = {
        row.canonical_name: row for row in artifact.coverage_comparisons
    }
    assert by_name["Acme Analyzer"].status == "presentBoth"
    assert by_name["Trust Layer"].status == "missingOnSubject"
    assert by_name["Trust Layer"].on_subject is False
    assert by_name["Trust Layer"].on_competitor is True
    assert any("Trust Layer" in item.action for item in artifact.recommendations)
    assert all(item.evidence_ids for item in artifact.recommendations)


def test_schema_markup_generates_only_visible_applicable_shapes() -> None:
    request = SchemaMarkupRequest(
        document=_readiness_request().document,
        requestedTypes=["Article", "FAQPage", "HowTo", "Product"],
    )
    artifact = DiagnosticService().schema_markup(request)

    assert artifact.artifact_type == "schemaMarkup.v1"
    assert [node["@type"] for node in artifact.json_ld] == [
        "Article",
        "FAQPage",
        "HowTo",
        "Product",
    ]
    assert all(finding.valid for finding in artifact.validation)
    encoded = json.dumps(artifact.json_ld)
    assert "ratingValue" not in encoded
    assert '"review":' not in encoded


def test_cross_contract_round_trip_and_strict_versions() -> None:
    service = DiagnosticService()
    document = _readiness_request().document
    artifacts = [
        service.readiness_score(ReadinessScoreRequest(document=document)),
        service.fact_density(FactDensityRequest(document=document)),
        service.entity_map(EntityMapRequest(document=document)),
        service.schema_markup(SchemaMarkupRequest(document=document)),
    ]
    contracts = [
        ReadinessScoreArtifact,
        FactDensityArtifact,
        EntityMapArtifact,
        SchemaMarkupArtifact,
    ]
    for artifact, contract in zip(artifacts, contracts, strict=True):
        payload = json.loads(artifact.model_dump_json(by_alias=True))
        assert contract.model_validate(payload) == artifact

    payload = json.loads(FIXTURE.read_text())
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ReadinessScoreRequest.model_validate(payload)
    payload.pop("unexpected")
    payload["contractVersion"] = "aiReadinessInput.v2"
    with pytest.raises(ValidationError):
        ReadinessScoreRequest.model_validate(payload)


def test_query_provenance_requires_source_for_real_query_intelligence() -> None:
    payload = json.loads(FIXTURE.read_text())
    payload["document"]["queries"][0].pop("sourceId")
    with pytest.raises(ValidationError, match="source provenance"):
        ReadinessScoreRequest.model_validate(payload)


def test_service_authenticated_diagnostic_routes_are_registered() -> None:
    expected = {
        "/v1/diagnostics/readiness-score",
        "/v1/diagnostics/fact-density",
        "/v1/diagnostics/entity-map",
        "/v1/diagnostics/schema-markup",
    }
    routes = {route.path: route for route in app.routes if route.path in expected}
    assert set(routes) == expected
    assert all(route.dependant.dependencies for route in routes.values())
