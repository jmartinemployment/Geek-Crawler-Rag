from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from geek_crawler_rag.app import app
from geek_crawler_rag.intelligence import IntelligenceService
from geek_crawler_rag.intelligence_models import (
    CompetitorAuditArtifact,
    CompetitorAuditRequest,
    CompetitorPageAnalysisArtifact,
    CompetitorPageAnalysisRequest,
    CompetitorPositioningArtifact,
    CompetitorPositioningRequest,
    ContentGapArtifact,
    ContentGapRequest,
    QueryPlanArtifact,
    QueryPlannerRequest,
    ReadinessComparisonArtifact,
    ReadinessComparisonRequest,
)

FIXTURES = Path(__file__).parent / "fixtures"
QUERY_FIXTURE = FIXTURES / "query-planner-golden.v1.json"
COMPETITOR_FIXTURE = FIXTURES / "competitor-intelligence-golden.v1.json"
READINESS_FIXTURE = FIXTURES / "readiness-comparison-golden.v1.json"
POSITIONING_FIXTURE = FIXTURES / "competitor-positioning-golden.v1.json"
POSITIONING_MULTI_INPUT = FIXTURES / "competitor-positioning-multi.input.json"
CONTENT_GAP_PARTIAL_INPUT = FIXTURES / "content-gap-partial.input.json"
PAGE_PARTIAL_INPUT = FIXTURES / "competitor-page-partial.input.json"


def _query_request() -> QueryPlannerRequest:
    return QueryPlannerRequest.model_validate(json.loads(QUERY_FIXTURE.read_text()))


def _gap_request() -> ContentGapRequest:
    payload = json.loads(COMPETITOR_FIXTURE.read_text())
    payload["contractVersion"] = "contentGapInput.v1"
    return ContentGapRequest.model_validate(payload)


def _page_request() -> CompetitorPageAnalysisRequest:
    payload = json.loads(COMPETITOR_FIXTURE.read_text())
    return CompetitorPageAnalysisRequest(
        contractVersion="competitorPageAnalysisInput.v1",
        page=payload["competitorPages"][0],
    )


def _readiness_request() -> ReadinessComparisonRequest:
    return ReadinessComparisonRequest.model_validate(
        json.loads(READINESS_FIXTURE.read_text())
    )


def _audit_request() -> CompetitorAuditRequest:
    payload = json.loads(COMPETITOR_FIXTURE.read_text())
    payload["contractVersion"] = "competitorAuditInput.v1"
    return CompetitorAuditRequest.model_validate(payload)


def _positioning_request() -> CompetitorPositioningRequest:
    return CompetitorPositioningRequest.model_validate(
        json.loads(POSITIONING_FIXTURE.read_text())
    )


def test_query_plan_golden_is_deterministic_and_evidence_labeled() -> None:
    service = IntelligenceService()
    first = service.query_plan(_query_request())
    second = service.query_plan(_query_request())

    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json", by_alias=True
    )
    assert first.artifact_type == "queryPlan.v1"
    assert first.methodology.methodology_id == "query-priority-heuristic.v1"
    assert first.methodology.formula == (
        "provenanceScore + intentScore + specificityScore + stageScore"
    )
    assert len(first.queries) == 4
    origins = [item.provenance.origin.value for item in first.queries]
    assert origins.count("observed") == 1
    assert origins.count("imported") == 1
    assert origins.count("generatedHypothesis") == 2
    assert all(
        item.provenance.source_reference
        == "deterministic-template:query-hypotheses.v1"
        for item in first.queries
        if item.provenance.origin.value == "generatedHypothesis"
    )
    observed = next(
        item
        for item in first.queries
        if item.provenance.origin.value == "observed"
    )
    assert observed.priority_score == 90
    assert observed.priority_signals.provenance_score == 40
    assert any("not observed demand" in warning for warning in first.warnings)
    assert any("no search volume" in warning.lower() for warning in first.warnings)


def test_query_planner_rejects_unsourced_or_caller_generated_demand() -> None:
    payload = json.loads(QUERY_FIXTURE.read_text())
    payload["queries"][0]["sourceId"] = "missing-source"
    with pytest.raises(ValidationError, match="missing from sources"):
        QueryPlannerRequest.model_validate(payload)

    payload = json.loads(QUERY_FIXTURE.read_text())
    payload["queries"][0]["origin"] = "generatedHypothesis"
    with pytest.raises(ValidationError, match="created only by the planner"):
        QueryPlannerRequest.model_validate(payload)


def test_query_planner_honors_zero_generated_queries() -> None:
    request = _query_request()
    request.max_generated_queries = 0

    artifact = IntelligenceService().query_plan(request)

    assert len(artifact.queries) == len(request.queries)
    assert all(
        item.provenance.origin.value != "generatedHypothesis"
        for item in artifact.queries
    )


def test_competitor_page_analysis_is_supplied_only_and_evidence_first() -> None:
    artifact = IntelligenceService().competitor_page_analysis(_page_request())

    assert artifact.artifact_type == "competitorPageAnalysis.v1"
    assert len(artifact.dimensions) == 7
    assert all(item.present is True for item in artifact.dimensions)
    assert all(item.evidence_ids for item in artifact.dimensions)
    assert artifact.opportunities == []
    assert artifact.provenance.sources[0].source_id == "competitor-page-1"
    assert "supplied-price-evidence" in artifact.provenance.evidence_ids
    assert any("does not infer traffic" in warning for warning in artifact.warnings)


def test_page_analysis_validates_exact_evidence_and_partial_absence() -> None:
    request = _page_request()
    request.page.evidence[0].quote = "Fabricated traffic claim with 1M visits."
    request.page.content_completeness = "partial"

    artifact = IntelligenceService().competitor_page_analysis(request)

    assert "supplied-price-evidence" not in artifact.provenance.evidence_ids
    assert any("failed exact" in warning for warning in artifact.warnings)
    assert any("partial" in warning.lower() for warning in artifact.warnings)

    empty_partial = request.model_copy(deep=True)
    empty_partial.page.visible_content = "A supplied fragment without full-page context."
    empty_partial.page.evidence = []
    partial_artifact = IntelligenceService().competitor_page_analysis(empty_partial)
    assert all(item.present is None for item in partial_artifact.dimensions)
    assert partial_artifact.opportunities == []


def test_content_gap_golden_has_comparisons_and_evidence_linked_gaps() -> None:
    service = IntelligenceService()
    first = service.content_gap(_gap_request())
    second = service.content_gap(_gap_request())

    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json", by_alias=True
    )
    assert first.artifact_type == "contentGapAnalysis.v1"
    assert len(first.dimensions) == 7
    assert {gap.dimension.value for gap in first.gaps} == {
        "topics",
        "questions",
        "proof",
        "pricing",
        "process",
        "callsToAction",
    }
    assert all(gap.status == "supportedGap" for gap in first.gaps)
    assert all(gap.evidence_ids for gap in first.gaps)
    assert all(
        evidence_id in first.provenance.evidence_ids
        for gap in first.gaps
        for evidence_id in gap.evidence_ids
    )
    assert [source.source_id for source in first.provenance.sources] == [
        "subject-page-1",
        "competitor-page-1",
    ]


def test_partial_subject_never_becomes_asserted_absent_or_supported_gap() -> None:
    request = _gap_request()
    request.subject_pages[0].content_completeness = "partial"
    artifact = IntelligenceService().content_gap(request)

    uncovered = [
        item for item in artifact.dimensions if item.subject_coverage is not True
    ]
    assert uncovered
    assert all(item.subject_coverage is None for item in uncovered)
    assert all(gap.status == "coverageUnknown" for gap in artifact.gaps)
    assert all(gap.confidence == "unknown" for gap in artifact.gaps)
    assert any("coverageUnknown" in warning for warning in artifact.warnings)


def test_readiness_comparison_reuses_diagnostic_rubric_and_is_deterministic() -> None:
    service = IntelligenceService()
    first = service.readiness_comparison(_readiness_request())
    second = service.readiness_comparison(_readiness_request())

    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json", by_alias=True
    )
    assert first.artifact_type == "readinessComparison.v1"
    assert first.rubric_version == "ai-readiness-heuristic.v1"
    assert first.methodology == "heuristic"
    assert first.subject.role == "subject"
    assert len(first.competitors) == 1
    assert first.competitors[0].competitor_id == "competitor-1"
    assert len(first.dimension_deltas) == 7
    assert first.subject.overall_score is not None
    assert any("not traffic" in warning.lower() for warning in first.warnings)
    assert any(
        "ai-readiness-heuristic.v1" in warning for warning in first.warnings
    )


def test_readiness_comparison_partial_subject_keeps_unknown_scores() -> None:
    request = _readiness_request()
    request.subject_page.content_completeness = "partial"
    request.subject_page.visible_content = "A short supplied fragment."
    request.subject_page.evidence = []
    request.subject_page.existing_json_ld = None
    request.subject_page.technical = None

    artifact = IntelligenceService().readiness_comparison(request)

    assert artifact.subject.overall_score is None
    assert any(item.subject_score is None for item in artifact.dimension_deltas)
    assert any("partial" in warning.lower() for warning in artifact.warnings)


def test_competitor_audit_composes_pages_gaps_and_prioritized_actions() -> None:
    service = IntelligenceService()
    first = service.competitor_audit(_audit_request())
    second = service.competitor_audit(_audit_request())

    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json", by_alias=True
    )
    assert first.artifact_type == "competitorAudit.v1"
    assert len(first.page_analyses) == 1
    assert first.page_analyses[0].artifact_type == "competitorPageAnalysis.v1"
    assert first.content_gap.artifact_type == "contentGapAnalysis.v1"
    assert first.prioritized_actions
    assert all(action.evidence_ids for action in first.prioritized_actions)
    assert all(
        evidence_id in first.provenance.evidence_ids
        for action in first.prioritized_actions
        for evidence_id in action.evidence_ids
    )
    assert any("not measure traffic" in warning.lower() for warning in first.warnings)
    priorities = [action.priority.value for action in first.prioritized_actions]
    assert priorities == sorted(
        priorities,
        key=lambda value: {"high": 0, "medium": 1, "low": 2}[value],
    )


def test_competitor_positioning_preserves_observations_and_labels_hypotheses() -> None:
    service = IntelligenceService()
    first = service.competitor_positioning(_positioning_request())
    second = service.competitor_positioning(_positioning_request())

    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json", by_alias=True
    )
    assert first.artifact_type == "competitorPositioning.v1"
    assert first.observations[0].model_or_engine == "example-engine/v1"
    assert first.observations[0].query == "best document analyzer"
    assert first.observations[0].raw_response.startswith("Competitor Inc.")
    assert first.observations[0].observed_at_utc == "2026-09-01T12:00:00Z"
    assert first.attribute_map
    assert any(gap.status == "supportedGap" for gap in first.perception_gaps)
    assert any(gap.status == "observationOnly" for gap in first.perception_gaps)
    assert first.messaging_hypotheses
    assert all(
        item.origin == "generatedHypothesis" for item in first.messaging_hypotheses
    )
    assert all(
        "not measured market perception" in item.disclaimer
        for item in first.messaging_hypotheses
    )
    assert any(
        "never be treated as measured market perception" in warning
        for warning in first.warnings
    )


def test_competitor_positioning_multi_competitor_cohort_warning() -> None:
    request = CompetitorPositioningRequest.model_validate(
        json.loads(POSITIONING_MULTI_INPUT.read_text())
    )
    artifact = IntelligenceService().competitor_positioning(request)
    assert any(
        "Positioning map considers 2 competitor pages." in warning
        for warning in artifact.warnings
    )
    assert any(
        attr.competitor_id == "alt-co"
        for attr in artifact.attribute_map
        if attr.party == "competitor"
    )


def test_partial_brand_positioning_never_asserts_absence() -> None:
    request = _positioning_request()
    request.brand_pages[0].content_completeness = "partial"
    artifact = IntelligenceService().competitor_positioning(request)

    content_gaps = [
        gap for gap in artifact.perception_gaps if gap.status != "observationOnly"
    ]
    assert content_gaps
    assert all(gap.status == "coverageUnknown" for gap in content_gaps)
    assert any("coverageUnknown" in warning for warning in artifact.warnings)


def test_multi_competitor_and_partial_crawl_goldens_match_live_service() -> None:
    """Committed multi/partial fixtures shared with GeekBackend contract tests."""
    service = IntelligenceService()
    cases = [
        (
            "competitorPositioning.multi.v1",
            CompetitorPositioningArtifact,
            service.competitor_positioning(
                CompetitorPositioningRequest.model_validate(
                    json.loads(POSITIONING_MULTI_INPUT.read_text())
                )
            ),
        ),
        (
            "contentGapAnalysis.partial.v1",
            ContentGapArtifact,
            service.content_gap(
                ContentGapRequest.model_validate(
                    json.loads(CONTENT_GAP_PARTIAL_INPUT.read_text())
                )
            ),
        ),
        (
            "competitorPageAnalysis.partial.v1",
            CompetitorPageAnalysisArtifact,
            service.competitor_page_analysis(
                CompetitorPageAnalysisRequest.model_validate(
                    json.loads(PAGE_PARTIAL_INPUT.read_text())
                )
            ),
        ),
    ]
    for artifact_type, contract, live in cases:
        path = FIXTURES / f"{artifact_type}.golden.json"
        golden = json.loads(path.read_text())
        assert golden["artifactType"] == artifact_type.replace(".multi", "").replace(
            ".partial", ""
        )
        assert contract.model_validate(golden) == live
        assert golden == json.loads(live.model_dump_json(by_alias=True))
        if "multi" in artifact_type:
            assert any(
                "Positioning map considers 2 competitor pages." in warning
                for warning in live.warnings
            )
        if "partial" in artifact_type and artifact_type.startswith("contentGap"):
            assert live.gaps
            assert all(gap.status == "coverageUnknown" for gap in live.gaps)
        if artifact_type.startswith("competitorPageAnalysis.partial"):
            assert all(item.present is None for item in live.dimensions)
            assert live.opportunities == []


def test_intelligence_contracts_round_trip_forbid_extras_and_pin_versions() -> None:
    service = IntelligenceService()
    artifacts = [
        (service.query_plan(_query_request()), QueryPlanArtifact),
        (service.competitor_page_analysis(_page_request()), CompetitorPageAnalysisArtifact),
        (service.content_gap(_gap_request()), ContentGapArtifact),
        (service.readiness_comparison(_readiness_request()), ReadinessComparisonArtifact),
        (service.competitor_audit(_audit_request()), CompetitorAuditArtifact),
        (
            service.competitor_positioning(_positioning_request()),
            CompetitorPositioningArtifact,
        ),
    ]
    for artifact, contract in artifacts:
        payload = artifact.model_dump(mode="json", by_alias=True)
        assert contract.model_validate(payload) == artifact
        payload["unexpected"] = True
        with pytest.raises(ValidationError):
            contract.model_validate(payload)

    query_payload = json.loads(QUERY_FIXTURE.read_text())
    query_payload["contractVersion"] = "queryPlannerInput.v2"
    with pytest.raises(ValidationError):
        QueryPlannerRequest.model_validate(query_payload)

    gap_payload = json.loads(COMPETITOR_FIXTURE.read_text())
    gap_payload["contractVersion"] = "contentGapInput.v1"
    gap_payload["subjectPages"][0]["unexpected"] = True
    with pytest.raises(ValidationError):
        ContentGapRequest.model_validate(gap_payload)

    readiness_payload = json.loads(READINESS_FIXTURE.read_text())
    readiness_payload["contractVersion"] = "readinessComparisonInput.v2"
    with pytest.raises(ValidationError):
        ReadinessComparisonRequest.model_validate(readiness_payload)


def test_intelligence_artifact_goldens_match_live_service_output() -> None:
    """Shared camelCase goldens consumed by GeekBackend intelligence contract tests."""
    service = IntelligenceService()
    expected = {
        "queryPlan.v1": (QueryPlanArtifact, service.query_plan(_query_request())),
        "competitorPageAnalysis.v1": (
            CompetitorPageAnalysisArtifact,
            service.competitor_page_analysis(_page_request()),
        ),
        "contentGapAnalysis.v1": (ContentGapArtifact, service.content_gap(_gap_request())),
        "readinessComparison.v1": (
            ReadinessComparisonArtifact,
            service.readiness_comparison(_readiness_request()),
        ),
        "competitorAudit.v1": (
            CompetitorAuditArtifact,
            service.competitor_audit(_audit_request()),
        ),
        "competitorPositioning.v1": (
            CompetitorPositioningArtifact,
            service.competitor_positioning(_positioning_request()),
        ),
    }
    for artifact_type, (contract, live) in expected.items():
        path = FIXTURES / f"{artifact_type}.golden.json"
        golden = json.loads(path.read_text())
        assert golden["artifactType"] == artifact_type
        assert contract.model_validate(golden) == live
        assert golden == json.loads(live.model_dump_json(by_alias=True))


def test_intelligence_routes_are_versioned_and_api_key_protected() -> None:
    expected = {
        "/v1/intelligence/query-plan",
        "/v1/intelligence/competitor-page",
        "/v1/intelligence/content-gap",
        "/v1/intelligence/readiness-comparison",
        "/v1/intelligence/competitor-audit",
        "/v1/intelligence/competitor-positioning",
    }
    routes = {route.path: route for route in app.routes if route.path in expected}
    assert set(routes) == expected
    assert all(route.dependant.dependencies for route in routes.values())
