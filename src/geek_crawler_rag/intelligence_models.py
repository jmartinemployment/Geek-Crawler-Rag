"""Strict versioned contracts for deterministic query and competitor intelligence."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from geek_crawler_rag.diagnostic_models import (
    DimensionScore,
    EvidenceReference,
    QueryOrigin,
    QueryProvenance,
    SourceProvenance,
    StrictContract,
    TechnicalSignals,
)


class SearchIntent(StrEnum):
    INFORMATIONAL = "informational"
    COMMERCIAL = "commercial"
    TRANSACTIONAL = "transactional"
    NAVIGATIONAL = "navigational"


class JourneyStage(StrEnum):
    AWARENESS = "awareness"
    CONSIDERATION = "consideration"
    DECISION = "decision"
    RETENTION = "retention"


class PriorityTier(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QueryPlannerRequest(StrictContract):
    contract_version: Literal["queryPlannerInput.v1"] = Field(
        "queryPlannerInput.v1", alias="contractVersion"
    )
    queries: list[QueryProvenance] = Field(default_factory=list, max_length=500)
    hypothesis_topics: list[str] = Field(
        default_factory=list, alias="hypothesisTopics", max_length=100
    )
    sources: list[SourceProvenance] = Field(default_factory=list, max_length=500)
    max_generated_queries: int = Field(
        20, alias="maxGeneratedQueries", ge=0, le=200
    )

    @model_validator(mode="after")
    def coherent_query_sources(self) -> QueryPlannerRequest:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Query planner sourceId values must be unique.")
        known = set(source_ids)
        for query in self.queries:
            if query.origin == QueryOrigin.GENERATED_HYPOTHESIS:
                raise ValueError(
                    "Caller queries cannot use generatedHypothesis origin; "
                    "hypotheses are created only by the planner."
                )
            if query.source_id and query.source_id not in known:
                raise ValueError(
                    f"Query sourceId={query.source_id} is missing from sources."
                )
        if not self.queries and not any(topic.strip() for topic in self.hypothesis_topics):
            raise ValueError("At least one sourced query or hypothesis topic is required.")
        return self


class QueryClassification(StrictContract):
    intent: SearchIntent
    stage: JourneyStage
    rationale: str


class QueryPrioritySignals(StrictContract):
    provenance_score: int = Field(alias="provenanceScore", ge=0, le=40)
    intent_score: int = Field(alias="intentScore", ge=0, le=25)
    specificity_score: int = Field(alias="specificityScore", ge=0, le=20)
    stage_score: int = Field(alias="stageScore", ge=0, le=15)


class PlannedQuery(StrictContract):
    query_id: str = Field(..., alias="queryId")
    query: str
    provenance: QueryProvenance
    classification: QueryClassification
    cluster_id: str = Field(..., alias="clusterId")
    cluster_label: str = Field(..., alias="clusterLabel")
    priority_score: int = Field(..., alias="priorityScore", ge=0, le=100)
    priority_tier: PriorityTier = Field(..., alias="priorityTier")
    priority_signals: QueryPrioritySignals = Field(..., alias="prioritySignals")

    @model_validator(mode="after")
    def query_matches_provenance(self) -> PlannedQuery:
        if self.query != self.provenance.query:
            raise ValueError("Planned query must exactly match its provenance query.")
        return self


class QueryPriorityMethodology(StrictContract):
    methodology_id: Literal["query-priority-heuristic.v1"] = Field(
        "query-priority-heuristic.v1", alias="methodologyId"
    )
    formula: str = (
        "provenanceScore + intentScore + specificityScore + stageScore"
    )
    provenance_weights: dict[Literal["observed", "imported", "generatedHypothesis"], int] = (
        Field(
            default_factory=lambda: {
                "observed": 40,
                "imported": 25,
                "generatedHypothesis": 5,
            },
            alias="provenanceWeights",
        )
    )
    intent_weights: dict[
        Literal["informational", "commercial", "transactional", "navigational"], int
    ] = Field(
        default_factory=lambda: {
            "informational": 10,
            "commercial": 20,
            "transactional": 25,
            "navigational": 15,
        },
        alias="intentWeights",
    )
    specificity_rule: str = (
        "20 for queries with at least five tokens, 10 for three or four, otherwise 5"
    )
    stage_weights: dict[
        Literal["awareness", "consideration", "decision", "retention"], int
    ] = Field(
        default_factory=lambda: {
            "awareness": 5,
            "consideration": 10,
            "decision": 15,
            "retention": 10,
        },
        alias="stageWeights",
    )
    tier_thresholds: str = "high >= 70; medium >= 45; low < 45"
    demand_disclaimer: str = Field(
        "Scores are deterministic planning heuristics, not traffic, volume, "
        "ranking, or demand measurements.",
        alias="demandDisclaimer",
    )


class QueryPlanArtifact(StrictContract):
    artifact_type: Literal["queryPlan.v1"] = Field(
        "queryPlan.v1", alias="artifactType"
    )
    planner_version: Literal["deterministic-query-planner.v1"] = Field(
        "deterministic-query-planner.v1", alias="plannerVersion"
    )
    methodology: QueryPriorityMethodology
    queries: list[PlannedQuery]
    cluster_count: int = Field(..., alias="clusterCount", ge=0)
    warnings: list[str]
    sources: list[SourceProvenance]

    @model_validator(mode="after")
    def validate_cluster_count(self) -> QueryPlanArtifact:
        if self.cluster_count != len({item.cluster_id for item in self.queries}):
            raise ValueError("clusterCount must match the distinct query clusters.")
        return self


class ContentCompleteness(StrEnum):
    FULL = "full"
    PARTIAL = "partial"


class PageSnapshot(StrictContract):
    source: SourceProvenance
    visible_content: str = Field(
        ..., alias="visibleContent", min_length=1, max_length=2_000_000
    )
    media_type: Literal["text/markdown", "text/plain", "text/html"] = Field(
        "text/markdown", alias="mediaType"
    )
    content_completeness: ContentCompleteness = Field(
        ContentCompleteness.FULL, alias="contentCompleteness"
    )
    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=1000)
    existing_json_ld: list[dict[str, Any]] | None = Field(
        None, alias="existingJsonLd", max_length=100
    )
    technical: TechnicalSignals | None = None

    @model_validator(mode="after")
    def evidence_belongs_to_snapshot(self) -> PageSnapshot:
        if any(item.source_id != self.source.source_id for item in self.evidence):
            raise ValueError("Snapshot evidence must reference its own sourceId.")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Snapshot evidenceId values must be unique.")
        return self


class CompetitorPageSnapshot(PageSnapshot):
    competitor_id: str = Field(..., alias="competitorId", min_length=1, max_length=200)
    competitor_name: str = Field(
        ..., alias="competitorName", min_length=1, max_length=300
    )


class IntelligenceDimension(StrEnum):
    TOPICS = "topics"
    QUESTIONS = "questions"
    CAPABILITIES = "capabilities"
    PROOF = "proof"
    PRICING = "pricing"
    PROCESS = "process"
    CALLS_TO_ACTION = "callsToAction"


class DimensionFinding(StrictContract):
    dimension: IntelligenceDimension
    present: bool | None
    summary: str
    signals: list[str]
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")


class IntelligenceProvenance(StrictContract):
    engine_version: Literal["deterministic-competitor-intelligence.v1"] = Field(
        "deterministic-competitor-intelligence.v1", alias="engineVersion"
    )
    sources: list[SourceProvenance]
    evidence_ids: list[str] = Field(alias="evidenceIds")
    evidence: list[EvidenceReference]

    @model_validator(mode="after")
    def validate_references(self) -> IntelligenceProvenance:
        source_ids = [source.source_id for source in self.sources]
        actual_evidence_ids = [item.evidence_id for item in self.evidence]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Artifact provenance sourceId values must be unique.")
        if len(actual_evidence_ids) != len(set(actual_evidence_ids)):
            raise ValueError("Artifact provenance evidenceId values must be unique.")
        if self.evidence_ids != actual_evidence_ids:
            raise ValueError("evidenceIds must match evidence records in order.")
        if any(item.source_id not in set(source_ids) for item in self.evidence):
            raise ValueError("Artifact evidence references an unknown sourceId.")
        return self


class CompetitorPageAnalysisRequest(StrictContract):
    contract_version: Literal["competitorPageAnalysisInput.v1"] = Field(
        "competitorPageAnalysisInput.v1", alias="contractVersion"
    )
    page: CompetitorPageSnapshot


class CompetitorPageAnalysisArtifact(StrictContract):
    artifact_type: Literal["competitorPageAnalysis.v1"] = Field(
        "competitorPageAnalysis.v1", alias="artifactType"
    )
    analyzer_version: Literal["competitor-page-heuristic.v1"] = Field(
        "competitor-page-heuristic.v1", alias="analyzerVersion"
    )
    methodology: str = (
        "Detect explicit visible-content signals with deterministic patterns; "
        "attach exact source spans; treat absences in partial inputs as unknown."
    )
    competitor_id: str = Field(alias="competitorId")
    dimensions: list[DimensionFinding]
    opportunities: list[str]
    warnings: list[str]
    provenance: IntelligenceProvenance


class ContentGapRequest(StrictContract):
    contract_version: Literal["contentGapInput.v1"] = Field(
        "contentGapInput.v1", alias="contractVersion"
    )
    subject_pages: list[PageSnapshot] = Field(
        ..., alias="subjectPages", min_length=1, max_length=100
    )
    competitor_pages: list[CompetitorPageSnapshot] = Field(
        ..., alias="competitorPages", min_length=1, max_length=100
    )

    @model_validator(mode="after")
    def unique_sources(self) -> ContentGapRequest:
        source_ids = [
            page.source.source_id
            for page in [*self.subject_pages, *self.competitor_pages]
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("All content-gap sourceId values must be unique.")
        return self


class ComparativeDimension(StrictContract):
    dimension: IntelligenceDimension
    subject_coverage: bool | None = Field(alias="subjectCoverage")
    competitor_coverage_count: int = Field(alias="competitorCoverageCount", ge=0)
    competitor_page_count: int = Field(alias="competitorPageCount", ge=1)
    subject_evidence_ids: list[str] = Field(alias="subjectEvidenceIds")
    competitor_evidence_ids: list[str] = Field(alias="competitorEvidenceIds")
    summary: str


class GapRecord(StrictContract):
    gap_id: str = Field(..., alias="gapId")
    dimension: IntelligenceDimension
    signal: str
    status: Literal["supportedGap", "coverageUnknown"]
    opportunity: str
    competitor_source_ids: list[str] = Field(alias="competitorSourceIds", min_length=1)
    evidence_ids: list[str] = Field(alias="evidenceIds", min_length=1)
    confidence: Literal["high", "medium", "unknown"]


class ContentGapArtifact(StrictContract):
    artifact_type: Literal["contentGapAnalysis.v1"] = Field(
        "contentGapAnalysis.v1", alias="artifactType"
    )
    analyzer_version: Literal["comparative-content-gap-heuristic.v1"] = Field(
        "comparative-content-gap-heuristic.v1", alias="analyzerVersion"
    )
    methodology: str = (
        "Compare deterministic visible-content signals by dimension. A supported gap "
        "requires complete subject input, absent subject signals, and supplied competitor "
        "evidence. Partial subject input yields coverageUnknown, never an asserted absence."
    )
    dimensions: list[ComparativeDimension]
    gaps: list[GapRecord]
    warnings: list[str]
    provenance: IntelligenceProvenance


class ReadinessComparisonRequest(StrictContract):
    contract_version: Literal["readinessComparisonInput.v1"] = Field(
        "readinessComparisonInput.v1", alias="contractVersion"
    )
    subject_page: PageSnapshot = Field(..., alias="subjectPage")
    competitor_pages: list[CompetitorPageSnapshot] = Field(
        ..., alias="competitorPages", min_length=1, max_length=4
    )

    @model_validator(mode="after")
    def unique_sources(self) -> ReadinessComparisonRequest:
        source_ids = [
            page.source.source_id
            for page in [self.subject_page, *self.competitor_pages]
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("All readiness-comparison sourceId values must be unique.")
        return self


class PageReadinessEntry(StrictContract):
    role: Literal["subject", "competitor"]
    source_id: str = Field(..., alias="sourceId")
    competitor_id: str | None = Field(None, alias="competitorId")
    competitor_name: str | None = Field(None, alias="competitorName")
    overall_score: float | None = Field(None, alias="overallScore", ge=0, le=100)
    dimensions: list[DimensionScore] = Field(..., min_length=7, max_length=7)
    prioritized_fixes: list[str] = Field(alias="prioritizedFixes")
    warnings: list[str]
    evidence_ids: list[str] = Field(alias="evidenceIds")


class CompetitorDimensionScore(StrictContract):
    source_id: str = Field(..., alias="sourceId")
    competitor_id: str = Field(..., alias="competitorId")
    score: float | None = Field(None, ge=0, le=100)


class ReadinessDimensionDelta(StrictContract):
    dimension: Literal[
        "headingHierarchy",
        "answerFirst",
        "faq",
        "schema",
        "factDensity",
        "eeat",
        "technicalCrawlabilityPerformance",
    ]
    subject_score: float | None = Field(None, alias="subjectScore", ge=0, le=100)
    competitor_scores: list[CompetitorDimensionScore] = Field(
        ..., alias="competitorScores"
    )
    best_competitor_score: float | None = Field(
        None, alias="bestCompetitorScore", ge=0, le=100
    )
    delta_vs_best_competitor: float | None = Field(
        None, alias="deltaVsBestCompetitor"
    )
    summary: str


class ReadinessComparisonArtifact(StrictContract):
    artifact_type: Literal["readinessComparison.v1"] = Field(
        "readinessComparison.v1", alias="artifactType"
    )
    analyzer_version: Literal["ai-readiness-comparison-heuristic.v1"] = Field(
        "ai-readiness-comparison-heuristic.v1", alias="analyzerVersion"
    )
    methodology: Literal["heuristic"] = "heuristic"
    rubric_version: Literal["ai-readiness-heuristic.v1"] = Field(
        "ai-readiness-heuristic.v1", alias="rubricVersion"
    )
    subject: PageReadinessEntry
    competitors: list[PageReadinessEntry]
    dimension_deltas: list[ReadinessDimensionDelta] = Field(
        ..., alias="dimensionDeltas", min_length=7, max_length=7
    )
    prioritized_fixes: list[str] = Field(alias="prioritizedFixes")
    warnings: list[str]
    provenance: IntelligenceProvenance


class CompetitorAuditRequest(StrictContract):
    contract_version: Literal["competitorAuditInput.v1"] = Field(
        "competitorAuditInput.v1", alias="contractVersion"
    )
    subject_pages: list[PageSnapshot] = Field(
        ..., alias="subjectPages", min_length=1, max_length=100
    )
    competitor_pages: list[CompetitorPageSnapshot] = Field(
        ..., alias="competitorPages", min_length=1, max_length=100
    )

    @model_validator(mode="after")
    def unique_sources(self) -> CompetitorAuditRequest:
        source_ids = [
            page.source.source_id
            for page in [*self.subject_pages, *self.competitor_pages]
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("All competitor-audit sourceId values must be unique.")
        return self


class PrioritizedAction(StrictContract):
    action_id: str = Field(..., alias="actionId")
    priority: PriorityTier
    dimension: IntelligenceDimension | None = None
    action: str
    rationale: str
    evidence_ids: list[str] = Field(alias="evidenceIds")
    gap_id: str | None = Field(None, alias="gapId")
    origin: Literal["supportedGap", "coverageUnknown", "pageOpportunity"]


class CompetitorAuditArtifact(StrictContract):
    artifact_type: Literal["competitorAudit.v1"] = Field(
        "competitorAudit.v1", alias="artifactType"
    )
    analyzer_version: Literal["competitor-audit-heuristic.v1"] = Field(
        "competitor-audit-heuristic.v1", alias="analyzerVersion"
    )
    methodology: str = (
        "Compose deterministic competitor-page dimension findings with comparative "
        "content-gap analysis and priority-ranked remediation actions from supplied "
        "documents only."
    )
    page_analyses: list[CompetitorPageAnalysisArtifact] = Field(
        ..., alias="pageAnalyses"
    )
    content_gap: ContentGapArtifact = Field(..., alias="contentGap")
    prioritized_actions: list[PrioritizedAction] = Field(
        ..., alias="prioritizedActions"
    )
    warnings: list[str]
    provenance: IntelligenceProvenance


class AiAnswerObservation(StrictContract):
    observation_id: str = Field(..., alias="observationId", min_length=1, max_length=200)
    model_or_engine: str = Field(
        ..., alias="modelOrEngine", min_length=1, max_length=300
    )
    query: str = Field(..., min_length=1, max_length=2000)
    raw_response: str = Field(..., alias="rawResponse", min_length=1, max_length=200_000)
    observed_at_utc: str = Field(..., alias="observedAtUtc", min_length=1, max_length=64)
    subject_mentioned: bool | None = Field(None, alias="subjectMentioned")
    competitor_ids_mentioned: list[str] = Field(
        default_factory=list, alias="competitorIdsMentioned", max_length=100
    )


class CompetitorPositioningRequest(StrictContract):
    contract_version: Literal["competitorPositioningInput.v1"] = Field(
        "competitorPositioningInput.v1", alias="contractVersion"
    )
    brand_pages: list[PageSnapshot] = Field(
        ..., alias="brandPages", min_length=1, max_length=100
    )
    competitor_pages: list[CompetitorPageSnapshot] = Field(
        ..., alias="competitorPages", min_length=1, max_length=100
    )
    ai_answer_observations: list[AiAnswerObservation] = Field(
        default_factory=list, alias="aiAnswerObservations", max_length=200
    )

    @model_validator(mode="after")
    def unique_sources_and_observations(self) -> CompetitorPositioningRequest:
        source_ids = [
            page.source.source_id
            for page in [*self.brand_pages, *self.competitor_pages]
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(
                "All competitor-positioning sourceId values must be unique."
            )
        observation_ids = [
            item.observation_id for item in self.ai_answer_observations
        ]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observationId values must be unique.")
        known_competitors = {page.competitor_id for page in self.competitor_pages}
        for observation in self.ai_answer_observations:
            unknown = [
                competitor_id
                for competitor_id in observation.competitor_ids_mentioned
                if competitor_id not in known_competitors
            ]
            if unknown:
                raise ValueError(
                    "aiAnswerObservations reference unknown competitorId values: "
                    f"{', '.join(sorted(unknown))}."
                )
        return self


class PositioningAttribute(StrictContract):
    attribute_id: str = Field(..., alias="attributeId")
    attribute: str
    party: Literal["brand", "competitor"]
    competitor_id: str | None = Field(None, alias="competitorId")
    competitor_name: str | None = Field(None, alias="competitorName")
    signals: list[str]
    evidence_ids: list[str] = Field(alias="evidenceIds")
    source_ids: list[str] = Field(alias="sourceIds")


class PositioningGap(StrictContract):
    gap_id: str = Field(..., alias="gapId")
    attribute: str
    status: Literal["supportedGap", "coverageUnknown", "observationOnly"]
    summary: str
    evidence_ids: list[str] = Field(alias="evidenceIds")
    observation_ids: list[str] = Field(default_factory=list, alias="observationIds")
    competitor_ids: list[str] = Field(default_factory=list, alias="competitorIds")


class MessagingHypothesis(StrictContract):
    hypothesis_id: str = Field(..., alias="hypothesisId")
    origin: Literal["generatedHypothesis"] = "generatedHypothesis"
    message: str
    target_query: str = Field(..., alias="targetQuery")
    related_attribute: str = Field(..., alias="relatedAttribute")
    evidence_ids: list[str] = Field(alias="evidenceIds")
    disclaimer: str = (
        "generatedHypothesis only; not measured market perception, demand, or ranking."
    )


class CompetitorPositioningArtifact(StrictContract):
    artifact_type: Literal["competitorPositioning.v1"] = Field(
        "competitorPositioning.v1", alias="artifactType"
    )
    analyzer_version: Literal["competitor-positioning-heuristic.v1"] = Field(
        "competitor-positioning-heuristic.v1", alias="analyzerVersion"
    )
    methodology: str = (
        "Map explicit brand and competitor attributes from supplied page content and "
        "optional AI-answer observations. Preserve observation model/engine, query, "
        "raw response, and observation date. Generated messaging and queries are labeled "
        "generatedHypothesis and are never presented as measured market perception."
    )
    attribute_map: list[PositioningAttribute] = Field(..., alias="attributeMap")
    perception_gaps: list[PositioningGap] = Field(..., alias="perceptionGaps")
    messaging_hypotheses: list[MessagingHypothesis] = Field(
        ..., alias="messagingHypotheses"
    )
    observations: list[AiAnswerObservation]
    warnings: list[str]
    provenance: IntelligenceProvenance
