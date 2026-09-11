"""Strict versioned contracts for deterministic generated-content task agents."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from geek_crawler_rag.diagnostic_models import (
    DiagnosticDocument,
    EvidenceReference,
    QueryOrigin,
    QueryProvenance,
    SourceProvenance,
    StrictContract,
)
from geek_crawler_rag.intelligence_models import (
    CompetitorPageSnapshot,
    IntelligenceProvenance,
    PageSnapshot,
)


class FaqGeneratorRequest(StrictContract):
    contract_version: Literal["faqGeneratorInput.v1"] = Field(
        "faqGeneratorInput.v1", alias="contractVersion"
    )
    topic: str = Field(..., min_length=1, max_length=500)
    queries: list[QueryProvenance] = Field(default_factory=list, max_length=100)
    hypothesis_topics: list[str] = Field(
        default_factory=list, alias="hypothesisTopics", max_length=50
    )
    source_document: DiagnosticDocument | None = Field(
        None, alias="sourceDocument"
    )
    max_pairs: int = Field(8, alias="maxPairs", ge=1, le=20)

    @model_validator(mode="after")
    def require_inputs(self) -> FaqGeneratorRequest:
        if self.source_document is None and not self.queries and not any(
            topic.strip() for topic in self.hypothesis_topics
        ):
            raise ValueError(
                "At least one source document, sourced query, or hypothesis topic is required."
            )
        for query in self.queries:
            if query.origin == QueryOrigin.GENERATED_HYPOTHESIS:
                raise ValueError(
                    "Caller queries cannot use generatedHypothesis origin; "
                    "hypotheses are created only by the generator."
                )
        return self


class FaqCitation(StrictContract):
    evidence_id: str = Field(..., alias="evidenceId")
    source_id: str = Field(..., alias="sourceId")
    quote: str = Field(..., min_length=12)
    url: str | None = None


class FaqPair(StrictContract):
    pair_id: str = Field(..., alias="pairId")
    question: str
    answer: str
    query_provenance: QueryProvenance = Field(..., alias="queryProvenance")
    citations: list[FaqCitation]
    verification_status: Literal["supported", "unsupported", "unverifiable"] = Field(
        ..., alias="verificationStatus"
    )


class ContentProvenance(StrictContract):
    engine_version: Literal["deterministic-content.v1"] = Field(
        "deterministic-content.v1", alias="engineVersion"
    )
    source: SourceProvenance | None = None
    queries: list[QueryProvenance]
    evidence_ids: list[str] = Field(alias="evidenceIds")
    evidence: list[EvidenceReference]

    @model_validator(mode="after")
    def validate_evidence(self) -> ContentProvenance:
        actual = [item.evidence_id for item in self.evidence]
        if self.evidence_ids != actual:
            raise ValueError("evidenceIds must match evidence records in order.")
        if len(actual) != len(set(actual)):
            raise ValueError("evidenceId values must be unique.")
        if self.source and any(item.source_id != self.source.source_id for item in self.evidence):
            raise ValueError("Evidence must reference the supplied source document.")
        return self


class FaqSetArtifact(StrictContract):
    artifact_type: Literal["faqSet.v1"] = Field("faqSet.v1", alias="artifactType")
    generator_version: Literal["faq-generator-heuristic.v1"] = Field(
        "faq-generator-heuristic.v1", alias="generatorVersion"
    )
    methodology: str = (
        "Build answer-first FAQ pairs from supplied queries and visible source content "
        "only. Answers must include exact evidence spans when a source document is "
        "provided; otherwise pairs are labeled unverifiable."
    )
    topic: str
    pairs: list[FaqPair]
    warnings: list[str]
    provenance: ContentProvenance


class CitableClaimsRequest(StrictContract):
    contract_version: Literal["citableClaimsInput.v1"] = Field(
        "citableClaimsInput.v1", alias="contractVersion"
    )
    source_document: DiagnosticDocument = Field(..., alias="sourceDocument")
    target_statements: list[str] = Field(
        default_factory=list, alias="targetStatements", max_length=50
    )
    max_claims: int = Field(20, alias="maxClaims", ge=1, le=50)
    insertion_target: str | None = Field(
        None, alias="insertionTarget", max_length=500
    )


class CitableClaim(StrictContract):
    claim_id: str = Field(..., alias="claimId")
    claim_text: str = Field(..., alias="claimText", min_length=1)
    claim_type: Literal[
        "quantifiableFact",
        "expertPosition",
        "attributableStatement",
        "rewrittenSpecific",
    ] = Field(..., alias="claimType")
    attribution: str | None = None
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")
    verification_status: Literal["supported", "unsupported", "unverifiable"] = Field(
        ..., alias="verificationStatus"
    )
    confidence: Literal["high", "medium", "low", "unknown"]
    contradiction_state: Literal["none", "possible", "unknown"] = Field(
        ..., alias="contradictionState"
    )
    insertion_location: str | None = Field(None, alias="insertionLocation")
    source_statement: str | None = Field(None, alias="sourceStatement")


class ClaimContradictionPair(StrictContract):
    left_claim_id: str = Field(..., alias="leftClaimId")
    right_claim_id: str = Field(..., alias="rightClaimId")
    reason: Literal["conflictingQuantities", "negationConflict"]


class ClaimLedgerArtifact(StrictContract):
    artifact_type: Literal["claimLedger.v1"] = Field(
        "claimLedger.v1", alias="artifactType"
    )
    generator_version: Literal["citable-claims-heuristic.v1"] = Field(
        "citable-claims-heuristic.v1", alias="generatorVersion"
    )
    methodology: str = (
        "Extract already-specific attributable statements from supplied visible "
        "content and rewrite vague target statements only when exact supporting "
        "quotes exist. Never invent statistics or unsupported specifics."
    )
    claims: list[CitableClaim]
    contradiction_pairs: list[ClaimContradictionPair] = Field(
        default_factory=list, alias="contradictionPairs"
    )
    warnings: list[str]
    provenance: ContentProvenance


class ComparisonBriefRequest(StrictContract):
    contract_version: Literal["comparisonBriefInput.v1"] = Field(
        "comparisonBriefInput.v1", alias="contractVersion"
    )
    subject_name: str = Field(..., alias="subjectName", min_length=1, max_length=300)
    competitor_name: str = Field(
        ..., alias="competitorName", min_length=1, max_length=300
    )
    subject_pages: list[PageSnapshot] = Field(
        ..., alias="subjectPages", min_length=1, max_length=100
    )
    competitor_pages: list[CompetitorPageSnapshot] = Field(
        ..., alias="competitorPages", min_length=1, max_length=100
    )
    decision_criteria: list[str] = Field(
        default_factory=list, alias="decisionCriteria", max_length=50
    )

    @model_validator(mode="after")
    def unique_sources(self) -> ComparisonBriefRequest:
        source_ids = [
            page.source.source_id
            for page in [*self.subject_pages, *self.competitor_pages]
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("All comparison-brief sourceId values must be unique.")
        return self


class ComparisonCriterionRow(StrictContract):
    criterion: str
    subject_signals: list[str] = Field(alias="subjectSignals")
    competitor_signals: list[str] = Field(alias="competitorSignals")
    evidence_ids: list[str] = Field(alias="evidenceIds")
    summary: str


class GeneratedHypothesisText(StrictContract):
    origin: Literal["generatedHypothesis"] = "generatedHypothesis"
    text: str = Field(..., min_length=1)
    disclaimer: str = (
        "generatedHypothesis only; not measured demand, ranking, or market perception."
    )


class RecommendedVerdict(StrictContract):
    framing: str
    disclaimer: str = (
        "Heuristic recommendation from supplied page signals only; not a measured "
        "ranking, win rate, or market outcome."
    )


class ComparisonBriefArtifact(StrictContract):
    artifact_type: Literal["comparisonBrief.v1"] = Field(
        "comparisonBrief.v1", alias="artifactType"
    )
    generator_version: Literal["comparison-brief-heuristic.v1"] = Field(
        "comparison-brief-heuristic.v1", alias="generatorVersion"
    )
    methodology: str = (
        "Compare deterministic visible-content signals across subject and competitor "
        "pages into a structured brief. Absences in partial inputs stay unknown; "
        "differentiators and angles are labeled generatedHypothesis."
    )
    subject_name: str = Field(..., alias="subjectName")
    competitor_name: str = Field(..., alias="competitorName")
    criteria: list[ComparisonCriterionRow]
    differentiators: list[GeneratedHypothesisText]
    proof_requirements: list[GeneratedHypothesisText] = Field(alias="proofRequirements")
    positioning_angles: list[GeneratedHypothesisText] = Field(alias="positioningAngles")
    recommended_verdict: RecommendedVerdict = Field(..., alias="recommendedVerdict")
    warnings: list[str]
    provenance: IntelligenceProvenance


class CompetitiveResponseRequest(StrictContract):
    contract_version: Literal["competitiveResponseInput.v1"] = Field(
        "competitiveResponseInput.v1", alias="contractVersion"
    )
    brand_pages: list[PageSnapshot] = Field(
        ..., alias="brandPages", min_length=1, max_length=100
    )
    competitor_pages: list[CompetitorPageSnapshot] = Field(
        ..., alias="competitorPages", min_length=1, max_length=100
    )
    response_mode: Literal["newPage", "targetedUpdate", "counterNarrative", "auto"] = (
        Field("auto", alias="responseMode")
    )
    focus_query: str | None = Field(None, alias="focusQuery", max_length=500)

    @model_validator(mode="after")
    def unique_sources(self) -> CompetitiveResponseRequest:
        source_ids = [
            page.source.source_id
            for page in [*self.brand_pages, *self.competitor_pages]
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(
                "All competitive-response sourceId values must be unique."
            )
        return self


class RequiredProofPoint(StrictContract):
    proof_id: str = Field(..., alias="proofId")
    statement: str
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")
    origin: Literal["suppliedEvidence", "generatedHypothesis"]


class ResponseOutlineSection(StrictContract):
    heading: str
    objective: str
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")


class CompetitiveResponseArtifact(StrictContract):
    artifact_type: Literal["competitiveResponse.v1"] = Field(
        "competitiveResponse.v1", alias="artifactType"
    )
    generator_version: Literal["competitive-response-heuristic.v1"] = Field(
        "competitive-response-heuristic.v1", alias="generatorVersion"
    )
    methodology: str = (
        "Select a response mode from supplied brand and competitor page signals, then "
        "emit brand-aligned angles, proof requirements, and outline sections. Does not "
        "copy competitor prose into draft body."
    )
    selected_mode: Literal["newPage", "targetedUpdate", "counterNarrative"] = Field(
        ..., alias="selectedMode"
    )
    rationale: str
    content_angles: list[GeneratedHypothesisText] = Field(alias="contentAngles")
    required_proof_points: list[RequiredProofPoint] = Field(
        alias="requiredProofPoints"
    )
    outline_sections: list[ResponseOutlineSection] = Field(alias="outlineSections")
    warnings: list[str]
    provenance: IntelligenceProvenance


class PillarOutlineRequest(StrictContract):
    contract_version: Literal["pillarOutlineInput.v1"] = Field(
        "pillarOutlineInput.v1", alias="contractVersion"
    )
    topic: str = Field(..., min_length=1, max_length=500)
    source_document: DiagnosticDocument | None = Field(
        None, alias="sourceDocument"
    )
    queries: list[QueryProvenance] = Field(default_factory=list, max_length=100)
    supporting_content_hints: list[str] = Field(
        default_factory=list, alias="supportingContentHints", max_length=50
    )

    @model_validator(mode="after")
    def reject_hypothesis_queries(self) -> PillarOutlineRequest:
        for query in self.queries:
            if query.origin == QueryOrigin.GENERATED_HYPOTHESIS:
                raise ValueError(
                    "Caller queries cannot use generatedHypothesis origin; "
                    "hypotheses are created only by the generator."
                )
        return self


class PillarOutlineSection(StrictContract):
    section_id: str = Field(..., alias="sectionId")
    heading: str
    objective: str
    answer_first_prompt: str = Field(..., alias="answerFirstPrompt")
    related_queries: list[str] = Field(default_factory=list, alias="relatedQueries")
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")


class SupportingContentPlanItem(StrictContract):
    content_type: Literal["faq", "comparison", "explainer"] = Field(
        ..., alias="contentType"
    )
    title: str
    rationale: str
    origin: Literal["generatedHypothesis"] = "generatedHypothesis"
    disclaimer: str = (
        "generatedHypothesis only; not measured demand or cluster coverage."
    )


class PillarOutlineArtifact(StrictContract):
    artifact_type: Literal["pillarOutline.v1"] = Field(
        "pillarOutline.v1", alias="artifactType"
    )
    generator_version: Literal["pillar-outline-heuristic.v1"] = Field(
        "pillar-outline-heuristic.v1", alias="generatorVersion"
    )
    methodology: str = (
        "Build a topic-cluster outline from the topic, optional sourced queries, and "
        "optional visible source headings/spans. Supporting content plans are labeled "
        "generatedHypothesis; this is not a full article draft."
    )
    topic: str
    sections: list[PillarOutlineSection]
    supporting_content_plan: list[SupportingContentPlanItem] = Field(
        alias="supportingContentPlan"
    )
    warnings: list[str]
    provenance: ContentProvenance


class PillarArticleRequest(StrictContract):
    contract_version: Literal["pillarArticleInput.v1"] = Field(
        "pillarArticleInput.v1", alias="contractVersion"
    )
    topic: str = Field(..., min_length=1, max_length=500)
    source_document: DiagnosticDocument | None = Field(
        None, alias="sourceDocument"
    )
    queries: list[QueryProvenance] = Field(default_factory=list, max_length=100)
    supporting_content_hints: list[str] = Field(
        default_factory=list, alias="supportingContentHints", max_length=50
    )

    @model_validator(mode="after")
    def reject_hypothesis_queries(self) -> PillarArticleRequest:
        for query in self.queries:
            if query.origin == QueryOrigin.GENERATED_HYPOTHESIS:
                raise ValueError(
                    "Caller queries cannot use generatedHypothesis origin; "
                    "hypotheses are created only by the generator."
                )
        return self


class PillarArticleSection(StrictContract):
    section_id: str = Field(..., alias="sectionId")
    heading: str
    body_markdown: str = Field(..., alias="bodyMarkdown")
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")
    grounded: bool


class PillarArticleArtifact(StrictContract):
    artifact_type: Literal["pillarArticle.v1"] = Field(
        "pillarArticle.v1", alias="artifactType"
    )
    generator_version: Literal["pillar-article-heuristic.v1"] = Field(
        "pillar-article-heuristic.v1", alias="generatorVersion"
    )
    methodology: str = (
        "Compose a full pillar Markdown draft from the deterministic topic-cluster outline. "
        "Section bodies use supplied source spans when available; otherwise they stay "
        "explicitly scaffolded without inventing statistics. Supporting content plans remain "
        "generatedHypothesis."
    )
    topic: str
    title: str
    markdown: str
    sections: list[PillarArticleSection]
    supporting_content_plan: list[SupportingContentPlanItem] = Field(
        alias="supportingContentPlan"
    )
    warnings: list[str]
    provenance: ContentProvenance
