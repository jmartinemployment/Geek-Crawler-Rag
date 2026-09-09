"""Strict versioned contracts for deterministic content diagnostics."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class StrictContract(BaseModel):
    model_config = {
        "populate_by_name": True,
        "ser_json_by_alias": True,
        "extra": "forbid",
    }


class QueryOrigin(StrEnum):
    OBSERVED = "observed"
    IMPORTED = "imported"
    GENERATED_HYPOTHESIS = "generatedHypothesis"


class QueryProvenance(StrictContract):
    query: str = Field(..., min_length=1, max_length=500)
    origin: QueryOrigin
    source_id: str | None = Field(None, alias="sourceId", min_length=1)
    source_reference: str | None = Field(None, alias="sourceReference", min_length=1)
    observed_at_utc: str | None = Field(None, alias="observedAtUtc", min_length=1)

    @model_validator(mode="after")
    def require_real_query_source(self) -> QueryProvenance:
        if self.origin in {QueryOrigin.OBSERVED, QueryOrigin.IMPORTED} and not (
            self.source_id or self.source_reference
        ):
            raise ValueError("Observed/imported queries require source provenance.")
        return self


class SourceProvenance(StrictContract):
    source_id: str = Field(..., alias="sourceId", min_length=1, max_length=200)
    url: str | None = Field(None, min_length=1)
    title: str | None = Field(None, min_length=1, max_length=500)
    retrieved_at_utc: str | None = Field(None, alias="retrievedAtUtc", min_length=1)
    source_digest: str | None = Field(
        None, alias="sourceDigest", pattern=r"^[0-9a-f]{64}$"
    )
    parser_id: str | None = Field(None, alias="parserId", min_length=1)
    parser_version: str | None = Field(None, alias="parserVersion", min_length=1)


class EvidenceReference(StrictContract):
    evidence_id: str = Field(..., alias="evidenceId", min_length=1, max_length=200)
    source_id: str = Field(..., alias="sourceId", min_length=1, max_length=200)
    quote: str = Field(..., min_length=12)
    start_char: int | None = Field(None, alias="startChar", ge=0)
    end_char: int | None = Field(None, alias="endChar", ge=1)

    @model_validator(mode="after")
    def validate_span(self) -> EvidenceReference:
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError(
                "Evidence coordinates must include both startChar and endChar."
            )
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char <= self.start_char
        ):
            raise ValueError("endChar must be greater than startChar.")
        return self


class TechnicalSignals(StrictContract):
    crawlable: bool | None = None
    status_code: int | None = Field(None, alias="statusCode", ge=100, le=599)
    load_time_ms: int | None = Field(None, alias="loadTimeMs", ge=0)
    canonical_url: str | None = Field(None, alias="canonicalUrl", min_length=1)


class DiagnosticDocument(StrictContract):
    source: SourceProvenance
    visible_content: str = Field(..., alias="visibleContent", min_length=1)
    media_type: Literal["text/markdown", "text/plain", "text/html"] = Field(
        "text/markdown", alias="mediaType"
    )
    content_completeness: Literal["full", "partial"] = Field(
        "full", alias="contentCompleteness"
    )
    queries: list[QueryProvenance] = Field(default_factory=list, max_length=100)
    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=1000)
    existing_json_ld: list[dict[str, Any]] | None = Field(
        None, alias="existingJsonLd", max_length=100
    )
    technical: TechnicalSignals | None = None


class ReadinessScoreRequest(StrictContract):
    contract_version: Literal["aiReadinessInput.v1"] = Field(
        "aiReadinessInput.v1", alias="contractVersion"
    )
    document: DiagnosticDocument


class FactDensityRequest(StrictContract):
    contract_version: Literal["factDensityInput.v1"] = Field(
        "factDensityInput.v1", alias="contractVersion"
    )
    document: DiagnosticDocument


class EntitySeed(StrictContract):
    canonical_name: str = Field(
        ..., alias="canonicalName", min_length=1, max_length=200
    )
    entity_type: Literal["person", "brand", "product", "concept", "topic", "other"] = (
        Field(alias="entityType")
    )
    aliases: list[str] = Field(default_factory=list, max_length=50)


class EntityMapRequest(StrictContract):
    contract_version: Literal["entityMapInput.v1"] = Field(
        "entityMapInput.v1", alias="contractVersion"
    )
    document: DiagnosticDocument
    seeds: list[EntitySeed] = Field(default_factory=list, max_length=200)


class SchemaMarkupRequest(StrictContract):
    contract_version: Literal["schemaMarkupInput.v1"] = Field(
        "schemaMarkupInput.v1", alias="contractVersion"
    )
    document: DiagnosticDocument
    requested_types: list[Literal["Article", "FAQPage", "HowTo", "Product"]] = Field(
        default_factory=list, alias="requestedTypes", max_length=4
    )


class ArtifactProvenance(StrictContract):
    engine_version: Literal["deterministic-diagnostics.v1"] = Field(
        "deterministic-diagnostics.v1", alias="engineVersion"
    )
    source: SourceProvenance
    queries: list[QueryProvenance]
    evidence_ids: list[str] = Field(alias="evidenceIds")
    evidence: list[EvidenceReference]


class DimensionScore(StrictContract):
    dimension: Literal[
        "headingHierarchy",
        "answerFirst",
        "faq",
        "schema",
        "factDensity",
        "eeat",
        "technicalCrawlabilityPerformance",
    ]
    score: float | None = Field(None, ge=0, le=100)
    explanation: str
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")


class ReadinessScoreArtifact(StrictContract):
    artifact_type: Literal["readinessScore.v1"] = Field(
        "readinessScore.v1", alias="artifactType"
    )
    methodology: Literal["heuristic"] = "heuristic"
    rubric_version: Literal["ai-readiness-heuristic.v1"] = Field(
        "ai-readiness-heuristic.v1", alias="rubricVersion"
    )
    overall_score: float | None = Field(None, alias="overallScore", ge=0, le=100)
    dimensions: list[DimensionScore] = Field(..., min_length=7, max_length=7)
    prioritized_fixes: list[str] = Field(alias="prioritizedFixes")
    warnings: list[str]
    provenance: ArtifactProvenance


class ClaimAssessment(StrictContract):
    claim_id: str = Field(..., alias="claimId")
    section_id: str = Field(..., alias="sectionId")
    text: str
    classification: Literal["specificFact", "generalStatement"]
    verification_status: Literal["supported", "unsupported", "notFactual"] = Field(
        alias="verificationStatus"
    )
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")


class SectionFactDensity(StrictContract):
    section_id: str = Field(..., alias="sectionId")
    heading: str
    sentence_count: int = Field(alias="sentenceCount", ge=0)
    specific_fact_count: int = Field(alias="specificFactCount", ge=0)
    supported_fact_count: int = Field(alias="supportedFactCount", ge=0)
    score: float | None = Field(None, ge=0, le=100)


class FactDensityArtifact(StrictContract):
    artifact_type: Literal["factDensityReport.v1"] = Field(
        "factDensityReport.v1", alias="artifactType"
    )
    methodology: Literal["heuristic"] = "heuristic"
    report_version: Literal["fact-density-heuristic.v1"] = Field(
        "fact-density-heuristic.v1", alias="reportVersion"
    )
    overall_score: float | None = Field(None, alias="overallScore", ge=0, le=100)
    sections: list[SectionFactDensity]
    claims: list[ClaimAssessment]
    unsupported_claim_ids: list[str] = Field(alias="unsupportedClaimIds")
    warnings: list[str]
    provenance: ArtifactProvenance


class CanonicalEntity(StrictContract):
    entity_id: str = Field(..., alias="entityId")
    canonical_name: str = Field(..., alias="canonicalName")
    entity_type: Literal["person", "brand", "product", "concept", "topic", "other"] = (
        Field(alias="entityType")
    )
    aliases: list[str]
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(alias="evidenceIds", min_length=1)


class EntityRelationship(StrictContract):
    relationship_id: str = Field(..., alias="relationshipId")
    source_entity_id: str = Field(..., alias="sourceEntityId")
    target_entity_id: str = Field(..., alias="targetEntityId")
    relation: Literal["coOccursWith"]
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(alias="evidenceIds", min_length=1)


class EntityMapArtifact(StrictContract):
    artifact_type: Literal["entityMap.v1"] = Field("entityMap.v1", alias="artifactType")
    mapper_version: Literal["canonical-entity-map.v1"] = Field(
        "canonical-entity-map.v1", alias="mapperVersion"
    )
    entities: list[CanonicalEntity]
    relationships: list[EntityRelationship]
    warnings: list[str]
    provenance: ArtifactProvenance


class SchemaValidationFinding(StrictContract):
    code: str
    schema_type: str = Field(alias="schemaType")
    valid: bool
    detail: str


class SchemaMarkupArtifact(StrictContract):
    artifact_type: Literal["schemaMarkup.v1"] = Field(
        "schemaMarkup.v1", alias="artifactType"
    )
    generator_version: Literal["visible-content-jsonld.v1"] = Field(
        "visible-content-jsonld.v1", alias="generatorVersion"
    )
    json_ld: list[dict[str, Any]] = Field(alias="jsonLd")
    validation: list[SchemaValidationFinding]
    warnings: list[str]
    provenance: ArtifactProvenance
