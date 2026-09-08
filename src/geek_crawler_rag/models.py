from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class IndexState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    # Legacy terminal value kept for reading old status docs only — never written.
    SKIPPED = "skipped"


# Admin POST may index any status; warn when outside this set.
TERMINAL_CRAWL_STATUSES = frozenset({"complete", "external"})


class IndexRunRequest(BaseModel):
    run_id: str = Field(..., alias="runId", min_length=1)

    model_config = {"populate_by_name": True}


class IndexStatusResponse(BaseModel):
    run_id: str = Field(..., alias="runId")
    state: IndexState
    crawl_type: str | None = Field(None, alias="crawlType")
    mongo_page_count: int | None = Field(None, alias="mongoPageCount")
    pages_seen: int = Field(0, alias="pagesSeen")
    pages_english: int = Field(0, alias="pagesEnglish")
    pages_skipped_lang: int = Field(0, alias="pagesSkippedLang")
    pages_skipped_empty: int = Field(0, alias="pagesSkippedEmpty")
    pages_deleted_locale: int = Field(0, alias="pagesDeletedLocale")
    pages_deleted_failure: int = Field(0, alias="pagesDeletedFailure")
    pages_deleted_empty: int = Field(0, alias="pagesDeletedEmpty")
    pages_deleted_non_english: int = Field(0, alias="pagesDeletedNonEnglish")
    chunks_upserted: int = Field(0, alias="chunksUpserted")
    attempt: int = 0
    trigger: str = "manual"
    embedding_rate_limit_retries: int = Field(0, alias="embeddingRateLimitRetries")
    embedding_wait_seconds: float = Field(0.0, alias="embeddingWaitSeconds")
    error: str | None = None
    started_at_utc: datetime | None = Field(None, alias="startedAtUtc")
    finished_at_utc: datetime | None = Field(None, alias="finishedAtUtc")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class IndexSchedulerStatus(BaseModel):
    enabled: bool
    interval_seconds: int = Field(..., alias="intervalSeconds")
    next_run_at_utc: datetime | None = Field(None, alias="nextRunAtUtc")
    last_enqueued_at_utc: datetime | None = Field(None, alias="lastEnqueuedAtUtc")
    last_run_id: str | None = Field(None, alias="lastRunId")
    last_error: str | None = Field(None, alias="lastError")
    last_selection_reason: str | None = Field(None, alias="lastSelectionReason")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class QueryRequest(BaseModel):
    need: str = Field(..., min_length=1)
    run_id: str = Field(..., alias="runId", min_length=1)
    crawl_type: str | None = Field(None, alias="crawlType")
    host: str | None = None
    top_k: int = Field(8, alias="topK", ge=1, le=50)
    prefer_parent: bool | None = Field(None, alias="preferParent")
    prefer_child: bool | None = Field(None, alias="preferChild")
    chunk_role: str | None = Field(None, alias="chunkRole")
    source_types: list[str] | None = Field(None, alias="sourceTypes")
    entity_names: list[str] | None = Field(None, alias="entityNames")
    categories: list[str] | None = Field(None, alias="categories")
    min_quality: float | None = Field(None, alias="minQuality", ge=0.0, le=1.0)
    # Phase D1 — "graph" for slide/strategy theme retrieval; default hybrid.
    retrieval_mode: str | None = Field(None, alias="retrievalMode")

    model_config = {"populate_by_name": True}


class ChunkHit(BaseModel):
    run_id: str = Field(..., alias="runId")
    crawl_type: str = Field(..., alias="crawlType")
    host: str
    url: str
    final_url: str = Field(..., alias="finalUrl")
    title: str | None = None
    chunk_index: int = Field(..., alias="chunkIndex")
    language: str
    text: str
    score: float
    page_id: str | None = Field(None, alias="pageId")
    entity_name: str | None = Field(None, alias="entityName")
    entity_id: str | None = Field(None, alias="entityId")
    source_type: str | None = Field(None, alias="sourceType")
    category: str | None = None
    content_intent: str | None = Field(None, alias="contentIntent")
    chunk_role: str | None = Field(None, alias="chunkRole")
    section_title: str | None = Field(None, alias="sectionTitle")
    quality_score: float | None = Field(None, alias="qualityScore")
    dense_score: float | None = Field(None, alias="denseScore")
    rerank_score: float | None = Field(None, alias="rerankScore")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class ThemeHit(BaseModel):
    label: str
    relationship: str | None = None
    entity: str | None = None
    related_entity: str | None = Field(None, alias="relatedEntity")
    url: str | None = None
    category: str | None = None
    crawl_type: str | None = Field(None, alias="crawlType")
    score: float | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class QueryResponse(BaseModel):
    run_id: str = Field(..., alias="runId")
    chunks: list[ChunkHit]
    warning: str | None = None
    retrieval: str | None = None
    themes: list[ThemeHit] | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class AdTemplateUpsertItem(BaseModel):
    id: str = ""
    name: str = "template"
    channel: str | None = None
    framework: str | None = None
    tone: str | None = None
    body: str = ""
    entity_tags: list[str] | None = Field(None, alias="entityTags")

    model_config = {"populate_by_name": True}


class AdTemplateIndexRequest(BaseModel):
    templates: list[AdTemplateUpsertItem]

    model_config = {"populate_by_name": True}


class AdTemplateIndexResponse(BaseModel):
    upserted: int
    warning: str | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class AdTemplateQueryRequest(BaseModel):
    need: str = Field(..., min_length=1)
    top_k: int = Field(5, alias="topK", ge=1, le=20)
    channel: str | None = None
    framework: str | None = None
    entity_tags: list[str] | None = Field(None, alias="entityTags")

    model_config = {"populate_by_name": True}


class AdTemplateHit(BaseModel):
    id: str
    name: str
    channel: str | None = None
    framework: str | None = None
    tone: str | None = None
    body: str
    score: float = 0.0
    entity_tags: list[str] = Field(default_factory=list, alias="entityTags")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class AdTemplateQueryResponse(BaseModel):
    templates: list[AdTemplateHit]
    warning: str | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class PageMarkdownResponse(BaseModel):
    page_id: str = Field(..., alias="pageId")
    run_id: str = Field(..., alias="runId")
    url: str
    final_url: str = Field(..., alias="finalUrl")
    title: str | None = None
    markdown: str
    excerpt: str | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class GenerateCitation(BaseModel):
    page_id: str | None = Field(None, alias="pageId")
    url: str
    title: str | None = None
    section_title: str | None = Field(None, alias="sectionTitle")
    quote: str
    crawl_type: str | None = Field(None, alias="crawlType")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class GenerateSource(BaseModel):
    url: str
    title: str | None = None
    entity: str | None = None
    crawl_type: str | None = Field(None, alias="crawlType")
    kind: str | None = None
    page_id: str | None = Field(None, alias="pageId")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class GenerateOutlineSection(BaseModel):
    key: str
    heading: str
    brief: str
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


APPROVED_GENERATION_MODELS = frozenset({"o1-pro", "o3"})
APPROVED_MODEL_POLICY_PRESETS = frozenset({"best-quality", "o3-only", "custom"})
APPROVED_MODEL_POLICY_STAGES = frozenset(
    {
        "researchPlanning",
        "outline",
        "section",
        "repair",
        "validation",
        "finalSynthesis",
        "complete",
    }
)
CURRENT_MODEL_POLICY_VERSION = "content-model-policy.v1"


class CanonicalBriefContext(BaseModel):
    """Versioned quality contract supplied by the unified content creator."""

    version: str = "gcc-v2-generation-brief.v1"
    title: str | None = None
    target_keyword: str | None = Field(None, alias="targetKeyword")
    content_type: str | None = Field(None, alias="contentType")
    primary_intent: str | None = Field(None, alias="primaryIntent")
    audience: str | list[str] | dict[str, Any] | None = None
    buying_stage: str | None = Field(None, alias="buyingStage")
    tone_of_voice: str | list[str] | None = Field(None, alias="toneOfVoice")
    brand: str | dict[str, Any] | None = None
    brand_kit: dict[str, Any] | None = Field(None, alias="brandKit")
    paa_questions: list[str] = Field(default_factory=list, alias="paaQuestions")
    required_topics: list[str] = Field(default_factory=list, alias="requiredTopics")
    operator_instructions: list[str] | str | None = Field(
        None, alias="operatorInstructions"
    )
    exclusions: list[str] = Field(default_factory=list)
    hierarchy: list[str] | dict[str, Any] | None = None
    internal_links: list[str] | list[dict[str, Any]] = Field(
        default_factory=list, alias="internalLinks"
    )
    output_requirements: list[str] | dict[str, Any] | str | None = Field(
        None, alias="outputRequirements"
    )
    channel_requirements: list[str] | dict[str, Any] | str | None = Field(
        None, alias="channelRequirements"
    )
    cta_requirements: list[str] | dict[str, Any] | str | None = Field(
        None, alias="ctaRequirements"
    )
    conversion_objective: str | None = Field(None, alias="conversionObjective")
    publishing_destination: str | None = Field(None, alias="publishingDestination")

    model_config = {"populate_by_name": True, "extra": "allow"}


class GenerateProvenance(BaseModel):
    generation_stage: str = Field(..., alias="generationStage")
    model_used: str = Field(..., alias="modelUsed")
    model_policy_preset: str | None = Field(None, alias="modelPolicyPreset")
    model_policy_version: str | None = Field(None, alias="modelPolicyVersion")
    prompt_version: str = Field(..., alias="promptVersion")
    retrieval: str
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class ValidationIssueCategory(StrEnum):
    UNSUPPORTED_CLAIM = "unsupportedClaim"
    SOURCE_CONFLICT = "sourceConflict"
    BRIEF_ALIGNMENT = "briefAlignment"
    BRAND_VOICE = "brandVoice"
    ORIGINALITY_REPETITION = "originalityRepetition"
    USEFULNESS = "usefulness"
    CTA = "cta"
    SEO_GEO = "seoGeo"
    CONTENT_TYPE_REQUIREMENTS = "contentTypeRequirements"


class GenerateValidationIssue(BaseModel):
    section_title: str | None = Field(None, alias="sectionTitle")
    category: ValidationIssueCategory
    detail: str = Field(..., min_length=1)
    repair_instruction: str = Field(..., alias="repairInstruction", min_length=1)

    model_config = {
        "populate_by_name": True,
        "ser_json_by_alias": True,
        "extra": "forbid",
    }


class GenerateValidation(BaseModel):
    approved: bool
    issues: list[GenerateValidationIssue]
    strengths: list[str]
    unsupported_claim_count: int = Field(..., alias="unsupportedClaimCount", ge=0)
    brief_alignment_score: float = Field(
        ..., alias="briefAlignmentScore", ge=0, le=100
    )
    evidence_coverage_score: float = Field(
        ..., alias="evidenceCoverageScore", ge=0, le=100
    )
    usefulness_score: float = Field(..., alias="usefulnessScore", ge=0, le=100)
    originality_score: float = Field(..., alias="originalityScore", ge=0, le=100)
    brand_alignment_score: float = Field(
        ..., alias="brandAlignmentScore", ge=0, le=100
    )

    model_config = {
        "populate_by_name": True,
        "ser_json_by_alias": True,
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def unsupported_claims_fail_approval(self) -> GenerateValidation:
        issue_count = sum(
            issue.category == ValidationIssueCategory.UNSUPPORTED_CLAIM
            for issue in self.issues
        )
        self.unsupported_claim_count = max(self.unsupported_claim_count, issue_count)
        if self.unsupported_claim_count:
            self.approved = False
        return self


class GenerateRequest(BaseModel):
    writing_intent: str = Field(..., alias="writingIntent", min_length=1)
    topic: str = Field(..., min_length=3)
    partner_run_id: str | None = Field(None, alias="partnerRunId")
    competitor_run_id: str | None = Field(None, alias="competitorRunId")
    target_entities: list[str] | None = Field(None, alias="targetEntities")
    ad_templates: list[AdTemplateUpsertItem] | None = Field(None, alias="adTemplates")
    graph_enabled: bool = Field(True, alias="graphEnabled")
    generation_stage: str = Field("complete", alias="generationStage")
    outline: list[GenerateOutlineSection] | None = None
    section_key: str | None = Field(None, alias="sectionKey")
    section_heading: str | None = Field(None, alias="sectionHeading")
    section_brief: str | None = Field(None, alias="sectionBrief")
    completed_section_summaries: list[str] | None = Field(
        None, alias="completedSectionSummaries"
    )
    draft_content: str | None = Field(None, alias="draftContent")
    input_sources: list[GenerateSource] | None = Field(None, alias="sources")
    canonical_brief: CanonicalBriefContext | None = Field(None, alias="canonicalBrief")
    model_policy_preset: str | None = Field(None, alias="modelPolicyPreset")
    model_policy_version: str | None = Field(None, alias="modelPolicyVersion")
    stage_model_overrides: dict[str, str] | None = Field(
        None, alias="stageModelOverrides"
    )

    # Keep unknown-field behavior compatible with existing standalone callers.
    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def validate_model_policy(self) -> GenerateRequest:
        raw_stage = self.generation_stage.strip()
        stage_key = "".join(character for character in raw_stage.casefold() if character.isalnum())
        stages = {
            "complete": "complete",
            "outline": "outline",
            "section": "section",
            "validation": "validation",
            "finalsynthesis": "finalSynthesis",
        }
        stage = stages.get(stage_key)
        if stage is None:
            raise ValueError(
                f"Unsupported generationStage '{self.generation_stage}'. "
                "Use complete, outline, section, validation, or finalSynthesis."
            )
        self.generation_stage = stage

        if stage in {"validation", "finalSynthesis"}:
            if not self.draft_content or not self.draft_content.strip():
                raise ValueError(
                    f"generationStage '{stage}' requires non-empty draftContent."
                )
            if self.canonical_brief is None:
                raise ValueError(
                    f"generationStage '{stage}' requires canonicalBrief."
                )
            if not (
                self.partner_run_id
                or self.competitor_run_id
                or self.input_sources
            ):
                raise ValueError(
                    f"generationStage '{stage}' requires sources, "
                    "partnerRunId, or competitorRunId."
                )
            if self.model_policy_preset is None:
                raise ValueError(
                    f"generationStage '{stage}' requires modelPolicyPreset."
                )
            if self.model_policy_version is None:
                raise ValueError(
                    f"generationStage '{stage}' requires modelPolicyVersion."
                )

        if (
            self.model_policy_preset is None
            and self.model_policy_version is None
            and self.stage_model_overrides is None
        ):
            return self

        preset = self.model_policy_preset or "best-quality"
        if preset not in APPROVED_MODEL_POLICY_PRESETS:
            approved = ", ".join(sorted(APPROVED_MODEL_POLICY_PRESETS))
            raise ValueError(
                f"Unapproved modelPolicyPreset '{preset}'. Approved presets: {approved}."
            )
        self.model_policy_preset = preset

        version = self.model_policy_version or CURRENT_MODEL_POLICY_VERSION
        if version != CURRENT_MODEL_POLICY_VERSION:
            raise ValueError(
                f"Unsupported modelPolicyVersion '{version}'. "
                f"Use '{CURRENT_MODEL_POLICY_VERSION}'."
            )
        self.model_policy_version = version

        overrides = self.stage_model_overrides or {}
        invalid_stages = sorted(set(overrides) - APPROVED_MODEL_POLICY_STAGES)
        if invalid_stages:
            raise ValueError(
                "Unapproved stageModelOverrides stage(s): "
                f"{', '.join(invalid_stages)}. Approved stages: "
                f"{', '.join(sorted(APPROVED_MODEL_POLICY_STAGES))}."
            )
        invalid_models = sorted(set(overrides.values()) - APPROVED_GENERATION_MODELS)
        if invalid_models:
            raise ValueError(
                "Unapproved stage model(s): "
                f"{', '.join(invalid_models)}. Approved models: "
                f"{', '.join(sorted(APPROVED_GENERATION_MODELS))}."
            )
        if preset != "custom" and overrides:
            raise ValueError(
                "stageModelOverrides requires modelPolicyPreset 'custom'; "
                "use an approved preset or explicitly select custom."
            )
        if preset == "custom" and not overrides:
            raise ValueError(
                "modelPolicyPreset 'custom' requires at least one stageModelOverrides entry."
            )
        if preset == "custom" and stage not in overrides:
            raise ValueError(
                f"Custom model policy has no override for generation stage '{stage}'. "
                f"Add stageModelOverrides.{stage} using o1-pro or o3."
            )
        return self


class GenerateResponse(BaseModel):
    intent: str
    content: str | None = None
    variations: list[str] | None = None
    battlecard: dict[str, Any] | None = None
    citations: list[GenerateCitation] = Field(default_factory=list)
    sources: list[GenerateSource] = Field(default_factory=list)
    themes: list[ThemeHit] | None = None
    outline: list[GenerateOutlineSection] | None = None
    warnings: list[str] = Field(default_factory=list)
    evidence_warnings: list[str] = Field(default_factory=list, alias="evidenceWarnings")
    model_used: str | None = Field(None, alias="modelUsed")
    retrieval: str | None = None
    provenance: GenerateProvenance | None = None
    validation: GenerateValidation | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
