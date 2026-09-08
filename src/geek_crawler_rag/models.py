from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import hashlib
from typing import Any
from uuid import UUID, uuid4

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
CURRENT_EXECUTION_VERSION = "rag-generate.v2"
CURRENT_SKILL_ENVELOPE_VERSION = "gcc-skill-envelope.v1"
CURRENT_SKILL_CATALOG_VERSION = "gcc-safe-skills.2026-09-08"
MAX_SKILL_ENVELOPE_BYTES = 16 * 1024
PINNED_SKILL_HASHES = {
    "seo-fundamentals": "f42ff5a164422e1526cdc3460fdff3cc2d5ab5508f6e6f6055ea07f1babc51b6",
    "geo-direct-answer": "44734869d4539621ce326b8d4b1087ab6f8286133742ec553de283ccf31763e7",
    "citation-discipline": "61da5cdcecc35a2c2838250370287c7b4092b138fbbaa0563ca0cd222cdbbbbc",
    "comparison-evidence": "d46466a0d413e40edf2465c80152c409f7be906b216b11b29a119c30435ef06b",
    "case-study-proof": "947fdb6fe9a2a364c05b4c02d34644783baaeb6b15b6afa8b63a166da1e5bb1b",
    "technical-depth": "5fb04e195d9be87a1fb1a2243f43abbb3aed5a00cc97ed5e136b0b218ee0e71c",
    "brand-voice": "76e245ac0401621836a5a9f77f27d21fa1a7a785bab82afc11ef856b82f5c326",
    "cta-alignment": "b7b4c49b20bfe4f8eeef70f1b38272825608c41235ec63f41a7b377d3e0a5440",
    "anti-repetition": "0ce61afbc02bf3e284d77a72d0e768e28849e7f0e35542ed5e0ec9c07ee95a8f",
    "linkedin-document-structure": "aac60102b0c1777734b7734a54b762b48e963b971542f32eb0ed03b974e64b11",
}
SUPPORTED_GENERATION_STAGES = frozenset(
    {"outline", "section", "repair", "validation", "finalSynthesis", "complete"}
)


class SkillDefinition(BaseModel):
    """Reviewed declarative instructions. Tool declarations are intentionally forbidden."""

    id: str = Field(..., min_length=1, max_length=80)
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    source: str = Field(..., min_length=1, max_length=200)
    license: str = Field(..., min_length=1, max_length=120)
    reviewer: str = Field(..., min_length=1, max_length=120)
    supported_stages: list[str] = Field(..., alias="supportedStages", min_length=1)
    supported_content_types: list[str] = Field(
        ..., alias="supportedContentTypes", min_length=1
    )
    order: int = Field(..., ge=0, le=10_000)
    conflicts: list[str] = Field(default_factory=list)
    prompt_instructions: str = Field(
        ..., alias="promptInstructions", min_length=1, max_length=2000
    )
    retrieval_hints: str = Field(
        ..., alias="retrievalHints", min_length=1, max_length=2000
    )
    output_requirements: str = Field(
        ..., alias="outputRequirements", min_length=1, max_length=2000
    )
    validation_checks: str = Field(
        ..., alias="validationChecks", min_length=1, max_length=2000
    )

    model_config = {"populate_by_name": True, "extra": "forbid"}

    @property
    def canonical_content(self) -> str:
        return "\n".join(
            (
                self.prompt_instructions,
                self.retrieval_hints,
                self.output_requirements,
                self.validation_checks,
            )
        )


class SkillExecutionEnvelope(BaseModel):
    envelope_version: str = Field(..., alias="envelopeVersion")
    catalog_version: str = Field(..., alias="catalogVersion")
    snapshot_hash: str = Field(..., alias="snapshotHash", pattern=r"^[0-9a-f]{64}$")
    content_type: str = Field(..., alias="contentType", min_length=1)
    skills: list[SkillDefinition] = Field(..., max_length=10)
    resolved_at_utc: datetime = Field(..., alias="resolvedAtUtc")

    model_config = {"populate_by_name": True, "extra": "forbid"}

    @model_validator(mode="after")
    def validate_immutable_envelope(self) -> SkillExecutionEnvelope:
        if self.envelope_version != CURRENT_SKILL_ENVELOPE_VERSION:
            raise ValueError("Unsupported skill envelope version.")
        if self.catalog_version != CURRENT_SKILL_CATALOG_VERSION:
            raise ValueError("Unsupported skill catalog version.")
        ids = [skill.id for skill in self.skills]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate skills are not allowed.")
        expected_order = sorted(self.skills, key=lambda skill: (skill.order, skill.id))
        if self.skills != expected_order:
            raise ValueError("Skills are not in deterministic catalog order.")
        selected = set(ids)
        for skill in self.skills:
            if PINNED_SKILL_HASHES.get(skill.id) != skill.sha256:
                raise ValueError(
                    f"Skill '{skill.id}' is not pinned in the reviewed catalog."
                )
            actual = hashlib.sha256(skill.canonical_content.encode()).hexdigest()
            if actual != skill.sha256:
                raise ValueError(f"Skill '{skill.id}' hash mismatch.")
            conflict = next((item for item in skill.conflicts if item in selected), None)
            if conflict:
                raise ValueError(f"Skill '{skill.id}' conflicts with '{conflict}'.")
        canonical = "\n".join(
            [self.catalog_version, self.content_type]
            + [
                f"{skill.order}|{skill.id}|{skill.version}|{skill.sha256}"
                for skill in self.skills
            ]
        )
        if hashlib.sha256(canonical.encode()).hexdigest() != self.snapshot_hash:
            raise ValueError("Skill execution snapshot hash mismatch.")
        if len(self.model_dump_json(by_alias=True).encode()) > MAX_SKILL_ENVELOPE_BYTES:
            raise ValueError("Skill execution envelope exceeds the size limit.")
        return self


class SkillProvenance(BaseModel):
    envelope_version: str = Field(..., alias="envelopeVersion")
    catalog_version: str = Field(..., alias="catalogVersion")
    snapshot_hash: str = Field(..., alias="snapshotHash")
    stage: str
    skill_versions: list[str] = Field(default_factory=list, alias="skillVersions")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class ProducerCapabilities(BaseModel):
    execution_versions: list[str] = Field(
        default_factory=lambda: [CURRENT_EXECUTION_VERSION],
        alias="executionVersions",
    )
    skill_envelope_versions: list[str] = Field(
        default_factory=lambda: [CURRENT_SKILL_ENVELOPE_VERSION],
        alias="skillEnvelopeVersions",
    )
    generation_stages: list[str] = Field(
        default_factory=lambda: sorted(SUPPORTED_GENERATION_STAGES),
        alias="generationStages",
    )
    specialist_executors: list[str] = Field(
        default_factory=lambda: [
            "researchPlanning",
            "outline",
            "section",
            "finalSynthesis",
            "validation",
            "repair",
        ],
        alias="specialistExecutors",
    )
    specialist_executor_version: str = Field(
        "bounded-specialists.v1", alias="specialistExecutorVersion"
    )
    tools_allowed: bool = Field(False, alias="toolsAllowed")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


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
    specialist_executor: str = Field(
        "legacy-complete", alias="specialistExecutor"
    )
    specialist_executor_version: str = Field(
        "bounded-specialists.v1", alias="specialistExecutorVersion"
    )
    execution_version: str = Field("rag-generate.v1", alias="executionVersion")
    attempt_id: str = Field(default_factory=lambda: str(uuid4()), alias="attemptId")
    skills: SkillProvenance = Field(
        default_factory=lambda: SkillProvenance(
            envelopeVersion="none",
            catalogVersion="none",
            snapshotHash=hashlib.sha256(b"").hexdigest(),
            stage="complete",
            skillVersions=[],
        )
    )

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
    execution_version: str = Field("rag-generate.v1", alias="executionVersion")
    attempt_id: str = Field(default_factory=lambda: str(uuid4()), alias="attemptId")
    skill_execution: SkillExecutionEnvelope | None = Field(
        None, alias="skillExecution"
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
            "repair": "repair",
            "validation": "validation",
            "finalsynthesis": "finalSynthesis",
        }
        stage = stages.get(stage_key)
        if stage is None:
            raise ValueError(
                f"Unsupported generationStage '{self.generation_stage}'. "
                "Use complete, outline, section, repair, validation, or finalSynthesis."
            )
        self.generation_stage = stage

        try:
            UUID(self.attempt_id)
        except ValueError as ex:
            raise ValueError("attemptId must be a UUID.") from ex
        if self.skill_execution is not None:
            if self.execution_version != CURRENT_EXECUTION_VERSION:
                raise ValueError(
                    f"skillExecution requires executionVersion '{CURRENT_EXECUTION_VERSION}'."
                )
            if stage not in SUPPORTED_GENERATION_STAGES:
                raise ValueError(f"Skills do not support generation stage '{stage}'.")
            for skill in self.skill_execution.skills:
                if stage not in skill.supported_stages:
                    continue
                if self.skill_execution.content_type not in skill.supported_content_types:
                    raise ValueError(
                        f"Skill '{skill.id}' does not support content type "
                        f"'{self.skill_execution.content_type}'."
                    )
        elif self.execution_version == CURRENT_EXECUTION_VERSION:
            raise ValueError("Current executionVersion requires skillExecution.")

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
