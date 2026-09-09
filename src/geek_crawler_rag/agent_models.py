"""Strict v3 agent-runtime contracts.

These models deliberately contain no transport credentials or caller supplied
authority.  Runtime authority is assembled by the service from the generation
request after its immutable snapshot has been verified.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


AGENT_TRACE_VERSION = "agent-trace.v1"
AGENT_TOOLS_VERSION = "agent-tools.v1"
AGENT_EXECUTOR_VERSION = "function-agents.v1"
AGENT_WORKFLOW_VERSION = "citeable-agent-workflow.v1"


class StrictModel(BaseModel):
    model_config = {
        "populate_by_name": True,
        "ser_json_by_alias": True,
        "extra": "forbid",
    }


class AgentStopReason(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budgetExhausted"
    CANCELLED = "cancelled"
    TIMED_OUT = "timedOut"
    TOOL_DENIED = "toolDenied"
    SKILL_ACTIVATION_FAILED = "skillActivationFailed"
    INVALID_STRUCTURED_OUTPUT = "invalidStructuredOutput"
    EVIDENCE_FAILURE = "evidenceFailure"
    INCOMPATIBLE_PROTOCOL = "incompatibleProtocol"
    UPSTREAM_FAILURE = "upstreamFailure"


class SpecialistRole(StrEnum):
    CONTRIBUTOR = "contributor"
    PRODUCER = "producer"
    REVIEWER = "reviewer"


class SkillAssignment(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class SpecialistCitation(StrictModel):
    page_id: str | None = Field(..., alias="pageId", max_length=120)
    url: str = Field(..., min_length=1, max_length=2_000)
    title: str | None = Field(..., max_length=500)
    section_title: str | None = Field(..., alias="sectionTitle", max_length=500)
    quote: str = Field(..., min_length=12, max_length=4_000)
    crawl_type: Literal["partner", "competitors"] | None = Field(
        ..., alias="crawlType"
    )


class ContributionQuery(StrictModel):
    need: str = Field(..., min_length=3, max_length=1_000)
    corpus: Literal["partner", "competitors"]


class ContributionOutlineSection(StrictModel):
    key: str = Field(..., min_length=1, max_length=120)
    heading: str = Field(..., min_length=1, max_length=300)
    brief: str = Field(..., min_length=1, max_length=2_000)
    evidence_ids: list[str] = Field(
        default_factory=list, alias="evidenceIds", max_length=20
    )


class SpecialistContribution(StrictModel):
    contract_version: Literal["contributorOutput.v1"] = Field(
        ..., alias="contractVersion"
    )
    stage: Literal[
        "researchPlanning",
        "outline",
        "section",
        "repair",
        "validation",
        "finalSynthesis",
    ]
    summary: str = Field(..., min_length=1, max_length=2_000)
    proposed_content: str | None = Field(
        None, alias="proposedContent", min_length=1, max_length=64_000
    )
    proposed_outline: list[ContributionOutlineSection] | None = Field(
        None, alias="proposedOutline", max_length=20
    )
    proposed_queries: list[ContributionQuery] | None = Field(
        None, alias="proposedQueries", max_length=20
    )
    recommendations: list[str] | None = Field(None, max_length=30)
    citations: list[SpecialistCitation] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def stage_payload_is_typed(self) -> "SpecialistContribution":
        expected = {
            "researchPlanning": self.proposed_queries is not None,
            "outline": self.proposed_outline is not None,
            "section": self.proposed_content is not None,
            "repair": self.proposed_content is not None,
            "finalSynthesis": self.proposed_content is not None,
            "validation": self.recommendations is not None,
        }[self.stage]
        if not expected:
            raise ValueError("Contributor payload does not match its declared stage.")
        return self


class SpecialistReviewIssue(StrictModel):
    category: Literal[
        "unsupportedClaim",
        "sourceConflict",
        "briefAlignment",
        "brandVoice",
        "originalityRepetition",
        "usefulness",
        "cta",
        "seoGeo",
        "contentTypeRequirements",
        "contractViolation",
    ]
    severity: Literal["info", "warning", "error"]
    section_title: str | None = Field(..., alias="sectionTitle", max_length=500)
    detail: str = Field(..., min_length=1, max_length=2_000)
    recommendation: str = Field(..., min_length=1, max_length=2_000)
    citation: SpecialistCitation | None = None


class SpecialistReview(StrictModel):
    contract_version: Literal["reviewerOutput.v1"] = Field(
        ..., alias="contractVersion"
    )
    stage: Literal[
        "researchPlanning",
        "outline",
        "section",
        "repair",
        "validation",
        "finalSynthesis",
    ]
    decision: Literal["approved", "changesRequired", "rejected"]
    summary: str = Field(..., min_length=1, max_length=2_000)
    issues: list[SpecialistReviewIssue] = Field(default_factory=list, max_length=100)
    citations: list[SpecialistCitation] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def decision_matches_issues(self) -> "SpecialistReview":
        if self.decision == "approved" and any(
            issue.severity == "error" for issue in self.issues
        ):
            raise ValueError("An approved review cannot contain error-severity issues.")
        if self.decision != "approved" and not self.issues:
            raise ValueError("A non-approved review requires at least one issue.")
        return self


class AgentBudgetLimits(StrictModel):
    max_turns: int = Field(8, alias="maxTurns", ge=1, le=24)
    max_tool_calls: int = Field(24, alias="maxToolCalls", ge=1, le=64)
    max_retrieved_pages: int = Field(8, alias="maxRetrievedPages", ge=1, le=20)
    max_active_skills: int = Field(6, alias="maxActiveSkills", ge=0, le=10)
    max_skill_bytes: int = Field(32_768, alias="maxSkillBytes", ge=0, le=131_072)
    max_resource_bytes: int = Field(32_768, alias="maxResourceBytes", ge=0, le=131_072)
    max_output_tokens: int = Field(16_000, alias="maxOutputTokens", ge=128, le=64_000)
    max_stage_seconds: float = Field(300.0, alias="maxStageSeconds", gt=0, le=900)
    max_repair_attempts: int = Field(2, alias="maxRepairAttempts", ge=0, le=5)


class AgentBudgetUsage(StrictModel):
    turns: int = 0
    tool_calls: int = Field(0, alias="toolCalls")
    retrieved_pages: int = Field(0, alias="retrievedPages")
    active_skills: int = Field(0, alias="activeSkills")
    skill_bytes: int = Field(0, alias="skillBytes")
    resource_bytes: int = Field(0, alias="resourceBytes")
    output_tokens: int = Field(0, alias="outputTokens")
    input_tokens: int = Field(0, alias="inputTokens")
    total_tokens: int = Field(0, alias="totalTokens")


class AgentToolTrace(StrictModel):
    sequence: int = Field(..., ge=1)
    tool_id: str = Field(..., alias="toolId", pattern=r"^[a-z][a-z0-9_.-]{1,79}$")
    tool_version: str = Field(AGENT_TOOLS_VERSION, alias="toolVersion")
    classification: Literal["readOnly", "outputOnly"]
    argument_digest: str = Field(..., alias="argumentDigest", pattern=r"^[0-9a-f]{64}$")
    result_digest: str | None = Field(None, alias="resultDigest", pattern=r"^[0-9a-f]{64}$")
    result_count: int | None = Field(None, alias="resultCount", ge=0)
    duration_ms: int = Field(..., alias="durationMs", ge=0)
    error_class: str | None = Field(None, alias="errorClass", max_length=120)
    budget_after: AgentBudgetUsage = Field(..., alias="budgetAfter")


class SkillActivationProvenance(StrictModel):
    skill_id: str = Field(..., alias="skillId")
    version: str
    activation_id: str = Field(..., alias="activationId")
    package_digest: str = Field(..., alias="packageDigest", pattern=r"^[0-9a-f]{64}$")
    resource_paths: list[str] = Field(default_factory=list, alias="resourcePaths")


class AgentExecutionProvenance(StrictModel):
    trace_version: str = Field(AGENT_TRACE_VERSION, alias="traceVersion")
    tools_version: str = Field(AGENT_TOOLS_VERSION, alias="toolsVersion")
    workflow_version: str = Field(AGENT_WORKFLOW_VERSION, alias="workflowVersion")
    executor_version: str = Field(AGENT_EXECUTOR_VERSION, alias="executorVersion")
    stage: str
    agent: str
    job_id: str = Field(..., alias="jobId")
    coordinator_execution_id: str = Field(..., alias="coordinatorExecutionId")
    stage_execution_id: str = Field(..., alias="stageExecutionId")
    idempotency_key: str = Field(..., alias="idempotencyKey")
    selected_agent_id: str = Field(..., alias="selectedAgentId")
    selected_agent_version: str = Field(..., alias="selectedAgentVersion")
    selected_agent_digest: str = Field(
        ..., alias="selectedAgentDigest", pattern=r"^[0-9a-f]{64}$"
    )
    role: SpecialistRole
    output_contract: str = Field(..., alias="outputContract")
    prompt_version: str = Field(..., alias="promptVersion")
    stop_reason: AgentStopReason = Field(..., alias="stopReason")
    limits: AgentBudgetLimits
    usage: AgentBudgetUsage
    tool_calls: list[AgentToolTrace] = Field(default_factory=list, alias="toolCalls")
    activated_skills: list[SkillActivationProvenance] = Field(
        default_factory=list, alias="activatedSkills"
    )
    assigned_skills: list["AgentSkillReference"] = Field(
        default_factory=list, alias="assignedSkills"
    )
    artifact_inputs: list["ArtifactInputReference"] = Field(
        default_factory=list, alias="artifactInputs"
    )
    retry_of_stage_execution_id: str | None = Field(
        None, alias="retryOfStageExecutionId"
    )
    attempt_number: int = Field(..., alias="attemptNumber", ge=1)


class AgentFailure(StrictModel):
    stop_reason: AgentStopReason = Field(..., alias="stopReason")
    error_class: str = Field(..., alias="errorClass", max_length=120)
    detail: str = Field(..., max_length=500)
    retryable: bool = False
    job_id: str | None = Field(None, alias="jobId")
    coordinator_execution_id: str | None = Field(
        None, alias="coordinatorExecutionId"
    )
    stage_execution_id: str | None = Field(None, alias="stageExecutionId")
    selected_agent_id: str | None = Field(None, alias="selectedAgentId")
    selected_agent_version: str | None = Field(None, alias="selectedAgentVersion")
    selected_agent_digest: str | None = Field(
        None, alias="selectedAgentDigest", pattern=r"^[0-9a-f]{64}$"
    )
    role: SpecialistRole | None = None
    usage: AgentBudgetUsage | None = None
    partial_trace: list[AgentToolTrace] = Field(
        default_factory=list, alias="partialTrace"
    )


class SkillResourceSnapshot(StrictModel):
    path: str = Field(..., min_length=1, max_length=240)
    media_type: str = Field("text/plain", alias="mediaType", max_length=120)
    byte_count: int = Field(..., alias="byteCount", ge=0, le=131_072)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    content: str = Field(..., max_length=131_072)

    @field_validator("path")
    @classmethod
    def normalized_relative_path(cls, value: str) -> str:
        if (
            value.startswith(("/", "\\", "~"))
            or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("Skill resource path must be normalized and relative.")
        if not value.startswith(("references/", "assets/")):
            raise ValueError("Only reviewed references/ and assets/ resources are readable.")
        return value


class SkillSnapshotV2(StrictModel):
    id: str = Field(..., min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(..., min_length=1, max_length=500)
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    activation_id: str = Field(..., alias="activationId", min_length=16, max_length=160)
    package_digest: str = Field(..., alias="packageDigest", pattern=r"^[0-9a-f]{64}$")
    skill_md_digest: str = Field(..., alias="skillMdDigest", pattern=r"^[0-9a-f]{64}$")
    skill_md: str = Field(..., alias="skillMd", min_length=1, max_length=32_768)
    source_repository: str = Field(..., alias="sourceRepository", min_length=1, max_length=300)
    source_ref: str = Field(..., alias="sourceRef", min_length=7, max_length=100)
    lifecycle: Literal["published", "deprecated"]
    supported_stages: list[str] = Field(..., alias="supportedStages", min_length=1)
    supported_content_types: list[str] = Field(
        ..., alias="supportedContentTypes", min_length=1
    )
    approved_tool_ids: list[str] = Field(default_factory=list, alias="approvedToolIds")
    order: int = Field(0, ge=0, le=10_000)
    resources: list[SkillResourceSnapshot] = Field(default_factory=list, max_length=32)
    scripts: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def verify_content_digests(self) -> "SkillSnapshotV2":
        import hashlib

        safe_tool_ids = {
            "search_corpus",
            "load_evidence_page",
            "get_brief_context",
            "get_outline_context",
            "get_completed_section_summaries",
            "get_specialist_artifacts",
            "activate_skill",
            "read_skill_resource",
        }
        denied = sorted(set(self.approved_tool_ids) - safe_tool_ids)
        if denied:
            raise ValueError(
                f"Skill requests unapproved runtime tool(s): {', '.join(denied)}."
            )
        encoded = self.skill_md.encode("utf-8")
        if len(encoded) > 32_768:
            raise ValueError("SKILL.md exceeds the runtime disclosure limit.")
        if hashlib.sha256(encoded).hexdigest() != self.skill_md_digest:
            raise ValueError("SKILL.md digest mismatch.")
        for resource in self.resources:
            body = resource.content.encode("utf-8")
            if len(body) != resource.byte_count:
                raise ValueError(f"Resource byte count mismatch: {resource.path}")
            if hashlib.sha256(body).hexdigest() != resource.sha256:
                raise ValueError(f"Resource digest mismatch: {resource.path}")
        return self


class SignedSkillExecutionEnvelopeV2(StrictModel):
    envelope_version: Literal["gcc-skill-envelope.v2"] = Field(
        ..., alias="envelopeVersion"
    )
    catalog_version: str = Field(..., alias="catalogVersion", min_length=1, max_length=120)
    snapshot_digest: str = Field(..., alias="snapshotDigest", pattern=r"^[0-9a-f]{64}$")
    signature: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    signature_key_id: str = Field(..., alias="signatureKeyId", min_length=1, max_length=120)
    content_type: str = Field(..., alias="contentType", min_length=1, max_length=120)
    job_id: str = Field(..., alias="jobId", min_length=1, max_length=120)
    attempt_id: str = Field(..., alias="attemptId", min_length=1, max_length=120)
    resolved_at_utc: datetime = Field(..., alias="resolvedAtUtc")
    skills: list[SkillSnapshotV2] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def immutable_snapshot_is_canonical(self) -> "SignedSkillExecutionEnvelopeV2":
        ids = [item.id for item in self.skills]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate skills are not allowed.")
        if self.skills != sorted(self.skills, key=lambda item: (item.order, item.id)):
            raise ValueError("Skills are not in deterministic snapshot order.")
        if any(skill.lifecycle != "published" for skill in self.skills):
            raise ValueError("New v2 snapshots may contain only published skills.")
        if any(skill.scripts for skill in self.skills):
            # Scripts may remain in quarantine/registry records, but executable or
            # embedded script payloads are never accepted by this runtime.
            raise ValueError("Marketplace skill scripts are not executable runtime content.")
        return self


class AgentSkillReference(StrictModel):
    skill_id: str = Field(..., alias="skillId", min_length=1, max_length=80)
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    activation_id: str = Field(..., alias="activationId", min_length=16, max_length=160)
    package_digest: str = Field(
        ..., alias="packageDigest", pattern=r"^[0-9a-f]{64}$"
    )
    assignment: SkillAssignment


class ArtifactInputReference(StrictModel):
    artifact_id: str = Field(..., alias="artifactId", min_length=1, max_length=160)
    artifact_type: Literal[
        "canonicalBrief",
        "outline",
        "draftContent",
        "sources",
        "completedSectionSummaries",
        "specialistContribution",
        "specialistReview",
    ] = Field(..., alias="artifactType")
    digest: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class SelectedAgentSnapshot(StrictModel):
    id: str = Field(..., min_length=2, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_.-]+$")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    digest: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    name: str = Field(..., min_length=2, max_length=120)
    role: SpecialistRole
    instructions: str = Field(..., min_length=1, max_length=16_000)
    instructions_digest: str = Field(
        ..., alias="instructionsDigest", pattern=r"^[0-9a-f]{64}$"
    )
    policy: str = Field(..., min_length=1, max_length=8_000)
    policy_digest: str = Field(
        ..., alias="policyDigest", pattern=r"^[0-9a-f]{64}$"
    )
    stages: list[str] = Field(..., min_length=1, max_length=8)
    tool_ids: list[str] = Field(..., alias="toolIds", min_length=1, max_length=16)
    model_ids: list[str] = Field(..., alias="modelIds", min_length=1, max_length=8)

    @model_validator(mode="after")
    def verify_identity_digests(self) -> "SelectedAgentSnapshot":
        import hashlib
        import json

        if hashlib.sha256(self.instructions.encode()).hexdigest() != self.instructions_digest:
            raise ValueError("Selected agent instructions digest mismatch.")
        if hashlib.sha256(self.policy.encode()).hexdigest() != self.policy_digest:
            raise ValueError("Selected agent policy digest mismatch.")
        payload = self.model_dump(by_alias=True, mode="json", exclude={"digest"})
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != self.digest:
            raise ValueError("Selected agent digest mismatch.")
        if len(self.stages) != len(set(self.stages)):
            raise ValueError("Selected agent stages must be unique.")
        if len(self.tool_ids) != len(set(self.tool_ids)):
            raise ValueError("Selected agent tools must be unique.")
        if len(self.model_ids) != len(set(self.model_ids)):
            raise ValueError("Selected agent models must be unique.")
        return self


class AgentExecutionRequest(StrictModel):
    contract_version: Literal["specialist-team-execution.v1"] = Field(
        ..., alias="contractVersion"
    )
    snapshot_digest: str = Field(
        ..., alias="snapshotDigest", pattern=r"^[0-9a-f]{64}$"
    )
    signature: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    signature_key_id: str = Field(
        ..., alias="signatureKeyId", min_length=1, max_length=120
    )
    job_id: str = Field(..., alias="jobId", min_length=1, max_length=120)
    attempt_id: str = Field(..., alias="attemptId", min_length=1, max_length=120)
    coordinator_execution_id: str = Field(
        ..., alias="coordinatorExecutionId", min_length=1, max_length=120
    )
    stage_execution_id: str = Field(
        ..., alias="stageExecutionId", min_length=1, max_length=120
    )
    idempotency_key: str = Field(..., alias="idempotencyKey", min_length=16, max_length=200)
    issued_at_utc: datetime = Field(..., alias="issuedAtUtc")
    expires_at_utc: datetime = Field(..., alias="expiresAtUtc")
    attempt_number: int = Field(..., alias="attemptNumber", ge=1, le=100)
    retry_of_stage_execution_id: str | None = Field(
        None, alias="retryOfStageExecutionId", min_length=1, max_length=120
    )
    stage: str = Field(..., min_length=1, max_length=80)
    selected_agent: SelectedAgentSnapshot = Field(..., alias="selectedAgent")
    output_contract: Literal[
        "contributorOutput.v1", "producerOutput.v1", "reviewerOutput.v1"
    ] = Field(..., alias="outputContract")
    assigned_skills: list[AgentSkillReference] = Field(
        default_factory=list, alias="assignedSkills", max_length=10
    )
    artifact_inputs: list[ArtifactInputReference] = Field(
        default_factory=list, alias="artifactInputs", max_length=32
    )
    limits: AgentBudgetLimits = Field(default_factory=AgentBudgetLimits)
    cancelled: bool = False
    repair_attempt: int = Field(0, alias="repairAttempt", ge=0, le=100)

    @model_validator(mode="after")
    def validate_execution_contract(self) -> "AgentExecutionRequest":
        if self.repair_attempt > self.limits.max_repair_attempts:
            raise ValueError("Repair attempt exceeds the declared agent budget.")
        for value, field in (
            (self.attempt_id, "attemptId"),
            (self.coordinator_execution_id, "coordinatorExecutionId"),
            (self.stage_execution_id, "stageExecutionId"),
        ):
            try:
                UUID(value)
            except ValueError as ex:
                raise ValueError(f"{field} must be a UUID.") from ex
        if self.issued_at_utc.tzinfo is None or self.expires_at_utc.tzinfo is None:
            raise ValueError("Execution timestamps must include a UTC offset.")
        if self.expires_at_utc <= self.issued_at_utc:
            raise ValueError("expiresAtUtc must be after issuedAtUtc.")
        if self.attempt_number == 1 and self.retry_of_stage_execution_id is not None:
            raise ValueError("Initial execution cannot declare retryOfStageExecutionId.")
        if self.attempt_number > 1 and self.retry_of_stage_execution_id is None:
            raise ValueError("Retry execution requires retryOfStageExecutionId.")
        expected_output = {
            SpecialistRole.CONTRIBUTOR: "contributorOutput.v1",
            SpecialistRole.PRODUCER: "producerOutput.v1",
            SpecialistRole.REVIEWER: "reviewerOutput.v1",
        }[self.selected_agent.role]
        if self.output_contract != expected_output:
            raise ValueError("Output contract does not match selected agent role.")
        refs = [(item.skill_id, item.version, item.activation_id) for item in self.assigned_skills]
        if len(refs) != len(set(refs)):
            raise ValueError("Assigned skill references must be unique.")
        artifacts = [item.artifact_id for item in self.artifact_inputs]
        if len(artifacts) != len(set(artifacts)):
            raise ValueError("Artifact input identifiers must be unique.")
        return self


def canonical_snapshot_payload(envelope: SignedSkillExecutionEnvelopeV2) -> dict[str, Any]:
    """Return the exact immutable fields covered by digest and signature."""
    return envelope.model_dump(
        by_alias=True,
        mode="json",
        exclude={"snapshot_digest", "signature"},
    )


def canonical_agent_execution_payload(execution: AgentExecutionRequest) -> dict[str, Any]:
    """Return the exact specialist/team fields covered by digest and signature."""
    return execution.model_dump(
        by_alias=True,
        mode="json",
        exclude={"snapshot_digest", "signature", "cancelled"},
    )
