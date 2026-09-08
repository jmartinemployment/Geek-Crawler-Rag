"""Deterministic contract-level quality checks for citeable generation."""

from __future__ import annotations

import re
from typing import Any

from geek_crawler_rag.generate import verify_citations
from geek_crawler_rag.models import (
    GenerateCitation,
    GenerateProvenance,
    GenerateRequest,
    GenerateSource,
    GenerateValidation,
)

REQUIRED_BRIEF_PROMPT_FIELDS = (
    "version",
    "title",
    "targetKeyword",
    "contentType",
    "primaryIntent",
    "audience",
    "buyingStage",
    "toneOfVoice",
    "brandKit",
    "paaQuestions",
    "requiredTopics",
    "operatorInstructions",
    "exclusions",
    "hierarchy",
    "internalLinks",
    "outputRequirements",
    "channelRequirements",
    "ctaRequirements",
    "conversionObjective",
    "publishingDestination",
)
REQUIRED_PROVENANCE_FIELDS = (
    "generationStage",
    "modelUsed",
    "modelPolicyPreset",
    "modelPolicyVersion",
    "promptVersion",
    "retrieval",
    "evidenceIds",
)
STRICT_PROVENANCE_FIELDS = (
    "specialistExecutor",
    "specialistExecutorVersion",
    "executionVersion",
    "attemptId",
    "skills",
)
REQUIRED_VALIDATION_FIELDS = (
    "approved",
    "issues",
    "strengths",
    "unsupportedClaimCount",
    "briefAlignmentScore",
    "evidenceCoverageScore",
    "usefulnessScore",
    "originalityScore",
    "brandAlignmentScore",
)
_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")


def evaluate_quality_contract(
    *,
    request: GenerateRequest,
    system_prompt: str,
    user_prompt: str,
    citations: list[GenerateCitation],
    sources: list[GenerateSource],
    pages: list[dict[str, Any]],
    provenance: GenerateProvenance | dict[str, Any],
    validation: GenerateValidation | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score only quality signals that can be proven without a model call."""
    combined_prompt = f"{system_prompt}\n{user_prompt}"
    covered_brief_fields = [
        field
        for field in REQUIRED_BRIEF_PROMPT_FIELDS
        if f'"{field}"' in user_prompt
    ]
    brief_score = len(covered_brief_fields) / len(REQUIRED_BRIEF_PROMPT_FIELDS)

    verified, dropped = verify_citations(citations, sources, pages)
    citation_score = len(verified) / len(citations) if citations else 0.0

    source_ids = {source.page_id or source.url for source in sources}
    cited_ids = {citation.page_id or citation.url for citation in verified}
    evidence_score = (
        len(source_ids & cited_ids) / len(source_ids) if source_ids else 0.0
    )

    headings = [
        f"{marker} {title}" for marker, title in _HEADING.findall(request.draft_content or "")
    ]
    heading_instruction_present = (
        "Preserve every Markdown heading exactly" in combined_prompt
        and "same order and at the same heading level" in combined_prompt
    )
    heading_score = (
        1.0
        if heading_instruction_present
        and headings
        and all(heading in user_prompt for heading in headings)
        else 0.0
    )

    provenance_data = (
        provenance.model_dump(by_alias=True)
        if isinstance(provenance, GenerateProvenance)
        else provenance
    )
    required_provenance_fields = REQUIRED_PROVENANCE_FIELDS
    if provenance_data.get("executionVersion") == "rag-generate.v2":
        required_provenance_fields += STRICT_PROVENANCE_FIELDS
    complete_provenance_fields = [
        field
        for field in required_provenance_fields
        if provenance_data.get(field) not in (None, "", [])
    ]
    if provenance_data.get("executionVersion") == "rag-generate.v2":
        skills = provenance_data.get("skills")
        strict_skill_provenance = (
            isinstance(skills, dict)
            and all(
                skills.get(field) not in (None, "", [])
                for field in (
                    "envelopeVersion",
                    "catalogVersion",
                    "snapshotHash",
                    "stage",
                    "skillVersions",
                )
            )
        )
        if not strict_skill_provenance and "skills" in complete_provenance_fields:
            complete_provenance_fields.remove("skills")
    provenance_score = len(complete_provenance_fields) / len(required_provenance_fields)

    signals = {
        "briefPromptCoverage": round(brief_score, 4),
        "citationExactness": round(citation_score, 4),
        "evidenceCoverage": round(evidence_score, 4),
        "provenanceCompleteness": round(provenance_score, 4),
    }
    if request.generation_stage == "finalSynthesis":
        signals["headingPreservationRequirements"] = round(heading_score, 4)
    complete_validation_fields: list[str] = []
    if request.generation_stage == "validation":
        validation_data = (
            validation.model_dump(by_alias=True)
            if isinstance(validation, GenerateValidation)
            else validation or {}
        )
        complete_validation_fields = [
            field for field in REQUIRED_VALIDATION_FIELDS if field in validation_data
        ]
        validation_score = len(complete_validation_fields) / len(
            REQUIRED_VALIDATION_FIELDS
        )
        issues = validation_data.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(issue, dict)
            or not all(
                field in issue
                for field in ("sectionTitle", "category", "detail", "repairInstruction")
            )
            for issue in issues
        ):
            validation_score = 0.0
        signals["validationCompleteness"] = round(validation_score, 4)

    return {
        "score": round(sum(signals.values()) / len(signals), 4),
        "signals": signals,
        "details": {
            "coveredBriefFields": covered_brief_fields,
            "verifiedCitationCount": len(verified),
            "droppedCitationCount": dropped,
            "coveredEvidenceIds": sorted(source_ids & cited_ids),
            "draftHeadings": headings,
            "completeProvenanceFields": complete_provenance_fields,
            "completeValidationFields": complete_validation_fields,
        },
    }
