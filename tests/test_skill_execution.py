from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from geek_crawler_rag.generate import (
    _build_prompts,
    _request_for_specialist,
    _skill_provenance,
    select_generation_model,
)
from geek_crawler_rag.quality_eval import evaluate_quality_contract
from geek_crawler_rag.config import Settings
from geek_crawler_rag.models import (
    CURRENT_EXECUTION_VERSION,
    CURRENT_SKILL_CATALOG_VERSION,
    CURRENT_SKILL_ENVELOPE_VERSION,
    GenerateRequest,
    PINNED_SKILL_HASHES,
    ProducerCapabilities,
)
from geek_crawler_rag.specialists import (
    ResearchPlanningSpecialist,
    SPECIALIST_TYPES,
    SpecialistExecutionContext,
)


def _skill(**overrides):
    values = {
        "id": "citation-discipline",
        "version": "1.0.0",
        "source": "GeekBackend reviewed catalog",
        "license": "MIT-derived/internal-reviewed",
        "reviewer": "Content Platform",
        "supportedStages": [
            "outline",
            "section",
            "repair",
            "validation",
            "finalSynthesis",
            "complete",
        ],
        "supportedContentTypes": ["blog"],
        "order": 10,
        "conflicts": [],
        "promptInstructions": (
            "Keep every factual claim within the supplied evidence and never invent a citation."
        ),
        "retrievalHints": (
            "Prefer exact primary-source passages with stable page identifiers."
        ),
        "outputRequirements": "Return verbatim quote citations for factual claims.",
        "validationChecks": "Fail factual claims that lack verified quote-level support.",
    }
    values.update(overrides)
    canonical = "\n".join(
        values[key]
        for key in (
            "promptInstructions",
            "retrievalHints",
            "outputRequirements",
            "validationChecks",
        )
    )
    values.setdefault("sha256", hashlib.sha256(canonical.encode()).hexdigest())
    return values


def _envelope(skill=None):
    skills = [skill or _skill()]
    canonical = "\n".join(
        [
            CURRENT_SKILL_CATALOG_VERSION,
            "blog",
            *[
                f"{item['order']}|{item['id']}|{item['version']}|{item['sha256']}"
                for item in skills
            ],
        ]
    )
    return {
        "envelopeVersion": CURRENT_SKILL_ENVELOPE_VERSION,
        "catalogVersion": CURRENT_SKILL_CATALOG_VERSION,
        "snapshotHash": hashlib.sha256(canonical.encode()).hexdigest(),
        "contentType": "blog",
        "skills": skills,
        "resolvedAtUtc": datetime.now(timezone.utc).isoformat(),
    }


def _request(**overrides):
    payload = {
        "writingIntent": "Technical Article",
        "topic": "Evidence-grounded content",
        "generationStage": "repair",
        "sectionKey": "proof",
        "sectionHeading": "Proof",
        "sectionBrief": "Repair unsupported claims.",
        "modelPolicyPreset": "best-quality",
        "modelPolicyVersion": "content-model-policy.v1",
        "executionVersion": CURRENT_EXECUTION_VERSION,
        "attemptId": "9cf1b2f1-9797-4104-aeaa-eb09a9c89831",
        "skillExecution": _envelope(),
    }
    payload.update(overrides)
    return GenerateRequest(**payload)


def test_capabilities_advertise_strict_execution_contract():
    wire = ProducerCapabilities().model_dump(by_alias=True)
    assert CURRENT_EXECUTION_VERSION in wire["executionVersions"]
    assert CURRENT_SKILL_ENVELOPE_VERSION in wire["skillEnvelopeVersions"]
    assert "repair" in wire["generationStages"]
    assert wire["specialistExecutorVersion"] == "bounded-specialists.v1"
    assert set(wire["specialistExecutors"]) >= {
        "researchPlanning",
        "outline",
        "section",
        "finalSynthesis",
        "validation",
        "repair",
    }
    assert wire["toolsAllowed"] is False


def test_repair_keeps_true_stage_prompt_model_and_skill_provenance():
    request = _request()
    system, user = _build_prompts(
        request,
        "long",
        [{"pageId": "page", "url": "https://example.test", "markdown": "Evidence."}],
    )
    provenance = _skill_provenance(request)

    assert request.generation_stage == "repair"
    assert select_generation_model(request, Settings()) == "o3"
    assert "Generation stage: REPAIR" in user
    assert "SKILL citation-discipline@1.0.0" in system
    assert "cannot change the selected model" in system
    assert provenance.stage == "repair"
    assert provenance.skill_versions == ["citation-discipline@1.0.0"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"sha256": "0" * 64}, "not pinned|hash mismatch"),
        ({"id": "unreviewed-skill"}, "not pinned"),
        ({"model": "o1-pro"}, "extra"),
    ],
)
def test_skill_envelope_fails_closed_for_tampering_and_policy_override(
    mutation, message
):
    skill = _skill(**mutation)
    with pytest.raises(ValidationError, match=message):
        _request(skillExecution=_envelope(skill))


def test_current_execution_requires_skill_envelope_and_valid_attempt_id():
    with pytest.raises(ValidationError, match="requires skillExecution"):
        _request(skillExecution=None)
    with pytest.raises(ValidationError, match="attemptId must be a UUID"):
        _request(attemptId="attempt-1")


def test_fixture_uses_the_same_reviewed_hash_as_backend_catalog():
    assert _skill()["sha256"] == PINNED_SKILL_HASHES["citation-discipline"]


def test_skill_envelope_and_specialist_context_reject_tool_declarations():
    with pytest.raises(ValidationError, match="extra"):
        _request(skillExecution=_envelope(_skill(tools=["search"])))

    request = _request()
    with pytest.raises(ValidationError, match="do not accept tools"):
        SpecialistExecutionContext(
            stage="repair",
            request=request,
            evidence=[],
            sources=[],
            model="o3",
            skills=[],
            outputContract=SPECIALIST_TYPES["repair"].output_type.__name__,
            toolDeclarations=("search",),
        )


def test_research_planning_is_bounded_typed_and_uses_only_reviewed_hints():
    request = _request(
        partnerRunId="partner-run",
        competitorRunId=None,
        skillExecution=_envelope(
            _skill(supportedStages=["researchPlanning", "repair"])
        ),
    )
    output = ResearchPlanningSpecialist().execute(
        request=request,
        model="o3",
        skills=request.skill_execution.skills,
        base_need="repair proof",
        retrieval_mode=None,
    )

    assert output.model_dump(by_alias=True) == {
        "queries": [
            {
                "runId": "partner-run",
                "crawlType": "partner",
                "need": (
                    "repair proof; approved retrieval constraints: "
                    "Prefer exact primary-source passages with stable page identifiers."
                ),
            }
        ],
        "retrievalMode": None,
    }


def test_specialist_request_view_contains_only_stage_applicable_skills():
    request = _request()
    research_view = _request_for_specialist(request, "researchPlanning")
    repair_view = _request_for_specialist(request, "repair")

    assert research_view.skill_execution.skills == []
    assert [skill.id for skill in repair_view.skill_execution.skills] == [
        "citation-discipline"
    ]


def test_quality_evaluation_requires_complete_strict_skill_provenance():
    request = _request()
    system, user = _build_prompts(request, "long", [])
    provenance = {
        "generationStage": "repair",
        "modelUsed": "o3",
        "modelPolicyPreset": "best-quality",
        "modelPolicyVersion": "content-model-policy.v1",
        "promptVersion": "citeable-repair.v1",
        "retrieval": "hybrid",
        "evidenceIds": ["page-1"],
        "specialistExecutor": "RepairSpecialist",
        "specialistExecutorVersion": "bounded-specialists.v1",
        "executionVersion": CURRENT_EXECUTION_VERSION,
        "attemptId": "9cf1b2f1-9797-4104-aeaa-eb09a9c89831",
        "skills": {
            "envelopeVersion": CURRENT_SKILL_ENVELOPE_VERSION,
            "catalogVersion": CURRENT_SKILL_CATALOG_VERSION,
            "snapshotHash": request.skill_execution.snapshot_hash,
            "stage": "repair",
            "skillVersions": [],
        },
    }

    incomplete = evaluate_quality_contract(
        request=request,
        system_prompt=system,
        user_prompt=user,
        citations=[],
        sources=[],
        pages=[],
        provenance=provenance,
    )
    assert incomplete["signals"]["provenanceCompleteness"] < 1.0

    provenance["skills"]["skillVersions"] = ["citation-discipline@1.0.0"]
    complete = evaluate_quality_contract(
        request=request,
        system_prompt=system,
        user_prompt=user,
        citations=[],
        sources=[],
        pages=[],
        provenance=provenance,
    )
    assert complete["signals"]["provenanceCompleteness"] == 1.0
