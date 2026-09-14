from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

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


def test_capabilities_do_not_advertise_removed_generate_endpoint():
    wire = ProducerCapabilities().model_dump(by_alias=True)
    assert wire["executionVersions"] == []
    assert wire["skillEnvelopeVersions"] == []
    assert wire["generationStages"] == []
    assert wire["agentGenerationStages"] == []
    assert wire["specialistExecutors"] == []

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