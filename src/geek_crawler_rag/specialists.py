"""Bounded, typed stage specialists used by the citeable generation workflow.

These executors are not autonomous agents: they have no tool registry, cannot
delegate, and perform exactly one typed stage operation per workflow invocation.
Durable orchestration and approval decisions remain in GeekBackend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from geek_crawler_rag.models import (
    GenerateCitation,
    GenerateOutlineSection,
    GenerateRequest,
    GenerateSource,
    GenerateValidation,
    SkillDefinition,
)


class ResearchQueryPlan(BaseModel):
    run_id: str = Field(..., alias="runId")
    crawl_type: Literal["partner", "competitors"] = Field(..., alias="crawlType")
    need: str

    model_config = {"populate_by_name": True, "extra": "forbid"}


class ResearchPlanningOutput(BaseModel):
    queries: list[ResearchQueryPlan]
    retrieval_mode: str | None = Field(None, alias="retrievalMode")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class OutlineSpecialistOutput(BaseModel):
    outline: list[GenerateOutlineSection]
    citations: list[GenerateCitation] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class SectionSpecialistOutput(BaseModel):
    content: str
    citations: list[GenerateCitation] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class FinalSynthesisSpecialistOutput(SectionSpecialistOutput):
    pass


class RepairSpecialistOutput(SectionSpecialistOutput):
    pass


class ValidationSpecialistOutput(BaseModel):
    validation: GenerateValidation
    citations: list[GenerateCitation] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class CompleteSpecialistOutput(BaseModel):
    content: str | None = None
    variations: list[str] | None = None
    battlecard: dict[str, Any] | None = None
    citations: list[GenerateCitation] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class SpecialistExecutionContext(BaseModel):
    """The complete authority available to one bounded specialist invocation."""

    stage: str
    request: GenerateRequest
    evidence: list[dict[str, Any]]
    sources: list[GenerateSource]
    model: str
    skills: list[SkillDefinition]
    output_contract: str = Field(..., alias="outputContract")
    tool_declarations: tuple[str, ...] = Field(default=(), alias="toolDeclarations")

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def no_tools_allowed(self) -> SpecialistExecutionContext:
        if self.tool_declarations:
            raise ValueError("Bounded generation specialists do not accept tools.")
        return self


PromptBuilder = Callable[[GenerateRequest, str, list[dict[str, Any]]], tuple[str, str]]
ChatTransport = Callable[[str, str, str], Awaitable[str]]
JsonParser = Callable[[str, str], dict[str, Any]]


class BoundedStageSpecialist(ABC):
    """Single-call specialist with a fixed stage and typed output contract."""

    stage: str
    output_type: type[BaseModel]

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        chat: ChatTransport,
        parser: JsonParser,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._chat = chat
        self._parser = parser

    async def execute(
        self, context: SpecialistExecutionContext, family: str
    ) -> BaseModel:
        if context.stage != self.stage:
            raise ValueError(
                f"{type(self).__name__} cannot execute stage '{context.stage}'."
            )
        if context.output_contract != self.output_type.__name__:
            raise ValueError("Specialist output contract mismatch.")
        system, user = self._prompt_builder(context.request, family, context.evidence)
        raw = await self._chat(context.model, system, user)
        return self.output_type.model_validate(self._parser(raw, family))


class OutlineSpecialist(BoundedStageSpecialist):
    stage = "outline"
    output_type = OutlineSpecialistOutput


class SectionWritingSpecialist(BoundedStageSpecialist):
    stage = "section"
    output_type = SectionSpecialistOutput


class FinalSynthesisSpecialist(BoundedStageSpecialist):
    stage = "finalSynthesis"
    output_type = FinalSynthesisSpecialistOutput


class ValidationSpecialist(BoundedStageSpecialist):
    stage = "validation"
    output_type = ValidationSpecialistOutput


class RepairSpecialist(BoundedStageSpecialist):
    stage = "repair"
    output_type = RepairSpecialistOutput


class CompleteSpecialist(BoundedStageSpecialist):
    """Compatibility executor for legacy one-shot generation callers."""

    stage = "complete"
    output_type = CompleteSpecialistOutput


class ResearchPlanningSpecialist:
    """Deterministically plans retrieval; it has no model or network authority."""

    stage = "researchPlanning"
    output_type = ResearchPlanningOutput

    def execute(
        self,
        *,
        request: GenerateRequest,
        model: str,
        skills: list[SkillDefinition],
        base_need: str,
        retrieval_mode: str | None,
    ) -> ResearchPlanningOutput:
        # Model is part of the reviewed policy context even though this bounded
        # deterministic specialist intentionally performs no LLM call.
        if not model:
            raise ValueError("Research planning requires an explicit model policy.")
        hints = "; ".join(skill.retrieval_hints for skill in skills)
        queries = [
            ResearchQueryPlan(
                runId=run_id,
                crawlType=crawl_type,
                need=f"{base_need}; approved retrieval constraints: {hints[:1200]}"
                if hints
                else base_need,
            )
            for run_id, crawl_type in (
                (request.partner_run_id, "partner"),
                (request.competitor_run_id, "competitors"),
            )
            if run_id
        ]
        return ResearchPlanningOutput(
            queries=queries,
            retrievalMode=retrieval_mode,
        )


SPECIALIST_TYPES: dict[str, type[BoundedStageSpecialist]] = {
    "outline": OutlineSpecialist,
    "section": SectionWritingSpecialist,
    "finalSynthesis": FinalSynthesisSpecialist,
    "validation": ValidationSpecialist,
    "repair": RepairSpecialist,
    "complete": CompleteSpecialist,
}

