"""Citeable generate: retrieve → read Mongo Markdown → draft → verify citations.

Uses LlamaIndex Workflow steps as the orchestration starting point (no LlamaParse).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from openai import AsyncOpenAI

from geek_crawler_rag.config import Settings
from geek_crawler_rag.models import (
    ChunkHit,
    GenerateCitation,
    GenerateOutlineSection,
    GenerateProvenance,
    GenerateRequest,
    GenerateResponse,
    GenerateSource,
    GenerateValidation,
    QueryRequest,
    ThemeHit,
)
from geek_crawler_rag.mongo import MongoCorpus
from geek_crawler_rag.query import QueryService

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")

_LONG_FORM = frozenset({"technical article", "case study"})
_SHORT_FORM = frozenset({"social ad", "short form"})
_BATTLECARD = frozenset({"competitive battlecard"})
_SLIDES = frozenset({"pitch slides", "strategy theme"})
_PROMPT_VERSION = "citeable-generate.v2"
_BEST_QUALITY_MODELS = {
    "researchPlanning": "o3",
    "outline": "o1-pro",
    "section": "o3",
    "repair": "o3",
    "validation": "o3",
    "finalSynthesis": "o1-pro",
    "complete": "o3",
}


class RetrievedEvent(Event):
    chunks: list[ChunkHit]
    themes: list[ThemeHit]
    retrieval: str
    warnings: list[str]


class PagesLoadedEvent(Event):
    pages: list[dict[str, Any]]
    themes: list[ThemeHit]
    retrieval: str
    warnings: list[str]


class DraftedEvent(Event):
    content: str | None
    variations: list[str] | None
    battlecard: dict[str, Any] | None
    outline: list[GenerateOutlineSection] | None
    citations: list[GenerateCitation]
    sources: list[GenerateSource]
    themes: list[ThemeHit]
    retrieval: str
    warnings: list[str]
    model_used: str
    validation: GenerateValidation | None


def _family(intent: str) -> str:
    key = intent.strip().lower()
    if key in _SHORT_FORM:
        return "short"
    if key in _BATTLECARD:
        return "battlecard"
    if key in _SLIDES:
        return "slides"
    return "long"


def _stage(request: GenerateRequest) -> str:
    return request.generation_stage or "complete"


def _normalize_ws(text: str) -> str:
    return _WS.sub(" ", (text or "").strip()).lower()


def quote_in_markdown(quote: str, markdown: str) -> bool:
    """True when quote (normalized) appears in markdown."""
    q = _normalize_ws(quote)
    if len(q) < 12:
        return False
    body = _normalize_ws(markdown)
    return q in body


def select_generation_model(request: GenerateRequest, settings: Settings) -> str:
    """Resolve an explicit policy without silently substituting another model."""
    stage = _stage(request)
    preset = request.model_policy_preset
    if preset is None:
        # Preserve standalone callers that predate the unified model policy.
        return (
            settings.openai_longform_model
            if _family(request.writing_intent) == "long"
            else settings.openai_standard_model
        )
    if preset == "o3-only":
        return "o3"
    if preset == "best-quality":
        return _BEST_QUALITY_MODELS[stage]
    override = (request.stage_model_overrides or {}).get(stage)
    if not override:
        raise ValueError(
            f"Custom model policy has no override for generation stage '{stage}'. "
            f"Add stageModelOverrides.{stage} using o1-pro or o3."
        )
    return override


def verify_citations(
    citations: list[GenerateCitation],
    sources: list[GenerateSource],
    pages: list[dict[str, Any]],
) -> tuple[list[GenerateCitation], int]:
    """Keep only citations whose URL and verbatim quote match loaded evidence."""
    md_by_url = {
        str(page.get("url") or "").lower(): str(page.get("markdown") or "")
        for page in pages
        if page.get("url") and page.get("markdown")
    }
    allowed_urls = {source.url.lower() for source in sources if source.url}
    kept: list[GenerateCitation] = []
    dropped = 0
    for citation in citations:
        url_key = citation.url.lower()
        body = md_by_url.get(url_key) or ""
        if (
            (allowed_urls and url_key not in allowed_urls)
            or not body
            or not quote_in_markdown(citation.quote, body)
        ):
            dropped += 1
            continue
        kept.append(citation)
    return kept, dropped


def _responses_output_text(response: Any) -> str:
    """Extract text from Responses API convenience or typed output shapes."""
    direct = (
        response.get("output_text")
        if isinstance(response, dict)
        else getattr(response, "output_text", None)
    )
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    output = (
        response.get("output", [])
        if isinstance(response, dict)
        else getattr(response, "output", [])
    )
    parts: list[str] = []
    for item in output or []:
        content = (
            item.get("content", [])
            if isinstance(item, dict)
            else getattr(item, "content", [])
        )
        for block in content or []:
            block_type = (
                block.get("type")
                if isinstance(block, dict)
                else getattr(block, "type", None)
            )
            text = (
                block.get("text")
                if isinstance(block, dict)
                else getattr(block, "text", None)
            )
            if block_type in {"output_text", "text"} and isinstance(text, str):
                parts.append(text)
    extracted = "".join(parts).strip()
    if not extracted:
        raise ValueError("OpenAI Responses API returned no text output.")
    return extracted


class CiteableGenerateWorkflow(Workflow):
    """Multi-step citeable draft over partner/competitor runs."""

    def __init__(
        self,
        *,
        mongo: MongoCorpus,
        query: QueryService,
        settings: Settings,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self._mongo = mongo
        self._query = query
        self._settings = settings
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key or "missing")

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrievedEvent:
        req: GenerateRequest = ev.get("request")
        await ctx.store.set("request", req)
        family = _family(req.writing_intent)
        prefer_parent = family != "short"
        prefer_child = family == "short"
        top_k = 5 if family == "short" else (8 if family == "battlecard" else 10)
        retrieval_mode = "graph" if family == "slides" and req.graph_enabled else None
        min_q = self._settings.generate_min_quality

        warnings: list[str] = []
        chunks: list[ChunkHit] = []
        themes: list[ThemeHit] = []
        retrieval_labels: list[str] = []

        entities = [e.strip() for e in (req.target_entities or []) if e and e.strip()][:12]
        brief_retrieval_context = _brief_retrieval_context(req)

        for run_id, crawl_type in (
            (req.partner_run_id, "partner"),
            (req.competitor_run_id, "competitors"),
        ):
            if not run_id:
                continue
            need = (
                f"{'competitor differentiation' if crawl_type == 'competitors' else 'partner tool'} "
                f"research; writing intent: {req.writing_intent}; topic: {req.topic[:200]}"
            )
            if brief_retrieval_context:
                need += f"; canonical brief: {brief_retrieval_context}"
            if _stage(req) == "section" and req.section_heading:
                need += f"; section: {req.section_heading[:160]}"
            if entities:
                need += f"; entities: {', '.join(entities[:8])}"
            qreq = QueryRequest(
                need=need,
                run_id=run_id,
                crawl_type=crawl_type,
                top_k=top_k,
                prefer_parent=prefer_parent,
                prefer_child=prefer_child,
                entity_names=entities or None,
                min_quality=min_q,
                retrieval_mode=retrieval_mode,
            )
            resp = await self._query.query(qreq)
            if resp.warning:
                warnings.append(resp.warning)
            if resp.retrieval:
                retrieval_labels.append(resp.retrieval)
            chunks.extend(resp.chunks)
            if resp.themes:
                themes.extend(resp.themes)

        if not chunks and not req.input_sources:
            warnings.append("No RAG chunks for partner/competitor runs.")

        return RetrievedEvent(
            chunks=chunks,
            themes=themes,
            retrieval=retrieval_mode or (retrieval_labels[0] if retrieval_labels else "hybrid"),
            warnings=warnings,
        )

    @step
    async def load_pages(self, ctx: Context, ev: RetrievedEvent) -> PagesLoadedEvent:
        req: GenerateRequest = await ctx.store.get("request")
        max_pages = self._settings.generate_max_pages
        max_chars = self._settings.generate_markdown_chars
        seen: set[str] = set()
        pages: list[dict[str, Any]] = []

        # Final synthesis can carry the exact source manifest used by earlier stages.
        # Reload its Markdown here so citation verification remains mandatory.
        for source in req.input_sources or []:
            page = await self._mongo.get_page(source.page_id) if source.page_id else None
            if page is None and source.url:
                crawl_type = (source.crawl_type or "").strip().casefold()
                if crawl_type in {"competitor", "competitors"}:
                    run_id = req.competitor_run_id
                elif crawl_type == "partner":
                    run_id = req.partner_run_id
                else:
                    run_id = req.partner_run_id or req.competitor_run_id
                if run_id:
                    page = await self._mongo.get_page_by_url(
                        run_id=run_id, url=source.url
                    )
            if page is None or not page.markdown:
                continue
            key = page.id or page.final_url or page.url
            if not key or key in seen:
                continue
            seen.add(key)
            pages.append(
                {
                    "pageId": page.id or source.page_id,
                    "url": page.final_url or page.url or source.url,
                    "title": page.title or source.title,
                    "sectionTitle": None,
                    "crawlType": source.crawl_type,
                    "entityName": source.entity,
                    "markdown": page.markdown[:max_chars],
                    "preview": page.markdown[:500],
                }
            )
            if len(pages) >= max_pages:
                break

        # Prefer unique pages by pageId, then url.
        ordered = sorted(ev.chunks, key=lambda c: c.score, reverse=True)
        for hit in ordered:
            if len(pages) >= max_pages:
                break
            key = hit.page_id or hit.final_url or hit.url
            if not key or key in seen:
                continue
            seen.add(key)

            markdown = None
            title = hit.title
            page_id = hit.page_id
            url = hit.final_url or hit.url
            if hit.page_id:
                page = await self._mongo.get_page(hit.page_id)
                if page and page.markdown:
                    markdown = page.markdown
                    title = page.title or title
                    url = page.final_url or page.url or url
                    page_id = page.id
            if not markdown:
                # Fallback: parent/chunk text only (still citeable unit, not Excerpt).
                markdown = hit.text
            if not markdown or len(markdown.strip()) < 40:
                continue

            q = hit.quality_score
            if q is not None and q < self._settings.generate_min_quality:
                continue

            pages.append(
                {
                    "pageId": page_id,
                    "url": url,
                    "title": title,
                    "sectionTitle": hit.section_title,
                    "crawlType": hit.crawl_type,
                    "entityName": hit.entity_name,
                    "markdown": markdown[:max_chars],
                    "preview": (hit.text or "")[:500],
                }
            )
        warnings = list(ev.warnings)
        if not pages:
            warnings.append("No page Markdown available for citation context.")

        await ctx.store.set("pages", pages)
        return PagesLoadedEvent(
            pages=pages,
            themes=ev.themes,
            retrieval=ev.retrieval,
            warnings=warnings,
        )

    @step
    async def draft(self, ctx: Context, ev: PagesLoadedEvent) -> DraftedEvent:
        req: GenerateRequest = await ctx.store.get("request")
        family = _family(req.writing_intent)
        model = select_generation_model(req, self._settings)

        if not self._settings.openai_api_key:
            return DraftedEvent(
                content=None,
                variations=None,
                battlecard=None,
                outline=None,
                citations=[],
                sources=_sources_from_pages(ev.pages),
                themes=ev.themes,
                retrieval=ev.retrieval,
                warnings=ev.warnings + ["OPENAI_API_KEY unset; generate skipped."],
                model_used=model,
                validation=None,
            )

        if not ev.pages:
            return DraftedEvent(
                content=None,
                variations=None,
                battlecard=None,
                outline=None,
                citations=[],
                sources=[],
                themes=ev.themes,
                retrieval=ev.retrieval,
                warnings=ev.warnings,
                model_used=model,
                validation=None,
            )

        system, user = _build_prompts(req, family, ev.pages)
        raw = await self._chat(model, system, user)
        parsed = _parse_llm_json(raw, family)
        citations = [
            GenerateCitation(
                page_id=c.get("pageId"),
                url=str(c.get("url") or ""),
                title=c.get("title"),
                section_title=c.get("sectionTitle"),
                quote=str(c.get("quote") or "").strip(),
                crawl_type=c.get("crawlType"),
            )
            for c in parsed.get("citations") or []
            if str(c.get("quote") or "").strip() and str(c.get("url") or "").strip()
        ]
        outline = [
            GenerateOutlineSection(
                key=str(section.get("key") or f"section-{i + 1}"),
                heading=str(section.get("heading") or "").strip(),
                brief=str(section.get("brief") or "").strip(),
                evidence_ids=[
                    str(evidence_id)
                    for evidence_id in (section.get("evidenceIds") or [])
                    if evidence_id
                ],
            )
            for i, section in enumerate(parsed.get("outline") or [])
            if isinstance(section, dict)
            and str(section.get("heading") or "").strip()
        ][:12]
        validation = (
            GenerateValidation.model_validate(parsed.get("validation"))
            if _stage(req) == "validation"
            else None
        )

        return DraftedEvent(
            content=parsed.get("content"),
            variations=parsed.get("variations"),
            battlecard=parsed.get("battlecard"),
            outline=outline or None,
            citations=citations,
            sources=_sources_from_pages(ev.pages),
            themes=ev.themes,
            retrieval=ev.retrieval,
            warnings=list(ev.warnings),
            model_used=model,
            validation=validation,
        )

    @step
    async def verify(self, ctx: Context, ev: DraftedEvent) -> StopEvent:
        req: GenerateRequest = await ctx.store.get("request")
        pages: list[dict[str, Any]] = await ctx.store.get("pages", default=[])
        kept, dropped = verify_citations(ev.citations, ev.sources, pages or [])

        warnings = list(ev.warnings)
        evidence_warnings = [
            warning
            for warning in warnings
            if "RAG chunks" in warning or "page Markdown" in warning
        ]
        if dropped:
            warning = f"Dropped {dropped} citation(s) that failed Markdown/URL verify."
            warnings.append(warning)
            evidence_warnings.append(warning)
        has_output = bool(
            ev.content
            or ev.variations
            or ev.battlecard
            or ev.outline
            or ev.validation
        )
        if has_output and not kept:
            warning = "Generated output has no verified citations; treat factual claims as unsupported."
            warnings.append(warning)
            evidence_warnings.append(warning)

        evidence_ids = list(
            dict.fromkeys(source.page_id or source.url for source in ev.sources)
        )

        return StopEvent(
            result=GenerateResponse(
                intent=req.writing_intent,
                content=ev.content,
                variations=ev.variations,
                battlecard=ev.battlecard,
                outline=ev.outline,
                citations=kept,
                sources=ev.sources,
                themes=ev.themes or None,
                warnings=warnings,
                evidence_warnings=evidence_warnings,
                model_used=ev.model_used,
                retrieval=ev.retrieval,
                provenance=GenerateProvenance(
                    generation_stage=_stage(req),
                    model_used=ev.model_used,
                    model_policy_preset=req.model_policy_preset,
                    model_policy_version=(
                        req.model_policy_version
                        if req.model_policy_preset
                        else None
                    ),
                    prompt_version=_PROMPT_VERSION,
                    retrieval=ev.retrieval,
                    evidence_ids=evidence_ids,
                ),
                validation=ev.validation,
            )
        )

    async def _chat(self, model: str, system: str, user: str) -> str:
        # o1-pro is Responses-API-only. o3 uses the same transport so every
        # approved reasoning stage follows the recommended new-project path.
        if model.lower().startswith(("o1", "o3", "o4")):
            response = await self._openai.responses.create(
                model=model,
                instructions=system,
                input=user,
                reasoning={"effort": "high"},
                max_output_tokens=(
                    self._settings.openai_reasoning_max_completion_tokens
                ),
                store=False,
            )
            return _responses_output_text(response)

        # Historical standalone callers may still select a non-reasoning
        # configured model; preserve their Chat Completions JSON mode.
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }
        resp = await self._openai.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()


class GenerateService:
    def __init__(
        self,
        mongo: MongoCorpus,
        query: QueryService,
        settings: Settings,
    ) -> None:
        self._mongo = mongo
        self._query = query
        self._settings = settings

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        if not self._settings.generate_enabled:
            return GenerateResponse(
                intent=request.writing_intent,
                warnings=["Rag generate soft-disabled (GENERATE_ENABLED=false)."],
            )
        wf = CiteableGenerateWorkflow(
            mongo=self._mongo,
            query=self._query,
            settings=self._settings,
        )
        result = await wf.run(request=request)
        if isinstance(result, GenerateResponse):
            return result
        return GenerateResponse(
            intent=request.writing_intent,
            warnings=["Generate workflow returned unexpected result."],
        )


def _sources_from_pages(pages: list[dict[str, Any]]) -> list[GenerateSource]:
    out: list[GenerateSource] = []
    for p in pages:
        out.append(
            GenerateSource(
                url=str(p.get("url") or ""),
                title=p.get("title"),
                entity=p.get("entityName"),
                crawl_type=p.get("crawlType"),
                kind="page",
                page_id=p.get("pageId"),
            )
        )
    return out


def _brief_retrieval_context(req: GenerateRequest) -> str:
    brief = req.canonical_brief
    if brief is None:
        return ""
    parts: list[str] = []
    for label, value in (
        ("title", brief.title),
        ("content type", brief.content_type),
        ("keyword", brief.target_keyword),
        ("intent", brief.primary_intent),
        ("audience", brief.audience),
        ("buying stage", brief.buying_stage),
        ("tone", brief.tone_of_voice),
        ("brand", brief.brand_kit or brief.brand),
        ("PAA", brief.paa_questions),
        ("required topics", brief.required_topics),
        ("operator instructions", brief.operator_instructions),
        ("exclusions", brief.exclusions),
        ("hierarchy", brief.hierarchy),
        ("internal links", brief.internal_links),
        ("output requirements", brief.output_requirements),
        ("channel requirements", brief.channel_requirements),
        ("CTA requirements", brief.cta_requirements),
        ("conversion objective", brief.conversion_objective),
        ("publishing destination", brief.publishing_destination),
    ):
        if value:
            rendered = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            parts.append(f"{label}: {rendered}")
    return "; ".join(parts)[:4000]


def _build_prompts(
    req: GenerateRequest, family: str, pages: list[dict[str, Any]]
) -> tuple[str, str]:
    corpus_parts = []
    for i, p in enumerate(pages, 1):
        corpus_parts.append(
            f"### Source {i}\n"
            f"pageId: {p.get('pageId')}\n"
            f"url: {p.get('url')}\n"
            f"title: {p.get('title')}\n"
            f"section: {p.get('sectionTitle')}\n"
            f"crawlType: {p.get('crawlType')}\n"
            f"markdown:\n{p.get('markdown')}\n"
        )
    corpus = "\n".join(corpus_parts)

    system = (
        "You are a B2B content writer. Use ONLY the provided source markdown. "
        "Every factual claim must be supportable by a verbatim quote from a source. "
        "Return strict JSON. Do not invent URLs or quotes. "
        "Never use meta descriptions or marketing fluff as quotes when a concrete claim exists."
    )

    stage = _stage(req)
    if stage == "outline":
        shape = (
            '{"outline":[{"key":"stable-slug","heading":"section heading",'
            '"brief":"what this section must accomplish",'
            '"evidenceIds":["source pageId allocated to this section"]}],'
            '"citations":[{"pageId":"","url":"","title":"","sectionTitle":"",'
            '"quote":"verbatim span","crawlType":""}]}'
        )
    elif stage == "section":
        shape = (
            '{"content":"markdown for this section only",'
            '"citations":[{"pageId":"","url":"","title":"","sectionTitle":"",'
            '"quote":"verbatim span","crawlType":""}]}'
        )
    elif stage == "validation":
        shape = (
            '{"validation":{"approved":false,"issues":[{"sectionTitle":"",'
            '"category":"unsupportedClaim|sourceConflict|briefAlignment|brandVoice|'
            'originalityRepetition|usefulness|cta|seoGeo|contentTypeRequirements",'
            '"detail":"specific finding","repairInstruction":"actionable repair only"}],'
            '"strengths":["specific strength"],"unsupportedClaimCount":0,'
            '"briefAlignmentScore":0,"evidenceCoverageScore":0,"usefulnessScore":0,'
            '"originalityScore":0,"brandAlignmentScore":0},'
            '"citations":[{"pageId":"","url":"","title":"","sectionTitle":"",'
            '"quote":"verbatim span","crawlType":""}]}'
        )
    elif stage == "finalSynthesis":
        shape = (
            '{"content":"the complete editorially synthesized markdown document",'
            '"citations":[{"pageId":"","url":"","title":"","sectionTitle":"",'
            '"quote":"verbatim span","crawlType":""}]}'
        )
    else:
        shape = {
        "long": (
            '{"content":"markdown article","citations":[{"pageId":"","url":"","title":"",'
            '"sectionTitle":"","quote":"verbatim span","crawlType":""}]}'
        ),
        "short": (
            '{"variations":["ad1","ad2","ad3"],"citations":[{"pageId":"","url":"",'
            '"title":"","sectionTitle":"","quote":"verbatim span","crawlType":""}]}'
        ),
        "battlecard": (
            '{"battlecard":{"partnerSummary":"","competitorSummary":"",'
            '"differentiators":[],"risks":[]},'
            '"citations":[{"pageId":"","url":"","title":"","sectionTitle":"",'
            '"quote":"verbatim span","crawlType":""}]}'
        ),
        "slides": (
            '{"content":"markdown slide outline","citations":[{"pageId":"","url":"",'
            '"title":"","sectionTitle":"","quote":"verbatim span","crawlType":""}]}'
        ),
        }[family]

    templates = ""
    if family == "short" and req.ad_templates:
        bodies = "\n---\n".join(
            f"{t.name}: {t.body}" for t in req.ad_templates[:3] if t.body
        )
        if bodies:
            templates = f"\nFew-shot ad templates:\n{bodies}\n"

    stage_instructions = ""
    if stage == "outline":
        stage_instructions = (
            "\nGeneration stage: OUTLINE. Return 5–10 ordered sections. "
            "Each brief must define a distinct job and identify facts/evidence needed. "
            "Do not draft the article yet.\n"
        )
    elif stage == "section":
        outline = "\n".join(
            f"- {s.key}: {s.heading} — {s.brief}" for s in (req.outline or [])
        )
        completed = "\n".join(
            f"- {summary[:500]}" for summary in (req.completed_section_summaries or [])[:10]
        )
        stage_instructions = (
            "\nGeneration stage: SECTION. Draft only the requested section; do not repeat "
            "other sections, the requested heading, or a document title.\n"
            f"Requested key: {req.section_key or '(none)'}\n"
            f"Requested heading: {req.section_heading or '(none)'}\n"
            f"Section brief: {req.section_brief or '(none)'}\n"
            f"Full outline:\n{outline or '(none)'}\n"
            f"Previously completed section summaries:\n{completed or '(none)'}\n"
        )
    elif stage == "finalSynthesis":
        stage_instructions = (
            "\nGeneration stage: FINAL SYNTHESIS. Edit the entire supplied draft as one "
            "document for whole-document editorial coherence and exact alignment with "
            "every applicable canonical-brief requirement. Preserve every Markdown "
            "heading exactly, in the same order and at the same heading level. Improve "
            "transitions, organization, voice, originality, and usefulness; remove "
            "repetition without deleting unique supported substance. Preserve the exact "
            "meaning, numbers, product names, qualifications, and evidence behind every "
            "factual claim. Do not invent, infer, strengthen, or add facts, URLs, quotes, "
            "or citations. Every factual claim in the final document must remain "
            "supported by provided evidence and every citation quote must be copied "
            "verbatim from source Markdown. Set each citation sectionTitle to the exact "
            "Markdown heading of the section it supports. Return the complete document, not a summary "
            "or change list.\n"
            f"Current full draft (edit this complete document):\n{req.draft_content}\n"
        )
    elif stage == "validation":
        stage_instructions = (
            "\nGeneration stage: VALIDATION. Review the entire supplied draft against "
            "every applicable canonical-brief requirement and all loaded full-page "
            "evidence. Do not rewrite, edit, or return replacement content. Return only "
            "the structured validation and exact evidence citations. Inspect every "
            "section for unsupported claims, source conflicts, brief alignment, brand "
            "voice, originality or repetition, usefulness, CTA quality, SEO/GEO quality, "
            "and content-type requirements. Use exactly these issue category values: "
            "unsupportedClaim, sourceConflict, briefAlignment, brandVoice, "
            "originalityRepetition, usefulness, cta, seoGeo, contentTypeRequirements. "
            "Every issue must identify its section when possible, explain the concrete "
            "problem, and provide a repair instruction without performing the repair. "
            "Scores are numeric from 0 to 100. Any unsupported claim requires approved=false "
            "and must be counted in unsupportedClaimCount. Do not invent facts, conflicts, "
            "quotes, URLs, or citations; citation quotes must be verbatim source Markdown.\n"
            f"Current full draft (review this complete document):\n{req.draft_content}\n"
        )

    user = (
        f"Writing intent: {req.writing_intent}\n"
        f"Topic: {req.topic}\n"
        f"Entities: {', '.join(req.target_entities or []) or '(none)'}\n"
        f"Canonical brief ({req.canonical_brief.version if req.canonical_brief else 'none'}):\n"
        f"{json.dumps(req.canonical_brief.model_dump(by_alias=True, exclude_none=True), ensure_ascii=False, indent=2) if req.canonical_brief else '(none)'}\n"
        f"{templates}{stage_instructions}\n"
        f"Sources:\n{corpus}\n\n"
        f"JSON shape: {shape}\n"
        "Each citations[].quote MUST be copied verbatim from that source's markdown."
    )
    return system, user


def _parse_llm_json(raw: str, family: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Generate LLM returned non-JSON; wrapping as content")
        if family == "short":
            return {"variations": [raw], "citations": []}
        return {"content": raw, "citations": []}
    if not isinstance(data, dict):
        return {"content": raw, "citations": []}
    return data
