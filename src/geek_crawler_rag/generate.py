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
    GenerateRequest,
    GenerateResponse,
    GenerateSource,
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
    citations: list[GenerateCitation]
    sources: list[GenerateSource]
    themes: list[ThemeHit]
    retrieval: str
    warnings: list[str]
    model_used: str


def _family(intent: str) -> str:
    key = intent.strip().lower()
    if key in _SHORT_FORM:
        return "short"
    if key in _BATTLECARD:
        return "battlecard"
    if key in _SLIDES:
        return "slides"
    return "long"


def _normalize_ws(text: str) -> str:
    return _WS.sub(" ", (text or "").strip()).lower()


def quote_in_markdown(quote: str, markdown: str) -> bool:
    """True when quote (normalized) appears in markdown."""
    q = _normalize_ws(quote)
    if len(q) < 12:
        return False
    body = _normalize_ws(markdown)
    if q in body:
        return True
    # Allow slight truncation: require 80% contiguous match of first 80 chars of quote.
    probe = q[:80]
    return len(probe) >= 12 and probe in body


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

        if not chunks:
            warnings.append("No RAG chunks for partner/competitor runs.")

        return RetrievedEvent(
            chunks=chunks,
            themes=themes,
            retrieval=retrieval_mode or (retrieval_labels[0] if retrieval_labels else "hybrid"),
            warnings=warnings,
        )

    @step
    async def load_pages(self, ctx: Context, ev: RetrievedEvent) -> PagesLoadedEvent:
        max_pages = self._settings.generate_max_pages
        max_chars = self._settings.generate_markdown_chars
        seen: set[str] = set()
        pages: list[dict[str, Any]] = []

        # Prefer unique pages by pageId, then url.
        ordered = sorted(ev.chunks, key=lambda c: c.score, reverse=True)
        for hit in ordered:
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
            if len(pages) >= max_pages:
                break

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
        model = (
            self._settings.openai_longform_model
            if family == "long"
            else self._settings.openai_standard_model
        )

        if not self._settings.openai_api_key:
            return DraftedEvent(
                content=None,
                variations=None,
                battlecard=None,
                citations=[],
                sources=_sources_from_pages(ev.pages),
                themes=ev.themes,
                retrieval=ev.retrieval,
                warnings=ev.warnings + ["OPENAI_API_KEY unset; generate skipped."],
                model_used=model,
            )

        if not ev.pages:
            return DraftedEvent(
                content=None,
                variations=None,
                battlecard=None,
                citations=[],
                sources=[],
                themes=ev.themes,
                retrieval=ev.retrieval,
                warnings=ev.warnings,
                model_used=model,
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

        return DraftedEvent(
            content=parsed.get("content"),
            variations=parsed.get("variations"),
            battlecard=parsed.get("battlecard"),
            citations=citations,
            sources=_sources_from_pages(ev.pages),
            themes=ev.themes,
            retrieval=ev.retrieval,
            warnings=list(ev.warnings),
            model_used=model,
        )

    @step
    async def verify(self, ctx: Context, ev: DraftedEvent) -> StopEvent:
        req: GenerateRequest = await ctx.store.get("request")
        pages: list[dict[str, Any]] = await ctx.store.get("pages", default=[])
        md_by_url: dict[str, str] = {}
        for p in pages or []:
            url = str(p.get("url") or "").lower()
            md = str(p.get("markdown") or "")
            if url and md:
                md_by_url[url] = md

        kept: list[GenerateCitation] = []
        dropped = 0
        allowed_urls = {s.url.lower() for s in ev.sources if s.url}
        for cite in ev.citations:
            url_key = cite.url.lower()
            if allowed_urls and url_key not in allowed_urls:
                dropped += 1
                continue
            body = md_by_url.get(url_key) or ""
            if not body or not quote_in_markdown(cite.quote, body):
                dropped += 1
                continue
            kept.append(cite)

        warnings = list(ev.warnings)
        if dropped:
            warnings.append(f"Dropped {dropped} citation(s) that failed Markdown/URL verify.")

        return StopEvent(
            result=GenerateResponse(
                intent=req.writing_intent,
                content=ev.content,
                variations=ev.variations,
                battlecard=ev.battlecard,
                citations=kept,
                sources=ev.sources,
                themes=ev.themes or None,
                warnings=warnings,
                model_used=ev.model_used,
                retrieval=ev.retrieval,
            )
        )

    async def _chat(self, model: str, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Reasoning models often reject temperature.
        if not model.lower().startswith(("o1", "o3", "o4")):
            kwargs["temperature"] = 0.3
            kwargs["response_format"] = {"type": "json_object"}
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

    user = (
        f"Writing intent: {req.writing_intent}\n"
        f"Topic: {req.topic}\n"
        f"Entities: {', '.join(req.target_entities or []) or '(none)'}\n"
        f"{templates}\n"
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
