from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


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
    error: str | None = None
    started_at_utc: datetime | None = Field(None, alias="startedAtUtc")
    finished_at_utc: datetime | None = Field(None, alias="finishedAtUtc")

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

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


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

    model_config = {"populate_by_name": True}


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
    model_used: str | None = Field(None, alias="modelUsed")
    retrieval: str | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
