from __future__ import annotations

from datetime import UTC, datetime
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
    # Indexing no longer deletes, so there is no ranked deletion taxonomy to
    # report — one total of what this run could not use, beside the lang/empty
    # breakdown that already existed.
    pages_skipped_unusable: int = Field(0, alias="pagesSkippedUnusable")
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
    owner_id: str = Field("system:crawler", alias="ownerId", min_length=1)
    visibility: str = Field("service", min_length=1)
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
    point_id: str | None = Field(None, alias="pointId")
    chunk_id: str | None = Field(None, alias="chunkId")
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
    lexical_score: float | None = Field(None, alias="lexicalScore")
    source_digest: str | None = Field(None, alias="sourceDigest")
    parser_id: str | None = Field(None, alias="parserId")
    parser_version: str | None = Field(None, alias="parserVersion")
    chunker_id: str | None = Field(None, alias="chunkerId")
    chunker_version: str | None = Field(None, alias="chunkerVersion")
    embedding_model: str | None = Field(None, alias="embeddingModel")
    retrieval_policy_version: str | None = Field(None, alias="retrievalPolicyVersion")
    rank: int | None = None

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
    owner_id: str = Field("system:content-creator", alias="ownerId", min_length=1)
    visibility: str = Field("service", min_length=1)

    model_config = {"populate_by_name": True}


class AdTemplateIndexResponse(BaseModel):
    upserted: int
    warning: str | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class AdTemplateQueryRequest(BaseModel):
    need: str = Field(..., min_length=1)
    owner_id: str = Field("system:content-creator", alias="ownerId", min_length=1)
    visibility: str = Field("service", min_length=1)
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


class PageTextResponse(BaseModel):
    page_id: str = Field(..., alias="pageId")
    run_id: str = Field(..., alias="runId")
    url: str
    final_url: str = Field(..., alias="finalUrl")
    title: str | None = None
    # Plain text derived from the page's typed blocks — the same projection the
    # chunker embeds, so a quote taken from a chunk matches here.
    text: str
    excerpt: str | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class GenerateCitation(BaseModel):
    page_id: str | None = Field(None, alias="pageId")
    run_id: str | None = Field(None, alias="runId")
    url: str
    title: str | None = None
    section_title: str | None = Field(None, alias="sectionTitle")
    section_key: str | None = Field(None, alias="sectionKey")
    quote: str
    crawl_type: str | None = Field(None, alias="crawlType")
    source_digest: str | None = Field(
        None, alias="sourceDigest", pattern=r"^[0-9a-f]{64}$"
    )
    source_rights: str | None = Field(
        None,
        alias="sourceRights",
        pattern=r"^(consented|licensed|unknown|prohibited)$",
    )
    coordinates: dict[str, Any] | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class GenerateSource(BaseModel):
    url: str
    title: str | None = None
    entity: str | None = None
    crawl_type: str | None = Field(None, alias="crawlType")
    kind: str | None = None
    page_id: str | None = Field(None, alias="pageId")
    source_digest: str | None = Field(
        None, alias="sourceDigest", pattern=r"^[0-9a-f]{64}$"
    )

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}

class ProducerCapabilities(BaseModel):
    """Library capability advertisement. Generate versions/stages stay empty."""

    product: str = "geek-rag-library"
    features: list[str] = Field(
        default_factory=lambda: ["index", "query", "pages", "templates"],
    )
    execution_versions: list[str] = Field(
        default_factory=list,
        alias="executionVersions",
    )
    skill_envelope_versions: list[str] = Field(
        default_factory=list,
        alias="skillEnvelopeVersions",
    )
    generation_stages: list[str] = Field(
        default_factory=list,
        alias="generationStages",
    )
    agent_generation_stages: list[str] = Field(
        default_factory=list,
        alias="agentGenerationStages",
    )
    specialist_executors: list[str] = Field(
        default_factory=list,
        alias="specialistExecutors",
    )
    specialist_executor_version: str = Field("", alias="specialistExecutorVersion")
    tools_allowed: bool = Field(False, alias="toolsAllowed")
    agent_executor_versions: list[str] = Field(
        default_factory=list,
        alias="agentExecutorVersions",
    )
    agent_trace_versions: list[str] = Field(
        default_factory=list,
        alias="agentTraceVersions",
    )
    agent_tool_versions: list[str] = Field(
        default_factory=list,
        alias="agentToolVersions",
    )
    stage_scoped_tools_allowed: bool = Field(False, alias="stageScopedToolsAllowed")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


def utc_now() -> datetime:
    return datetime.now(UTC)


class HostIndexRequest(BaseModel):
    """URLs to check. Hosts are derived here; callers pass what the operator typed."""

    urls: list[str] = Field(..., min_length=1, max_length=100)

    model_config = {"populate_by_name": True}


class HostIndexResult(BaseModel):
    url: str
    host: str | None = None
    indexed: bool
    run_id: str | None = Field(None, alias="runId")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class HostIndexResponse(BaseModel):
    results: list[HostIndexResult]

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}
