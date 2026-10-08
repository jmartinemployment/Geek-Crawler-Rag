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
    # Chunks not written because an earlier page of the same run already carried the exact text.
    # Site chrome repeats on every page; a 2,025-page run held 7,046 such copies, and identical
    # vectors fill the dense candidate list before anything downstream can collapse them.
    chunks_skipped_repeat: int = Field(0, alias="chunksSkippedRepeat")
    attempt: int = 0
    trigger: str = "manual"
    embedding_rate_limit_retries: int = Field(0, alias="embeddingRateLimitRetries")
    embedding_wait_seconds: float = Field(0.0, alias="embeddingWaitSeconds")
    error: str | None = None
    started_at_utc: datetime | None = Field(None, alias="startedAtUtc")
    finished_at_utc: datetime | None = Field(None, alias="finishedAtUtc")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class IndexEnqueueResponse(IndexStatusResponse):
    """The POST /v1/index answer: the job's status, plus whether this call queued it.

    Separate from IndexStatusResponse because that model is what gets persisted to
    `rag_index_jobs`, and "was this particular call accepted" is not a property of a
    stored job -- it belongs to one request.

    Why it exists at all: claim(force=True) will not take a row that is already
    pending/running with a live lease, but `_enqueue` discarded its accepted flag, so a
    refused claim returned **HTTP 200 carrying the stale row** and queued nothing. Callers
    could not tell a fresh claim from a refusal, and on 2026-09-26 a re-post that looked
    successful (200, state=pending) had in fact done nothing -- the only tell was that
    `attempt` had not incremented. `accepted` is now stated rather than inferred.
    """

    accepted: bool = True


class IndexSchedulerStatus(BaseModel):
    """Scheduler cadence, and whether an operator has paused it.

    ``enabled`` and ``paused`` answer different questions and have different
    owners. ``enabled`` is configuration -- ``INDEX_SCHEDULER_ENABLED`` -- and
    ``scheduler_status`` re-syncs the stored value to the env var on every read,
    so nothing at runtime can hold a change to it. ``paused`` is the operator's,
    lives only in Mongo, and no config read touches it. Writing a pause into
    ``enabled`` would therefore be reverted by the next status call, which is
    why this is a separate field rather than a reused one.
    """

    enabled: bool
    interval_seconds: int = Field(..., alias="intervalSeconds")
    next_run_at_utc: datetime | None = Field(None, alias="nextRunAtUtc")
    last_enqueued_at_utc: datetime | None = Field(None, alias="lastEnqueuedAtUtc")
    last_run_id: str | None = Field(None, alias="lastRunId")
    last_error: str | None = Field(None, alias="lastError")
    last_selection_reason: str | None = Field(None, alias="lastSelectionReason")
    paused: bool = False
    paused_at_utc: datetime | None = Field(None, alias="pausedAtUtc")
    paused_reason: str | None = Field(None, alias="pausedReason")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class IndexKillRequest(BaseModel):
    """Why this job is being killed. Recorded as the job's error."""

    reason: str = Field(..., min_length=1, max_length=500)

    model_config = {"populate_by_name": True}


class IndexKillResponse(BaseModel):
    """The job's status plus what the kill actually did.

    ``outcome`` is carried explicitly because the status alone cannot distinguish a kill
    from a no-op, and because ``stopping`` is not ``failed`` yet -- a running job stops at
    its next checkpoint, so a caller that read only ``state`` would believe a still-running
    job was already dead.
    """

    outcome: str
    status: IndexStatusResponse | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class IndexIntakePauseRequest(BaseModel):
    """Why new index jobs are being refused."""

    reason: str = Field(..., min_length=1, max_length=500)

    model_config = {"populate_by_name": True}


class IndexIntakeStatus(BaseModel):
    """Whether new index jobs are being accepted, and why not."""

    paused: bool
    paused_reason: str | None = Field(None, alias="pausedReason")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class SchedulerPauseRequest(BaseModel):
    """Why the scheduler is being paused, so the next operator is not guessing."""

    reason: str = Field(..., min_length=1, max_length=500)

    model_config = {"populate_by_name": True}


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


class ChunkAnchor(BaseModel):
    """A link under the chunk's heading.

    The crawler stores these as {label, href} objects, not strings. Typing the field ``list[str]``
    made /v1/query return 500 for every chunk that had any -- validation failed before the response
    was built, so a change meant to expose more metadata took retrieval down instead.
    """

    label: str | None = None
    href: str | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


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
    # Stamped into the Qdrant payload by llama_nodes.py and, until now, never returned. The consumer
    # is therefore blind to parent/child structure: a retrieved child arrives as an isolated
    # fragment, even though the parent text it belongs to is sitting in the same payload and needs
    # no second fetch. anchors is the same loss for the crawler's link structure.
    parent_text: str | None = Field(None, alias="parentText")
    child_text: str | None = Field(None, alias="childText")
    anchors: list[ChunkAnchor] = Field(default_factory=list)
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


class QuoteCheck(BaseModel):
    page_id: str = Field(..., alias="pageId", min_length=1)
    quote: str = Field(..., min_length=1)

    model_config = {"populate_by_name": True}


class VerifyQuotesRequest(BaseModel):
    """`POST /v1/verify`. One quote is a list of one; there is no second shape."""

    run_id: str = Field(..., alias="runId", min_length=1)
    quotes: list[QuoteCheck] = Field(..., min_length=1, max_length=200)

    model_config = {"populate_by_name": True}


class QuoteVerdict(BaseModel):
    page_id: str = Field(..., alias="pageId")
    quote: str
    found: bool
    # None when found. Otherwise one of: "page_not_found" (no such page in this run, or no text),
    # "page_not_citable:<reason>" (the crawler's own signals condemn it), "not_on_page".
    reason: str | None = None
    # sha256 of the page's contentHtml (llama_nodes.page_source_digest), the value stamped on every
    # point cut from it. Identification only, never a verdict. Absent when the page was not read or
    # has no contentHtml.
    source_digest: str | None = Field(None, alias="sourceDigest")

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class VerifyQuotesResponse(BaseModel):
    run_id: str = Field(..., alias="runId")
    results: list[QuoteVerdict]

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
    """URLs to check. Hosts are derived here; callers pass what the operator typed.

    `crawlType` is the list the URLs came from (`partner`, `competitors`, the project site). A host
    is not a run: a site on two lists has two runs, and only the type says which one is wanted.
    Without it a host indexed under more than one type is refused as ambiguous, never guessed.
    """

    urls: list[str] = Field(..., min_length=1, max_length=100)
    crawl_type: str | None = Field(None, alias="crawlType")

    model_config = {"populate_by_name": True}


class HostIndexResult(BaseModel):
    url: str
    host: str | None = None
    indexed: bool
    run_id: str | None = Field(None, alias="runId")
    # The crawl type of the run answered, so the caller can see it got the run it asked for.
    crawl_type: str | None = Field(None, alias="crawlType")
    # Why `indexed` is false when the host has points but the answer is withheld.
    reason: str | None = None

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}


class HostIndexResponse(BaseModel):
    results: list[HostIndexResult]

    model_config = {"populate_by_name": True, "ser_json_by_alias": True}
