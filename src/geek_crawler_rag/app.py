"""Geek-Crawler-Rag HTTP API — FastAPI shell over LlamaIndex ingest/query."""

from __future__ import annotations

import logging
import secrets
from urllib.parse import urlsplit
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi import (
    Query as ApiQuery,
)
from fastapi.responses import JSONResponse

from geek_crawler_rag.ad_templates import AdTemplateIndexService, normalize_upsert_items
from geek_crawler_rag.asset_context import AssetContextService
from geek_crawler_rag.config import Settings, get_settings
from geek_crawler_rag.context_models import (
    AssetDeleteRequest,
    AssetIndexRequest,
    AssetIndexResponse,
    ManifestQueryRequest,
    ManifestQueryResponse,
    RuntimeManifestQueryRequest,
    TrustedAssetDeleteRequest,
    TrustedAssetIndexRequest,
)
from geek_crawler_rag.diagnostic_models import (
    EntityMapArtifact,
    EntityMapRequest,
    FactDensityArtifact,
    FactDensityRequest,
    ReadinessScoreArtifact,
    ReadinessScoreRequest,
    SchemaMarkupArtifact,
    SchemaMarkupRequest,
)
from geek_crawler_rag.diagnostics import DiagnosticService
from geek_crawler_rag.indexer import IndexService
from geek_crawler_rag.intelligence import IntelligenceService
from geek_crawler_rag.intelligence_models import (
    CompetitorAuditArtifact,
    CompetitorAuditRequest,
    CompetitorPageAnalysisArtifact,
    CompetitorPageAnalysisRequest,
    CompetitorPositioningArtifact,
    CompetitorPositioningRequest,
    ContentGapArtifact,
    ContentGapRequest,
    QueryPlanArtifact,
    QueryPlannerRequest,
    ReadinessComparisonArtifact,
    ReadinessComparisonRequest,
)
from geek_crawler_rag.llama_engine import LlamaIndexEngine
from geek_crawler_rag.block_text import derive_plaintext_from_blocks
from geek_crawler_rag.models import (
    AdTemplateIndexRequest,
    AdTemplateIndexResponse,
    AdTemplateQueryRequest,
    AdTemplateQueryResponse,
    HostIndexRequest,
    HostIndexResponse,
    HostIndexResult,
    IndexIntakePauseRequest,
    IndexIntakeStatus,
    IndexKillRequest,
    IndexKillResponse,
    IndexRunRequest,
    IndexSchedulerStatus,
    IndexEnqueueResponse,
    IndexStatusResponse,
    PageTextResponse,
    ProducerCapabilities,
    QueryRequest,
    QueryResponse,
    SchedulerPauseRequest,
)
from geek_crawler_rag.mongo import MongoCorpus
from geek_crawler_rag.qdrant_store import QdrantStore
from geek_crawler_rag.unusable import classify_unusable_page
from geek_crawler_rag.query import QueryService
from geek_crawler_rag.rerank import Reranker
from geek_crawler_rag.scheduler import IndexScheduler
from geek_crawler_rag.vector_cache import VectorCache
from geek_crawler_rag.status_store import IndexStatusStore
from geek_crawler_rag.webhook import IndexStatusWebhook

logger = logging.getLogger(__name__)


class AppState:
    settings: Settings
    mongo: MongoCorpus
    store: QdrantStore
    llama: LlamaIndexEngine
    indexer: IndexService
    status_store: IndexStatusStore
    query: QueryService
    templates: AdTemplateIndexService
    webhook: IndexStatusWebhook
    scheduler: IndexScheduler
    assets: AssetContextService
    diagnostics: DiagnosticService
    intelligence: IntelligenceService


state = AppState()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    if not settings.api_key and not settings.local_test_mode:
        raise RuntimeError(
            "API_KEY is required unless LOCAL_TEST_MODE=true is explicitly configured."
        )
    state.settings = settings
    state.mongo = MongoCorpus(settings.mongo_crawler_url, settings.mongo_db_name)
    state.store = QdrantStore(
        settings.qdrant_url,
        collection=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
        vector_size=settings.embedding_dimensions,
    )
    state.llama = LlamaIndexEngine(settings)
    # Cross-run embedding reuse. Wired after Mongo because the cache lives
    # there; absent it, indexing behaves exactly as it did before.
    vector_cache = VectorCache(
        state.mongo.db,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    await vector_cache.ensure_indexes()
    state.llama.set_vector_cache(vector_cache)
    status_store = IndexStatusStore(state.mongo.db)
    state.status_store = status_store
    state.webhook = IndexStatusWebhook(
        settings.index_status_webhook_url,
        settings.index_status_webhook_key or settings.api_key,
    )
    state.indexer = IndexService(
        state.mongo,
        state.store,
        settings,
        llama=state.llama,
        status_store=status_store,
        webhook=state.webhook,
    )
    reranker = Reranker(
        settings.cohere_api_key,
        model=settings.cohere_rerank_model,
        enabled=settings.rerank_enabled,
    )
    state.query = QueryService(
        state.store, settings, llama=state.llama, reranker=reranker
    )
    state.templates = AdTemplateIndexService(settings, state.llama)
    state.assets = AssetContextService(state.store, state.llama, settings)
    state.diagnostics = DiagnosticService()
    state.intelligence = IntelligenceService(state.diagnostics)
    state.scheduler = IndexScheduler(
        state.mongo,
        status_store,
        state.indexer,
        settings,
        owner=state.indexer.owner,
    )
    await state.mongo.ensure_indexes()
    await state.store.ensure_collection()
    await state.templates.ensure_collection()
    await state.indexer.start()
    await state.scheduler.start()
    logger.info(
        "Geek-Crawler-Rag listening (collection=%s templates=%s engine=LlamaIndex webhook=%s rerank=%s)",
        settings.qdrant_collection,
        settings.qdrant_ad_templates_collection,
        "on" if state.webhook.enabled else "off",
        "on" if reranker.enabled else "off",
    )
    yield
    await state.scheduler.stop()
    await state.indexer.stop()
    await state.webhook.close()
    await state.templates.close()
    await state.llama.close()
    await state.store.close()
    await state.mongo.close()


app = FastAPI(
    title="Geek-Crawler-Rag",
    description=(
        "Index and query Geek-Crawler Mongo pages via LlamaIndex + Qdrant "
        "(English only; parent/child chunks; hybrid + optional Cohere rerank; "
        "graph themes + ad-template index)."
    ),
    version="0.5.0",
    lifespan=lifespan,
)


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-Api-Key")] = None,
) -> None:
    expected = settings.api_key
    if not expected:
        if settings.local_test_mode:
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service authentication is not configured.",
        )
    supplied = x_api_key or ""
    if not secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Api-Key",
        )


@app.exception_handler(Exception)
async def redact_unhandled_error(request: Request, error: Exception) -> JSONResponse:
    logger.exception("Unhandled API error path=%s", request.url.path, exc_info=error)
    return JSONResponse(
        {"detail": "Internal service error."},
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


@app.get("/health")
async def health() -> JSONResponse:
    mongo_ok = False
    qdrant_ok = False
    scheduler_status = None
    errors: list[str] = []
    try:
        mongo_ok = await state.mongo.ping()
    except Exception as ex:
        logger.warning("Mongo health check failed", exc_info=ex)
        errors.append("mongo unavailable")
    collection_present: bool | None = None
    try:
        qdrant_ok = await state.store.ping()
    except Exception as ex:
        logger.warning("Qdrant health check failed", exc_info=ex)
        errors.append("qdrant unavailable")
    if qdrant_ok:
        # Reachable is not the same as present. Startup calls ensure_collection, so a missing
        # collection here means something dropped it under a running process — which happened
        # on 2026-09-24 — and every query now returns nothing while rag_index_jobs still
        # reports runs complete with their old chunk counts.
        collection_present = await state.store.collection_present()
        if collection_present is False:
            errors.append(
                f"qdrant collection {state.settings.qdrant_collection} is MISSING — "
                "the corpus is empty and every query will return nothing; re-index is required"
            )
    if mongo_ok:
        try:
            scheduler_status = (await state.scheduler.status()).model_dump(
                by_alias=True, mode="json"
            )
        except Exception as ex:
            logger.warning("Scheduler health check failed", exc_info=ex)
            errors.append("scheduler unavailable")

    # `status` stays a reachability verdict and a missing collection does not flip it, on
    # purpose. The container healthcheck urlopen()s this endpoint, so 503 marks the container
    # unhealthy — and the thing that recreates the collection is an index job served by this
    # same container. Failing health would block the only path that heals it. The condition
    # travels in `errors`, which is the field for exactly this.
    healthy = mongo_ok and qdrant_ok
    body = {
        "status": "ok" if healthy else "degraded",
        "mongo": mongo_ok,
        "qdrant": qdrant_ok,
        "qdrantCollection": collection_present,
        "engine": "llamaindex",
        "features": ["hybrid", "graph", "ad-templates", "pages"],
        "embeddingThrottle": state.llama.embedding_stats(),
        "scheduler": scheduler_status,
        "errors": errors or None,
    }
    return JSONResponse(body, status_code=200 if healthy else 503)


@app.get(
    "/v1/capabilities",
    response_model=ProducerCapabilities,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def capabilities() -> ProducerCapabilities:
    """Declare strict generation and skill-envelope versions supported by this producer."""
    return ProducerCapabilities()


@app.post(
    "/v1/index",
    response_model=IndexEnqueueResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def start_index(body: IndexRunRequest) -> IndexEnqueueResponse:
    """Enqueue a full-run English index (idempotent rebuild per runId).

    Answers with `accepted` so a refusal cannot pass for success. A run already
    pending/running under a live lease is refused, and that is correct -- but it used to
    look identical to a fresh claim.
    """
    # The gate, before the claim. A paused intake must leave no job row at all: creating one
    # and refusing to run it would be a job that never reaches a terminal state, which is
    # exactly what the job rules forbid.
    #
    # 503 rather than a quiet 200: GeekAPI's EnqueueIndexAsync fails closed on a non-2xx --
    # it returns null and logs -- so the crawl still closes healthy and the loss is visible.
    # Answering 200 with accepted=false would be read as "queued" by anything that checks
    # only the status code, and the run would be silently unindexed.
    intake_paused = await state.indexer.intake_pause_reason()
    if intake_paused is not None:
        logger.warning(
            "Refused index enqueue for runId=%s: intake paused (%s)",
            body.run_id,
            intake_paused,
        )
        raise HTTPException(
            # 503 literal: the name `status` is shadowed by this function's local
            # `status, accepted = ...` unpack, so fastapi.status is unreachable here.
            status_code=503,
            detail=(
                f"Index intake is paused: {intake_paused}. "
                f"Nothing was queued for runId={body.run_id}. "
                "Resume with POST /v1/index-intake/resume, then re-post this run."
            ),
        )

    status, accepted = await state.indexer.enqueue(body.run_id)
    if not accepted:
        logger.warning(
            "Index enqueue REFUSED for runId=%s: state=%s already holds the lease, "
            "nothing was queued (attempt stays %s)",
            status.run_id,
            status.state,
            status.attempt,
        )
    return IndexEnqueueResponse(**status.model_dump(), accepted=accepted)


@app.get(
    "/v1/index/{run_id}",
    response_model=IndexStatusResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def index_status(run_id: str) -> IndexStatusResponse:
    status_row = await state.indexer.get_status(run_id)
    if status_row is None:
        raise HTTPException(status_code=404, detail=f"No index job for runId={run_id}")
    return status_row


@app.post(
    "/v1/index/{run_id}/kill",
    response_model=IndexKillResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def kill_index_job(run_id: str, body: IndexKillRequest) -> IndexKillResponse:
    """Kill one index job, leaving every other job running.

    Until now the only way to stop an index job was to stop the process, which took every
    queued job with it: on 2026-09-28 stopping one run killed fourteen, thirteen of which
    had not started. This kills exactly one.

    ``outcome`` says what happened, and the status code follows it:

    * ``dequeued`` -- 200. It was queued and never started; it is ``failed`` now and the
      worker will skip it. No vectors were written.
    * ``stopping`` -- 200. It is running. The run stops at its next page or embed-batch
      boundary, so it is **not** terminal when this returns; poll
      ``GET /v1/index/{run_id}`` for ``failed``. Latency is one page or one batch, not
      instant, because the kill is cooperative rather than a task cancellation -- see
      ``JobKilled``.
    * ``already_terminal`` -- 409. It had already finished. Nothing was killed, and
      answering 200 would tell an operator they stopped a run that in the ``complete``
      case is live and citable.
    * ``absent`` -- 404. No job row for this run.

    Points already upserted are kept, exactly as on quarantine. Discarding a partial index
    is ``DELETE /v1/index/runs/{run_id}``, deliberately a separate call: "stop this" must
    not silently also mean "destroy what it built".
    """
    status_row, outcome = await state.indexer.kill(run_id, reason=body.reason)
    if outcome == "absent":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No index job for runId={run_id}",
        )
    if outcome == "already_terminal":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Index job for runId={run_id} is already "
                f"{status_row.state if status_row else 'terminal'}; nothing was killed."
            ),
        )
    return IndexKillResponse(outcome=outcome, status=status_row)


@app.get(
    "/v1/index-intake",
    response_model=IndexIntakeStatus,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def index_intake_status() -> IndexIntakeStatus:
    """Whether new index jobs are being accepted, and the reason if not."""
    reason = await state.indexer.intake_pause_reason()
    return IndexIntakeStatus(paused=reason is not None, paused_reason=reason)


@app.post(
    "/v1/index-intake/pause",
    response_model=IndexIntakeStatus,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def pause_index_intake(body: IndexIntakePauseRequest) -> IndexIntakeStatus:
    """Refuse all new index jobs until resumed. The emergency brake.

    This is the one that stops the traffic that actually floods the queue. Every job in
    flight on 2026-09-28 was ``trigger=manual`` -- GeekAPI posting a run as each crawl
    committed -- so the scheduler pause, which deliberately leaves ``POST /v1/index``
    working, would have stopped none of them.

    Paused intake refuses **both** entrances: the manual/GeekAPI route and the scheduler's
    own enqueue. No job row is created for a refused run, so nothing is left ``pending``
    with no worker that will ever take it.

    The trade, stated rather than discovered later: GeekAPI's ``EnqueueIndexAsync`` fails
    closed on the 503, so crawls will keep completing **unindexed** while this is on, each
    one logging the refusal. Those runs need a deliberate re-post afterwards -- nothing
    re-drives them, by design.

    It does not stop work already queued or running. That is
    ``POST /v1/index/{runId}/kill``, one job at a time.
    """
    paused, reason = await state.status_store.set_intake_paused(
        paused=True, reason=body.reason
    )
    return IndexIntakeStatus(paused=paused, paused_reason=reason)


@app.post(
    "/v1/index-intake/resume",
    response_model=IndexIntakeStatus,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def resume_index_intake() -> IndexIntakeStatus:
    """Accept new index jobs again.

    Runs refused while paused are **not** replayed -- no row was created for them, and
    nothing re-drives a run this service never accepted. Re-post them deliberately, or use
    ``scripts/list_unindexed_runs.py`` to find content-ready runs with no job.
    """
    paused, reason = await state.status_store.set_intake_paused(
        paused=False, reason=None
    )
    return IndexIntakeStatus(paused=paused, paused_reason=reason)


@app.get(
    "/v1/index-scheduler",
    response_model=IndexSchedulerStatus,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def index_scheduler_status() -> IndexSchedulerStatus:
    """Return persisted scheduler cadence, most recent enqueue, and pause state."""
    return await state.scheduler.status()


@app.post(
    "/v1/index-scheduler/pause",
    response_model=IndexSchedulerStatus,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def pause_index_scheduler(body: SchedulerPauseRequest) -> IndexSchedulerStatus:
    """Stop the scheduler enqueuing new runs, until an explicit resume.

    What this does and does not stop, because the difference is the whole point:

    * Stopped -- the scheduler claiming its next due tick, so no *further* run is
      enqueued. Durable in Mongo, so a deploy that recreates the container does not
      lift it.
    * Not stopped -- the job already running, and anything already in the worker
      queue. Those hold their own leases and finish. Pausing the feed is the safe
      operation; killing live indexing is not, and is what recreating the container
      already does by accident.
    * Also not stopped -- `POST /v1/index`. A manual enqueue is someone deciding to
      index one run right now, and this pauses the automatic feed, not the operator.
      Read the name narrowly: it is the scheduler that is paused, not indexing.

    A reason is required. A pause with no reason is a pause the next operator cannot
    judge, and the state is durable enough to outlive whoever set it.
    """
    return await state.scheduler.pause(body.reason)


@app.post(
    "/v1/index-scheduler/resume",
    response_model=IndexSchedulerStatus,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def resume_index_scheduler() -> IndexSchedulerStatus:
    """Let the scheduler claim due ticks again, on its existing cadence.

    `nextRunAtUtc` is untouched, so resuming does not itself trigger an enqueue --
    a resume that fired a tick immediately would make "unpause" and "index now"
    the same button, and they are not.
    """
    return await state.scheduler.resume()


@app.post(
    "/v1/query",
    response_model=QueryResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def query(body: QueryRequest) -> QueryResponse:
    """Hybrid retrieve via LlamaIndex dense + BM25/text RRF (+ optional rerank).

    Pass ``retrievalMode: graph`` for Phase D1 theme/relationship overlay.
    """
    return await state.query.query(body)


@app.post(
    "/v1/templates/index",
    response_model=AdTemplateIndexResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def index_templates(body: AdTemplateIndexRequest) -> AdTemplateIndexResponse:
    """Upsert few-shot ad templates (owned by content-creator-v2)."""
    cleaned = normalize_upsert_items(body.templates)
    return await state.templates.index(
        AdTemplateIndexRequest(
            templates=cleaned,
            ownerId=body.owner_id,
            visibility=body.visibility,
        )
    )


@app.post(
    "/v1/templates/query",
    response_model=AdTemplateQueryResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def query_templates(body: AdTemplateQueryRequest) -> AdTemplateQueryResponse:
    """Retrieve top ad-template exemplars for short-form few-shot generate."""
    return await state.templates.query(body)


@app.get(
    "/v1/pages/{page_id}",
    response_model=PageTextResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def get_page_text(
    page_id: str,
    run_id: Annotated[str, ApiQuery(alias="runId", min_length=1)],
) -> PageTextResponse:
    """Return the page's plaintext projection for citation reads (404 if empty or unusable).

    This is the read side of citation verification: GeekAPI's
    `GccV2PartnerExtractionVerify.VerifyAgainstLibraryAsync` fetches text here and matches a
    model's quote against it, then stamps the citation verified.

    So an unusable page must 404 here, not just be excluded at index time. Until 2026-09-29
    the only checks were "does the page exist, does the run match, is the text non-empty" --
    and a 4xx error page passes all three. Its body ("Sorry, we could not find that page")
    clears every prose floor the pipeline has, so a page indexed before the reject gate
    existed could be retrieved by /v1/query, quoted, confirmed against this endpoint, and
    stamped QuoteVerified=true on a citation to a URL the server said it did not serve.

    Index-time rejection does not cover it, for two reasons: points already in Qdrant carry
    no status in their payload, so retrieval cannot filter them out; and this endpoint reads
    Mongo directly, so it never consults the index at all.

    `classify_unusable_page` is the one classifier, reused rather than re-deriving a status
    check here -- the whole defect it now closes was two readers of one field disagreeing.
    """
    page = await state.mongo.get_page(page_id)
    text = derive_plaintext_from_blocks(page.blocks) if page is not None else ""
    if page is None or page.run_id != run_id or not text:
        raise HTTPException(
            status_code=404,
            detail="No text for the authorized page.",
        )
    reject = classify_unusable_page(
        url=page.url,
        final_url=page.final_url,
        failure_reason=page.failure_reason,
        robots_allowed=page.robots_allowed,
        blocks=page.blocks,
        status_code=page.status_code,
    )
    if reject is not None and reject != "no_content":
        # no_content is excluded: an empty body is already covered by `not text` above, and
        # this service re-adjudicating "readable" is what destroyed 5,274 pages on
        # 2026-09-18. What is refused here is a page the CRAWLER's own signals condemn.
        logger.warning(
            "Refusing citation read for unusable page pageId=%s runId=%s reason=%s status=%s",
            page.id,
            page.run_id,
            reject,
            page.status_code,
        )
        raise HTTPException(
            status_code=404,
            detail=f"Page is not citable ({reject}).",
        )
    return PageTextResponse(
        page_id=page.id,
        run_id=page.run_id,
        url=page.url,
        final_url=page.final_url or page.url,
        title=page.title,
        text=text,
        excerpt=None,
    )


@app.get(
    "/v1/pages",
    response_model=PageTextResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def get_page_text_by_url(
    run_id: Annotated[str, ApiQuery(alias="runId", min_length=1)],
    url: Annotated[str, ApiQuery(min_length=1)],
) -> PageTextResponse:
    """Lookup a page's plaintext projection by runId + Url/FinalUrl.

    `runId` is aliased, matching the sibling handler and README.md. It was a bare `run_id`,
    so the documented `GET /v1/pages?runId=...&url=...` answered 422 -- the endpoint could
    not be called the way its own contract specifies. Nothing noticed because GeekAPI only
    ever calls the by-id form (HttpGeekCrawlerRagClient) and so does
    scripts/staging_citation_smoke.py, which is also why the missing status gate below
    survived here: this adapter has never been exercised in production.

    Same unusable-page refusal as the by-id handler, for the same reason.
    """
    page = await state.mongo.get_page_by_url(run_id=run_id, url=url)
    text = derive_plaintext_from_blocks(page.blocks) if page is not None else ""
    if page is None or not text:
        raise HTTPException(
            status_code=404,
            detail=f"No text for runId={run_id} url={url}",
        )
    reject = classify_unusable_page(
        url=page.url,
        final_url=page.final_url,
        failure_reason=page.failure_reason,
        robots_allowed=page.robots_allowed,
        blocks=page.blocks,
        status_code=page.status_code,
    )
    if reject is not None and reject != "no_content":
        logger.warning(
            "Refusing citation read for unusable page pageId=%s runId=%s reason=%s status=%s",
            page.id,
            page.run_id,
            reject,
            page.status_code,
        )
        raise HTTPException(
            status_code=404,
            detail=f"Page is not citable ({reject}).",
        )
    return PageTextResponse(
        page_id=page.id,
        run_id=page.run_id,
        url=page.url,
        final_url=page.final_url or page.url,
        title=page.title,
        text=text,
        excerpt=None,
    )


@app.post(
    "/v1/diagnostics/readiness-score",
    response_model=ReadinessScoreArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def readiness_score(body: ReadinessScoreRequest) -> ReadinessScoreArtifact:
    """Run the versioned seven-dimension heuristic without external side effects."""
    return state.diagnostics.readiness_score(body)


@app.post(
    "/v1/diagnostics/fact-density",
    response_model=FactDensityArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def fact_density(body: FactDensityRequest) -> FactDensityArtifact:
    """Classify section-level fact density and exact supplied evidence support."""
    return state.diagnostics.fact_density(body)


@app.post(
    "/v1/diagnostics/entity-map",
    response_model=EntityMapArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def entity_map(body: EntityMapRequest) -> EntityMapArtifact:
    """Build a canonical evidence-linked entity/co-occurrence map."""
    return state.diagnostics.entity_map(body)


@app.post(
    "/v1/diagnostics/schema-markup",
    response_model=SchemaMarkupArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def schema_markup(body: SchemaMarkupRequest) -> SchemaMarkupArtifact:
    """Generate and validate JSON-LD only from supplied visible content."""
    return state.diagnostics.schema_markup(body)


@app.post(
    "/v1/intelligence/query-plan",
    response_model=QueryPlanArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def query_plan(body: QueryPlannerRequest) -> QueryPlanArtifact:
    """Plan sourced queries and explicitly labeled deterministic hypotheses."""
    return state.intelligence.query_plan(body)


@app.post(
    "/v1/intelligence/competitor-page",
    response_model=CompetitorPageAnalysisArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def competitor_page_analysis(
    body: CompetitorPageAnalysisRequest,
) -> CompetitorPageAnalysisArtifact:
    """Analyze one supplied competitor page without external enrichment."""
    return state.intelligence.competitor_page_analysis(body)


@app.post(
    "/v1/intelligence/content-gap",
    response_model=ContentGapArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def content_gap(body: ContentGapRequest) -> ContentGapArtifact:
    """Compare supplied subject and competitor documents with exact evidence."""
    return state.intelligence.content_gap(body)


@app.post(
    "/v1/intelligence/readiness-comparison",
    response_model=ReadinessComparisonArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def readiness_comparison(
    body: ReadinessComparisonRequest,
) -> ReadinessComparisonArtifact:
    """Compare subject and competitor pages under ai-readiness-heuristic.v1."""
    return state.intelligence.readiness_comparison(body)


@app.post(
    "/v1/intelligence/competitor-audit",
    response_model=CompetitorAuditArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def competitor_audit(body: CompetitorAuditRequest) -> CompetitorAuditArtifact:
    """Compose page analyses, content gaps, and prioritized actions from supplied docs."""
    return state.intelligence.competitor_audit(body)


@app.post(
    "/v1/intelligence/competitor-positioning",
    response_model=CompetitorPositioningArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def competitor_positioning(
    body: CompetitorPositioningRequest,
) -> CompetitorPositioningArtifact:
    """Map narrative attributes and labeled hypotheses without claiming market perception."""
    return state.intelligence.competitor_positioning(body)



@app.post(
    "/v1/assets/index",
    response_model=AssetIndexResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def index_asset(body: AssetIndexRequest) -> AssetIndexResponse:
    """Index one immutable Knowledge resource; GeekRepository remains authoritative."""
    return await state.assets.index(body)


@app.post(
    "/v1/context/assets/index",
    response_model=AssetIndexResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def index_trusted_asset(body: TrustedAssetIndexRequest) -> AssetIndexResponse:
    """Index one GeekAPI-authoritative revision over the authenticated service boundary."""
    if not state.settings.context_asset_indexing_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return await state.assets.index_trusted(body)


@app.post(
    "/v1/context/assets/delete",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_api_key)],
)
async def delete_trusted_asset(body: TrustedAssetDeleteRequest) -> None:
    """Tombstone one exact owner-scoped resource after authoritative revocation."""
    if not state.settings.context_asset_indexing_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    await state.assets.delete_trusted(body)


@app.post(
    "/v1/context/assets/query",
    response_model=ManifestQueryResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def query_runtime_assets(
    body: RuntimeManifestQueryRequest,
) -> ManifestQueryResponse:
    """Retrieve only revisions authorized by GeekAPI's persisted signed manifest."""
    if not state.settings.context_manifest_query_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return await state.assets.query_runtime(body)


@app.post(
    "/v1/assets/delete",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_api_key)],
)
async def delete_asset(body: AssetDeleteRequest) -> None:
    """Delete only an exact revision authorized by a verified manifest entry."""
    await state.assets.delete(body)


@app.post(
    "/v1/assets/query",
    response_model=ManifestQueryResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def query_assets(body: ManifestQueryRequest) -> ManifestQueryResponse:
    """Query only exact entries from a digest- and signature-verified manifest."""
    return await state.assets.query(body)


@app.delete(
    "/v1/index/runs/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_api_key)],
)
async def delete_run_index(run_id: str) -> None:
    """Remove every crawler-owned vector for one run, and its job row.

    Called by GeekAPI as the first step of a run delete: vectors go before the
    pages they cite, so retrieval can never return a chunk whose source no
    longer exists.

    Absence is a no-op in all three of its forms, and each is enforced rather than
    asserted: no points for the run (a filter delete matching nothing succeeds), no
    collection at all (`delete_by_run_id` catches it), and no job row
    (`status_store.delete` returns False rather than raising). A real failure -- an
    unreachable Qdrant, a timeout -- still raises and still becomes a non-2xx, because
    GeekAPI deletes the cited pages once this returns success.

    The job row goes with them. Purging vectors while leaving it behind meant
    GET /v1/index/{runId} kept reporting ``complete`` with the page and chunk
    counts of a corpus that had been deleted, and every caller that trusts that
    status — the indexed-runs report among them — repeated those numbers.
    """
    if not run_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    await state.store.delete_by_run_id(
        run_id,
        owner_id=state.settings.crawler_owner_id,
        visibility=state.settings.crawler_visibility,
    )
    await state.status_store.delete(run_id)


def _host_candidates(raw: str) -> list[str]:
    """
    Hosts to try for a typed URL, most specific first.

    A URL that will not parse yields nothing, which is the correct answer on its own: it was never
    crawled, so no index can exist for it. That is why no separate syntax check is needed.

    www and bare forms are stored as distinct payload values, so both are tried — otherwise a bare
    domain reports no index for a site indexed under its www host.

    Both directions, which is the whole point and was half-implemented until 2026-09-23: the old
    body stripped "www." but never added it, so a www-indexed host answered a www query and nothing
    else. Which form a host lands under is decided by whatever seed was crawled, not normalised at
    write time -- the store holds www.medius.com and www.stampli.com beside tipalti.com and
    melio.com -- so a one-directional lookup makes "is this indexed" depend on the operator typing
    the same form as the crawl seed. https://www.medius.com reported indexed while
    https://medius.com reported not indexed, for one run, at the same moment.
    """
    text = (raw or "").strip()
    if not text:
        return []
    if "://" not in text:
        text = f"https://{text}"

    try:
        host = (urlsplit(text).hostname or "").lower()
    except ValueError:
        return []
    if not host or "." not in host:
        return []

    bare = host[4:] if host.startswith("www.") else host
    candidates = [host]
    for alternate in (bare, f"www.{bare}"):
        if alternate not in candidates:
            candidates.append(alternate)
    return candidates


@app.post(
    "/v1/index/hosts",
    response_model=HostIndexResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def host_index_exists(body: HostIndexRequest) -> HostIndexResponse:
    """Whether an index exists for each URL's host, and which run indexed it."""
    results: list[HostIndexResult] = []
    for url in body.urls:
        found = None
        run_id = None
        for host in _host_candidates(url):
            payload = await state.store.find_host_index_payload(host)
            if payload is not None:
                found = host
                run_id = payload.get("runId")
                break
        results.append(
            HostIndexResult(url=url, host=found, indexed=found is not None, run_id=run_id)
        )
    return HostIndexResponse(results=results)
