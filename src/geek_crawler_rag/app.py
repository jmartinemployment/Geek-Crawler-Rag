"""Geek-Crawler-Rag HTTP API — FastAPI shell over LlamaIndex ingest/query."""

from __future__ import annotations

import logging
import secrets
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
from geek_crawler_rag.content import ContentService
from geek_crawler_rag.content_models import (
    ClaimLedgerArtifact,
    CitableClaimsRequest,
    ComparisonBriefArtifact,
    ComparisonBriefRequest,
    CompetitiveResponseArtifact,
    CompetitiveResponseRequest,
    FaqGeneratorRequest,
    FaqSetArtifact,
    PillarOutlineArtifact,
    PillarOutlineRequest,
)
from geek_crawler_rag.diagnostics import DiagnosticService
from geek_crawler_rag.generate import GenerateService
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
from geek_crawler_rag.models import (
    AdTemplateIndexRequest,
    AdTemplateIndexResponse,
    AdTemplateQueryRequest,
    AdTemplateQueryResponse,
    GenerateRequest,
    GenerateResponse,
    IndexRunRequest,
    IndexSchedulerStatus,
    IndexStatusResponse,
    PageMarkdownResponse,
    ProducerCapabilities,
    QueryRequest,
    QueryResponse,
)
from geek_crawler_rag.mongo import MongoCorpus
from geek_crawler_rag.qdrant_store import QdrantStore
from geek_crawler_rag.query import QueryService
from geek_crawler_rag.rerank import Reranker
from geek_crawler_rag.scheduler import IndexScheduler
from geek_crawler_rag.status_store import IndexStatusStore
from geek_crawler_rag.webhook import IndexStatusWebhook

logger = logging.getLogger(__name__)


class AppState:
    settings: Settings
    mongo: MongoCorpus
    store: QdrantStore
    llama: LlamaIndexEngine
    indexer: IndexService
    query: QueryService
    templates: AdTemplateIndexService
    webhook: IndexStatusWebhook
    generate: GenerateService
    scheduler: IndexScheduler
    assets: AssetContextService
    diagnostics: DiagnosticService
    intelligence: IntelligenceService
    content: ContentService


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
    status_store = IndexStatusStore(state.mongo.db)
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
    state.generate = GenerateService(
        state.mongo, state.query, settings, assets=state.assets
    )
    state.diagnostics = DiagnosticService()
    state.intelligence = IntelligenceService(state.diagnostics)
    state.content = ContentService()
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
        "Geek-Crawler-Rag listening (collection=%s templates=%s engine=LlamaIndex webhook=%s rerank=%s generate=%s)",
        settings.qdrant_collection,
        settings.qdrant_ad_templates_collection,
        "on" if state.webhook.enabled else "off",
        "on" if reranker.enabled else "off",
        "on" if settings.generate_enabled else "off",
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
    try:
        qdrant_ok = await state.store.ping()
    except Exception as ex:
        logger.warning("Qdrant health check failed", exc_info=ex)
        errors.append("qdrant unavailable")
    if mongo_ok:
        try:
            scheduler_status = (await state.scheduler.status()).model_dump(
                by_alias=True, mode="json"
            )
        except Exception as ex:
            logger.warning("Scheduler health check failed", exc_info=ex)
            errors.append("scheduler unavailable")

    healthy = mongo_ok and qdrant_ok
    body = {
        "status": "ok" if healthy else "degraded",
        "mongo": mongo_ok,
        "qdrant": qdrant_ok,
        "engine": "llamaindex",
        "features": ["hybrid", "graph", "ad-templates", "pages", "generate"],
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
    response_model=IndexStatusResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def start_index(body: IndexRunRequest) -> IndexStatusResponse:
    """Enqueue a full-run English index (idempotent rebuild per runId)."""
    return await state.indexer.enqueue(body.run_id)


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


@app.get(
    "/v1/index-scheduler",
    response_model=IndexSchedulerStatus,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def index_scheduler_status() -> IndexSchedulerStatus:
    """Return persisted scheduler cadence and most recent enqueue."""
    return await state.scheduler.status()


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
    response_model=PageMarkdownResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def get_page_markdown(
    page_id: str,
    run_id: Annotated[str, ApiQuery(alias="runId", min_length=1)],
) -> PageMarkdownResponse:
    """Return Mongo Markdown for citation reads (404 if missing or empty)."""
    page = await state.mongo.get_page(page_id)
    if page is None or page.run_id != run_id or not page.markdown:
        raise HTTPException(
            status_code=404,
            detail="No Markdown for the authorized page.",
        )
    return PageMarkdownResponse(
        page_id=page.id,
        run_id=page.run_id,
        url=page.url,
        final_url=page.final_url or page.url,
        title=page.title,
        markdown=page.markdown,
        excerpt=None,
    )


@app.get(
    "/v1/pages",
    response_model=PageMarkdownResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def get_page_markdown_by_url(run_id: str, url: str) -> PageMarkdownResponse:
    """Lookup page Markdown by runId + Url/FinalUrl."""
    page = await state.mongo.get_page_by_url(run_id=run_id, url=url)
    if page is None or not page.markdown:
        raise HTTPException(
            status_code=404,
            detail=f"No Markdown for runId={run_id} url={url}",
        )
    return PageMarkdownResponse(
        page_id=page.id,
        run_id=page.run_id,
        url=page.url,
        final_url=page.final_url or page.url,
        title=page.title,
        markdown=page.markdown,
        excerpt=None,
    )


@app.post(
    "/v1/generate",
    response_model=GenerateResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def generate(body: GenerateRequest) -> GenerateResponse:
    """Citeable multi-step generate: retrieve → read Markdown → draft → verify."""
    return await state.generate.generate(body)


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
    "/v1/content/faq-set",
    response_model=FaqSetArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def faq_set(body: FaqGeneratorRequest) -> FaqSetArtifact:
    """Generate answer-first FAQ pairs from supplied queries and visible source content."""
    return state.content.faq_set(body)


@app.post(
    "/v1/content/citable-claims",
    response_model=ClaimLedgerArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def citable_claims(body: CitableClaimsRequest) -> ClaimLedgerArtifact:
    """Convert vague statements into attributable claims without inventing statistics."""
    return state.content.citable_claims(body)


@app.post(
    "/v1/content/comparison-brief",
    response_model=ComparisonBriefArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def comparison_brief(body: ComparisonBriefRequest) -> ComparisonBriefArtifact:
    """Build a structured X vs Y brief from supplied brand and competitor pages."""
    return state.content.comparison_brief(body)


@app.post(
    "/v1/content/competitive-response",
    response_model=CompetitiveResponseArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def competitive_response(
    body: CompetitiveResponseRequest,
) -> CompetitiveResponseArtifact:
    """Convert competitor coverage into a brand-aligned response strategy outline."""
    return state.content.competitive_response(body)


@app.post(
    "/v1/content/pillar-outline",
    response_model=PillarOutlineArtifact,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def pillar_outline(body: PillarOutlineRequest) -> PillarOutlineArtifact:
    """Produce a deterministic topic-cluster pillar outline (not a full article)."""
    return state.content.pillar_outline(body)


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
