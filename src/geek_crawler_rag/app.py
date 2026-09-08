"""Geek-Crawler-Rag HTTP API — FastAPI shell over LlamaIndex ingest/query."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from geek_crawler_rag.ad_templates import AdTemplateIndexService, normalize_upsert_items
from geek_crawler_rag.config import Settings, get_settings
from geek_crawler_rag.indexer import IndexService
from geek_crawler_rag.llama_engine import LlamaIndexEngine
from geek_crawler_rag.generate import GenerateService
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
    state.generate = GenerateService(state.mongo, state.query, settings)
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
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Api-Key",
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
        errors.append(f"mongo: {ex}")
    try:
        qdrant_ok = await state.store.ping()
    except Exception as ex:
        errors.append(f"qdrant: {ex}")
    if mongo_ok:
        try:
            scheduler_status = (
                await state.scheduler.status()
            ).model_dump(by_alias=True, mode="json")
        except Exception as ex:
            errors.append(f"scheduler: {ex}")

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
    return await state.templates.index(AdTemplateIndexRequest(templates=cleaned))


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
async def get_page_markdown(page_id: str) -> PageMarkdownResponse:
    """Return Mongo Markdown for citation reads (404 if missing or empty)."""
    page = await state.mongo.get_page(page_id)
    if page is None or not page.markdown:
        raise HTTPException(
            status_code=404,
            detail=f"No Markdown for pageId={page_id}",
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
