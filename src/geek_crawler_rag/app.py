"""Geek-Crawler-Rag HTTP API — index + query for the Geek-Crawler Mongo corpus."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from geek_crawler_rag.config import Settings, get_settings
from geek_crawler_rag.embed import Embedder
from geek_crawler_rag.indexer import IndexService
from geek_crawler_rag.models import (
    IndexRunRequest,
    IndexStatusResponse,
    QueryRequest,
    QueryResponse,
)
from geek_crawler_rag.mongo import MongoCorpus
from geek_crawler_rag.qdrant_store import QdrantStore
from geek_crawler_rag.query import QueryService
from geek_crawler_rag.status_store import IndexStatusStore
from geek_crawler_rag.webhook import IndexStatusWebhook

logger = logging.getLogger(__name__)


class AppState:
    settings: Settings
    mongo: MongoCorpus
    store: QdrantStore
    embedder: Embedder
    indexer: IndexService
    query: QueryService
    webhook: IndexStatusWebhook


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
    state.embedder = Embedder(
        settings.openai_api_key,
        model=settings.openai_embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embed_batch_size,
    )
    status_store = IndexStatusStore(state.mongo.db)
    state.webhook = IndexStatusWebhook(
        settings.index_status_webhook_url,
        settings.index_status_webhook_key or settings.api_key,
    )
    state.indexer = IndexService(
        state.mongo,
        state.store,
        state.embedder,
        settings,
        status_store=status_store,
        webhook=state.webhook,
    )
    state.query = QueryService(state.store, state.embedder)
    await state.store.ensure_collection()
    await state.indexer.start()
    logger.info(
        "Geek-Crawler-Rag listening (collection=%s webhook=%s)",
        settings.qdrant_collection,
        "on" if state.webhook.enabled else "off",
    )
    yield
    await state.indexer.stop()
    await state.webhook.close()
    await state.store.close()
    await state.mongo.close()


app = FastAPI(
    title="Geek-Crawler-Rag",
    description="Index and query Geek-Crawler Mongo HTML via Qdrant (English only).",
    version="0.1.0",
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
    errors: list[str] = []
    try:
        mongo_ok = await state.mongo.ping()
    except Exception as ex:
        errors.append(f"mongo: {ex}")
    try:
        qdrant_ok = await state.store.ping()
    except Exception as ex:
        errors.append(f"qdrant: {ex}")

    healthy = mongo_ok and qdrant_ok
    body = {
        "status": "ok" if healthy else "degraded",
        "mongo": mongo_ok,
        "qdrant": qdrant_ok,
        "errors": errors or None,
    }
    return JSONResponse(body, status_code=200 if healthy else 503)


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


@app.post(
    "/v1/query",
    response_model=QueryResponse,
    response_model_by_alias=True,
    dependencies=[Depends(require_api_key)],
)
async def query(body: QueryRequest) -> QueryResponse:
    """Retrieve top-k English chunks for a need, filtered by runId (+ optional host/crawlType)."""
    return await state.query.query(body)
