"""Index pipeline: Mongo → extract → English-only → chunk → embed → Qdrant.

Index concurrency = 1 (single worker). Rebuild = delete-by-runId then full pass.
"""

from __future__ import annotations

import asyncio
import logging

from geek_crawler_rag.chunk import chunk_text
from geek_crawler_rag.config import Settings
from geek_crawler_rag.embed import Embedder
from geek_crawler_rag.extract import extract_text_and_title, host_from_origin_or_url
from geek_crawler_rag.language import is_english
from geek_crawler_rag.models import TERMINAL_CRAWL_STATUSES, IndexState, IndexStatusResponse, utc_now
from geek_crawler_rag.mongo import MongoCorpus
from geek_crawler_rag.qdrant_store import QdrantStore, point_id
from geek_crawler_rag.status_store import IndexStatusStore
from geek_crawler_rag.webhook import IndexStatusWebhook

logger = logging.getLogger(__name__)


class IndexService:
    def __init__(
        self,
        mongo: MongoCorpus,
        store: QdrantStore,
        embedder: Embedder,
        settings: Settings,
        status_store: IndexStatusStore | None = None,
        webhook: IndexStatusWebhook | None = None,
    ) -> None:
        self._mongo = mongo
        self._store = store
        self._embedder = embedder
        self._settings = settings
        self._status_store = status_store
        self._webhook = webhook
        self._statuses: dict[str, IndexStatusResponse] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._enqueue_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop(), name="index-worker")

    async def stop(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        for status in self._statuses.values():
            if status.state in (IndexState.PENDING, IndexState.RUNNING):
                status.state = IndexState.FAILED
                status.error = "Indexer shut down while job was in flight"
                status.finished_at_utc = utc_now()
                await self._persist(status)
        self._worker_task = None

    async def get_status(self, run_id: str) -> IndexStatusResponse | None:
        cached = self._statuses.get(run_id)
        if cached is not None:
            return cached
        if self._status_store is None:
            return None
        loaded = await self._status_store.get(run_id)
        if loaded is not None:
            self._statuses[run_id] = loaded
        return loaded

    async def enqueue(self, run_id: str) -> IndexStatusResponse:
        run_id = run_id.strip()
        if not run_id:
            raise ValueError("runId is required")

        async with self._enqueue_lock:
            existing = self._statuses.get(run_id)
            if existing is None and self._status_store is not None:
                existing = await self._status_store.get(run_id)
                if existing is not None:
                    self._statuses[run_id] = existing
            if existing and existing.state in (IndexState.PENDING, IndexState.RUNNING):
                return existing

            status = IndexStatusResponse(
                run_id=run_id,
                state=IndexState.PENDING,
            )
            self._statuses[run_id] = status
            await self._persist(status)
            await self._queue.put(run_id)
            logger.info("Enqueued index job for runId=%s", run_id)
            return status

    async def _persist(self, status: IndexStatusResponse) -> None:
        if self._status_store is not None:
            try:
                await self._status_store.save(status)
            except Exception:
                logger.exception("Failed to persist index status for runId=%s", status.run_id)
        if self._webhook is not None:
            await self._webhook.notify(status)

    async def _worker_loop(self) -> None:
        logger.info("Index worker started (concurrency=1)")
        while True:
            run_id = await self._queue.get()
            try:
                async with self._lock:
                    await self._index_run(run_id)
            except asyncio.CancelledError:
                status = self._statuses.get(run_id)
                if status and status.state in (IndexState.PENDING, IndexState.RUNNING):
                    status.state = IndexState.FAILED
                    status.error = "Indexer cancelled while job was in flight"
                    status.finished_at_utc = utc_now()
                    await self._persist(status)
                raise
            except Exception:
                logger.exception("Unhandled index failure for runId=%s", run_id)
                status = self._statuses.get(run_id)
                if status and status.state not in (
                    IndexState.COMPLETE,
                    IndexState.FAILED,
                    IndexState.SKIPPED,
                ):
                    status.state = IndexState.FAILED
                    status.error = "Unhandled indexer exception"
                    status.finished_at_utc = utc_now()
                    await self._safe_cleanup(run_id)
                    await self._persist(status)
            finally:
                self._queue.task_done()

    async def _safe_cleanup(self, run_id: str) -> None:
        try:
            await self._store.delete_by_run_id(run_id)
        except Exception:
            logger.exception("Failed to clean Qdrant points after failure for runId=%s", run_id)

    async def _index_run(self, run_id: str) -> None:
        status = self._statuses.get(run_id) or IndexStatusResponse(
            run_id=run_id, state=IndexState.PENDING
        )
        self._statuses[run_id] = status
        status.state = IndexState.RUNNING
        status.started_at_utc = utc_now()
        status.finished_at_utc = None
        status.error = None
        status.pages_seen = 0
        status.pages_english = 0
        status.pages_skipped_lang = 0
        status.pages_skipped_empty = 0
        status.chunks_upserted = 0
        await self._persist(status)

        run = await self._mongo.get_run(run_id)
        if run is None:
            status.state = IndexState.FAILED
            status.error = f"Run not found: {run_id}"
            status.finished_at_utc = utc_now()
            logger.error("Index failed — %s", status.error)
            await self._persist(status)
            return

        status.crawl_type = run.crawl_type
        if run.status and run.status.lower() not in TERMINAL_CRAWL_STATUSES:
            logger.warning(
                "Indexing non-terminal crawl status runId=%s status=%s (admin POST allowed)",
                run_id,
                run.status,
            )

        mongo_page_count = await self._mongo.count_pages(run_id)
        status.mongo_page_count = mongo_page_count
        # Required by plan/rules: log Mongo page count at index start.
        logger.info(
            "Indexing runId=%s crawlType=%s status=%s mongoPageCount=%s",
            run_id,
            run.crawl_type,
            run.status,
            mongo_page_count,
        )
        await self._persist(status)

        if mongo_page_count == 0:
            await self._store.delete_by_run_id(run_id)
            status.state = IndexState.SKIPPED
            status.error = "No Mongo pages for run"
            status.finished_at_utc = utc_now()
            logger.warning("Index skipped for runId=%s — no pages", run_id)
            await self._persist(status)
            return

        await self._store.ensure_collection()
        await self._store.delete_by_run_id(run_id)

        pending_ids: list[str] = []
        pending_texts: list[str] = []
        pending_payloads: list[dict] = []

        try:
            async for pages in self._mongo.iter_pages(
                run_id, batch_size=self._settings.page_batch_size
            ):
                for page in pages:
                    status.pages_seen += 1
                    text, title = extract_text_and_title(page.html)
                    if not text:
                        status.pages_skipped_empty += 1
                        continue
                    if not is_english(text):
                        status.pages_skipped_lang += 1
                        continue

                    status.pages_english += 1
                    host = host_from_origin_or_url(page.origin, page.url)
                    chunks = chunk_text(
                        text,
                        size_tokens=self._settings.chunk_size_tokens,
                        overlap_tokens=self._settings.chunk_overlap_tokens,
                    )
                    for idx, chunk in enumerate(chunks):
                        pending_ids.append(point_id(run_id, page.id, idx))
                        pending_texts.append(chunk)
                        pending_payloads.append(
                            {
                                "runId": run_id,
                                "crawlType": run.crawl_type,
                                "host": host,
                                "url": page.url,
                                "finalUrl": page.final_url or page.url,
                                "chunkIndex": idx,
                                "language": "en",
                                "title": title,
                                "text": chunk,
                                "pageId": page.id,
                            }
                        )

                    if len(pending_texts) >= self._settings.embed_batch_size:
                        await self._flush(status, pending_ids, pending_texts, pending_payloads)
                        pending_ids, pending_texts, pending_payloads = [], [], []

            if pending_texts:
                await self._flush(status, pending_ids, pending_texts, pending_payloads)

        except Exception as ex:
            status.state = IndexState.FAILED
            status.error = str(ex)
            status.finished_at_utc = utc_now()
            logger.exception("Index failed for runId=%s: %s", run_id, ex)
            # Always scrub after a failed rebuild so consumers never see a partial runId.
            await self._safe_cleanup(run_id)
            await self._persist(status)
            return

        if status.pages_english == 0:
            status.state = IndexState.SKIPPED
            status.error = "No English pages to embed"
            status.finished_at_utc = utc_now()
            logger.warning(
                "Index skipped for runId=%s — pagesSeen=%s skippedLang=%s skippedEmpty=%s",
                run_id,
                status.pages_seen,
                status.pages_skipped_lang,
                status.pages_skipped_empty,
            )
            await self._persist(status)
            return

        status.state = IndexState.COMPLETE
        status.finished_at_utc = utc_now()
        logger.info(
            "Index complete for runId=%s chunksUpserted=%s pagesEnglish=%s",
            run_id,
            status.chunks_upserted,
            status.pages_english,
        )
        await self._persist(status)

    async def _flush(
        self,
        status: IndexStatusResponse,
        ids: list[str],
        texts: list[str],
        payloads: list[dict],
    ) -> None:
        vectors = await self._embedder.embed(texts)
        await self._store.upsert(ids=ids, vectors=vectors, payloads=payloads)
        status.chunks_upserted += len(ids)
        # Progress push for SignalR bridge (webhook); durable store optional mid-run.
        if self._webhook is not None:
            await self._webhook.notify(status)
