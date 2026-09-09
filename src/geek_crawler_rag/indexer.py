"""Index pipeline via LlamaIndex (nodes + embed + Qdrant) under FastAPI job queue.

Index concurrency = 1 (single worker). Rebuild = delete-by-runId then full pass.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Protocol

from llama_index.core.schema import TextNode

from geek_crawler_rag.config import Settings
from geek_crawler_rag.extract import host_from_origin_or_url
from geek_crawler_rag.llama_nodes import page_to_nodes
from geek_crawler_rag.models import (
    TERMINAL_CRAWL_STATUSES,
    IndexState,
    IndexStatusResponse,
    utc_now,
)
from geek_crawler_rag.mongo import CrawlPage, MongoCorpus
from geek_crawler_rag.qdrant_store import QdrantStore
from geek_crawler_rag.status_store import IndexStatusStore
from geek_crawler_rag.unusable import classify_unusable_page
from geek_crawler_rag.webhook import IndexStatusWebhook

logger = logging.getLogger(__name__)


class LeaseLostError(RuntimeError):
    """The worker no longer owns the Mongo index-job lease."""


class NodeUpserter(Protocol):
    async def embed_and_upsert(self, nodes: list[TextNode]) -> int: ...


class IndexService:
    def __init__(
        self,
        mongo: MongoCorpus,
        store: QdrantStore,
        settings: Settings,
        llama: NodeUpserter,
        status_store: IndexStatusStore | None = None,
        webhook: IndexStatusWebhook | None = None,
    ) -> None:
        self._mongo = mongo
        self._store = store
        self._settings = settings
        self._llama = llama
        self._status_store = status_store
        self._webhook = webhook
        self._statuses: dict[str, IndexStatusResponse] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._enqueue_lock = asyncio.Lock()
        self.owner = str(uuid.uuid4())

    async def start(self) -> None:
        if self._status_store is not None:
            await self._status_store.ensure_indexes()
            recovered = await self._status_store.claim_recoverable(
                owner=self.owner,
                lease_seconds=self._settings.index_job_lease_seconds,
                max_attempts=self._settings.index_scheduler_max_attempts,
            )
            for status in recovered:
                self._statuses[status.run_id] = status
                await self._queue.put(status.run_id)
            if recovered:
                logger.warning("Recovered %s stale index job(s)", len(recovered))
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._worker_loop(), name="index-worker"
            )

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
        if self._status_store is not None:
            loaded = await self._status_store.get(run_id)
            if loaded is not None:
                self._statuses[run_id] = loaded
            return loaded
        return self._statuses.get(run_id)

    async def enqueue(self, run_id: str) -> IndexStatusResponse:
        status, _ = await self._enqueue(run_id, trigger="manual", force=True)
        return status

    async def enqueue_scheduled(self, run_id: str) -> bool:
        _, accepted = await self._enqueue(run_id, trigger="scheduled", force=False)
        return accepted

    async def _enqueue(
        self, run_id: str, *, trigger: str, force: bool
    ) -> tuple[IndexStatusResponse, bool]:
        run_id = run_id.strip()
        if not run_id:
            raise ValueError("runId is required")

        async with self._enqueue_lock:
            existing = self._statuses.get(run_id)
            if existing is None and self._status_store is not None:
                existing = await self._status_store.get(run_id)
                if existing is not None:
                    self._statuses[run_id] = existing
            if (
                self._status_store is None
                and existing
                and existing.state in (IndexState.PENDING, IndexState.RUNNING)
            ):
                return existing, False

            if self._status_store is not None:
                claimed = await self._status_store.claim(
                    run_id,
                    owner=self.owner,
                    lease_seconds=self._settings.index_job_lease_seconds,
                    trigger=trigger,
                    force=force,
                )
                if claimed is None:
                    existing = await self._status_store.get(run_id)
                    if existing is None:
                        raise RuntimeError(f"Could not claim index job runId={run_id}")
                    self._statuses[run_id] = existing
                    return existing, False
                status = claimed
            else:
                status = IndexStatusResponse(
                    run_id=run_id,
                    state=IndexState.PENDING,
                    attempt=(existing.attempt + 1 if existing else 1),
                    trigger=trigger,
                )

            self._statuses[run_id] = status
            await self._persist(status)
            await self._queue.put(run_id)
            logger.info("Enqueued index job for runId=%s trigger=%s", run_id, trigger)
            return status, True

    async def _persist(self, status: IndexStatusResponse) -> bool:
        if self._status_store is not None:
            try:
                saved = await self._status_store.save(status, owner=self.owner)
                if not saved:
                    logger.warning(
                        "Rejected stale status write for runId=%s owner=%s",
                        status.run_id,
                        self.owner,
                    )
                    return False
            except Exception:
                logger.exception(
                    "Failed to persist index status for runId=%s", status.run_id
                )
                return False
        if self._webhook is not None:
            await self._webhook.notify(status)
        return True

    async def _worker_loop(self) -> None:
        logger.info("Index worker started (concurrency=1, engine=LlamaIndex)")
        while True:
            run_id = await self._queue.get()
            try:
                async with self._lock:
                    await self._run_claimed_job(run_id)
            except LeaseLostError:
                logger.error(
                    "Stopped stale index worker after lease loss runId=%s",
                    run_id,
                )
            except asyncio.CancelledError:
                status = self._statuses.get(run_id)
                if status and status.state in (IndexState.PENDING, IndexState.RUNNING):
                    status.state = IndexState.FAILED
                    status.error = "Indexer cancelled while job was in flight"
                    status.finished_at_utc = utc_now()
                    await self._safe_cleanup(run_id)
                    await self._persist(status)
                raise
            except Exception:
                logger.exception("Unhandled index failure for runId=%s", run_id)
                status = self._statuses.get(run_id)
                if status and status.state not in (
                    IndexState.COMPLETE,
                    IndexState.FAILED,
                ):
                    status.state = IndexState.FAILED
                    status.error = "Unhandled indexer exception"
                    status.finished_at_utc = utc_now()
                    await self._safe_cleanup(run_id)
                    await self._persist(status)
            finally:
                if self._status_store is not None:
                    try:
                        await self._status_store.release(
                            run_id,
                            owner=self.owner,
                            retry_seconds=self._settings.index_scheduler_retry_seconds,
                        )
                    except Exception:
                        logger.exception(
                            "Failed releasing index lease for runId=%s", run_id
                        )
                self._queue.task_done()

    async def _run_claimed_job(self, run_id: str) -> None:
        if self._status_store is None:
            await self._index_run(run_id)
            return
        if not await self._status_store.is_owned(run_id, owner=self.owner):
            raise LeaseLostError(f"Index lease expired before execution: {run_id}")

        index_task = asyncio.create_task(
            self._index_run(run_id), name=f"index-run-{run_id}"
        )
        heartbeat = asyncio.create_task(
            self._heartbeat_loop(run_id), name=f"index-heartbeat-{run_id}"
        )
        try:
            done, _ = await asyncio.wait(
                {index_task, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if index_task in done:
                await index_task
                return
            if heartbeat in done:
                error = heartbeat.exception()
                index_task.cancel()
                try:
                    await index_task
                except asyncio.CancelledError:
                    pass
                if error is not None:
                    raise error
                raise LeaseLostError(f"Index heartbeat stopped: {run_id}")
            await index_task
        finally:
            for task in (index_task, heartbeat):
                if not task.done():
                    task.cancel()
            for task in (index_task, heartbeat):
                try:
                    await task
                except (asyncio.CancelledError, LeaseLostError):
                    pass
                except Exception:
                    pass

    async def _heartbeat_loop(self, run_id: str) -> None:
        last_success = time.monotonic()
        failure_limit = max(
            1,
            self._settings.index_job_lease_seconds
            - self._settings.index_job_heartbeat_seconds,
        )
        while True:
            await asyncio.sleep(self._settings.index_job_heartbeat_seconds)
            if self._status_store is None:
                return
            try:
                renewed = await self._status_store.heartbeat(
                    run_id,
                    owner=self.owner,
                    lease_seconds=self._settings.index_job_lease_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                if time.monotonic() - last_success >= failure_limit:
                    raise LeaseLostError(
                        f"Could not renew index lease before expiry: {run_id}"
                    )
                logger.warning(
                    "Transient index heartbeat error runId=%s", run_id, exc_info=True
                )
                continue
            if not renewed:
                raise LeaseLostError(f"Lost index job lease: {run_id}")
            last_success = time.monotonic()

    def _embedding_stats(self) -> tuple[int, float]:
        getter = getattr(self._llama, "embedding_stats", None)
        if not callable(getter):
            return 0, 0.0
        stats = getter()
        return int(stats.get("rateLimitRetries", 0)), float(
            stats.get("waitSeconds", 0.0)
        )

    def _sync_embedding_stats(
        self,
        status: IndexStatusResponse,
        baseline: tuple[int, float],
    ) -> None:
        retries, wait = self._embedding_stats()
        status.embedding_rate_limit_retries = max(0, retries - baseline[0])
        status.embedding_wait_seconds = round(max(0.0, wait - baseline[1]), 3)

    async def _safe_cleanup(self, run_id: str) -> None:
        try:
            await self._delete_run_points(run_id)
        except Exception:
            logger.exception(
                "Failed to clean Qdrant points after failure for runId=%s", run_id
            )

    async def _delete_run_points(self, run_id: str) -> None:
        await self._store.delete_by_run_id(
            run_id,
            owner_id=self._settings.crawler_owner_id,
            visibility=self._settings.crawler_visibility,
        )

    async def _delete_page_points(self, page_id: str) -> None:
        await self._store.delete_by_page_id(
            page_id,
            owner_id=self._settings.crawler_owner_id,
            visibility=self._settings.crawler_visibility,
        )

    async def _delete_unusable(
        self, page: CrawlPage, reason: str, status: IndexStatusResponse
    ) -> None:
        if reason == "locale":
            status.pages_deleted_locale += 1
        elif reason == "failure":
            status.pages_deleted_failure += 1
        elif reason == "non_english":
            status.pages_deleted_non_english += 1
        else:
            status.pages_deleted_empty += 1
        try:
            await self._mongo.delete_page(page.id)
        except Exception:
            logger.exception(
                "Failed deleting unusable Mongo page id=%s reason=%s", page.id, reason
            )
        try:
            await self._delete_page_points(page.id)
        except Exception:
            logger.debug("Qdrant pageId delete skipped id=%s", page.id, exc_info=True)

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
        status.pages_deleted_locale = 0
        status.pages_deleted_failure = 0
        status.pages_deleted_empty = 0
        status.pages_deleted_non_english = 0
        status.chunks_upserted = 0
        status.embedding_rate_limit_retries = 0
        status.embedding_wait_seconds = 0.0
        embedding_baseline = self._embedding_stats()
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
        logger.info(
            "Indexing runId=%s crawlType=%s status=%s mongoPageCount=%s engine=LlamaIndex",
            run_id,
            run.crawl_type,
            run.status,
            mongo_page_count,
        )
        await self._persist(status)

        max_pages = 50_000
        if mongo_page_count > max_pages:
            status.state = IndexState.FAILED
            status.error = (
                f"mongoPageCount={mongo_page_count} exceeds safety cap {max_pages}; "
                "refusing index (re-POST after crawl is complete if intentional)"
            )
            status.finished_at_utc = utc_now()
            logger.warning("Index failed for runId=%s — %s", run_id, status.error)
            await self._persist(status)
            return

        if mongo_page_count == 0:
            await self._delete_run_points(run_id)
            status.state = IndexState.COMPLETE
            status.error = "No Mongo pages for run"
            status.finished_at_utc = utc_now()
            logger.warning("Index complete (empty) for runId=%s — no pages", run_id)
            await self._persist(status)
            return

        await self._store.ensure_collection()
        await self._delete_run_points(run_id)

        pending: list[TextNode] = []
        entity_cache: dict[str, object] = {}

        try:
            async for pages in self._mongo.iter_pages(
                run_id, batch_size=self._settings.page_batch_size
            ):
                for page in pages:
                    status.pages_seen += 1
                    reject = classify_unusable_page(
                        url=page.url,
                        final_url=page.final_url,
                        failure_reason=page.failure_reason,
                        robots_allowed=page.robots_allowed,
                    )
                    if reject:
                        await self._delete_unusable(page, reject, status)
                        continue

                    host_key = host_from_origin_or_url(page.origin, page.url)
                    if host_key not in entity_cache:
                        entity_cache[host_key] = await self._mongo.resolve_entity(
                            host=host_key, crawl_type=run.crawl_type
                        )
                    entity = entity_cache[host_key]
                    nodes, skip = page_to_nodes(
                        page=page,
                        run_id=run_id,
                        crawl_type=run.crawl_type,
                        entity=entity,  # type: ignore[arg-type]
                        settings=self._settings,
                    )
                    if skip == "empty":
                        await self._delete_unusable(page, "extract_empty", status)
                        continue
                    if skip == "lang":
                        await self._delete_unusable(page, "non_english", status)
                        continue

                    status.pages_english += 1
                    pending.extend(nodes)
                    if len(pending) >= self._settings.embed_batch_size:
                        n = await self._llama.embed_and_upsert(pending)
                        status.chunks_upserted += n
                        self._sync_embedding_stats(status, embedding_baseline)
                        pending = []
                        if self._settings.qdrant_upsert_delay_seconds > 0:
                            await asyncio.sleep(
                                self._settings.qdrant_upsert_delay_seconds
                            )
                        if self._webhook is not None:
                            await self._webhook.notify(status)

            if pending:
                n = await self._llama.embed_and_upsert(pending)
                status.chunks_upserted += n
                self._sync_embedding_stats(status, embedding_baseline)
                if self._webhook is not None:
                    await self._webhook.notify(status)

        except Exception as ex:
            self._sync_embedding_stats(status, embedding_baseline)
            status.state = IndexState.FAILED
            status.error = "Indexing failed due to an internal service error."
            status.finished_at_utc = utc_now()
            logger.exception("Index failed for runId=%s: %s", run_id, ex)
            await self._safe_cleanup(run_id)
            await self._persist(status)
            return

        if status.pages_english == 0:
            status.state = IndexState.COMPLETE
            status.error = "No English pages to embed (unusable pages deleted)"
            status.finished_at_utc = utc_now()
            logger.warning(
                "Index complete (empty corpus) for runId=%s — pagesSeen=%s deletedLocale=%s "
                "deletedFailure=%s deletedEmpty=%s deletedNonEnglish=%s",
                run_id,
                status.pages_seen,
                status.pages_deleted_locale,
                status.pages_deleted_failure,
                status.pages_deleted_empty,
                status.pages_deleted_non_english,
            )
            await self._persist(status)
            return

        status.state = IndexState.COMPLETE
        self._sync_embedding_stats(status, embedding_baseline)
        status.finished_at_utc = utc_now()
        logger.info(
            "Index complete for runId=%s chunksUpserted=%s pagesEnglish=%s "
            "deletedLocale=%s deletedFailure=%s deletedEmpty=%s deletedNonEnglish=%s",
            run_id,
            status.chunks_upserted,
            status.pages_english,
            status.pages_deleted_locale,
            status.pages_deleted_failure,
            status.pages_deleted_empty,
            status.pages_deleted_non_english,
        )
        await self._persist(status)
