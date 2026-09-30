"""Mongo access to geek_crawler crawl_pages / crawl_runs / entities.

GeekRepository stores PG-export-shaped documents: PascalCase fields, Guid as
string ("d" format). Corpus reads are primary; deletes remove unusable pages
that leaked past the crawler (locale / failure / extract-empty).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from geek_crawler_rag.metadata import (
    EntityRef,
    entity_from_crawl,
    entity_from_doc,
    normalize_host,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrawlRun:
    id: str
    crawl_type: str
    status: str


@dataclass(frozen=True)
class SchedulableRun:
    id: str
    page_count: int
    ready_at: Any = None


@dataclass(frozen=True)
class SchedulableRunScan:
    candidate: SchedulableRun | None
    missing_ready_marker: int = 0
    zero_pages: int = 0
    safety_cap: int = 0
    excluded: int = 0

    def summary(self) -> str:
        if self.candidate is not None:
            return (
                f"candidate runId={self.candidate.id} pages={self.candidate.page_count}"
            )
        return (
            "no_candidate "
            f"missing_ready_marker={self.missing_ready_marker} "
            f"zero_pages={self.zero_pages} "
            f"safety_cap={self.safety_cap} "
            f"excluded={self.excluded}"
        )


@dataclass(frozen=True)
class CrawlPage:
    id: str
    run_id: str
    origin: str
    url: str
    final_url: str
    html: str | None
    # The corpus body. `blocks` is the typed structure the chunker and the
    # verification projection both read; `content_html` is the clean fragment,
    # kept for display and audit.
    content_html: str | None = None
    blocks: list[dict[str, Any]] = field(default_factory=list)
    title: str | None = None
    crawled_at: str | None = None
    failure_reason: str | None = None
    robots_allowed: bool | None = None
    # The response status the crawler recorded. 0 means the row predates the
    # field; it is not an error and must not be read as one.
    status_code: int | None = None


class MongoCorpus:
    def __init__(self, url: str, db_name: str = "geek_crawler") -> None:
        self._client = AsyncIOMotorClient(url)
        self._db: AsyncIOMotorDatabase = self._client[db_name]
        self._entity_cache: list[dict[str, Any]] | None = None

    @property
    def db(self) -> AsyncIOMotorDatabase:
        return self._db

    async def close(self) -> None:
        self._client.close()

    async def ping(self) -> bool:
        await self._client.admin.command("ping")
        return True

    async def ensure_indexes(self) -> None:
        """Create RAG-owned covered indexes without indexing large page bodies."""
        await self._db["crawl_runs"].create_index(
            [("Status", 1), ("ContentReadyAt", 1), ("Id", 1)],
            name="ix_crawl_runs_content_ready",
        )
        await self._db["rag_execution_replays"].create_index(
            [("stageExecutionId", 1)],
            name="ux_rag_execution_replays_stage",
            unique=True,
        )
        await self._db["rag_execution_replays"].create_index(
            [("expiresAtUtc", 1)],
            name="ttl_rag_execution_replays_expiry",
            expireAfterSeconds=0,
        )

    async def claim_stage_execution(
        self,
        *,
        stage_execution_id: str,
        idempotency_key: str,
        expires_at_utc: datetime,
    ) -> bool:
        """Atomically persist a replay claim across replicas and restarts."""
        try:
            await self._db["rag_execution_replays"].insert_one(
                {
                    "stageExecutionId": stage_execution_id,
                    "idempotencyKey": idempotency_key,
                    "expiresAtUtc": expires_at_utc,
                }
            )
            return True
        except DuplicateKeyError:
            return False

    async def get_run(self, run_id: str) -> CrawlRun | None:
        doc = await self._db["crawl_runs"].find_one({"Id": run_id})
        if doc is None:
            return None
        return CrawlRun(
            id=str(doc.get("Id", run_id)),
            crawl_type=str(doc.get("CrawlType") or ""),
            status=str(doc.get("Status") or ""),
        )

    async def count_pages(self, run_id: str) -> int:
        return int(await self._db["crawl_pages"].count_documents({"RunId": run_id}))

    async def find_oldest_content_ready_run(
        self,
        *,
        excluded_run_ids: set[str],
        maximum_pages: int = 50_000,
    ) -> SchedulableRunScan:
        """Return the OLDEST completed, content-ready crawl not yet indexed.

        Oldest by `ContentReadyAt`, tie-broken on `Id` so the choice is stable. This picked the
        smallest by page count until `384fa35`; the method kept the old name until 2026-09-30, which
        is worse than the original bug -- the old name was itself a
        live claim, and this one said the scheduler optimises for throughput. It does not, deliberately: content is
        produced in the order sites were crawled, so indexing a 24-page site ahead of a 500-page one
        that has been waiting reorders the work queue behind it.

        Readiness is `ContentReadyAt`, the marker GeekAPI stamps once every persisted page of the run
        carries extracted content. The `hint` names the index `ensure_indexes` creates at startup —
        Mongo errors on a hint naming an index that does not exist, so the two must change together.
        """
        run_filter: dict[str, Any] = {
            "Status": {"$in": ["complete", "external"]},
            "ContentReadyAt": {"$exists": True, "$nin": [None, ""]},
            "Id": {"$type": "string", "$ne": ""},
        }
        missing_ready_marker = await self._db["crawl_runs"].count_documents(
            {
                "Status": {"$in": ["complete", "external"]},
                "$or": [
                    {"ContentReadyAt": {"$exists": False}},
                    {"ContentReadyAt": None},
                    {"ContentReadyAt": ""},
                ],
            }
        )
        cursor = self._db["crawl_runs"].find(
            run_filter,
            {"Id": 1, "ContentReadyAt": 1, "_id": 0},
            hint="ix_crawl_runs_content_ready",
        )

        candidates: list[SchedulableRun] = []
        zero_pages = 0
        safety_cap = 0
        excluded = 0
        async for doc in cursor:
            run_id = str(doc.get("Id") or "")
            if not run_id:
                continue
            if run_id in excluded_run_ids:
                excluded += 1
                continue
            page_count = await self.count_pages(run_id)
            if page_count <= 0:
                zero_pages += 1
            elif page_count > maximum_pages:
                safety_cap += 1
            else:
                candidates.append(
                    SchedulableRun(run_id, page_count, doc.get("ContentReadyAt"))
                )
        # Oldest ready first, not smallest first.
        #
        # Smallest-first was chosen so a pipeline fault surfaced on the cheapest run rather than
        # after the most expensive one. That was worth having while the pipeline was unproven, and it
        # is a debugging property, not an operating one -- it silently reorders the corpus against
        # whatever order the operator crawled in, which is the order they are writing content in.
        # Jeff, 2026-09-30: "I never liked it picking smallest first, as I am trying to produce
        # content in a different order."
        #
        # FIFO on ContentReadyAt makes the order controllable by the one thing the operator already
        # controls: when they crawl. A run with no marker cannot be selected at all (the filter
        # requires it), so the fallback here only orders ties.
        candidate = (
            min(
                candidates,
                key=lambda item: (item.ready_at or datetime.max.replace(tzinfo=timezone.utc), item.id),
            )
            if candidates
            else None
        )
        return SchedulableRunScan(
            candidate=candidate,
            missing_ready_marker=int(missing_ready_marker),
            zero_pages=zero_pages,
            safety_cap=safety_cap,
            excluded=excluded,
        )

    async def iter_pages(
        self,
        run_id: str,
        *,
        batch_size: int = 25,
    ) -> AsyncIterator[list[CrawlPage]]:
        """Paginate pages for a run. Batches Html/ContentHtml/Blocks deliberately (large docs)."""
        projection = {
            "Id": 1,
            "RunId": 1,
            "Origin": 1,
            "Url": 1,
            "FinalUrl": 1,
            "Html": 1,
            "ContentHtml": 1,
            "contentHtml": 1,
            "Blocks": 1,
            "blocks": 1,
            "Title": 1,
            "title": 1,
            "CrawledAtUtc": 1,
            "FailureReason": 1,
            "RobotsAllowed": 1,
            "StatusCode": 1,
            "_id": 0,
        }
        cursor = (
            self._db["crawl_pages"]
            .find({"RunId": run_id}, projection)
            .sort("CrawledAtUtc", 1)
            .batch_size(batch_size)
        )

        batch: list[CrawlPage] = []
        async for doc in cursor:
            batch.append(_page_from_doc(doc, run_id))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    async def resolve_entity(self, *, host: str, crawl_type: str) -> EntityRef:
        """Match Mongo entities.domains to host; else crawlType + host fallback."""
        host_n = normalize_host(host)
        fallback = entity_from_crawl(crawl_type, host_n)
        if not host_n:
            return fallback
        try:
            entities = await self._load_entities()
        except Exception:
            # Not a fallback path: this value is computed unconditionally above
            # and is also the legitimate answer when no entity matches. What is
            # worth saying is that the collection did not load, so a failed load
            # is not read as a host that simply has no entity.
            logger.exception(
                "Entities collection did not load for host=%s; "
                "answering from crawlType only",
                host_n,
            )
            return fallback

        for doc in entities:
            domains_raw = doc.get("domains") or doc.get("Domains") or []
            if isinstance(domains_raw, str):
                domains = [domains_raw]
            elif isinstance(domains_raw, list):
                domains = [str(d) for d in domains_raw if d]
            else:
                domains = []
            for domain in domains:
                d = normalize_host(domain)
                if not d:
                    continue
                if host_n == d or host_n.endswith("." + d):
                    return entity_from_doc(
                        doc, fallback_host=host_n, crawl_type=crawl_type
                    )
        return fallback

    async def _load_entities(self) -> list[dict[str, Any]]:
        if self._entity_cache is not None:
            return self._entity_cache
        names = await self._db.list_collection_names()
        if "entities" not in names:
            self._entity_cache = []
            return self._entity_cache
        cursor = self._db["entities"].find({}).limit(5000)
        self._entity_cache = [doc async for doc in cursor]
        logger.info("Loaded %s entities for domain matching", len(self._entity_cache))
        return self._entity_cache

    async def get_page(self, page_id: str) -> CrawlPage | None:
        """Load one page by Id (Guid string). Projects `Blocks` for citation reads."""
        if not page_id:
            return None
        projection = {
            "Id": 1,
            "RunId": 1,
            "Origin": 1,
            "Url": 1,
            "FinalUrl": 1,
            "Html": 1,
            "ContentHtml": 1,
            "contentHtml": 1,
            "Blocks": 1,
            "blocks": 1,
            "Title": 1,
            "title": 1,
            "Excerpt": 1,
            "CrawledAtUtc": 1,
            "FailureReason": 1,
            "RobotsAllowed": 1,
            "StatusCode": 1,
            "_id": 0,
        }
        doc = await self._db["crawl_pages"].find_one({"Id": page_id}, projection)
        if doc is None:
            return None
        return _page_from_doc(doc, str(doc.get("RunId") or ""))

    async def get_page_by_url(self, *, run_id: str, url: str) -> CrawlPage | None:
        """Lookup by run + Url or FinalUrl (exact match)."""
        if not run_id or not url:
            return None
        projection = {
            "Id": 1,
            "RunId": 1,
            "Origin": 1,
            "Url": 1,
            "FinalUrl": 1,
            "Html": 1,
            "ContentHtml": 1,
            "contentHtml": 1,
            "Blocks": 1,
            "blocks": 1,
            "Title": 1,
            "title": 1,
            "Excerpt": 1,
            "CrawledAtUtc": 1,
            "FailureReason": 1,
            "RobotsAllowed": 1,
            "StatusCode": 1,
            "_id": 0,
        }
        doc = await self._db["crawl_pages"].find_one(
            {
                "RunId": run_id,
                "$or": [{"Url": url}, {"FinalUrl": url}],
            },
            projection,
        )
        if doc is None:
            return None
        return _page_from_doc(doc, run_id)

    async def delete_page(self, page_id: str) -> int:
        """Delete one crawl_pages doc and its crawl_links. Returns links removed."""
        if not page_id:
            return 0
        link_res = await self._db["crawl_links"].delete_many({"PageId": page_id})
        await self._db["crawl_pages"].delete_one({"Id": page_id})
        return int(link_res.deleted_count)


def _as_int(value: Any) -> int | None:
    """An int from either shape GeekAPI writes, or None when it is not readable.

    GeekAPI's Mongo class map stores this field as a STRING -- `LegacyStringInt32Serializer`
    does `WriteString(...)` -- so `StatusCode` arrives as "404", not 404. Measured
    2026-09-29: 4,144 of 4,144 `crawl_pages` rows hold it as a string.

    The previous read was `status if isinstance(status, int) else None`, which is False for
    every real value, so the http_error rejection gate could never fire and a 404 body was
    chunked, embedded and quotable under a URL the server said it did not serve. The same
    mistake is why the robots gate below has never fired either.

    Both shapes are accepted rather than only the string, because the encoding is a legacy
    artefact being removed: when it goes, this keeps working instead of inverting the bug.
    A value that is neither is returned as None and logged -- "I could not read it" is not
    "it was 200", and silently defaulting is what made this invisible the first time.

    `bool` is excluded explicitly: it is a subclass of int in Python, so True would
    otherwise sail through as 1.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            logger.warning("Unreadable StatusCode value %r; treating as unknown", value)
            return None
    if value is not None:
        logger.warning("Unreadable StatusCode value %r; treating as unknown", value)
    return None


def _as_bool(value: Any) -> bool | None:
    """A bool from either shape GeekAPI writes, or None when it is not readable.

    `LegacyStringBooleanSerializer` writes "t"/"f", so `RobotsAllowed` arrives as "t" and
    `isinstance(value, bool)` is False for it. Measured 2026-09-29: all 4,144 rows hold "t",
    so `if robots_allowed is False` in unusable.py has never once matched.

    Reviving it changes nothing today -- zero rows hold a denied value, because the crawler
    rejects robots-denied URLs before it saves and passes `robotsAllowed: true` literally --
    but a gate that cannot fire is not a gate, and the next producer will not have that
    property.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("t", "true", "1", "y", "yes"):
            return True
        if text in ("f", "false", "0", "n", "no"):
            return False
        if not text:
            return None
        logger.warning("Unreadable RobotsAllowed value %r; treating as unknown", value)
        return None
    if isinstance(value, int):
        return bool(value)
    if value is not None:
        logger.warning("Unreadable RobotsAllowed value %r; treating as unknown", value)
    return None


def _page_from_doc(doc: dict[str, Any], run_id: str) -> CrawlPage:
    html = doc.get("Html")
    if html is not None and not isinstance(html, str):
        html = str(html)
    content_html = doc.get("ContentHtml")
    if content_html is None:
        content_html = doc.get("contentHtml")
    if content_html is not None and not isinstance(content_html, str):
        content_html = str(content_html)
    # Stored as a native BSON array, so pymongo hands back a list of dicts.
    # Anything else is treated as absent rather than coerced: a page whose
    # blocks did not survive storage is not one this service can index.
    raw_blocks = doc.get("Blocks")
    if raw_blocks is None:
        raw_blocks = doc.get("blocks")
    blocks = [b for b in raw_blocks if isinstance(b, dict)] if isinstance(raw_blocks, list) else []
    title = doc.get("Title")
    if title is None:
        title = doc.get("title")
    if title is not None and not isinstance(title, str):
        title = str(title)
    crawled = doc.get("CrawledAtUtc")
    crawled_at = None
    if crawled is not None:
        crawled_at = (
            crawled.isoformat() if hasattr(crawled, "isoformat") else str(crawled)
        )
    failure = doc.get("FailureReason")
    failure_reason = (
        failure.strip() if isinstance(failure, str) and failure.strip() else None
    )
    robots_allowed = _as_bool(doc.get("RobotsAllowed"))
    status_code = _as_int(doc.get("StatusCode"))
    return CrawlPage(
        id=str(doc.get("Id") or ""),
        run_id=str(doc.get("RunId") or run_id),
        origin=str(doc.get("Origin") or ""),
        url=str(doc.get("Url") or ""),
        final_url=str(doc.get("FinalUrl") or doc.get("Url") or ""),
        html=html,
        content_html=content_html.strip()
        if isinstance(content_html, str) and content_html.strip()
        else None,
        blocks=blocks,
        title=title.strip() if isinstance(title, str) and title.strip() else None,
        crawled_at=crawled_at,
        failure_reason=failure_reason,
        robots_allowed=robots_allowed,
        status_code=status_code,
    )
