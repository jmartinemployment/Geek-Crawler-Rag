# Cleanup: delete unusable crawl data (this repo)

Status: **implemented** — ops procedure for corpus hygiene when junk leaks past Geek-Crawler-v2.

Sibling: [Geek-Crawler-v2 `plans/crawl-reject-unusable-pages.md`](../../Geek-Crawler-v2/plans/crawl-reject-unusable-pages.md) (prevent at crawl). This repo **deletes** what still arrives.

## Unusable classes (delete)

| Reason | Detect |
|--------|--------|
| `locale` | Non-US region / non-English first path segment (`unusable.py`, same rules as Crawler-v2) |
| `failure` | Non-empty `FailureReason`, or `RobotsAllowed: false` (includes Cloudflare challenge rows) |
| `extract_empty` | Readability/markdown empty at backfill, or index `page_to_nodes` empty skip |
| `non_english` | Index-time `langdetect` skip (deleted, not left in Mongo) |

## Procedures

### 1. Bulk purge (Hostinger Mongo)

```bash
export MONGO_CRAWLER_URL='mongodb://…@2.24.101.90:27017/?authSource=admin'
# Dry-run sample
uv run python scripts/cleanup_unusable_pages.py --dry-run --limit 500
# Full delete
uv run python scripts/cleanup_unusable_pages.py --write
```

- Deletes matching `crawl_pages` and related `crawl_links` (`PageId`).
- Idempotent; safe to re-run.
- Prefer pausing long backfill jobs during a full write pass to reduce races.

### 2. Ongoing — markdown backfill

[`scripts/backfill_markdown.py`](../scripts/backfill_markdown.py) **deletes** locale / failure / extract-empty candidates instead of skip-marking them.

### 3. Ongoing — index sweeper

[`indexer.py`](../src/geek_crawler_rag/indexer.py) on `POST /v1/index`:

1. Classify each page before embed
2. **Delete** Mongo page + links; drop any Qdrant points with that `pageId`
3. Report on job status / webhook: `pagesDeletedLocale`, `pagesDeletedFailure`, `pagesDeletedEmpty`, `pagesDeletedNonEnglish` (legacy `pagesSkippedLang` / `pagesSkippedEmpty` still incremented for compatibility)

### 4. After purge

Reindex affected runs (`POST /v1/index`) so Qdrant matches the cleaned Mongo corpus. Full reindex already `delete_by_run_id` then rebuild.

## Code

- Shared rules: [`src/geek_crawler_rag/unusable.py`](../src/geek_crawler_rag/unusable.py)
- Script: [`scripts/cleanup_unusable_pages.py`](../scripts/cleanup_unusable_pages.py)

## Out of scope

- Preventing insert at crawl (Crawler-v2)
- Cloudflare bypass
