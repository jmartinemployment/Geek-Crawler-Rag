# RAG baseline inventory (Phase 0)

Captured: 2026-09-06 (Hostinger live + code). Gate for Phase B in [`rag-content-writing-pipeline.md`](./rag-content-writing-pipeline.md).

## Live stack

| Check | Result |
|-------|--------|
| `GET http://2.24.101.90:8080/health` | `{"status":"ok","mongo":true,"qdrant":true}` |
| Image | `ghcr.io/jmartinemployment/geek-crawler-rag:latest` |
| Collection | `geek_crawler_chunks` |

## Index jobs (`rag_index_jobs`)

From ops snapshot prior to Phase B:

| State | Count (approx) |
|-------|----------------|
| complete | 58 |
| failed | 10 (OpenAI 429 TPM during bulk partner embed) |
| skipped | 3–4 (no pages / no English / safety cap) |
| not indexed | ~15 eligible complete/external runs |

Sample complete job shape: `runId`, `state`, `crawlType`, `mongoPageCount`, `pagesEnglish`, `chunksUpserted`, timestamps.

## Qdrant payload (pre–Phase B)

Indexed fields today:

```text
runId, crawlType, host, url, finalUrl, chunkIndex, language, title, text, pageId
```

Payload indexes: `runId`, `crawlType`, `host`, `language` (KEYWORD).  
Vectors: dense cosine, `text-embedding-3-small` dim 1536. No sparse / BM25 / rerank.

## Sample query contract (pre–Phase B)

`POST /v1/query`:

```json
{ "need": "...", "runId": "...", "crawlType": "partner", "host": null, "topK": 8 }
```

Response chunks: `runId`, `crawlType`, `host`, `url`, `finalUrl`, `title`, `chunkIndex`, `language`, `text`, `score`.

## Mongo `crawl_pages` markdown?

| Field | Status |
|-------|--------|
| `Html` | Present (GeekRepository `GeekCrawlerPage`) |
| `Markdown` / `markdown` / `Title` | **Not on PG entity yet**; Crawler-v2 Phase A planned. Indexer must prefer markdown when present and fall back to HTML. |

## Entities seed (`entities.domains`)

| Check | Status |
|-------|--------|
| `geek_crawler.entities` | May be empty / owned by Backend later |
| Resolver behavior | Match host → `domains[]`; else fall back to `crawlType` + host as `sourceType` / `entityName` |

Seed list for domain-match tests: use fixtures in unit tests (`acme.com` → Acme) until Backend seeds production.

## Crawl types in live data

`partner` and `competitors` (plus schedules producing `complete` / `external` / `failed` / `cancelled`).

## Phase B readiness

| Gate | Ready? |
|------|--------|
| Qdrant + API up | Yes |
| Reindex path exists | Yes (`POST /v1/index`) |
| Markdown corpus | Partial — code path required; data backfill is Crawler-v2 |
| Entities corpus | Soft — resolver + heuristics without hard dependency |

**Decision:** Proceed with Phase B in this repo (prefer-markdown, parent/child, hybrid, rerank, query contract). Reindex runs after deploy to populate new payloads.
