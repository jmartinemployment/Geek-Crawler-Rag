# RAG pipeline upgrades (Geek-Crawler-Rag scope)

Status: **Phase 0 + B + E + D implemented** (LlamaIndex under FastAPI; GraphRAG themes; ad-template index); **Phase M (markdown backfill) next**.  
Sibling plans: Geek-Crawler-v2 (markdown ingest), GeekBackend (`/api/rag/generate`), content-creator-v2 (consume generate).

## This repo owns

- Indexing Mongo `crawl_pages` → Qdrant
- Chunking, embeddings, hybrid retrieval, rerank
- **LlamaIndex** as the RAG framework (ingestion, indices, query pipelines) **inside** this service
- `POST /v1/index`, `GET /v1/index/{runId}`, `POST /v1/query`
- Reading Mongo `entities` for domain → `entityName` / `sourceType` (collection owned by Backend; this service consumes it)

## Today (after Phase D)

- Prefer Mongo `Markdown`/`Title` when present; else BeautifulSoup HTML → text; EN-only `langdetect`
- Parent (~1000 tok) + child (~200 tok) chunks; both embedded as separate LlamaIndex `TextNode`s
- **LlamaIndex** `OpenAIEmbedding` + `QdrantVectorStore` drive ingest/dense query inside FastAPI
- OpenAI `text-embedding-3-small` → Qdrant `geek_crawler_chunks`
- Rich payload: run/host/url + `parentText`/`childText`/`chunkRole` + entity/category/quality fields
- Hybrid query: LlamaIndex dense + BM25/text RRF; optional Cohere rerank (`COHERE_API_KEY`)
- Query filters: `preferParent`/`preferChild`/`chunkRole`/`sourceTypes`/`entityNames`/`categories`/`minQuality`
- **Graph mode:** `retrievalMode: graph` → parent-biased hybrid + `themes[]` (entity/category/co-occurrence)
- **Ad templates:** `/v1/templates/index` + `/v1/templates/query` on collection `geek_ad_templates`
- FastAPI remains the HTTP/deploy shell (`v1/*` stable for GeekAPI)

## Locked decisions

- **Keep Qdrant** (not Pinecone/Chroma/Mongo vectors)
- **FastAPI + LlamaIndex are complementary** — FastAPI stays the HTTP/API shell; LlamaIndex owns ingestion, indices, and query pipelines (not “pick one”)
- **Adopt LlamaIndex in this repo** (Phase E) — current custom index/query is an interim shape, not the end state
- Hybrid retrieval (vector + BM25 + RRF) stays required; implement/keep it via LlamaIndex (or LlamaIndex + Qdrant) under the existing `v1/*` contracts
- Crawlee stays the crawler (Firecrawl not adopted)
- OpenAI remains the writer (in GeekAPI), not in this service — Claude/Gemini from research are optional later, not blocking

## Phase 0 (shared gate — data first)

Before coding Phase B here:

1. Inventory Qdrant point count + payload field set + recent `rag_index_jobs`
2. Sample indexed `runId`s and a `v1/query` response
3. Confirm whether any pages already have `markdown` in Mongo (after Crawler-v2 Phase A)
4. Entity seed list available for domain match tests

Artifact: `plans/rag-baseline-inventory.md` (this repo or Crawler-v2).

## Phase B — This repo’s main work

### B1. Prefer markdown when present — done

- If page has `markdown` / `title` from ingest → chunk that
- Else fall back to current HTML extract

### B2. Parent-child chunking + dual embed — done

- **Child** ~200 tokens (feature/stat pinpoint)
- **Parent** = heading section or ~800–1200 tokens
- Embed both; store both texts in payload
- Long-form path: match on child → return **parent** text to caller
- Short-form path: return **child** text

### B3. Richer payload + filters — done

```text
runId, crawlType, host, url, pageId, title,
parentText, childText, sectionTitle, chunkIndex, language,
sourceType, entityName, entityId?,
category, contentIntent, tags[], qualityScore,
isEvergreen, lastCrawled
```

- Resolve entity via Mongo `entities.domains` match; else `crawlType` + host
- Category/intent heuristics from URL + text (backfill old HTML pages on reindex)

### B4. Hybrid retrieval (vector + BM25 + RRF) — done

- Dense Qdrant search + keyword/BM25 (exact product names, codes, stats)
- Reciprocal Rank Fusion merge
- Extends `POST /v1/query`

### B5. Rerank — done

- Initial pool k=20–30 → Cohere v3 (or configured reranker) → top 5
- Config via env (soft-disable if key missing)

### B6. Query API contract extensions — done

Request additions (camelCase):

- `preferParent` | `preferChild` (or `chunkRole`)
- `sourceTypes[]`, `entityNames[]`, `categories[]`, `minQuality`
- Keep existing `need`, `runId`, `crawlType`, `host`, `topK`

Response: include `url`, chosen text, `entityName`, `category`, scores.

## Phase E — LlamaIndex under FastAPI (this repo — done)

LlamaIndex and FastAPI are different layers: FastAPI = routes/deploy; LlamaIndex = ingest/indices/query pipelines.

1. [x] Add LlamaIndex dependency; keep existing `v1/index` / `v1/query` HTTP contracts stable for GeekAPI
2. [x] Move Mongo → document/node pipeline (markdown-preferring, parent/child) into LlamaIndex nodes (`llama_nodes.py`)
3. [x] Wire Qdrant as the vector store behind LlamaIndex (`llama_engine.py` + `QdrantVectorStore`)
4. [x] Route hybrid (dense via LlamaIndex + BM25/text RRF) + optional Cohere rerank under FastAPI query service
5. [x] Thin bespoke indexer/query to orchestrate LlamaIndex (filters, preferParent/Child, entity metadata preserved)

**Not in Phase E:** replacing FastAPI, changing GeekAPI writer, or Firecrawl.

## Phase M — One-time markdown backfill (this repo)

Draft-research cleaning for **existing** Mongo HTML (no re-crawl). New crawls already clean in Geek-Crawler-v2 via Readability → Turndown.

1. Script: `scripts/backfill_markdown.py` (deps: `readability-lxml`, `markdownify`)
2. For pages with `Html` and empty `Markdown`: extract → write `Title` / `Markdown` / `Excerpt` / `MarkdownBackfilledAt`
3. Locale filter ported from Crawler-v2 `locale-path.ts` (KEEP `/us/`, DROP foreign regions/non-English)
4. Dry-run first, then write; skip robots-denied / failures / already-backfilled
5. After write: operators reindex affected runs via `POST /v1/index`

Prerequisite: GeekBackend page docs accept those fields on ingest (and script may write Mongo directly on Hostinger).

## Phase D — GraphRAG + few-shot ad template index (this repo — done)

### D1. GraphRAG (slides / strategy themes) — done

- Theme overlay on hybrid hits: entity nodes, category themes, co-occurrence edges (`graph_retrieve.py`)
- `POST /v1/query` with `retrievalMode: "graph"` (defaults to parent sections; returns `themes[]` + chunks)
- Vector hybrid remains default for technical deep dives

### D2. Few-shot ad template index — done

- Separate Qdrant collection `geek_ad_templates` (configurable)
- `POST /v1/templates/index` — upsert templates (corpus owned by content-creator-v2)
- `POST /v1/templates/query` — retrieve exemplars by need + optional channel/framework/entityTags
- Metadata: channel, framework, tone, entityTags

## Out of scope here

- GeekAPI `/api/rag/generate` prompts and **OpenAI o1/o3 model routing** (Backend)
- Content Creator **template product ownership** / picker UX (content-creator-v2) — this repo only indexes what it sends
- Replacing Crawlee
- Deleting historical dirty crawl URLs
- Running LlamaIndex from Geek-Crawler-v2 or content-creator-v2

## Success criteria

- [x] Reindex run produces parent+child points with metadata
- [x] `v1/query` supports hybrid + rerank + preferParent/Child + entity filters
- [x] Backward compatible: old clients omitting new fields still get results
- [x] Phase 0 artifact: [`rag-baseline-inventory.md`](./rag-baseline-inventory.md)
- [x] Phase E: LlamaIndex drives ingest + query pipelines inside FastAPI; hybrid + filters parity; GeekAPI clients unchanged
- [x] Phase D1: GraphRAG retrieval usable for slide/strategy intents (`retrievalMode: graph` → `themes`)
- [x] Phase D2: Ad template index query returns few-shot exemplars for short-form (`/v1/templates/*`)
- [ ] Phase M: one-time Readability markdown backfill over existing HTML; dry-run then write

**Ops note:** Deploy new image, then `POST /v1/index` for runs that should get parent/child payloads. Pre-B points lack new fields; hybrid/filters degrade gracefully.
