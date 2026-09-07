# Geek-Crawler-Rag

Standalone **Python** retrieval product for the Geek-Crawler Mongo corpus.
**FastAPI** exposes `v1/*`; **LlamaIndex** owns ingest/embed/dense retrieval into Qdrant
(parent/child chunks, hybrid BM25 RRF, optional Cohere rerank).

See [`architecture.md`](./architecture.md), [`plans/geek-crawler-rag.md`](./plans/geek-crawler-rag.md),
and [`plans/rag-content-writing-pipeline.md`](./plans/rag-content-writing-pipeline.md).

## Product overview

Geek-Crawler-Rag turns partner and competitor website crawls into searchable evidence and citation-backed content. It combines semantic and keyword retrieval, hierarchical context, entity-aware filtering, and source verification for technical articles, case studies, ads, battlecards, and strategy presentations.

### Capabilities

- English-only parent/child chunking for pinpoint and section-level context
- OpenAI embeddings and deterministic Qdrant vector records
- LlamaIndex dense retrieval combined with BM25/text search and reciprocal-rank fusion
- Optional Cohere reranking
- Entity, source, category, quality, host, and chunk-role filters
- Graph-style entity/category/co-occurrence themes
- Few-shot advertising-template indexing and retrieval
- Full-page Markdown reads for source verification
- Editable outlines and section-by-section generation
- Citation validation that rejects quotes not found in source Markdown
- Corpus hygiene, historical Markdown backfill, and idempotent run-level reindexing

### Technology

Python, FastAPI, Pydantic, LlamaIndex, MongoDB, Qdrant, OpenAI, BM25, Cohere, Readability, Docker, and GHCR.

## Place in the Geek content platform

```text
Geek-Crawler-v2 → MongoDB → Geek-Crawler-Rag/Qdrant
                                  ↓
                         GeekAPI → Content Creator v2
```

**Geek-Crawler-v2** produces the clean crawl corpus. This service owns corpus hygiene, indexing, retrieval, themes, and citation-aware generation. **Content Creator v2** provides the operator-facing writing, editing, publishing, and export workflow.

## What this is / is not

| Is | Is not |
|----|--------|
| Index + query for `geek_crawler` pages | A crawler |
| FastAPI + LlamaIndex + Qdrant on Hostinger | Cloud vector DB / pgvector |
| English-only embed (`text-embedding-3-small`) | Spanish indexing |
| Owned by this repo | Logic inside phi or GeekAPI |

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Mongo + Qdrant liveness (`engine`, `features`) |
| `POST` | `/v1/index` | Enqueue full-run index `{ "runId": "…" }` |
| `GET` | `/v1/index/{runId}` | Index job status (ops/debug; UI uses SignalR, not polling) |
| `POST` | `/v1/query` | Hybrid or graph retrieve (see below) |
| `POST` | `/v1/templates/index` | Upsert ad-template exemplars (Content Creator owns corpus) |
| `POST` | `/v1/templates/query` | Retrieve few-shot templates by need (+ channel/framework/tags) |
| `GET` | `/v1/pages/{pageId}` | Mongo Markdown for citation reads |
| `GET` | `/v1/pages?runId=&url=` | Same lookup by run + URL |
| `POST` | `/v1/generate` | Citeable multi-step draft (retrieve → Markdown → draft → verify) |

`POST /v1/query` body (camelCase; new fields optional / backward compatible):

```json
{
  "need": "…",
  "runId": "…",
  "crawlType": "partner",
  "host": null,
  "topK": 8,
  "preferParent": true,
  "preferChild": false,
  "chunkRole": "child",
  "sourceTypes": ["partner"],
  "entityNames": ["acme.com"],
  "categories": ["pricing"],
  "minQuality": 0.4,
  "retrievalMode": "hybrid"
}
```

- Default `retrievalMode`: `hybrid` (LlamaIndex dense + BM25/text RRF + optional Cohere).
- `retrievalMode: "graph"`: parent-biased hybrid plus `themes[]` (entity / category / co-occurrence) for slides/strategy.
- Index writes parent + child LlamaIndex nodes with `parentText` / `childText` and entity metadata.
- `GET /health` includes `"engine": "llamaindex"` and `"features": ["hybrid","graph","ad-templates"]`.

`POST /v1/templates/index` example:

```json
{
  "templates": [
    {
      "id": "pas-linkedin",
      "name": "PAS LinkedIn",
      "channel": "linkedin",
      "framework": "pas",
      "tone": "professional",
      "body": "Problem… Agitate… Solution…",
      "entityTags": ["acme"]
    }
  ]
}
```

`POST /v1/templates/query`: `{ "need": "…", "topK": 5, "channel": "linkedin", "framework": "pas" }`.

Optional auth: set `API_KEY` and send header `X-Api-Key`.

Index concurrency is **1**. Rebuild deletes all Qdrant points for `runId`, then reindexes.
At index start the service logs **`mongoPageCount`**. Runs with `mongoPageCount` above **50 000** are skipped (Hostinger safety cap).

### Index status push (no UI polling)

On status transitions the API POSTs to GeekAPI (optional):

- `INDEX_STATUS_WEBHOOK_URL` — e.g. `https://api.geekatyourspot.com/api/geek-crawler/internal/rag/index-status`
- `INDEX_STATUS_WEBHOOK_KEY` — `GEEK_BACKEND_API_KEY` (falls back to `API_KEY` if unset)

GeekAPI fans out SignalR **`GeekCrawlerRagIndexEvent`**. The Geek-Crawler UI listens on that hub method; it does **not** poll `GET /v1/index`.

## Local run

```bash
cp .env.example .env   # set MONGO_CRAWLER_URL, OPENAI_API_KEY
docker compose up -d qdrant
uv sync
uv run geek-crawler-rag
```

Or full stack:

```bash
docker compose up --build -d
```

## Hostinger

Live stack on KVM 2 (alongside Mongo):

- Health: `http://2.24.101.90:8080/health`
- Image: `ghcr.io/jmartinemployment/geek-crawler-rag:latest`
- Compose: [`deploy/hostinger-compose.yml`](./deploy/hostinger-compose.yml) (Qdrant + API; Mongo via `host.docker.internal`)

Caps: Qdrant `mem_limit: 3g` + `MAX_SEARCH_THREADS=1`, API `mem_limit: 2g`.
Point `MONGO_CRAWLER_URL` at the existing Hostinger Mongo `geek_crawler` database. Normal indexing reads the corpus; controlled cleanup procedures may delete unusable crawl pages and related links.

GeekAPI: set `GEEK_CRAWLER_RAG_URL` / optional `GEEK_CRAWLER_RAG_API_KEY`.

## Consumers

- **gcc-v2 WRITE** (via GeekAPI): HTTP query client — resolve `runId`, call `/v1/query`, inject chunks; on miss **notify-and-skip** (do not block generate).
- Thin trigger: `POST /v1/index` when a crawl reaches **`complete`**.
- Progress: webhook → GeekAPI → SignalR (see above).
