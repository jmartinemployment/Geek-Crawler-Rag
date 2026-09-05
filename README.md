"""Geek-Crawler-Rag

Standalone **Python** retrieval product for the Geek-Crawler Mongo corpus.
Indexes English HTML into Qdrant and serves a thin query API for consumers
(content-creator-v2 WRITE, optional GeekAPI glue).

See [`architecture.md`](./architecture.md) and [`plans/geek-crawler-rag.md`](./plans/geek-crawler-rag.md).

## What this is / is not

| Is | Is not |
|----|--------|
| Index + query for `geek_crawler` HTML | A crawler |
| Qdrant colocated with Mongo on Hostinger | Cloud vector DB / pgvector |
| English-only embed (`text-embedding-3-small`) | Spanish indexing |
| Owned by this repo | Logic inside phi or GeekAPI |

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Mongo + Qdrant liveness |
| `POST` | `/v1/index` | Enqueue full-run index `{ "runId": "…" }` |
| `GET` | `/v1/index/{runId}` | Index job status (ops/debug; UI uses SignalR, not polling) |
| `POST` | `/v1/query` | Retrieve chunks `{ "need", "runId", "crawlType?", "host?", "topK?" }` |

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
Point `MONGO_CRAWLER_URL` at the existing Hostinger Mongo `geek_crawler` database (read-only for crawl HTML).

GeekAPI: set `GEEK_CRAWLER_RAG_URL` / optional `GEEK_CRAWLER_RAG_API_KEY`.

## Consumers

- **gcc-v2 WRITE** (via GeekAPI): HTTP query client — resolve `runId`, call `/v1/query`, inject chunks; on miss **notify-and-skip** (do not block generate).
- Thin trigger: `POST /v1/index` when a crawl reaches **`complete`**.
- Progress: webhook → GeekAPI → SignalR (see above).
"""
