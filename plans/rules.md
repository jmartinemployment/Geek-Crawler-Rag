# Rules

**Correctness over expediency.**

Authoritative rules for **Geek-Crawler-Rag**. When this file conflicts with an older plan snippet, **this file wins** unless the owner overrides in chat.

| Doc | Role |
|-----|------|
| **This file** | Hard rules — pass/fail |
| [`geek-crawler-rag.md`](./geek-crawler-rag.md) | Product / implementation plan |
| [`../architecture.md`](../architecture.md) | Topology and boundaries |

---

## 1. Workspace and repos

| Rule | Detail |
|------|--------|
| **Geek-Crawler-Rag workspace** | `/Users/jeffmartin/development/Geek-Crawler-Rag` — this repo only for indexer/query implementation |
| **Preserve** | `plans/` and `architecture.md` at repo root |
| **Not Content Creator** | Do **not** implement RAG inside `/Users/jeffmartin/development/content-creator-v2` |
| **Not GeekAPI** | Do **not** bury chunk/embed/Qdrant logic under `GeekAPI/Services/ContentCreatorV2` |
| **Sibling products** | Geek-Crawler, GeekBackend remain separate; Geek-Crawler-Rag **reads** Mongo, does not own crawl |

---

## 2. Isolation — zero diffs where forbidden

Fail any change that:

| Forbidden | Notes |
|-----------|-------|
| Adds crawl/fetch of partner or competitor URLs in this repo | Corpus comes from Mongo only |
| Writes crawl HTML as a second source of truth | Mongo via GeekRepository remains durable store |
| Puts Qdrant or embed code in phi `src/` | Phi is consumer only |
| Defaults to .NET for the RAG worker “because GeekAPI is C#” | Python is locked for AI ecosystem fit |

**Allowed consumer glue (other repos, thin only):** HTTP client to Geek-Crawler-Rag index/query from GeekAPI or content-creator-v2 BFF — no business logic duplication.

---

## 3. Correctness — no silent failures

- Index jobs must reach an explicit **complete**, **failed** (with error), or **skipped** (e.g. no English pages) state — never hang without status.
- Log **Mongo page count** per `runId` at index start (Cheerio UI may not expose counts).
- Do not report success if embed/upsert partially failed without recording failure.
- Consumers: missing index/query → **notify-and-skip** research; **do not** block Content Creator generate.

---

## 4. Index and language

| Rule | Detail |
|------|--------|
| **Full RAG per `runId`** | Index all usable English pages in that run |
| **`crawlType`** | `partner` and `competitors` — same pipeline; type is payload/filter only |
| **English only** | Detect language; embed/upsert **`en` only**; skip others |
| **Spanish** | Out of scope (future thought only) |
| **Rebuild** | Delete all points for `runId`, then full reindex |
| **Concurrency** | Index concurrency = **1** on Hostinger (2 CPU) |

---

## 5. Hostinger / Qdrant

| Rule | Detail |
|------|--------|
| **Colocate** | Qdrant on same host as Mongo — avoid cloud vector egress |
| **Qdrant RAM** | Cap ~3 GB |
| **Search threads** | `MAX_SEARCH_THREADS=1` |
| **Worker RAM** | Cap ~2 GB |
| **No Playwright** required on this box for RAG |

---

## 6. Embeddings and generation

| Rule | Detail |
|------|--------|
| **Embed model** | OpenAI `text-embedding-3-small` |
| **Secrets** | `OPENAI_API_KEY` (and Mongo/Qdrant URLs) in env — never commit |
| **LLM generation** | Stays with consumers (GeekAPI / existing providers) — Geek-Crawler-Rag returns chunks, does not write articles |

---

## 7. Naming

| Avoid | Use |
|-------|-----|
| “phi RAG” / “gcc-v2 RAG service” | **Geek-Crawler-Rag** |
| Partner-only indexer | Index **all** Geek-Crawler `runId`s (`partner` + `competitors`) |
| JSON-LD / advert-pack as architecture center | **Retrieval** is the product |

---

## 8. Pass/fail checklist (agents)

```bash
test -f /Users/jeffmartin/development/Geek-Crawler-Rag/architecture.md
test -f /Users/jeffmartin/development/Geek-Crawler-Rag/plans/rules.md
test -f /Users/jeffmartin/development/Geek-Crawler-Rag/plans/geek-crawler-rag.md
# No RAG implementation under content-creator-v2/src
! rg -l 'qdrant|Qdrant' /Users/jeffmartin/development/content-creator-v2/src 2>/dev/null | head -1
```

[ ] Separate repo owns Python + Qdrant  
[ ] Full index per `runId`; English only  
[ ] `partner` and `competitors` same pipeline  
[ ] gcc-v2 / GeekAPI are consumers only  
[ ] No Creator crawl of tools/competitors from this project  
