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

## 3a. Correctness — fail closed

**No Retries. No Fallbacks. No Crappy Code.**

Applies to Geek-Crawler-Rag and sibling Geek-Crawler-v2 work. See also [`no-retries-no-fallbacks.md`](./no-retries-no-fallbacks.md).

- **No Retries** — Do not add or retain application-level retry loops, exponential backoff, or “try again later” wrappers around failures, including HTTP 5xx responses, timeouts, Mongo failures, OpenAI failures, and GeekAPI `pages/batch` or `links/batch` ingest failures. Fail the operation on its first failure, return the real diagnostic error, and fix the root cause.
- **No Fallbacks** — Do not turn a required-operation failure into apparent success by dropping fields, skipping persistence, swallowing exceptions, or continuing on a best-effort basis. Intentional product processing paths, such as selecting Playwright when static HTML is not viable, are permitted only when explicit, logged, and contract-preserving. They must never hide API or storage failure.
- **No Crappy Code** — Delete incorrect behavior instead of concealing it. Empty error responses, diagnostic truncation, silent catches, and fatal failures without actionable server-side detail are defects. Fix the behavior and preserve enough safe diagnostic context to identify the cause.

**Checklist (agents — pass/fail):**

- [ ] No application-level retry, backoff, or retry wrapper added or retained.
- [ ] Required persistence or ingest failure fails the operation.
- [ ] Failure responses retain actionable diagnostic detail.
- [ ] No exception is swallowed or converted to success.
- [ ] Designed alternate processing paths are explicit and logged.

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

[x] Separate repo owns Python + Qdrant  
[x] Full index per `runId`; English only  
[x] `partner` and `competitors` same pipeline  
[x] gcc-v2 / GeekAPI are consumers only  
[x] No Creator crawl of tools/competitors from this project  
[x] Index status is push (webhook → SignalR); UI does not poll  
[x] No Retries / No Fallbacks / No Crappy Code (§3a) documented and Cursor-enforced  
