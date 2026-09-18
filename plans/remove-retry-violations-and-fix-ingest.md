# Remove retry violations and fix ingest root cause

Status: **Code-complete locally** (2026-09-14) — not committed/deployed. Policy law in [`rules.md`](./rules.md) §3a.

Repos: Geek-Crawler-v2, GeekBackend (GeekAPI), GeekRepository (page persistence).  
This file lives in Geek-Crawler-Rag as the cross-repo tracking plan; cited paths are in sibling repos under `~/development/`.

## Law

**No Retries. No Fallbacks. No Crappy Code.** Do not “fix” this work by adding retries, soft-succeeding after ingest failure, truncating payloads, or mapping distinct failures onto opaque 502/empty 500 responses.

## Problem (Avid / Stampli class)

Runs failed on `POST …/pages/batch` with **502** / empty **500** after large crawls because:

1. Uncapped raw HTML (plus a capped Markdown body, since retired) was sent inline per page → Mongo BSON **16 MiB** document overflows and/or API/repo request-size mismatch.
2. GeekAPI mapped repository non-2xx → **502**, and timeouts / unhandled exceptions → **500** with empty body — concealing the real cause.
3. Application-level retry/split-after-failure hid those root causes instead of failing closed.

---

## Locked decisions (ingest contract)

### 1. Raw HTML policy (not “if appropriate”)

| Rule | Decision |
|------|----------|
| Truncation | **Prohibited** for `html`, `contentHtml` and `blocks`. Never shorten content to fit. |
| Corpus body | Typed **`blocks`** (+ `contentHtml` for display/audit), with `title` / `excerpt`. **Markdown is forbidden** — the crawler does not send it, ingest must not accept it, and no hop may convert HTML to it. Superseded the `markdown` field this plan originally specified. |
| Raw HTML | **Retained only when the full serialized page document stays under the per-document limit** (see §2). Not stored by reference in this plan (no blob/object-store phase). |
| When HTML would exceed budget | Set `html` to `null`, persist the rest of the page (`contentHtml` + `blocks` always), and record `htmlOmittedBytes` + reason in structured logs / optional response metadata. This is an **explicit contract path**, not a silent fallback. |
| When the page **without** HTML still exceeds the per-document limit | **Reject** that page (server: fail the batch per §5; client: must not submit it). |

Do not rename or relocate uncapped HTML into another inline field. Any new string field that holds page body content is subject to the same size rules.

### 2. Layered size limits (authoritative numbers)

Limits are measured in **bytes of the would-be Mongo/BSON document** (or the repository’s equivalent serialized document), not UTF-16/`string.Length` character counts. HTTP batch size is measured as **raw request body bytes** (`Content-Length` / buffered body length).

| Layer | Limit | Authority |
|-------|-------|-----------|
| Mongo hard cap | **16 MiB** (`16_777_216`) BSON document | MongoDB — immutable ceiling |
| **Per-page document** | **14 MiB** (`14_680_064`) | `GeekCrawlerIngestLimits.MaxPageDocumentBytes` (GeekAPI) + mirrored `ingest-limits.ts` (Crawler-v2) + repo belt check |
| **Batch HTTP body** | **28 MiB** (`29_360_128`) | Same constant family; Kestrel `MaxRequestBodySize` aligned in GeekAPI `Program.cs` |
| Safety margin | 2 MiB under Mongo 16 MiB for per-page; batch cap is independent (many docs per request) | Prevents API accepting what Mongo will reject |

**Behavior when exceeded:**

| Condition | Outcome |
|-----------|---------|
| Single page BSON estimate > 14 MiB even with `html: null` | Client: do not include in batch; throw before network. Server: **reject entire batch** with **422**. |
| Single page fits only if `html` omitted | Client/server apply §1 omit-HTML path **before** persistence; success with `html: null`. |
| Batch body > 28 MiB | **Do not send.** Client throws before network. Server: **413** with JSON body. |
| Truncation to fit | Never. |

Estimator: field overhead + UTF-8 byte lengths of string fields + fixed metadata (`GeekCrawlerPageDocumentSizer` / `estimatePageDocumentBytes`).

### 3. Split vs retry (explicit)

| Pattern | Allowed? |
|---------|----------|
| **Pre-network batching**: partition pages into independently valid batches **before** the first network call | **Yes** |
| **Post-failure split/retry** | **No** — removed |
| Application retry loops / `LINK_BATCH_ATTEMPTS` | **No** — absent |

GeekBackend: `SavePagesBatchAsync` / `SaveLinksBatchAsync` (renamed from `*WithRetryAsync`); single repository attempt.

### 4. Error response contract

Every non-2xx from GeekAPI ingest endpoints returns **non-empty JSON**:

```json
{
  "error": {
    "code": "page_document_too_large",
    "message": "human-readable summary",
    "requestId": "…",
    "details": { }
  }
}
```

| Condition | HTTP | `error.code` |
|-----------|------|----------------|
| Malformed / schema-invalid request | **400** | `invalid_request` |
| One or more pages exceed per-document limit | **422** | `page_document_too_large` |
| Whole request body exceeds batch limit | **413** | `batch_body_too_large` |
| Upstream / repository non-success | mapped status (422/413/503/504/502…) | `repository_error` or preserved code |
| Timeout | **504** | `upstream_timeout` |
| Cancellation | **400** | `request_cancelled` |
| Unhandled exception | **500** | `internal_error` |

`requestId` from `X-Request-Id` or `TraceIdentifier`.

### 5. Partial-success behavior (atomic batches)

**`pages/batch` and `links/batch` are atomic.** Success ACK length/count equals submitted length. Oversized page → entire batch rejected; nothing persisted.

### 6. Observability

Structured logs on `pages/batch`: `requestId`, `runId`, `pageCount`, `batchBodyBytes`, `maxPageBytes`, `htmlOmittedCount`, `htmlOmittedBytes`. Client logs `ingest_html_omitted` when omitting. No page body contents in logs.

### 7. Caller and transport retry audit (completed 2026-09-14)

| Surface | Result |
|---------|--------|
| Geek-Crawler-v2 `GeekApiClient.request` | **Pass** — single `fetch`; tests assert attempts === 1 on 5xx |
| Crawler wrappers around `createPagesBatch` / `createLinksBatch` | **Pass** — `persist.ts` coordinator runs one call; failure throws |
| `LINK_BATCH_ATTEMPTS` | **Pass** — absent; `npm run check:fail-closed` ok |
| GeekBackend `HttpGeekCrawlerRepository` / `GeekRepository` HttpClient | **Pass** — no `AddStandardResilienceHandler` / Polly on GeekAPI ingest client |
| `GeekCrawlerPageBatchWriter` | **Pass** — single attempt; tests assert PostCount === 1 on 5xx/504 |
| Reverse proxies / ingress | **Documented risk** — Hostinger/nginx/Cloudflare POST retries not controlled in-repo; do not enable app-level retries to compensate |
| Soft-success after ingest failure | **Pass** — PersistenceError fails the page/run path |

### 8. Code changes (shipped)

| Repo / location | Change |
|-----------------|--------|
| GeekAPI `GeekCrawlerIngestLimits` / `GeekCrawlerPageDocumentSizer` / `GeekCrawlerIngestErrorResults` | Limits, estimator, JSON errors |
| GeekAPI `GeekCrawlerIngestController` | Enforce limits; atomic 422/413; structured errors |
| GeekAPI `GeekCrawlerPageBatchWriter` | Single-attempt `SavePagesBatchAsync` / `SaveLinksBatchAsync` |
| GeekAPI `Program.cs` | Kestrel max body = 28 MiB |
| GeekRepository `GeekCrawlerPagesController` | Belt-and-suspenders 14 MiB reject (422) |
| Geek-Crawler-v2 `ingest-limits.ts` + `geek-api-client.ts` | Pre-send omit/guards; JSON error parsing |

---

## Acceptance criteria / tests

| # | Criterion | Evidence |
|---|-----------|----------|
| 1 | `LINK_BATCH_ATTEMPTS` removed | `check:fail-closed` + client tests |
| 2 | `SavePagesWithRetryAsync` removed | Renamed; grep clean |
| 3 | Exactly one outbound save on 5xx/timeout | `GeekCrawlerPageBatchWriterTests` + v2 client tests |
| 4 | Oversized HTML cannot BSON-overflow | sizer + `applyHtmlOmit` tests under 16 MiB |
| 5 | Oversized batches fail with structured JSON | controller 413/422 paths + client pre-send throws |
| 6 | Valid pages within limits succeed | omit-then-ACK client test |
| 7 | Atomic oversize reject | server builds oversized list → 422, no persist |
| 8 | Audit artifact | §7 above |

---

## Out of scope

- Policy prose already in [`rules.md`](./rules.md) §3a and `.cursor/rules/no-retries-no-fallbacks.mdc`.
- Object-store / by-reference HTML blobs.
- Adding retries of any kind.
- Changing infra proxy retry behavior (documented only).
