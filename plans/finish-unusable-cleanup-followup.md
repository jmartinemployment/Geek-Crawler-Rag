# Finish unusable-page cleanup follow-up

Status: **Not started**

Closes remaining work after the Hostinger bulk purge and local sweeper fixes
that lived in the former `cleanup-unusable-pages.md`.

**Policy:** there is one corpus representation and nothing to backfill, so no
backfill may run. A page is unusable when it carries no typed `blocks`
(`no_content`), never a missing legacy body. Repair the corpus by **re-crawling**;
indexing itself no longer deletes anything (`indexer._skip_unusable`). Former
pipeline Phase M is cancelled and its backfill scripts were removed from this
repo.

Also carries **verification caveats** / open ops from former plans:

- `crawl-duplicate-pages-prefetch-and-content.md` (Geek-Crawler-v2 `c2f7f06`)
- `oversized-directory-page-extraction.md` (gallery strip in `extract-content.ts`)
- `s6-rag-credential-containment.md` (Hostinger RAG public HTTP / key rotate — **do immediately**)

## Remaining

### 1. Deploy indexer sweeper fixes

Local `indexer.py` already:

- Increments legacy `pagesSkippedLang` / `pagesSkippedEmpty` on empty / non-English deletes
- Fails closed on Mongo / Qdrant page delete errors (no swallowed exceptions)

Ship the current image to Hostinger so production matches that behavior.

### 2. Reconcile Qdrant after the purge

The write pass deleted **4** locale pages + **202** links; `runId`s were not logged.

Pick one:

- **A (preferred if runIds recoverable):** `POST /v1/index` for each affected run (full reindex already `delete_by_run_id` then rebuild).
- **B:** Scan Qdrant for `pageId`s missing from Mongo and delete those points (or reindex every run that still has points).

Do not paper over orphans with retries.

### 3. Caveat — crawl duplicate dedup verification (Crawler-v2)

Dedup code is shipped; these checks were never evidenced as run:

- **E2E:** baseline then re-crawl `n8n.io`; compare per-layer metrics
  (`enqueueSuppressed*`, `httpRequests`, `browserRenders`, `extractionInvocations`,
  `pagesSaved`, skip causes). Wall-clock is a benchmark only.
- **Corpus audit:** count duplicate `Url` per `RunId` in Mongo; re-crawl before
  re-indexing runs that still carry pre-dedup duplicates (e.g. `b51485e9` should
  not be indexed as-is).

Out of scope for that ship (unchanged): GeekAPI uniqueness, legacy ledger
backfill, retracting queued duplicates, promoting canonical after a variant.

### 4. Caveat — oversized directory / gallery extraction (Crawler-v2)

Selector strip is implemented (`GALLERY_CONTAINER_SELECTORS`,
`.grid:has(a.card--default)` before Readability) with unit tests. Plan status
was stale (“not yet implemented”).

Live verification still not evidenced:

- Re-crawl n8n; confirm `/integrations/if/` and `/integrations/set/` extracted
  content drops below the cap and `truncated` does not fire
- Guardrail sample of low-gallery `/integrations/*` pages (selector no-op)
- Spot-check retained prose is the node description, not empty

Do not re-propose Nuxt `__NUXT_DATA__` as the cause without re-testing — already
falsified. Generic density heuristics stay deferred.

### 5. Caveat — S6 RAG credential / transport containment (do immediately)

Do **not** wait on crawl SSRF polish. Live Hostinger still publishes
`0.0.0.0:8080` while repo `deploy/hostinger-compose.yml` already uses
`127.0.0.1:8080:8080`.

**Do not apply the loopback bind alone** while GeekAPI still calls
`http://2.24.101.90:8080` — that breaks production RAG. Order:

1. TLS terminator (Caddy/nginx) on VPS `:443` → `127.0.0.1:8080`
   (`deploy/Caddyfile`, `deploy/caddy-compose.yml`, `deploy/README-caddy.md`)
2. Point GeekAPI `GEEK_CRAWLER_RAG_URL` at `https://rag.geekatyourspot.com`
   and clear `GEEK_CRAWLER_RAG_ALLOW_INSECURE_HTTP_HOSTS`
3. Apply compose bind to loopback + restart (edit on-box
   `/docker/geek-crawler-rag/docker-compose.yml` — Hostinger MCP recreate does
   **not** push repo compose)
4. Rotate `API_KEY` and Railway `GEEK_CRAWLER_RAG_API_KEY` in the same window;
   confirm old key → 401

Same window (same class of exposure): colocated Mongo publishes
`0.0.0.0:27017` — bind to `127.0.0.1` or firewall drop.

**Ops constraint:** this agent machine has no SSH key to the VPS
(`Permission denied (publickey)`). Caddy + bind + rotate must run from a shell
that can reach `root@2.24.101.90` (or Hostinger panel terminal).

Done when: zero successful RAG calls with the retired key; GeekAPI logs show
`https://` RAG base; public probe of bare-IP `:8080` fails.

## Out of scope

- Prevent-at-crawl (Geek-Crawler-v2 `src/crawl/reject.ts`)
- Cloudflare bypass
- Rewriting the bulk cleanup script (already idempotent; safe to re-run)

## Verification

- Hostinger API image includes fail-closed `_delete_unusable` + legacy skip counters
- No Qdrant points remain for Mongo pages removed by the purge
- Dedup E2E / corpus audit either completed or explicitly deferred with owner note
- Gallery extraction live checks either completed or explicitly deferred with owner note
- RAG API no longer reachable on public plaintext `:8080`; TLS hostname + rotated key in use
