# Stop fetching and saving duplicate pages in Geek-Crawler-v2

Status: **Complete** — shipped in Geek-Crawler-v2 `c2f7f06` (2026-09-12)
Found: **2026-09-12** — the `36f8785` dedup fix did not reduce duplicates or crawl time
Revised twice after design review — see *Revisions from review* at the end

> Code changes landed in `/Users/jeffmartin/development/Geek-Crawler-v2`, not this repo.

## Context

Duplicate and near-duplicate pages are still being saved, and the crawl is still
slow. The previous fix (`Geek-Crawler-v2` @ `36f8785`, surfaced in `df9486b`)
did not address either symptom, for three structural reasons:

1. **It dedups too late.** `savePage` (`src/storage/persist.ts:235`) checks its
   key *after* the request was fetched, `isViableHtml` ran, and
   `extractCleanContent` put the HTML through JSDOM + Readability + Turndown —
   and for promoted URLs, after a full Playwright render. It skips only the
   database row. Every duplicate still costs a request, a parse, a pool slot, and
   `maxRequestsPerCrawl` budget. The crawl could not have gotten faster.

2. **The key is too weak for post-redirect convergence.** `normalizeCrawlUrl`
   (`src/crawl/sitemap.ts:95`) strips only the fragment and a 19-entry
   `TRACKING_PARAMS` list, applied at enqueue time to the *pre-redirect* URL.

3. **No content check, and no `rel=canonical` handling.** `grep -rn canonical src/`
   returns one hit, a code comment. Identical or near-identical bodies at
   different URLs — print views, AMP, faceted and paginated archives, template
   pages — are invisible to a URL-only key. This is the "virtually duplicated" class.

Secondary defect: `savedFinalUrls` is an in-memory `Set` in the persist closure.
`beginResume()` restores counters but not that set, and in `api` mode the local
run store never records per-page URLs, so a resumed run re-saves every duplicate.

**Intended outcome:** fewer duplicate fetches within a run; duplicate and
near-duplicate bodies not persisted; every page that *reaches a handler* carries
exactly one recorded outcome; a resumed run does not redo recorded work.
Exactly-once persistence across crashes is explicitly not claimed.

**Governing principle, applied consistently: fail toward duplication, never
toward loss.** A duplicate row is today's behavior and is recoverable. A silently
dropped page is not.

## Two planes, two vocabularies

The single most important correction from review: Crawlee suppresses a repeated
`uniqueKey` *before any handler runs*, and `filterEnqueueUrls`' local `seen` set
does the same. Neither can produce a ledger record. So "every skip has exactly
one recorded reason" was false as written. Two distinct planes:

**Enqueue plane (pre-fetch).** Duplicate URL candidates are suppressed before a
request exists. These are **deliberately not ledger records** — writing one per
link candidate would mean millions of lines. They are counted in aggregate:
`enqueueAttempts`, `enqueueSuppressedLocal` (our `seen` set),
`enqueueSuppressedQueue` (Crawlee's `uniqueKey` collision).

**Handler plane (post-fetch).** Every page that reaches a request handler ends in
exactly one outcome: `accepted`, or one of five skip reasons, and gets exactly one
ledger record. This is where the one-reason invariant holds.

## Design: two keys

`normalizeCrawlUrl`'s output is *requested* — it is the URL in `filterEnqueueUrls`
and `initialCrawlUrls`. It must stay conservative: unifying `www`/apex there would
request the apex on `www`-canonical sites, adding a redirect to every page. So:

- **`normalizeCrawlUrl`** (unchanged) — a *fetchable* URL.
- **`crawlDedupKey`** (new) — a lossy *comparison* key, never requested.

`crawlDedupKey` default rules, limited to actual equivalences:

- lowercase the host (host is case-insensitive per RFC 3986)
- drop the default port (`:80`/`:443`)
- drop the fragment
- drop `TRACKING_PARAMS` plus an expanded noise list

It does **not** unify `www`/apex, `http`/`https`, path case, trailing slash, or
`index.html`, and — corrected from the previous draft — it does **not sort query
parameters**. Parameter order can be meaningful, especially with repeated keys
(`?tag=a&tag=b`) and application-specific APIs; sorting is a practical heuristic,
not an RFC equivalence. Query order is preserved by default.

`AGGRESSIVE_URL_ALIASES=1` enables the host/protocol/slash/case/index rules **and
param sorting**. Off by default: the failure mode is silent page loss.

Pagination and facet params (`page`, `p`, `sort`, `variant`) are never dropped.
Those are the content layer's job, decided on the body rather than guessed from
the URL.

## What the evidence shows about speed — and its limits

The original n8n evidence (`/integrations/set/` six times) was
**redirect-convergent**: six distinct pre-redirect URLs resolving to one page.
Tracking params were already collapsed at enqueue. So aggressive URL aliasing
would not have prevented those fetches — only post-fetch evidence identifies them.

The fetch reduction comes from, in rough order of value:

1. **A redirect alias table learned at runtime.** When `finalUrl !== request.url`,
   record `dedupKey(request.url) → dedupKey(finalUrl)`; later enqueues resolve
   through it.
2. **Keying `promoteToPlaywright` by dedup key.** Today a `Set` of raw
   `request.url`, so two non-viable variants each get their own browser render —
   the most expensive duplicate in the pipeline.
3. **A raw-HTML hash gate before Readability/Turndown** (see the coverage caveat
   in §6 — it does *not* skip link extraction).
4. **An expanded noise-param list** — unambiguous marketing params only.

**Qualification, per review.** Runtime aliases are **best-effort and only take
effect after the first redirect is observed.** Variants already sitting in the
queue, or starting concurrently, still get fetched — Crawlee offers no clean way
to retract queued requests, and this plan does not attempt it. The handler-level
final-URL reservation prevents duplicate *persistence* for those; it does not
prevent the request. The honest claim is "duplicate fetches fall after alias
discovery," not "duplicate variants are never fetched."

## Changes

### 1. New `src/crawl/dedup.ts`

- `crawlDedupKey(url, opts?)` — as above; `opts.aggressive` adds the evidence-free
  rules including param sorting.
- `AliasTable` — variant key → representative key, `resolve()` chain-following
  (depth-capped, cycle-guarded), `learn()` rejecting self- and cross-site entries.
- `htmlHash(html)` — SHA-256 of raw wire HTML. Reuse the `createHash` idiom in
  `src/storage/raw-body.ts:12`.
- `contentHash(markdown)` — SHA-256, **whitespace collapsed only, case preserved**:
  folding case conflates code, commands, identifiers, and acronyms, and this hash
  asserts identity.
- `simhash64(markdown, shingle)` — 64-bit SimHash over word n-shingles.
  Tokenization may lowercase; this is a similarity signal, not identity.
- `SimhashIndex(threshold)` — banded lookup with **band count derived from the
  threshold**: with `b` bands the pigeonhole guarantee holds only for `d ≤ b - 1`,
  so `bands = threshold + 1` over 64 bits (uneven remainders fine).
  `NEAR_DUP_HAMMING` clamped `0..7`. Fixed 4×16 banding would silently stop
  comparing valid candidates at threshold 4+.

Env: `NEAR_DUP_HAMMING` (3), `NEAR_DUP_MIN_CHARS` (500), `NEAR_DUP_SHINGLE` (5),
`AGGRESSIVE_URL_ALIASES` (off). Thresholds only — no report/enforce/off switch.
Near-duplicate detection is always on and always skips.

### 2. New `src/storage/page-dedup.ts` — reservations and ledger

**Skip reasons, strict precedence, exactly one per page reaching a handler:**

| # | Reason | Test | Where |
|---|---|---|---|
| 1 | `duplicate_url` | alias-resolved final-URL key accepted | handler entry |
| 2 | `duplicate_html` | raw-HTML hash accepted | before Readability |
| 3 | `canonical_alias` | canonical group has a representative | after extraction |
| 4 | `duplicate_content` | exact markdown hash accepted | in `savePage` |
| 5 | `near_duplicate` | SimHash within threshold | in `savePage` |

`canonical_alias` is its own reason and its own ledger result — never folded into
`duplicate_url`.

#### Three-state reservation (the most important fix in this revision)

The previous draft returned a skip for an in-flight key. That is a **page-loss
bug**: handler A reserves, B is recorded as skipped, A then fails and releases —
and B is already gone. Reservation must distinguish three states:

```
reserve(keys) →
  | { state: 'accepted', reason }      // durable; safe to skip
  | { state: 'in_flight', settled }    // a Promise for the owner's outcome
  | { state: 'reserved' }              // we own it; proceed
```

On `in_flight`, the caller **awaits the owner's outcome** rather than skipping:

- owner commits → this page skips with the owner's real reason (durable fact)
- owner releases (extraction or persist failed) → this page **proceeds** and
  takes ownership
- wait exceeds a bound (aligned to `requestHandlerTimeoutSecs`, default 60s) →
  this page **proceeds anyway**, accepting a possible duplicate row

Awaiting is cheap: it yields the event loop, and there is no deadlock risk
because each key has exactly one owner and waiters never own anything the owner
needs. Every outcome either skips on a durable fact or proceeds — never on a
provisional one.

This applies identically to all four reservation kinds: URL key, HTML hash,
content hash, and SimHash. `commit` promotes and appends the ledger line;
`release` drops ownership and settles the promise so waiters wake. Every call
site wraps in `try { … } finally { if (!committed) release(keys) }`. No timeout
sweeper: a thrown handler runs `finally`, and a dead process ends the run.

#### Ledger record and commit boundary

One JSONL line per accepted page in `runs/<runId>/dedup.jsonl`, schema pinned so
resume and counters cannot drift across handlers or persist modes:

```jsonc
{ "v": 1, "pageId": "…", "at": "2026-09-12T…Z",
  "requestedUrlKey": "…", "finalUrlKey": "…",
  "aliasKeys": ["…"],                  // learned redirect variants
  "canonicalKey": "…" ,                // group this page represents, if any
  "htmlHash": "…", "contentHash": "…", "simhash": "…",
  "markdownLength": 12345 }
```

Skips append a parallel `runs/<runId>/dedup-skips.jsonl` line with the reason,
both URLs, and the discriminator `cause: 'accepted' | 'in_flight'` so the metrics
distinguish a durable duplicate from a concurrency collision.

**Durability is explicit and weak:** appends are `writeFile(…, {flag:'a'})`, not
`fsync`ed. A crash can lose the tail. That is precisely why `rehydrate()` tolerates
a truncated final record and why ordering is save-then-append (below).

### 3. Durability: what is and is not guaranteed

A JSONL ledger and an external API page store cannot be transactionally atomic.
**API save first, then ledger append.** A crash between them means resume may
re-save that page — a duplicate row, today's behavior, no regression. The reverse
order would let resume *suppress* a page that was never saved: silent loss.

Limits, stated rather than papered over:

- **Not exactly-once.** That needs a durable idempotency key or uniqueness
  constraint in the receiving store. Out of scope, so the outcome claimed is
  "does not redo recorded work," not "never persisted twice."
- **Single crawler process per run.** The ledger is per-process; two processes on
  one `runId` would duplicate across each other.

### 4. Legacy resume — confirmed no backfill path

`rehydrate()` has nothing to restore for runs created before this change.
**Verified this session:** `src/storage/geek-api-client.ts` exposes only
`createRun`, `patchRun`, `createPagesBatch`, `createLinksBatch` — it is
write-only. There is **no list-pages endpoint**, so backfilling a legacy ledger
from Mongo is impossible without new GeekAPI/.NET work. That is out of scope here.

Policy: resume **proceeds with a loud warning** (hard-refusing would break an
existing operator workflow), and sets `dedupLedgerBackfilled: false` on the run
record so the audit can identify untrustworthy resumed runs.

### 5. Enqueue plane

**First, inventory every queue insertion path** — missing one weakens the speed
claim. Cover: `crawler.run(startUrls)` (seeds + sitemap), each `enqueueLinks` in
`cheerio-runner.ts`, each `enqueueLinks` in `playwright-pool.ts`, the
`promoteToPlaywright` handoff, and any `addRequest`. Crawlee's internal retries
reuse the same `uniqueKey` and need no change; redirect following is not a queue
insert.

At each: set `uniqueKey: aliases.resolve(crawlDedupKey(url)) ?? url` via
`transformRequestFunction`. In `filterEnqueueUrls` / `initialCrawlUrls`, dedup the
local `seen` on the dedup key while still pushing the `normalizeCrawlUrl` form as
the URL to fetch. Key `promoteToPlaywright` by dedup key, value = URL to render.

Count `enqueueAttempts` and `enqueueSuppressedLocal` here; read
`enqueueSuppressedQueue` from Crawlee's queue stats (verify the available API
during implementation).

### 6. Handler plane

In `cheerio-runner.ts`'s `requestHandler`:

**URL key.** After the locale check, learn the redirect alias when
`finalUrl !== request.url`, then `reserve` the alias-resolved final-URL key. This
is the post-redirect catch no pre-fetch normalization can make.

**HTML hash — gate the expensive path only, not link discovery.** Corrected from
the previous draft, which skipped the whole handler. Identical bytes at two
different final URLs **do not imply identical discovered links**: relative hrefs
resolve against the page URL, so absent a `<base>` tag the same HTML yields
different absolute links, and skipping extraction would cut crawl coverage. So on
an HTML-hash duplicate:

- **still** run `extractHrefs` + `filterEnqueueUrls` + `enqueueLinks` (Cheerio `$`
  is already parsed by Crawlee — nearly free)
- **skip** `extractCleanContent` (JSDOM + Readability + Turndown) and persistence

The CPU win was always Readability/Turndown, never link extraction, so nothing is
lost by this correction.

**Canonical — a group with a representative.** Also corrected. Learning
`variantKey → canonicalKey` does not work: the canonical key is not *accepted*
until `/real` is itself crawled, so later variants resolve to a key with no
accepted page and fail to collapse. Instead:

- `canonicalGroups: Map<canonicalKey, { representativePageId, representativeKey }>`.
- Accepting a page that declares same-site canonical `C` registers group `C` with
  this page as representative, if the group has none.
- A later page whose URL key *or* declared canonical resolves to a group **with a
  representative** skips as `canonical_alias`.
- **Explicit exemption:** the canonical URL `C` itself is never suppressed by its
  own group. It stays **eligible to be fetched**; if its content is identical it
  is skipped by `duplicate_content`, leaving the variant as the v1 representative.
  (Previous wording said it "is still fetched and saved," which contradicted the
  content gate — corrected.)

Parse `rel` as a **token list**, case-insensitively: `rel="alternate canonical"`
and odd whitespace are common, and `[rel="canonical"]` misses them. Resolve
against `finalUrl`, strip the fragment, test same-site with `isSameSite` against
`scopeUrl` — **seed-anchored deliberately**, since anchoring to the post-redirect
page is precisely the scope-drift bug `36f8785` fixed. Off-site canonical is
ignored.

Promoting `C` to representative once it is crawled is a documented v1 wart, not a
loss: `canonicalUrl` is stored as page metadata so the Rag side can prefer or
repair it.

Apply the same additions in `playwright-pool.ts` (own `$`, own `savePage` at `:106`).

### 7. Near-duplicate skipping is unconditional — with a recoverable artifact

A page within `NEAR_DUP_HAMMING` of an accepted page is skipped. No mode switch,
no report-only pass; the threshold is the only knob.

Because there is no report mode, the audit trail has to be good enough to judge a
false positive **without a re-crawl**. URLs and a Hamming distance are not —
per review. Each `near_duplicate` skip records:

- both URLs, both content hashes, both markdown lengths, both titles
- the kept page's canonical URL, if any
- the Hamming distance
- a bounded excerpt (first ~500 chars) of each

and writes the **dropped markdown** to `runs/<runId>/near-dup-rejected/<hash>.md`,
capped in aggregate (e.g. 200 files / 50 MB, then counter-only). So a misfire is
recoverable from local disk instead of requiring the site to be crawled again.

Residual risk, stated plainly: unlike the URL/HTML/content hashes, SimHash is a
similarity judgment, so this check can drop a genuinely distinct templated or
paginated page. `NEAR_DUP_MIN_CHARS` keeps short boilerplate out of it. Tune
`NEAR_DUP_HAMMING` down if the samples show false positives.

### 8. Metrics

Per review, one metric per layer rather than a single conflated count:

| Metric | Layer |
|---|---|
| `enqueueAttempts`, `enqueueSuppressedLocal`, `enqueueSuppressedQueue` | enqueue plane |
| `httpRequests` | network |
| `browserRenders` | Playwright pool |
| `extractionInvocations` | Readability/Turndown |
| `pagesSaved` | persistence |
| `skipped{Url,Html,CanonicalAlias,Content,NearDuplicate}` | handler plane |
| `skipCause{Accepted,InFlight}` | reservation outcome |
| `aliasesLearned` | alias table |

Threaded `stats()` → `RunCrawlResult` → `CrawlRunMeta` + `recordRejectStats`
(`src/storage/runs.ts`) → `rejectStatsHostProgressEntry` (`src/crawl/reject.ts`)
→ `patchRun`'s `hostProgressJson`. Additive only; this is the path `df9486b`
already built for `duplicatePagesSkipped`.

## Files

| File | Change |
|---|---|
| `src/crawl/dedup.ts` | **new** — `crawlDedupKey`, `AliasTable`, hashes, `simhash64`, threshold-derived `SimhashIndex` |
| `src/storage/page-dedup.ts` | **new** — three-state reserve/commit/release, canonical groups, ledger schema, rehydrate, counters |
| `src/storage/persist.ts` | ledger wiring, canonical metadata, save-before-append ordering, stats |
| `src/crawl/cheerio-runner.ts` | `uniqueKey` transform, alias learning, URL key, HTML gate (links preserved), canonical, metrics |
| `src/crawl/playwright-pool.ts` | same gates, dedup-keyed promotion, `browserRenders` |
| `src/crawl/sitemap.ts` | `filterEnqueueUrls` / `initialCrawlUrls` compare on dedup key; enqueue counters |
| `src/storage/runs.ts` | new metric fields + `dedupLedgerBackfilled` on `CrawlRunMeta` |
| `src/crawl/reject.ts` | carry new counters in `rejectStatsHostProgressEntry` |
| `src/crawl/dedup.test.ts` | **new** — keys, aliases, fingerprints, banding |
| `src/storage/page-dedup.test.ts` | **new** — reservation states, concurrency, failure, truncation, canonical groups, legacy runs |
| `src/crawl/url-dedup.test.ts` | extend; its `dedupKey` helper must mirror the new key |

## Reuse (do not reimplement)

- `normalizeCrawlUrl` + `TRACKING_PARAMS` — `src/crawl/sitemap.ts:9,95`
- `isSameSite` / `hostnameKey` — `src/crawl/links.ts:26,17` (canonical scope test)
- `extractHrefs` / `sameOriginUrls` — `src/crawl/links.ts` (the HTML-gate link path)
- `RejectSampleLog` — `src/crawl/reject.ts` (capped sample logging, proven)
- `createHash` url-keying idiom — `src/storage/raw-body.ts:12`
- `withRunLock` + append-JSONL pattern — `src/storage/runs.ts:233`

## Verification

**Unit** — `npm test` (`tsx --test src/**/*.test.ts`) and `npm run typecheck`.

*URL keys and aliases*
1. Default key collapses host case, default port, fragment, and noise params.
2. Default key keeps `www`/apex, `http`/`https`, path case, trailing slash,
   `index.html`, **and query-param order** distinct. All collapse under
   `AGGRESSIVE_URL_ALIASES=1`.
3. Repeated keys (`?tag=a&tag=b`) survive unreordered by default.
4. `?page=1` vs `?page=2` stay distinct.
5. Redirect learns an alias; a later variant resolves through it; chains
   terminate; cycles do not hang.
6. Key output is never requested — `filterEnqueueUrls` still returns
   `normalizeCrawlUrl` forms.

*Reservation states — the page-loss regression*
7. Owner commits → waiter skips with the owner's reason.
8. **Owner releases after failure → waiter proceeds and saves.** This is the #2
   defect; it must fail loudly if reintroduced.
9. Owner never settles → waiter proceeds after the bound, producing a duplicate
   row rather than a lost page.
10. Two concurrent identical-HTML handlers: exactly one extracts; the other is
    recorded with `cause: 'in_flight'`, distinguishable from `'accepted'`.
11. Applies to all four kinds: URL, HTML, content, SimHash.

*HTML gate coverage*
12. Two different URLs with byte-identical HTML containing **relative** hrefs:
    both sets of absolute links are enqueued, and `extractCleanContent` runs once.
    (Guards the #6 coverage regression.)

*Canonical groups*
13. Variant accepted, declares canonical `C`; a second variant of `C` skips as
    `canonical_alias` and collapses onto the stored page.
14. `C` itself is **not** suppressed by its own group — it is fetched, then
    skipped by `duplicate_content` if identical, saved if different.
15. `rel="alternate canonical"`, odd casing, extra whitespace all recognized.
16. Off-site canonical ignored; page still saves.

*Content and near-dup*
17. `contentHash` collapses identical markdown at two URLs and **distinguishes
    bodies differing only in case**.
18. Band count tracks threshold — at `NEAR_DUP_HAMMING=5` a distance-5 pair is
    still found (fails with fixed 4×16 banding). Under `NEAR_DUP_MIN_CHARS`,
    never flagged.
19. A near-dup skip writes both hashes, lengths, titles, excerpts, and a
    recoverable `.md` artifact; the aggregate cap degrades to counter-only.

*Durability and legacy*
20. `rehydrate()` tolerates a truncated final JSONL record.
21. `rehydrate()` with no ledger warns, proceeds, marks
    `dedupLedgerBackfilled: false`.
22. Every handler-plane page has exactly one outcome record; enqueue-plane
    suppressions produce none and only move aggregate counters.

*Regression*
23. `reject.test.ts`, `viability-reject.test.ts`, `links-scope.test.ts` green.

**End-to-end** — re-crawl `n8n.io` (443 rows / 358 distinct;
`/integrations/set/` six times). Capture a baseline from the current build first,
then compare per layer, not with one conflated number:

```bash
cd /Users/jeffmartin/development/Geek-Crawler-v2
npm run crawl -- --seed https://n8n.io/ --crawl-type partner
```

- `enqueueAttempts` vs `enqueueSuppressed*` — pre-fetch layer is working.
- `httpRequests` falls vs baseline — the real fetch saving, expected to be
  **partial** given the alias-discovery qualification above.
- `browserRenders` falls — dedup-keyed promotion.
- `extractionInvocations` falls — HTML gate.
- `pagesSaved` ≤ distinct-URL count, usually strictly less (canonical + content
  dedup remove pages with distinct URLs). The earlier draft asserted equality,
  which was inconsistent with having a content layer.
- No two accepted pages share a URL key or a content hash.
- `skipCauseInFlight` should be small; a large value means concurrency is doing
  work twice and the bound needs review.
- Review `near_duplicate` samples and the retained `.md` artifacts by hand.

Wall-clock is a **benchmark, not an assertion** — site content, rate limiting, and
network variability dominate. Request, render, and extraction counts are what
attribute to this change.

Then re-run `--resume` on a run created after this change: no additional accepted
pages.

**Corpus audit** — existing Mongo runs already contain these duplicates; this
change is not retroactive. Count duplicate `Url` per `RunId` to find runs needing
a re-crawl before re-indexing. (Run `b51485e9` should not be indexed as-is.)

## Out of scope

- **Server-side uniqueness in GeekAPI / GeekRepository** — the only route to
  exactly-once across crashes and concurrent processes. Deferred: .NET changes in
  a second repo. Save-before-append keeps the failure mode at "duplicate row."
- **Legacy ledger backfill** — impossible without a GeekAPI list-pages endpoint,
  confirmed absent this session.
- **Retracting already-queued duplicate requests** — no clean Crawlee API.
- Re-crawling or re-indexing affected runs; the separate ~40% duplicate-embedding
  finding on the Rag side (chunk-level, tracked independently).
- Promoting a canonical URL to representative after a variant was stored first.
- Collapsing pagination/facet params by URL rule.

## Revisions from review

**Round 1 (accepted):** conservative-by-default URL key with an evidence-based
alias table; canonical no longer suppresses its own target; atomic
reserve/commit/release; softened durability with save-before-append; legacy-resume
policy; SimHash bands derived from threshold; case preserved in `contentHash`;
one-reason precedence with `canonical_alias` broken out; `rel` as tokens; queue
producer inventory; corrected verification assertions.

**Round 2 (accepted):**
1. Enqueue-plane suppressions defined as unrecorded aggregate counters; the
   one-reason invariant now scoped to the handler plane.
2. **Three-state reservation** — in-flight no longer means skip; waiters await the
   owner's outcome and proceed on release or timeout. This was a page-loss bug
   contradicting the plan's own principle.
3. Canonical rewritten as **groups with a representative**, plus an explicit
   exemption so the canonical URL is never suppressed by its own group.
4. Query-param sorting moved out of the default key into the aggressive set.
5. Redirect-alias speed claim qualified as best-effort post-discovery.
6. HTML-hash gate **preserves link extraction**, skipping only Readability/Turndown.
7. Canonical wording fixed: "eligible to be fetched," then content-deduped.
8. Near-dup audit records both hashes, lengths, titles, excerpts, and a
   recoverable dropped-markdown artifact.
9. Ledger record schema and commit boundary pinned; append durability stated as
   non-fsynced.
10. Per-layer metrics replacing one conflated count.

**Standing divergences, with reasons:**
- **No lock or timeout sweeper for reservations.** Node is single-threaded, so
  check-and-reserve is atomic; races exist only across `await`, covered by
  `try/finally` plus the settle promise.
- **Canonical scope stays seed-anchored**, not final-URL-anchored — anchoring to
  the post-redirect page is the scope-drift bug `36f8785` fixed.
- **Near-dup enforces unconditionally**, per explicit direction. Both the
  report/enforce modes and the per-run policy plumbing were removed at the user's
  instruction; the retained artifact in §7 is the mitigation that replaces them.
