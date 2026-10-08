# Partner evidence reaches the writer, and every claim about a tool has a quote

Written 2026-10-04 from the investigation of one Content Creator refusal. **Status corrected
2026-10-08.** The line that stood here — "Nothing here is built" — was false on the day it was
written: Stages 1, 2 and 3 and the Library half of Stage 4 were committed that morning, and
Stage 5 shipped on 2026-10-08 in a different shape. The investigation below is kept as the dated
record it is; each stage now carries its state, and the measurements carry the label of the
retrieval path they were taken on, which no longer exists.

## Status, 2026-10-08

| Stage | State |
|---|---|
| 1 — a repeated text is indexed once per run | **Built 2026-10-04**, Rag `1a25095` + `03b02c7`: `textDigest` on every point, `indexer._admit_page_nodes`, `chunksSkippedRepeat` on the status and the webhook. Done-when met on the old Ramp run: 33,728 → 26,683 points (7,045 repeats skipped); the five plan questions went from 1/1/3/1/1 passages to 32 each, from 20–28 distinct pages. That run (`f8a3aa8c`) has since been deleted; Ramp is `4563f7ec` now (2,150 pages, 28,150 points, 7,339 repeats skipped). |
| 2 — follow links the sitemap omits | **Built 2026-10-04**, Geek-Crawler-v2 `195e2df`: every same-origin link is a candidate; off-sitemap product and evidence links go to the front of the queue; other-tier off-sitemap pages get a per-directory cap; `offSitemapAdmitted` in the discovery ledger. Ramp was re-crawled on it as `4563f7ec`. **Its Done-when has not been measured** (`/products`, `/bill-pay`, `/accounting-automation`; bill.com `/pricing`). |
| 3 — no affiliate wording or category | **Built 2026-10-04**, GeekBackend `8917237` (freshness and disclosures dropped with it). `affiliate` returns one hit across every `.cs` file: the comment at `GccNicheFraming.cs:25`. Decision 3 is all that is left. |
| 4 — extracted items are verified before they count | **Library half built, GeekAPI half open.** `POST /v1/verify` exists (Rag `1a25095`, finished `9a0c901`; `app.py:731`; `tests/test_verify_route.py`) — decision 4 is taken. The FAQ prompt's "already verified" sentence was removed (GeekBackend `185df80`). **Nothing in GeekAPI calls `/v1/verify`** (zero hits in any `.cs`); `VerifyAgainstLibraryAsync` is still reachable only from the dormant `GccV2GeekCrawlerResearchResolver`; the live tool path trusts extraction. This is A1 full in `content-creator-v2/plans/fix-geekapi.md`. |
| 5 — one Library question per claim | **Superseded** by `plans/retrieval-from-the-brief.md` P5 (GeekBackend `6ea68fb`, 2026-10-08). A partner run is asked its core problem, then one question per **evidence row** (problem / vendor's solution / search terms), eight passages each — not one per paragraph; the "where they fail" paragraphs no longer exist as an input (`painPoints` is derived from the rows since content-creator-v2 `4677b69`). The results are merged by URL (`GccGroundingResolver.cs:207`): **no passage is kept with its claim**, which is the one property Stage 6 reads from this stage. |
| 6 — a gate per content type replaces "3 of N" | **Not built.** `HasSufficientPartnerData` (`GccGenerateService.cs:685`, `:1380`, `:1758`) and `CountPopulatedPartnerDataCategories` (`:698`, `:1767`) are live; the bar is 3 of 20 (22 before `8917237`). No fill-rate measurement exists. |

This file lives in `Geek-Crawler-Rag/plans` because the investigation ran here. Stage 1 and the
Stage 4 route are this repo's work and are built; the rest belongs to `Geek-Crawler-v2` (Stage 2)
and `GeekBackend` (Stages 3–6). The per-repo copies are `content-creator-v2/plans/fix-geek-crawler-rag.md`
(R1–R7), `fix-geek-crawler-v2.md` (C1–C5) and `fix-geekapi.md` (A1–A19); their headers still say
"Nothing in 'The work' is built", which is as stale as this file's was. Every `file:line` below was
read on 2026-10-04 and most have moved since.

## The refusal that started this

> Pillar names 4 of 5 partner tools, after a retry naming the omission. Missing: Ramp.

Ramp was crawled and indexed: 2,025 pages, 33,728 chunks (section 2 shows the crawl was not full —
the sitemap allowlist dropped its product pages). The refusal comes from the required-mentions check
(`GccGenerateService.cs:2455`), not from the partner-data gate (`HasSufficientPartnerData`, `:1560`)
and not from the index-size thresholds (25 pages / 250 chunks).

## What was found

### 1. The writer was handed one passage about Ramp

On 2026-10-04 Content Creator asked the Library one question per partner run, 32 passages each
(`GccGroundingResolver.cs:385-398`, `PartnerTopK = 32`), built from the keyword alone (`BuildNeed`,
`:515`). (Since `6ea68fb` a partner run is asked the brief's core problem and one question per
evidence row at 8 each; the keyword-only question at 32 is used only when the brief has no framing
for that host.) Sending that exact question on 2026-10-04:

| Partner | Automated Payment Execution | Automated Invoice Processing | Accounts Payable Automation |
|---|---|---|---|
| bill.com | 32 passages / 19 pages | 32 / 15 | 32 / 19 |
| tipalti.com | 32 / 15 | 32 / 16 | 32 / 17 |
| melio.com | 32 / 22 | 32 / 17 | 32 / 16 |
| stampli.com | 10 / 10 | 32 / 20 | 31 / 20 |
| **ramp.com** | **1 / 1** | **1 / 1** | 32 / 29 |

**Label, added 2026-10-08.** Both tables in this section were measured on a retrieval path that no
longer exists: dense-only (the hybrid's keyword half did not run until `7c47fa1`, 2026-10-08),
top-of-list selection (page-diverse since `3092ccd`), before index-time collapse (`1a25095`). The
"at least 10 passages from at least 10 pages" bar below was calibrated on Stampli's result here and
does not transfer; re-baseline on `4563f7ec` with current code before using either table as a floor.

The single Ramp passage is from `/charge-finder/albertsons`. Which keyword the refused draft used
was not established; for the third keyword Ramp returned 32 passages from 29 pages, so if the draft
used that one the mechanism below did not apply and the refusal needs another explanation.

**Cause, for the two keywords that returned one passage.** Ramp has 194 `/charge-finder` pages that
carry the same block of text. Identical text has identical vectors, so the copies filled the dense
candidate list (`query.py:115-145` on 2026-10-04; limit `max(topK*2, 30)`). The copies were removed
only afterwards, in `_select_ranked_candidates` (`if text in seen_text: continue`), which left one.
The real Ramp pages never entered the list. (The plan originally said the copies filled "both
candidate lists". The keyword scroll list returned nothing for four of five Ramp questions and was
deleted the same day, `5e622b6`; the hybrid's own keyword half was not running. One list was
flooded.)

Scale: 7,046 of Ramp's 33,728 points (21%) were byte-identical copies. One call-to-action sentence
appeared 1,088 times — that is site chrome (a footer call-to-action on every page), a second
phenomenon from the charge-finder block; both are what Stage 1's point-count drop measures.
Corpus-wide the figure was 15% on 2026-10-01.

This was the open item 2 in the indexing-throughput notes ("duplicates still skew fusion before
anything collapses them"), in its severe form.

**Questions shaped like a claim collapsed too.** Asked of the Ramp run at 32 passages, same path:

| Question | Passages returned |
|---|---|
| Ramp bill pay payment methods ACH check card wire | 32 (26 blog, 6 top-level) |
| Ramp syncs with QuickBooks NetSuite Xero accounting | 1 (`/integrations`) |
| Ramp approval workflow who approves and who releases payment | 1 (`/charge-finder`) |

Stage 1 shipped first (2026-10-04); the brief-driven questions shipped on 2026-10-08. After
`1a25095` the five plan questions returned 32 passages each (commit `03b02c7`).

### 2. Product pages were never fetched

When a site had a sitemap, the crawler dropped every discovered link that was not in it
(`Geek-Crawler-v2/src/crawl/sitemap.ts:273` on 2026-10-04), with no counter and no log line. Ramp's
sitemap (3,856 URLs) omits `/products` — linked from 1,997 of its 2,025 crawled pages — along with
`/product-releases`, `/bill-pay` and `/accounting-automation`. Fixed in `195e2df` (Stage 2).

Across the corpus: 54 of 65 runs had a sitemap; 39 dropped at least one product-tier page and 24
dropped five or more. Examples: bill.com `/pricing` and `/product/card`; every lightyear.cloud
`/features/*` page; 39 centime.com product pages; dext.com `/products/*`. In all, 16,005 distinct
paths were dropped, 589 of them product-tier and 264 evidence-tier. That is an upper bound: the
sitemaps were loaded on 2026-10-04, not at crawl time, and the count includes stale `-old` pages and
Stripe's foreign-locale copies.

The earlier conclusion that Ramp's missing product pages were "not a crawling-scope problem" was
wrong. The replay behind it replayed the section quotas only and could not see this filter.

### 3. Extracted items are trusted without a check

- The verify pass, `GccV2PartnerExtractionVerify.VerifyAgainstLibraryAsync`, has one caller:
  `GccV2GeekCrawlerResearchResolver:482`. That class is registered and nothing in GeekAPI uses it.
  (Still true on 2026-10-08: its caller `MergeExternalResearchAsync` has no caller.)
- `GccV2PartnerExtractionService.IsGrounded` (`:371`) has no caller at all. (Still true.)
- On the live path, extraction goes straight to the gate (`GccGenerateService.cs:1163-1168`, and
  the readiness path at `:540`), then into the FAQ prompt, which said *"Every answer below is
  already verified against the partner's own site"* (`ContentPromptBuilder.cs:2805`). **That
  sentence was removed in `185df80` (2026-10-04)**; the phrase returns zero hits now. The field is
  still named `VerifiedAnswer`.
- The verify pass, as written, returns the extraction unchanged when the Library client is disabled
  (`if (!rag.IsEnabled) return extraction;`, `GccV2PartnerExtractionVerify.cs:28`). That is
  fail-open, in code that is dormant.
- Block quotations on the tool page **are** checked: `GccToolQuoteGuard.FindViolations` — against
  spans GeekAPI cuts itself from GeekRepository blocks, not against the Library (see
  `plans/audit-content-creator.md`).

`CLAUDE.md` §2: never document a safety property that no check enforces. The prompt sentence was
one; it is gone. The gap it named — nothing verifies an extracted item — is Stage 4's open half.

### 4. The categories are a gate, not a structure

- Only `FaqBank` is read by name on the live path (`GccGenerateService.cs:1413`). The others reach
  the writer as one serialized block (`:1215`, `:1287`) and the structured-data builder (`:1520`).
- The gate counts each category equally and one item is enough (`:1564-1586`). One fact, one award
  and one call-to-action pass it. (On 2026-10-08: `>= 3` of 20, `GccGenerateService.cs:1760`.)
- Which category a sentence lands in is the model's choice. `GccToolQuoteGuard.cs:125` already
  recorded a page refused because findings were filed under features and pricing rather than
  testimonials and citables.
- `AppendPartnerExtractionNotes`, the per-category rendering, is called only inside
  `GccV2ContextAdapter`. The Create path does not use that adapter.

### 5. Affiliate wording contradicted the operator's decision — fixed

Jeff, 2026-10-04: *"I have edited out Affiliate/Partner from data I enter, and has no place in my
site."* `GccNicheFraming.cs:25` already recorded the same decision from 2026-10-02. Against that, on
2026-10-04, there were 19 mentions in GeekAPI: the extraction prompt opened *"A partner is a
third-party SaaS product that the operator promotes for affiliate revenue"*
(`GccV2PartnerExtractionService.cs:52`), `affiliateDisclosures` was an extraction category the gate
counted, and so on. **`8917237` removed them the same day.** One occurrence remains, the comment at
`GccNicheFraming.cs:25` that records the decision; whether it stays is decision 3.

### 6. Retrieval never saw the operator's framing — fixed in a different shape

The niche framing (core problem, the failures, the automation to pitch, diagnosis questions, and
the per-tool overrides) is what the page argues. Its own doc comment sets the rule: *"A page may say
the problem is late invoices because the operator says so; it may not attribute a capability to a
product on this basis."*

The per-tool boxes do attribute capabilities. The Melio example asserted that it connects to
QuickBooks Online, QuickBooks Desktop or Xero; pays by ACH, card or check; lets a bookkeeper prepare
and an owner release; and syncs payment status back. Each needs a quote from a Melio page.

On 2026-10-04 `GccNicheFraming` was read by `GccGenerateService` and `ToolPrompts` only;
`GccGroundingResolver` did not read it. Since `6ea68fb` the resolver asks each partner run the
brief's core problem and one question per evidence row (`GccGroundingResolver.cs:679-703`). Whether
a passage backing "Melio connects to QuickBooks" is retrieved is no longer chance — but it is still
not **kept with the claim**, so no gate can ask whether the claim has one (Stage 6).

## Ruled out, so it is not tried again

- **Directory caps.** The site budget (`MAX_PAGES_PER_SITE = 2500`) had not been reached on any run,
  so a cap frees nothing and adds no evidence. The 1,088-copy chrome sentence is not in one
  directory, so no directory cap reaches it. (Stage 2 did add a per-directory cap, but on
  *off-sitemap other-tier* admissions, as a trap defence — a different purpose.)
- **Removing `integrations` from the crawler's product list, or capping it.** Tried and reverted on
  2026-10-04. Integrations is evidence the page needs (two of the four Melio claims above). It was
  not the cause of the Ramp refusal. Across 35 sites, 38% of integrations pages carry 1,500
  characters of their own text; Parseur (98%), Avalara (74%) and Tipalti (71%) are real write-ups,
  Ramp (7%), Anrok (2%) and Numeral (0%) are not. The directory name does not tell them apart.
- **Treating the crawler's product/evidence list as a whitelist.** It is a priority list
  (`classify-path.ts`): it sets crawl order and exempts evidence from the 20% editorial share.
  Nothing is blocked for being off it. 62% of crawled pages are in directories on neither list.
- **The page-level similarity filter** (`dedup.ts`, 64-bit simhash, distance ≤ 3). It catches 0 of
  780 sibling pairs in every Ramp directory; sibling distances are 24–31 bits, the same as unrelated
  blog posts. It compares whole pages; the repetition is in blocks.
- **Text density, script share and sibling similarity as a page filter.** Generated directories and
  product pages overlap (median prose 2,721 vs 3,398 characters). Script share is a property of the
  site, not the page (Ramp 85%, Stripe 69%).
- **"Own text ≥ 1,500 characters, or an FAQ plus 600" as a filter.** It keeps 65% of 26,752 pages.
  It removes Ramp `/integrations` (8% kept) and `/rewards` (0%), but keeps `/expense-category`
  (78%) and also drops 27% of product pages and 15% of case-study pages corpus-wide. Useful as a
  measurement, not as a gate.
- **The thresholds.** Neither 25 pages / 250 chunks nor 3 of 22 caused the refusal.

## Stages

Order: 1 before 4 and 5 (a partner that returns one page has nothing to verify and nothing to
ask), 3 before 6, 4 before 6. Stages 1 and 2 are independent of each other and of 3. In the event:
1, 2, 3 and the Library half of 4 on 2026-10-04; 5 on 2026-10-08.

### Stage 1 — a repeated text is indexed once per run (`Geek-Crawler-Rag`) — built

**What was built** (`1a25095`, `03b02c7`). At index time, within a run, a chunk whose embedded text
has already been emitted by an earlier page of that run is not emitted again. The key is a SHA-256
digest of the exact string that is embedded, stored on the point as `textDigest`
(`llama_nodes.py:46-53`, `:273`; `indexer._admit_page_nodes`, `:863-896`). The set is built from
scratch on every attempt, because every attempt deletes the run's points first (`03b02c7`;
`indexer.py:669-680`) — there is no resume path, and none may be reintroduced. The index status
reports the skipped repeats as `chunksSkippedRepeat` (`models.py:46`; webhook contract both sides,
GeekBackend `1c553fc`).

Embeddings were already deduplicated (`782d0f0`) and duplicate parent points within a page removed
(`04310dd`); cross-page points were not. This closed that.

**Why at index time and not at query time.** On 2026-10-04 collapsing at query time would have
needed it in two places that worked differently — `query_points` and the `search_text` scroll. The
scroll was deleted the same day (`5e622b6`) and retrieval is one hybrid call now, but the reason
stands: one rule at index time removes the copies for every consumer, including the verify route.

**Cost, stated.** A repeated block is retrievable from one page of the run, the first that carried
it. Quote verification still holds, because that page does contain the text. If that page is later
deleted, the block is absent until the run is re-indexed.

**Done-when — met, 2026-10-04**, on the old Ramp run `f8a3aa8c` before it was deleted: 33,728 →
26,683 points (7,045 skipped); the two claim questions that returned one passage returned 32; the
five plan questions came from 20–28 pages each. A re-index today cannot repeat the point-count
measurement cleanly — stub chunks are dropped too since `b14c200` — so `chunksSkippedRepeat` on
`GET /v1/index/{runId}` is the figure to read.

**Still open from this stage: the near-copy measurement.** Whether the first 32 passages are now
filled by near-copies (the same paragraph with a merchant or app name swapped) has not been
measured on `4563f7ec`. The measure is known — a paragraph is templated when at least 60% of its
4-word phrases appear on at least 30% of sibling pages — and gets its own plan if the measurement
says so. Do not build it on speculation. (R3's second half in `fix-geek-crawler-rag.md`.)

**Constraints.** A push to this repo deploys and recreates the container; not while indexing is
running. Re-indexing needs no approval (the corpus is test data). No embedding retries.

### Stage 2 — follow links the sitemap omits, and count them (`Geek-Crawler-v2`) — built, unmeasured

**What was built** (`195e2df`). The allowlist test is gone: "Every link is a candidate, sitemap or
not" (`sitemap.ts:402`). A same-origin link not in the sitemap goes through the same admission as
any other; off-sitemap product and evidence links go to the front of the queue; other-tier
off-sitemap pages are capped per directory as a trap defence (`sitemap.ts:321`). The sitemap keeps
its other job, seeding the crawl in priority order. The discovery ledger counts
`offSitemapAdmitted` (`discovery-ledger.ts:82`).

**Not re-checked here.** Whether `cheerio-runner.ts` still sizes the crawl from the sitemap
(`maxRequestsPerCrawl = clamp(max(min(sitemap size, profile default), seeds))`). With off-sitemap
pages admitted, that ceiling must not be the sitemap's size, or the new pages only displace listed
ones. Also whether the ledger's counters reach Mongo: on 2026-10-04 they lived only in the local
`run.json`; `hostProgressJson` is sent to GeekAPI now and `GET crawls/{runId}` returns it untyped.

**Done-when — not measured.** Ramp was re-crawled as `4563f7ec` (2,150 pages) on this code; whether
it fetched `/products`, `/bill-pay` and `/accounting-automation`, and how many off-sitemap pages
were admitted, has not been read off the run. Same check on bill.com (`/pricing`). lightyear.cloud
is no longer offered as a check.

### Stage 3 — remove affiliate wording and the category (`GeekBackend`) — built

`8917237`: the extraction prompt's definition of a partner rewritten, `affiliateDisclosures` and the
freshness log removed from the schema, the document model and the gate, the "affiliate perk"
wording gone. Zero hits for `affiliate` across every `.cs` file except the one comment.

**Decision for Jeff (open).** `GccNicheFraming.cs:25` states the absence in a comment. By the rule in
`CLAUDE.md` §1a — the test is whether an occurrence can be read as evidence — that comment would go
as well. Say whether it stays as the record of the decision.

### Stage 4 — extracted items are verified before they count (`GeekBackend`) — route built, wiring open

**Built.** `POST /v1/verify` (`app.py:731`): "Is each quote on its page? The only place that question
is answered." It calls `verify_quote` (`citation_verify.py:57`), which normalises the two recorded
differences between the C# block projection and the Library's and otherwise compares the page's own
characters — curly quotes are not folded, by instruction. `tests/test_verify_route.py`. The FAQ
prompt's "already verified" sentence is out (`185df80`).

**Open — the GeekAPI half.** Run the verify pass between extraction and the gate, at both live sites
(`GccGenerateService.cs:540` and `:1163` on 2026-10-04). An item whose quote is not found in its
page's text is removed from the document. It does not count toward any gate and does not reach any
prompt. If the Library cannot be asked, the create is refused; the unverified document is never used
in its place. The pass calls `/v1/verify` and keeps no comparison of its own: the existing
`pageText.IndexOf(quote, OrdinalIgnoreCase)` over `GET /v1/pages/{page_id}`
(`GccV2PartnerExtractionVerify.cs:148`) is a second rule for one question and is deleted, not kept
as a fallback.

**Also open.** The tool page's block quotation is checked by `GccToolQuoteGuard` against spans GeekAPI
cuts from GeekRepository blocks (`GccQuoteCandidates`), never against the Library — a third rule for
the same question. Whether that check should also go through `/v1/verify` is for the audit's plan.

**Expect.** More refusals at first: the gate will count fewer items. That is the check working.

**Done when.** A test feeds an extraction with one item whose quote is not on the page and shows it
is absent from the gate count, the prompt and the structured data; and `grep v1/verify` finds the
call.

### Stage 5 — one Library question per claim in the framing (`GeekBackend`) — superseded

Shipped as `plans/retrieval-from-the-brief.md` P5 (`6ea68fb`): one question per **evidence row**,
not per paragraph. The rows (problem / the vendor's solution in the vendor's vocabulary / search
terms) are the operator's input now; the paragraph boxes this stage proposed to split are gone, and
`painPoints` is derived from the rows. Eight passages per row, page-diverse on the Library's side.

**The delta Stage 6 still needs.** The passages are merged into one pool per run, deduped by URL
(`GccGroundingResolver.cs:207`). Nothing records which row a passage answered, so the writer is not
shown which evidence backs which statement and no gate can ask whether a row has any. Keeping the
row id on each passage is additive and is the precondition for Stage 6's claim check.

**Open.** Whether the rows' passages should be banked with the extraction. The Library's `topK`
ceiling is 50 per call.

### Stage 6 — a gate per content type replaces "3 of 20" (`GeekBackend`) — not built

**Change.** Each content type declares what it needs, and is refused when that is missing — not
when fewer than three of twenty unrelated buckets are filled. Two kinds of requirement:

1. **Claims.** Every statement the framing makes about a specific tool — every evidence row — has at
   least one verified quote (Stage 4 plus Stage 5's delta), or the page is refused naming the row.
   No claim is written without one.
2. **Categories.** The verified categories that type's sections are built from.

A first mapping, for discussion — it is not in the code:

| Content type | Categories it would draw on |
|---|---|
| Tool page | features, pricing, integrations, limits, FAQ, compliance |
| Pillar, Blog | facts with numbers, case studies, testimonials |
| Comparison | comparisons, battlecards, features, pricing |
| Alternatives | alternatives |
| Case study | case studies, testimonials, facts with numbers |
| Ads | advertisements, offers |
| Email, Social | offers, facts with numbers |
| Listicle | awards, categories |

Who-it-is-for, use cases, disqualifiers and demo beats are what the operator's framing already
supplies; extracting them from vendor marketing duplicates it with a weaker source. Whether those
categories stay is a decision for Jeff (the freshness log and disclosures are already gone).

This stage is motivated by section 4, not by the Ramp refusal: the 3-of-N gate did not cause that.

**Measure first.** Before fixing any type's list, count how often each category fills, with verified
items only, across the stored extractions for the indexed partners. A type whose categories are
mostly empty on real sites cannot be enabled, whatever its gate says. Comparison and Alternatives
are the ones to check. No such count exists yet.

**Done when.** `HasSufficientPartnerData` and `CountPopulatedPartnerDataCategories` are gone, each
enabled type has its own requirement, and each refusal names the row or category that was missing.

## Decisions still open

1. Stage 3: whether the comment at `GccNicheFraming.cs:25` stays.
2. Stage 4: wiring — GeekAPI's verify pass calls `/v1/verify` and its `IndexOf` comparison is
   deleted (the route choice itself is made). And whether the tool-page quotation check joins it.
3. Stage 5's delta: keep the row id on each retrieved passage (needed before Stage 6).
4. Stage 6: a framing row with no verified quote refuses the page (recommended, and what
   `CLAUDE.md` §2 requires) — confirm, since it will refuse pages that are written today.
5. Stage 6: which of the remaining 20 categories are dropped outright.
6. Stage 2: what the crawl ceiling should be once off-sitemap pages are admitted, if
   `cheerio-runner.ts` still sizes it from the sitemap.

Decisions 1 (index-time collapse), 2 (admit every off-sitemap link — built with a third shape, the
trap defence) and 4 (a Library verify route) from the original list are taken in code.

## Related plans

- `plans/retrieval-from-the-brief.md` (this repo, 2026-10-08) — the shape Stage 5 shipped in; all
  seven of its stages are built.
- `plans/audit-content-creator.md` (this repo, 2026-10-08) — the audit that found the GeekAPI half
  of Stage 4 unwired and the quotation check running on a third rule.
- `content-creator-v2/plans/fix-overview.md` and its five per-repo files — the R/C/A/D/F stage
  numbers; their "nothing is built" headers are stale, and the retired-plans list there is where
  `remove-unwired-code.md`, `bank-extraction-and-legible-quote-guard.md` and
  `grounded-generation-and-serp.md` went.
- `GeekBackend/plans/competitor-partner-data-purpose.md` (2026-09-22) — describes partners as
  promoted products. Superseded on that point by the 2026-10-02 and 2026-10-04 decisions.
- `GeekBackend/plans/audit-does-the-writer-use-rag.md` — the earlier trace of the same path.
