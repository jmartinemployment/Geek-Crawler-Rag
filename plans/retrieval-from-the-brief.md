# Retrieval from the brief — plan, 2026-10-08

## Status, 2026-10-08 (same day)

| Stage | Commit | Deployed |
|---|---|---|
| P1 `keyword` on `/v1/query` → keyword half | Rag `d8a628e` | VPS 14:41 UTC; verified: `keyword: "payment reconciliation"` on pain 5 puts the reconciliation page in slot 1 |
| P2 second BM25 deleted | Rag `d8a628e` | VPS 14:41; `bm25_rank` absent from the image |
| P3 stub chunks dropped, `chunksSkippedStub` | Rag `b14c200` + GeekBackend `36283f1` | VPS 14:41, Railway 14:42; status carries the field (0 until a re-index) |
| P4 evidence rows in the brief | content-creator-v2 `290440c` + GeekBackend `8f3ead6` | Vercel on push; GeekAPI with P5 |
| P4, second cut (Jeff: enter a failure once) — rows **replace** "Where they fail"; `painPoints` derived from the rows' Problem column; paste-to-rows importer; the category's rows shown above a tool's own | content-creator-v2 `4677b69` + GeekBackend `288dde3` | Vercel and Railway on push |
| P5 partner runs asked from the brief | GeekBackend `6ea68fb` | Railway 14:57 UTC |
| P6 probe asks the brief; missing blockquote is a gap | GeekBackend `6ea68fb` | Railway 14:57 UTC |
| P7 crawler: non-content directories | Geek-Crawler-v2 `05b2649` | the crawler runs locally; the next crawl uses it |

**All seven stages are built.** Acceptance tests 1–4 still need, from Jeff: a Tipalti re-crawl
(on `05b2649`, so the legal pages are refused) and its re-index (so stubs are dropped and
`chunksSkippedStub` is reported), evidence rows entered on the brief for Tipalti, then a readiness
run. Test 5 holds on the live box: `hybrid_halves` on every query, `bm25_rank` absent from the
image, `chunksSkippedStub` on the status.

**Written for one session to implement, in order, across three repositories.** Decided with Jeff on
2026-10-08 after a day of live measurement on project `cc480d8c` ("Accounts Payable: Automated
Payment Execution"), scoped to Tipalti (`39bbce59`, the weakest partner) for the acceptance tests.
Every `file:line` is as of the commits named; re-check before editing.

## What was established today (do not re-derive)

- Retrieval was meaning-only until `7c47fa1` (the keyword half never ran); the second, in-process
  BM25 pass was disabled in `e2937f0`; one page could take most of the 32 slots until `3092ccd`
  (page-diverse selection); the 22 fixed words GeekAPI appended to every question were deleted in
  GeekBackend `bfd99c9` (the question is now the bare keyword); the hosts lookup resolves
  `(host, crawlType)` since `0eb139c` + `2d7420b`.
- **Input quality, measured on Tipalti.** Best: a solution description in the vendor's own
  vocabulary, one per pain point — it found the product page for all six problems, including
  reconciliation, tax-form onboarding and migration, which the brief's pain points missed. Next:
  the tool's own `coreProblem` (case studies, pricing). Pain points find the vendor *describing*
  the problem. Worst: generic text — the deleted 22 words, the category `coreProblem` on a tool
  query, the angle question "The cost, delay, error rate and frustration of doing {keyword}
  manually, and how it is fixed" (which still carries the automated-manually contradiction
  `684cc59` fixed in `BuildNeed`).
- **The blockquote probe** (`GccAngleQuoteProbe`, topK 8) fed the selector three stub passages
  (a bare title, a heading, a breadcrumb "Home / AP Automation / Payment Reconciliation"), two
  definitions and nothing from Tipalti's slice. The Library indexes heading-only chunks; nothing
  in `llama_nodes.py` / `chunk.py` has a minimum-content rule.
- **Tipalti's corpus**: 170 pages — blog 28, legal 25 + privacy 5, integrations 21, industries 18,
  ap-automation 14; one product page per pain topic; a services agreement surfaced as evidence.
- **Nothing the operator or Perplexity writes is ever quoted.** The brief is prompt input and
  retrieval input; quotes come only from crawled partner pages (`GccToolQuoteGuard`).

## Jeff's decisions (2026-10-08)

1. Loosen the blockquote: a tool page without a fitting partner quotation is **not refused**; the
   absence is reported. A quotation that is not a verbatim, correctly cited candidate still refuses.
2. **Delete** the second BM25 (it is disabled in place today; the deletion is this plan's).
3. The brief fields are defined here, by this plan, even where they differ from today's tests.

## The brief fields (content-creator-v2 `brief-catalog.ts`, GeekAPI `GccNicheFraming.cs`)

Keep every existing `NicheFraming` field; the writer still consumes them. Add to **`NicheFramingSet`**
(so it exists at category level and per tool, `perTool[host]`):

```ts
/** One retrieval question per failure. Each row is asked of the partner's crawl on its own. */
evidence: EvidenceRow[];
type EvidenceRow = {
  /** The reader's failure, in the reader's words (same voice as painPoints). */
  problem: string;
  /** What the vendor does about it, in the VENDOR'S vocabulary (Perplexity-sourced is fine;
   *  the operator reviews it). Goes to the meaning half. Never quoted. */
  solution: string;
  /** 2–5 distinctive terms from the vendor's vocabulary ("W-9 W-8", "payment reconciliation",
   *  "multi-entity"). Goes to the keyword half. */
  terms: string[];
};
```

UI (`NicheFramingPanel.tsx`, `FramingFields` at `:229-290`, rendered for the category and for each
tool): a rows table under the three existing boxes — **Problem | Vendor's solution | Search
terms** — add/remove row; the hint says the middle column is the vendor's own words and the
terms are what the keyword search matches. Per tool, rows **add to** the category's (same rule as
pain points). `painPoints` stays a paragraph blob for the writer; rows are the retrieval input.

## Stages, in order

**P1 — Library: `keyword` on `/v1/query`.** `models.py:150-168` `QueryRequest.keyword: str | None`
(alias `keyword`); `query.py` `DenseRetriever` protocol and `_query_hybrid` (`:117-130`) pass it;
`llama_engine.dense_query` (`:465-512`) embeds `need` and sets `query_str = keyword or need`.
Test through `dense_query` (extend `tests/test_dense_query_is_hybrid.py`): `query_str` is the
keyword when given, the need otherwise. Done when a live query with both logs `hybrid_halves`.

**P2 — Library: delete the second BM25.** Remove the commented block and the two commented imports
in `query.py` (`_query_hybrid`, module docstring), `_lexical_doc`; delete `bm25_rank.py`, `rrf.py`;
`tests/test_chunk.py:16-17` imports and `:279-290` `test_rrf_and_bm25`; `pyproject.toml:27`
`rank-bm25` then `uv lock` (CI runs `uv sync --locked`); README `:74`, `:150` and HANDOFF §7/§10
lose the "disabled, kept" wording. Done when `grep -rn "bm25_rank\|rrf" src tests` is empty and
the suite passes.

**P3 — Library: stop indexing stub chunks.** In `chunk.py` `parent_child_units` (`:220`) /
`split_blocks_into_sections` (`:147`), drop a unit whose child text (a) has fewer than 8 words, or
(b) equals its section title, or (c) matches a breadcrumb (`^[^/]{1,40}( / [^/]{1,40}){1,6}$`).
Count it as `chunksSkippedStub` in the index status (`models.py` `IndexStatusResponse`,
`status_store._from_doc`, and `contracts/rag-index-status/webhook.v1.json` with the GeekBackend
copy, same commit pair). Done when a Tipalti re-index (Jeff queues it) no longer returns the three
stubs above as passages and reports the count.

**P4 — Brief fields.** content-creator-v2: `brief-catalog.ts:219-234` type + `emptyNicheFramingSet`
+ the rows UI; GeekAPI `GccNicheFraming.cs` reader (`:422` region) reads `evidence` for category
and per tool; `GccNicheFramingReaderTests` round-trip. Done when rows saved from the form come
back from `GET repo/content-creator/projects/{id}` and the reader.

**P5 — GeekAPI: retrieval from the brief.** `HttpGeekCrawlerRagClient.QueryAsync` (`:528-567`)
gains `keyword`. In `GccGroundingResolver` (`:462-530`), for a **partner** run: one query with
`need = coreProblem` (tool's, else category's), then one query per evidence row with
`need = row.solution`, `keyword = join(row.terms)`, each `topK` 8 (new const `EvidenceTopK`);
the run's pages are the union, deduped by URL as today (`seenByCrawlType`). With no rows and no
`coreProblem`, fall back to `BuildNeed` (keyword alone) at `PartnerTopK` 32. Site and competitor
queries unchanged. Done when Tipalti's pool contains `/ap-automation/automated-payment-reconciliation/`,
`/mass-payments/self-service-onboarding/`, `/ap-automation/multi-entity/`, and the readiness
category count for Tipalti and AvidXchange is measured before/after (the 2-of-20 refusal).

**P6 — GeekAPI: the blockquote.** (a) The probe asks the brief's question: `GccController.cs:1003`
builds the spec; `GccAngleQuoteProbe.cs:127` sends `spec.Need` — use the tool's `coreProblem`
(else category's) as `need` and the first row's terms as `keyword`; `spec.Rule` unchanged.
(b) Loosen the guard: `GccToolQuoteGuard.FindViolations` returns the two "no block quotation"
messages (`:58-60`) as a **gap**, not a violation; `GccDraftGuard.cs:109-118` records
`GccGuardFinding("blockquote-missing", …, Refuses: false)` for it and keeps `Refuses: true` for a
quotation that is empty (`:86`), not a verbatim candidate, or cited to another page.
`GccGenerateService.cs:1611-1612` and the tool prompt say: include one block quotation only when a
supplied candidate answers the problem; otherwise none. Done when a tool page with no fitting
candidate ships with the `blockquote-missing` warning in `warnings` and the run log, and a
fabricated or misattributed quotation still refuses (both pinned by tests).

**P7 — Crawler: no legal, privacy or careers pages in a partner corpus.** Geek-Crawler-v2
`classify-path.ts:26` (`PageTier`) and `:95` (`classifyPath`): classify `/legal/`, `/privacy`,
`/terms`, `/careers` as `non_content`; `reject.ts:10-40` gains `non_content_directory`, counted in
the ledger like `locale_excluded`. Done when a Tipalti re-crawl has zero such pages and
`refused.non_content_directory` reports them; check `/resources/customer-stories/` coverage in
the same run (it is where Tipalti's outcome evidence lives).

## Order and gating

P2 → P1 → P3 (Library; push once, scheduler idle, no GeekAPI traffic — the usual guard) →
P4 → P5 → P6 (frontend + GeekAPI; contract commits paired) → P7 (crawler) → Jeff queues the Tipalti
re-crawl and re-index → the acceptance tests. Nothing here re-indexes or re-crawls on its own.

## Acceptance tests (all on Tipalti `39bbce59`, then the re-crawled run)

1. Pains 2, 5, 6 (tax-form onboarding, reconciliation, migration) each surface their product page
   in the top 8 of their own row's query. (Before: all three missed with the brief's pain points.)
2. No stub passage in any top 8.
3. The probe's candidate set for the problem_solution angle contains at least one passage from
   Tipalti's slice (global / multi-entity / payouts / tax), not generic AP.
4. A readiness run: Tipalti's category count, before and after.
5. `hybrid_halves` logged on every query; `grep bm25_rank` empty; `chunksSkippedStub` reported.

## Rules that bind this plan

Fail closed, no fallbacks, no Markdown (`CLAUDE.md` §1a, §2). The blockquote loosening is a
*reported gap*, not a fallback: nothing is substituted for the missing quotation. Nothing the
brief holds is ever quoted. Contract changes are committed on both sides together. Push only with
no index job running and no GeekAPI call in the last three minutes.
