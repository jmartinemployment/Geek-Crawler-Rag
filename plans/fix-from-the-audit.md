# Fix from the audit — plan, 2026-10-08

**Written for one session per repository to implement, in order.** Source: `plans/audit-content-creator.md`
(same day), whose F-ids this plan cites; every `file:line` is as of GeekBackend `288dde3`,
content-creator-v2 `4677b69`, Geek-Crawler-v2 `05b2649`, Geek-Crawler-Rag `e65dd5e` — re-check
before editing. Nothing here is built.

The audit's answer to "is this a best-in-class document writer" was no, for six reasons (audit §0).
This plan removes them in the order that buys the most trust per commit: first a run cannot be
lost to a provider fault; then everything a page says about a partner is verified by one rule;
then a page is refused for the reason that matters and the operator sees it; then the prompt says
each thing once and in words; then the dead weight goes.

## Stages

Each stage names its repository, the change, the files, and its done-when. **Every stage rewrites
every comment, message and hint in the files it touches so they say what the code does**
(CLAUDE.md §1a: a surviving name is read as a live claim). Audit F18's table is the checklist; the
rows are assigned to the stage that touches their file.

### X1 — A provider fault is a fault, not a lost run (GeekBackend) — F5, F6

**Change.** `AnthropicProvider` catches `OperationCanceledException` when the caller's token is not
cancelled — the `HttpClient` timeout, the same test `OpenAiCompatibleOutcome.IsTimeout` makes — and
throws `ContentGenerationException("Anthropic did not answer within {TimeoutSeconds} seconds.")`
(Anth:104-111 catches only `HttpRequestException` today; OAI:98-102 is the model). On every 200,
`stop_reason` is read (Anth:140-144 reads none); `max_tokens` becomes the same typed outcome as
OpenAI's `finish_reason=length` (`OpenAiCompatibleOutcome.cs:36-45`): the partial is kept, the run
log classifies it `fault`, the type refuses with the reason. `GccRunLog` records a call that ended
in a cancellation as a `fault` event instead of skipping it (Log:161). What is sent to the model
does not change in this stage; whether to send a thinking budget is decision 1 below.

**Done when.** Tests: a handler that answers after the timeout yields, through
`GccGenerationCoordinator` with a fake provider, a refusal of that type naming the provider and the
seconds, while the other requested types settle and save; a reply with `stop_reason: "max_tokens"`
is a `fault` with the partial recorded and the type refused; no `OperationCanceledException`
escapes `AttemptAsync` (Coord:442-478) for a timeout. The stale Ext:87-88 "never throws" is made
true or rewritten.

### X2 — One verifier: `/v1/verify` for every extracted item and every quotation (GeekBackend) — F1, F4

**Contract (exists, Library `models.py:341-370`, `app.py:731`).** `POST /v1/verify`
`{runId, quotes: [{pageId, quote}]}` (1–200 quotes) → `{runId, results: [{pageId, quote, found,
reason, sourceDigest}]}`. `found` is the verdict; `reason` is `page_not_found`,
`page_not_citable:<why>` or `not_on_page`; `sourceDigest` identifies, never judges. GeekAPI
compares nothing itself. `pageId` is the crawl page's Mongo id — the id GeekRepository's
`pages/by-seeds` returns and `GET /v1/pages/{id}` takes.

**Change.** `IGeekCrawlerRagClient.VerifyQuotesAsync(runId, quotes, ct)` in RAG, with the contract
pinned in `RagClientContractTests`. A verify pass on the live path at both sites —
`AssessPartnerToolReadinessAsync` after `ExtractFromPagesAsync` and before the gate (Svc:599-704,
659), and `GenerateToolPageAsync` (Svc:1373-1380): every extracted item that carries a source
quotation is sent with its page id; an item with `found: false` is removed from the document
before the gate counts it, before the prompt prints it and before the structured data is built;
an item without a source quotation is removed as well, because nothing can verify it. The bank
(`gcc_partner_extractions`, Svc:665-676) stores the verified document, so `CurrentExtractorVersion`
is bumped and old rows are re-extracted. If the Library cannot be asked, the partner is
"could not be assessed" and refused — the unverified document is never used in its place.

The tool page's block quotation goes through the same route: after a candidate number is resolved
(TQG:227-231), `FindViolations` sends the candidate's text and its page id and refuses on
`found: false`. The containment comparison (TQG:105-119) and the snap-by-typed-text that
substitutes a whole candidate for a fragment (TQG:233-242) are deleted — a quotation is a listed
candidate number or it is refused; no comparison of GeekAPI's own survives. `GccQuoteCandidate`
gains `PageId` from the typed-passage read. The dormant `GccV2PartnerExtractionVerify`
(`IndexOf`, :148) and `IsGrounded` (Ext:362) are deleted. The lede is in scope: a quote paragraph
in the tool lede is verified like one in the body, or the lede contract forbids it (DG:93 and
Svc:1662-1664 leave it unchecked today).

**Done when.** `grep -rn 'v1/verify' --include=*.cs` finds the client; `GccToolQuoteGuard` contains
no `Contains`, `IndexOf` or `Unquote`; a test feeds an extraction with one item the fake Library
reports `found: false` and shows it absent from the gate count, the PARTNER DATA block and the
JSON-LD; a test shows an unreachable Library refuses the partner with that reason; the `VerifiedAnswer`
field is verified or renamed; `partner-evidence-reaches-the-writer.md` Stage 4 is marked done.

### X3 — Measure category fill rates before any gate is designed (Jeff runs it; the query is written here) — F2

**Change.** One SQL over the extraction bank, after X2 has re-extracted the indexed partners: per
partner and per category, is the array non-empty. Shape (column names to be read off
`ContentCreatorDbContext.cs` when written):

```sql
select host,
       (extraction->'featureInventory')  is not null and jsonb_array_length(extraction->'featureInventory')  > 0 as features,
       -- one column per category, twenty in all (Svc:1767-1787)
       …
from content_creator.gcc_partner_extractions;
```

The table — twenty columns, one row per indexed partner, with verified items only — is pasted
into this file under Status. A type whose categories are mostly empty on real sites cannot be
enabled, whatever its gate says; the first mapping (`partner-evidence-reaches-the-writer.md`
Stage 6) is corrected from this table before X4 encodes it.

**Done when.** The table is in this file.

### X4 — Passages carry their row; the gate is per row and per type (GeekBackend) — F2, F3

**Change.** `GccGroundingResolver.PartnerQuestions` (Res:679-703) tags each question's results with
the evidence row that asked it; the merge by URL (Res:207, 347-352) keeps the set of row ids on
the page (`GccQuoteablePage.EvidenceRowIds`); `GccPartnerToolSlices` carries them; the tool prompt's
QUOTEABLE RESEARCH groups pages under "Evidence for: {row.Problem}" so the writer sees which
evidence backs which claim. The gate replaces `HasSufficientPartnerData`,
`CountPopulatedPartnerDataCategories` and `PartnerDataCategoryCount` (Svc:685, 1380, 1758-1787):

1. **Rows.** For a tool page, every evidence row of that tool's framing (`ForProduct`) has at least
   one passage from the partner's run that carries a verified item (X2); otherwise the page is
   refused naming the row — "Tipalti: no verified evidence for 'The company cannot reliably
   reconcile global payments…'". Decision D5 (refuse, 2026-10-04) already covers this.
2. **Categories.** The type's categories from X3's table, verified items only. Pillar and blog keep
   the existing "every declared partner returned a passage" refusal (Coord:125-133) and add their
   categories.

The gate runs on what retrieval and verification returned, before any writing call is paid; the
single-type run goes through the same path as the multi-type run (Coord:292-325 is deleted; one
`RunGenerateAsync` branch), so every run writes `grounding`, `outcome` and `settled` records and
catches the link check.

**Done when.** The three names return no hits; a refusal names the row or the category; a tool
prompt in the snapshot test shows passages grouped by row; `GccGroundingEvidenceQuestionsTests`
pins the row ids through the merge; a single-type and a multi-type run produce the same event
kinds.

### X5 — The branch, then Revise (GeekBackend; content-creator-v2 for F11) — F7, F8

**Jeff decides first** (decision 2): merge `fix-content-creator-stages`, cherry-pick A8 `849f9c9`,
D3 `9742608` and D4 `62ecfe4` onto `main` (resolving against the redone A11, A12 and D2), or delete
the branch and rebuild the three from their commit messages. Either way the branch is deleted
after, and `git branch -a` shows `main` alone.

**Change.** Revise is a guarded generate: `POST versions/{id}/revise-job` runs the type's generator
with section scope, the grounding resolved the way Generate resolves it (until X9's evidence row
exists, re-resolved; after, read from the row), `GuardedDraftAsync`, and the same settlement; the
old `versions/{id}/revise` (GccC:633) is deleted in the same commit the workspace switches to the
new route (F11, `gcc-api.ts:391`). Social, email and the generic long-form branch (Svc:964-1018)
either get `GccDraftGuard` or are refused as types (X8 decides which exist).

**Done when.** A tool Revise runs the draft guard (test through the coordinator); `grep -rn
'versions/{id:guid}/revise"'` is empty; `metadata_json` carries model ids, extractor version and
bank digests (D3); D4's data-plane rule test is on `main`; the stale comment Svc:1100-1103 is gone
with the code it described.

### X6 — What the operator sees (GeekBackend + content-creator-v2, paired contract commits) — F10, F11, F12

**Change.** A gap is an object, not a string: `{check, product, detail}` in the envelope's
`warnings`, the hub `warning` event (Notif), `resultJson.warnings` and `GccRunSettlement`'s refusal
lines; `GuardedDraftAsync` keeps the finding's name (Svc:3172-3175 drops it); `MissingQuotation`
names the product (TQG:56-59). The page shows "tool · Tipalti · blockquote-missing — {detail}" and
merges live and recorded items by (type, product, check), not by string (RD:274-277; PCW:512-538);
the run log renders for every run (PCW:1514-1518). The dead and contradictory copy goes: the
stale-grounding prompt and `acknowledgeStaleGrounding` on both sides (PCW:924-944; Ctl:789-794);
the unreachable related-pages block (PCW:819-824, 474-477); the "no site crawl… start the create
again" message checks what the server checks (Ctl:384-396) and says "project"; one sentence for
the save rule (PCW:1584-1585 agrees with 1457 and Coord:400-402); no GUID in the "already running"
line (Ctl:421-425); NFP:134 and NFP:176; the History "append-only" claim or the delete; the
`AGENTS.md` comment at PCW:1117. The hints say what the code does: the angle hint (CBP:529-532),
the CTA label hint (CBP:569-571, see X10c), the diagnosis-questions hint (NFP:137-140: the page
appends them, the writer never sees them), the core-problem hint (NFP:112-117, after X10c makes it
true). One quotation rule, stated once: DG:138 ("exactly one"), DG:119, DG:75-77, TQG:7-31,
TQG:287, QC:24-27, PRB:27-29 rewritten from the code.

**Done when.** Both suites green with the new warning shape; `grep -rn "exactly one"` in GeekBackend
is empty; `stalePrompt`, `saMissingPages` and `acknowledgeStaleGrounding` return no hits in either
repo; a test renders two tool pages with the same gap and asserts two lines.

### X7 — Authorization (GeekBackend) — F13

**Change.** `[Authorize(Policy = ContentCreatorAuthConstants.ManagePolicy)]` at class level on
`GccController`, as `GccProjectsController` has it (Ctl:30); the five per-method attributes become
redundant and go; the private `GccController.ToLlm` (GccC:1197-1200) is deleted. `ValidateAudience
= false` (Program.cs:216) is decision 3.

**Done when.** A test enumerates every action on both controllers and asserts the policy; the
anonymous `versions/{id}/revise-job`, `approve`, `seo`, `polish`, `repurpose`,
`brief/partner-quote-readiness` and `serp/parse` answer 401.

### X8 — Types are an allow-list; swallowed failures are refusals (GeekBackend) — F14, F15

**Change.** The API accepts exactly the types the UI offers (`content-types.ts:11-32` minus the
disabled thirteen); an unknown string is a 400, not the generic blog path (Coord:161-193,
644-653); `aiTool` is removed (Coord:755-756); `social` and `ads` are disabled with the thirteen
until a platform exists, or `linkedin` and `facebook` are offered in their place — decision 4;
the standalone image prompt stops being a content type (decision 5). Each row of audit F14 becomes
a refusal with its reason or a warning the operator sees: a bank read that is not 2xx refuses the
partner ("bank unreadable: {status}") instead of paying for re-extraction (Repo:390-394, 544-551);
a project read that is not 2xx fails the run (Res:317-330); a hub push that fails is in the run
log as a `fault` (Run:195-222); a job row that cannot be completed is retried once and then
reported, not left `running` (Run:176-187); a resolver that returns empty because of an error
(competitor headings, known tools, publisher profile) refuses the type that needed it; an
email/social image-prompt failure is a warning on the piece (Coord:985-1012); a question that
returned 0 pages keeps its RAG warning (Res:533-537).

**Done when.** A test per row; `GccGenerationCoordinator.GenerateOneAsync` has no default branch;
the UI's disabled list and the API's refusal list are one list, read by a test on both sides.

### X9 — Provenance (GeekBackend, GeekRepository) — F16

**Change.** D3's `metadata_json` (X5); extraction calls go through the recording provider (Ext:104
bypasses `GetLlm`, Svc:1862-1866) so the run log's "every model call" (Log:11-13) is true;
`gcc_version_evidence` is written by `PersistAllAsync` — the prompts, the passages with their row
ids, the candidate list, the extraction digest and the readiness result for that version — and
read by `GET …/versions/{id}/evidence`, or the table and its client are dropped (decision 6); the
job result is serialized with `JsonSerializerDefaults.Web` (GccJobsAndSeo.cs:48) so the stored
JSON is the camelCase the page reads; the merged research is recorded on the evidence row and
never on the create (Coord:74 stays unwritten). The F18 rows for Prov, Log, GccJobsAndSeo and
ProviderOptions are rewritten here.

**Done when.** A version's evidence is readable and names what X4's gate read; the run log shows
extraction calls; `JsonSerializer.Serialize(result)` with no options returns no hits in
ContentCreator.

### X10 — The prompt, in four commits (GeekBackend) — audit §2

**X10a — Markdown out, and each block once.** `  # {heading}` (CPB:633) becomes `Heading: …`;
`[{Title}] ({Url})` (Svc:402, 2952), `[Competitor page n: …]` (Svc:3013) and `[{label}]` (RBB:323)
become `Page: {Title} — {Url}`; `- ` bullets become labelled lines; `LeakedMarkupSyntax` also
catches a line-leading `# ` (LlmResponseJsonParser.cs:22-24); the stale "use the bold/italic/href
fields instead" (:362) is rewritten. The BRIEF block, BuildAudience, must-mention and `create.Notes`
are printed once per call (today up to five times, §2a; "Tool summary:" repeats them, CPB:2766-2770;
PROJECT SITE carries a second brand voice, RBB:160-206); the OWN SITE block holds own-site pages
and nothing else, and on the tool page it is omitted when no profile is passed (Svc:1460-1477)
rather than headed "what the publisher already says about themselves" over the brief.

**X10b — One rule per concern, stated once, matching the guard.** Quotation: tool pages "at most
one, a listed candidate, verified" (X2); pillar and blog "none" — and the `quote` paragraph type,
"A block quotation is the one exception", the OWN SITE blockquote sentence, the currency
exception and the `quote` lede type are removed from those contracts (§2c-7, 8). Length: one floor
per batch, computed from the slots, told to the model as the number it is measured against
(Svc:3484-3496 vs CPB:487-491, 2351, 2848); "fails outright" is true (a guard refuses under the
floor) or gone; email and social lengths equal the UI bands (Svc:2370 vs BC:704-709); the scorer's
floors equal the bands. Voice: the brand voice and the tone legend reconciled; the consultant
appendix's "JSONB, TDD" and four-phase method deleted (Svc:492-498) — the operator's framing is the
method. Links: one rule. Attribution: one rule — a page attributes nothing to a source except a
verified block quotation's cite. CTA: no CTA language on long-form prompts at all (CPB:1482, 644-645,
2859-2861); the `PageBuildsClosing` flag and `ClosingCallToActionInstruction` are deleted
(GenerationRequest.cs:84; CPB:1082-1113) so every caller gets the one line. The quiz phrases are
named in the prompt or DG:449-451 stops claiming they are. "CRITICAL: there is no case-study data"
(CPB:1728, 1862) is conditional on the evidence. The lede-type list is filtered per type (no
"question" or "directAddress" where the lede rule forbids them, CPB:1268-1269 vs 1648, 1652);
"hypothetical/illustrative" figures leave the prompt (CPB:1733, 1867, 2743); the tool lede is a tool
lede, not a "TechnicalArticle pillar" (CPB:1648), and "Pillar topic:" leaves the tool prompts
(CPB:2767, 2917, 2990); "Public path" shows the real path or nothing (CPB:2770 vs Svc:1697); "The
five forms are" lists five (CPB:570-575); the "advisory H2s below" line goes (CPB:2409-2412).

**X10c — The brief reaches the writer.** The pillar's body batch 1 receives the core problem
(today only slot 0's guidance, which the lede call renders as labels and the body skips —
Svc:2603); per-tool framing reaches pillar and blog under a PARTNER FRAMING block, one entry per
declared partner from `ForProduct` (Svc:2500, 2756 use `ForCategory`); `BriefJson`, `Topic` and
`Notes` leave `NumberEvidence` (Svc:3297-3303) — a figure is licensed by retrieved text and the
verified extraction only, never by what the operator typed; raw enum tokens are printed as words
(`commercial_investigation`, `in_market`, `problem_solution` → the `DescribeAngle` pattern), and
"Angle: {raw}" leaves the lede prompt that also says "Pick ONE ledeType" (CPB:1426, 1404-1409);
`ctaType` and `ctaLabel` either drive the long-form closing config or leave the long-form brief
(decision 7); the dead `lengthBand` lines go (Svc:275-276; CPB:1400, 1489); blog batch 1 shows the
pain points once (BlogPrompts.cs:84, 144); the continuity block carries the pillar's introduction,
not the lede alone (Svc:2601); `ledeType` is used (the "Hook type" line, CPB:854-857) or removed
from the contract.

**X10d — Each call carries only its own evidence (U7b).** Each pillar and blog batch receives the
passages for its own sections — a facet per slot chosen by the row ids X4 attached — instead of the
whole research block (Svc:2493, 2516, 2746); PARTNER DATA is cut to the categories the type draws
on (X3's table) instead of all twenty with every provenance field (CPB:2776-2796); the research
block's page cap is a stated rule (today none, Svc:389-391), and the six chunks shown per page are
the six best-scored or the six earliest by a stated choice (decision 8; today the client re-sorts
the kept chunks into reading order before `Take(6)`, RAG:976-982). The §7 cost figure (bytes per
`call` event, 100–290 KB today) is measured before and after and recorded here.

**Done when, for X10.** A prompt-snapshot test per type lists each block once and fails on a
line-leading `#`, on `[…](…)` or `] (`, on `- ` and on a block that appears twice; the ten
existing prompt test classes updated; `grep -n 'TechnicalArticle pillar\|Pillar topic' CPB` empty
on the tool path; the cost measurement in this file's Status.

### X11 — Dead code and the stale comments (GeekBackend) — F17, F18

**Change.** Delete: `ValidateImagePromptRequiresLongForm` (Svc:129-138); the `onlyProduct` branch
and `GccPartnerToolSlices.ForProduct`'s unused overload (Svc:758, 767-782; Slices:115-126);
`GccResearchCaps.MaxQuoteables`; the unused `section` parameter (Svc:3672); the two unreachable
refusals (Svc:792-796, 1416-1426). Rewrite from the code every F18 row not already taken by an
earlier stage. Last, A15: the ContentCreatorV2 cluster — 347 files, its hub (Program.cs:375) and
workers (ServiceRegistration.cs:96, 137) — is deleted after X2 has lifted the extraction service
and the schema-constrained generator into ContentCreator; `grep -rn ContentCreatorV2` then returns
nothing.

**Done when.** Each F17 and F18 line returns no hits; the solution builds; the suite is green.

### X12 — Library (Geek-Crawler-Rag) — F19, F20, F22

**The fusion cut and the order rule — built 2026-10-08 evening (decision 11, closed).** Until then
`dense_query` fetched `max(topK·2, 30)` per half and LlamaIndex cut the fused union at that same
number by min-max relative score, so 7–34 keyword-only and 12–44 meaning-only chunks per query were
dropped by curve shape before selection. Built, on the first real run's lines (HANDOFF §9b):
(a) `hybrid_top_k` = both fetches, nothing dropped at fusion; (b) reciprocal rank fusion, k 60, in
`logged_rank_fusion` with the composition logging kept — the "LlamaIndex's, unchanged" test
reversed on purpose; (c) the pool is the whole union unless the Cohere reranker is on. Remaining
done-when: the nine §9b questions re-run on the deployed code and appended to §9b, and the first
eight on tipalti and the reconciliation row read by Jeff.

**Then.** R3's first half: the exact-text and sibling collapse run inside the ranked list before
the pool cut (`query.py:164-168` cuts, 372-384 collapses after), so the pool holds `max(topK·2, 40)`
distinct texts, backfilled from the over-fetch. The near-copy measurement is a script,
`scripts/measure_near_copies.py`, run against one run id: a paragraph is templated when at least
60% of its 4-word phrases appear on at least 30% of its sibling pages; the table for Ramp
`4563f7ec` goes into HANDOFF §9. The `minQuality` floor GeekAPI sends (0.55, RAG:621) is documented
against the `qualityScore` formula (`metadata.py:164-181`) on the `/v1/query` route and in the
README; whether it stays is decision 9.

**Done when.** `tests/test_page_diverse_selection.py` has a flooded over-fetch whose pool is 40
distinct texts; the script's table is in HANDOFF; the README names the floor.

### X13 — Crawler (Geek-Crawler-v2) — F21

**Change.** Delete `enqueueSuppressedQueue` and `browserRenders`; `persist.ts` passes the real
value to `recordAcceptedPage`; read C1's done-when off Ramp `4563f7ec` (`/products`, `/bill-pay`,
`/accounting-automation` present; `offSitemapAdmitted`) and record it in
`partner-evidence-reaches-the-writer.md` Stage 2; whether `cheerio-runner.ts` still sizes the
crawl from the sitemap is checked and, if so, the ceiling is the site cap; the per-directory
breakdown of `refused.directoryCap` is decision 10.

**Done when.** `tests/KNOWN_GAPS.md`'s counters section is deleted because the counters are; the
Ramp numbers are in the plan; the crawler suite is green.

## Order and gating

X1 → X2 → X3 (Jeff) → X4 → X5 (Jeff's branch decision first) → X6 → X7 → X8 → X9 → X10a → X10b →
X10c → X10d → X11. X12 and X13 are independent and can run at any time in their own repositories.
Contract changes (X6's warning shape, X2's client, X9's evidence route) land on both sides in
paired commits. A GeekBackend push redeploys both Railway services and fails any Generate that is
running (`GccInterruptedJobsOnStartup`): push only with no `running` job row. A Rag push recreates
the container: push only with no index job pending or running. No stage re-crawls or re-indexes on
its own.

## Acceptance

After X1–X6, the seven-type Generate on "test" that is still owed (Jeff's third run): every
refusal names a row, a category or a provider; every tool quotation is `found: true` from
`/v1/verify`; every gap on screen carries the product and the check; a forced Anthropic timeout
(the timeout set to one second in a test environment) costs one type and the rest save; the run
log is present on a clean run. After X10: the snapshot test green on every type; the per-call
bytes before and after in this file. After X11: `grep -rn ContentCreatorV2` empty. The Library's
acceptance tests in `retrieval-from-the-brief.md` (Tipalti re-crawl, re-index, rows, readiness)
still stand and are not repeated here.

## Decisions for Jeff

| # | Decision | Recommendation |
|---|---|---|
| 1 | X1: send a thinking budget to Anthropic, or leave the payload as it is | Leave it; name `stop_reason` first and revisit when a truncation is actually seen |
| 2 | X5: merge, cherry-pick A8/D3/D4, or delete the branch and rebuild | Cherry-pick the three onto `main`, then delete the branch |
| 3 | X7: `ValidateAudience = false` — why it is off, and whether it stays | Turn it on unless the token issuer cannot set an audience |
| 4 | X8: `social` and `ads` — disable until a platform field exists, or offer `linkedin` and `facebook` | Disable; the generic "Professional tone, concise." is not a product |
| 5 | X8: the standalone image prompt as a content type | A property of a page, not a type |
| 6 | X9: `gcc_version_evidence` written, or dropped | Written — every diagnosis this month was archaeology on prompt text |
| 7 | X10c: `ctaType` / `ctaLabel` on long-form — drive the closing config, or leave the long-form brief | Leave the long-form brief; the closing is fixed wording by Jeff's 10-07 decision |
| 8 | X10d: the six chunks per page — six best-scored or six earliest | Six best-scored |
| 9 | X12: the `minQuality` 0.55 floor | Keep, documented; an untitled blockless page under 2,000 characters is not evidence |
| 10 | X13: per-directory breakdown of `refused.directoryCap` | Yes, as a map in the ledger — cheap, and it is the number C2 asks for |
| 11 | X12: the fusion — (a) no cut at fusion, (b) relative score or rank fusion, (c) pool = union when the reranker is off | **Taken and built 2026-10-08 evening** — all three, on Jeff's "do whatever necessary" after the first real run's lines (HANDOFF §9b). The first-eight lists on tipalti and the reconciliation row are still his to read |

## Status

Nothing built. Rows are added here as stages land, with commit, deploy time and the measurement
each stage's done-when asked for.

## Rules that bind this plan

Fail closed, no fallbacks, no auto-repair, no Markdown in the corpus or the prompts (CLAUDE.md §1a,
§1b, §2). A verifier that cannot be reached refuses. A snap that changes words is a refusal. A gap
is reported with its identity, never filled. Nothing the operator types is quoted or licenses a
figure. One rule per question, stated once, in the place the code enforces it.
