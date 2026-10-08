# Content Creator audit — 2026-10-08

Jeff asked, after the second plan shipped: *"Are you 100% sure no additional changes are needed to
actually make this application (Content-Creator) function as a best in class document writer?"* The
answer was no. This is the list.

**What was read.** Geek-Crawler-Rag `eaa8c8a`, GeekBackend `288dde3` (`main`), content-creator-v2
`4677b69`, Geek-Crawler-v2 `05b2649`. Five read-only sweeps — Generate orchestration and gates;
guards and quote verification; what the writing model is shown; the frontend; a reconciliation of
every open plan item against git — plus this repo's own retrieval numbers and two angles of the
code review Jeff ran on `partner-evidence-reaches-the-writer.md`. No Generate was run. Nothing in
code was changed; the documentation corrections made today are listed in §4. Every `file:line` is
as of the commits above; the headline claims were re-read at their lines before they went in here.

**Path key** (`B` = GeekBackend/GeekAPI, `CC` = content-creator-v2/src): **Ctl**
B/Controllers/ContentCreator/GccProjectsController.cs · **GccC** …/GccController.cs · **Run, Coord,
Res, Svc, Settle, TPR, Slices, Prov, Log, Notif** B/Services/ContentCreator/GccGenerateJobRunner.cs,
GccGenerationCoordinator.cs, GccGroundingResolver.cs, GccGenerateService.cs, GccRunSettlement.cs,
GccTypedPassageReader.cs, GccPartnerToolSlices.cs, GccVersionProvenance.cs, GccRunLog.cs,
GccGenerateNotifier.cs · **DG, TQG, QC, PRB** …/Guardrail/GccDraftGuard.cs, …/Guardrail/GccToolQuoteGuard.cs,
…/GccQuoteCandidates.cs, …/GccAngleQuoteProbe.cs · **RAG** B/Services/GeekCrawler/HttpGeekCrawlerRagClient.cs
· **Repo** B/HttpClients/HttpGccRepository.cs · **Ext** B/Services/ContentCreatorV2/Partner/GccV2PartnerExtractionService.cs
· **Anth, OAI** B/Services/Workflow/Providers/AnthropicProvider.cs, OpenAiProvider.cs · **CPB**
B/Services/Workflow/Services/PromptBuilders/ContentPromptBuilder.cs · **PCW, CBP, NFP** CC/components/content-creator/ProjectContentWorkspace.tsx,
ContentBriefPanel.tsx, NicheFramingPanel.tsx · **API, RD** CC/services/gcc-api.ts, CC/lib/content-creator/run-display.ts.

**Severity.** **S1** — a page can be wrong and nothing says so. **S2** — a run or a page is lost.
**S3** — the operator is misled, or cannot act on what the run found. **S4** — dead code and
statements that contradict the code.

## 0. The verdict in five lines

1. The writer is grounded on retrieved passages, but **nothing a page says about a partner is
   verified against the Library.** The one check that exists, the tool page's block quotation, runs
   on spans GeekAPI cuts for itself. `/v1/verify` has existed since 2026-10-04 and has no caller.
2. The only partner-data gate counts how many of twenty buckets a model chose to fill; three is
   enough. No gate asks whether a claim the brief makes has evidence, and it cannot: the passages
   are not kept with the row that retrieved them.
3. One Anthropic timeout kills the whole run, after every requested type's old page was deleted.
   A truncated Anthropic reply is not detected where it happens.
4. Revise runs with no guard and no evidence. The fix for that — and two other stages — sit on a
   GeekBackend branch that was never pushed.
5. What a run found is shown without its identity: a gap reaches the screen with no check name and
   no product name, identical gaps from different tool pages collapse into one line, and a clean
   run shows no record at all.
6. The brief reaches the writer in pieces. The pillar never sees the core problem; the form
   promises uses the prompt does not make (the angle "decides" a quotation it never shapes, the
   CTA label "overrides" wording it never touches); and the prompts contradict each other, the
   guards and the schema in twenty-two places — including Markdown heading markers in the block
   headed as the publisher's own site.

## 1. Findings, ranked

### S1 — a page can be wrong and nothing says so

**F1. Nothing in GeekAPI calls `POST /v1/verify`.** Zero hits for `v1/verify` in any `.cs` under
GeekAPI, GeekApplication or GeekRepository; the RAG client's routes are `v1/index`, `v1/query`,
`v1/templates/*`, `v1/pages/{id}`, `v1/capabilities` and the diagnostics trio (RAG:327-333, 356-859).
The Library route exists (`app.py:731`, since Rag `1a25095`/`9a0c901`; `tests/test_verify_route.py`).
Three rules answer "is this text on that page" today:

| Rule | Where | Reached from |
|---|---|---|
| `verify_quote` over the block projection, the two C#/Python projection differences normalised | Library `citation_verify.py:57`, `app.py:731` | nothing |
| Case-insensitive containment of the quote in a span GeekAPI cut from GeekRepository blocks | TQG:105-119, 236-242 | every tool page (DG:112-121) |
| `pageText.IndexOf(quote, OrdinalIgnoreCase)` over `GET /v1/pages/{id}` | GccV2PartnerExtractionVerify.cs:148 | only the dormant `GccV2GeekCrawlerResearchResolver:486`, whose caller `MergeExternalResearchAsync` has no caller |

Extraction items — every feature, price, integration, FAQ answer the tool page is built from — are
never verified at all (`IsGrounded` has no caller, Ext:362). The field is still called
`VerifiedAnswer`. **Change:** the verify pass runs between extraction and the gate at both live
sites and calls `/v1/verify`; an item not found is dropped from the document, the gate and the
prompt; if the Library cannot be asked the create is refused. The tool quotation goes through the
same route. The `IndexOf` and containment comparisons are deleted, not kept as fallbacks
(`partner-evidence-reaches-the-writer.md` Stage 4; `fix-geekapi.md` A1 full).

**F2. The partner-data gate is 3 of 20 equal buckets, and it is the only one.** Ready = extraction
present ∧ `PagesFailed == 0` ∧ ≥3 of 20 categories non-empty ∧ (FeatureInventory or Citables
non-empty) (Svc:685, 1758-1760; the twenty at 1767-1787). Which bucket a sentence lands in is the
model's choice; one item fills a bucket. It runs for tool pages only (Svc:685, 1380); pillar and
blog have no partner-data gate at all, and get zero typed passages (Coord:567-568, 580-581). **Change:**
a requirement per content type, and per evidence row (Stage 6), replacing `HasSufficientPartnerData`
and `CountPopulatedPartnerDataCategories`; measure fill rates per category over the banked
extractions first, because a type whose categories are empty on real sites cannot be enabled.

**F3. Passages are not kept with the row that retrieved them.** A partner run is asked its core
problem and one question per evidence row at topK 8 (Res:679-703), then the results are merged into
one pool deduped by URL (Res:207, 347-352). Nothing records which row a passage answered, so the
writer is not shown which evidence backs which claim and no gate can ask whether a row has any.
Passages and candidates exist afterwards only inside the `call.user` prompt text of the run log.
**Change:** carry the row id on each retrieved passage (additive); it is the precondition for F2.

**F4. The quotation check has no product, no page scope, and skips the lede.** `FindViolations`
accepts any in-range candidate number without reading its text or cite (TQG:102); a quote in the
tool lede is neither snapped nor checked by `FindViolations`/`MissingQuotation` (DG:93, snapping at
Svc:1662-1664 covers body sections only), yet is counted by `one-quotation` and scanned by the
numbers check. A shortened span is silently expanded to the full candidate and its formatting
dropped (TQG:233-242). The candidate list is not part of the figure evidence while quotes are
included in the numbers check (Svc:3297-3303, 1628-1632; DG:369). **Change:** one verifier (F1);
the lede is in scope or quotes are refused there; snapping that changes words is a refusal, not a
repair (CLAUDE.md §2).

### S2 — a run or a page is lost

**F5. An Anthropic timeout kills the whole run with nothing saved, after the delete.** The
Anthropic provider catches only `HttpRequestException` (Anth:104-111); the `HttpClient` timeout
(120 s, ProviderOptions.cs:60, 90) surfaces as a `TaskCanceledException`, which Ext:416-419,
Svc:606-609, Svc:849-852 and Coord:459-462 all rethrow. The job fails as
`"{ExceptionType}: {message}"` (Run:148), no type is settled or saved, the call is not in the run
log (Log:161) — and the old pages of every requested type were already deleted (Coord:224-227,
Jeff's 2026-10-06 "delete should happen first"). OpenAI converts the same timeout into a provider
fault for that call (OAI:98-102). The wait in the four-call concurrency gate is outside the timeout
(LlmConcurrencyGate.cs:18-28). **Change:** Anthropic reports a timeout as a `ContentGenerationException`
the way OpenAI does; the type refuses with "{provider} did not answer within N s", the others
settle.

**F6. Anthropic truncation is not detected where it happens.** On a 200 with content, `stop_reason`
is never read (Anth:140-144); it is read only in `NoBlock` (158-169). A reply cut at `max_tokens`
is noticed only by the JSON parser's "does not end with a closing brace" heuristic
(LlmResponseJsonParser.cs:677-697). OpenAI's `finish_reason=length` is a typed fault with the
partial kept (OpenAiCompatibleOutcome.cs:36-45). Neither provider payload carries a thinking or
effort field (Anth:59-75; OAI:63-81), so a model that thinks spends the single `max_tokens` budget
before writing (F-A19 in the content-creator-v2 handoff). **Change:** read `stop_reason` on every
reply; `max_tokens` is the same fault as OpenAI's; leave room after thinking.

**F7. Revise runs no guard and no evidence, and allows tool pages.** `ReviseAsync` (Svc:1047-1152)
runs `ContentGuardrail.Apply` only (1146); `GuardedDraftAsync` is called from the tool, pillar and
blog generators alone (Svc:1670, 2613, 2805). The route `versions/{id}/revise` (GccC:633) takes any
type. Social, email and the generic long-form branch (Svc:964-1018) never see `GccDraftGuard`
either. **Change:** A8 — Revise is a guarded generate (`revise-job`) — which exists only on the
unpushed branch (F8); until it lands, tool Revise returns 400 (the interim the plan named).

**F8. Nine GeekBackend commits sit on a local branch that was never pushed.**
`fix-content-creator-stages` (tip `849f9c9`, 2026-10-04 11:34–11:38, forked from `db918bb`): A13
`953ed1e`, D2 `c535382`, D3 `9742608`, D4 `62ecfe4`, A9 `c024cc1`, A10 `0b422bc`, A11 `e71706c`,
A12 `139f30d`, A8 `849f9c9`. `main` later redid A13, D2, A12 and A11 on its own. **A8, D3
(`metadata_json` with model ids, extractor version, bank digests), D4 (the data-plane rule test) and
A10's evidence writer never reached `main`.** The plans and the Rag handoff said these were "not
found in the log"; `git log --all` finds them. It also breaks the "no branches" rule for this repo.
**Change:** Jeff's call — merge, cherry-pick A8/D3/D4, or delete; then the plans say which.

**F9. The gate fires after the money is spent, and the type's page is already gone.** Tool
extraction is one paid call per partner page (Ext:155-156) before the 3-of-20 gate; the draft guard
fires after lede, every batch and the FAQ (Svc:3166-3169, no retry). A refused type ends with no
page because every requested type's old pages go first. In a single-type run any refusal fails the
job outright, the link check is uncaught (Coord:303) and no `grounding`/`outcome`/`settled` records
are written (Coord:292-325 vs 267-290) — two code paths for one run shape. **Change:** one path
(the multi-type one) for every run; the gate reads what retrieval returned before extraction is
bought (F2/F3 make that possible).

### S3 — the operator is misled, or cannot act on what the run found

**F10. A gap reaches the screen without its identity.** The check name is dropped when a gap becomes
a warning (Svc:3172-3175: `verdict.Gaps.Select(g => g.Detail)`); `MissingQuotation`'s text names no
product (TQG:56-59); the warning is prefixed only `"{contentType}: "` (Coord:404-409); the page merges
live and recorded items by exact string (RD:274-277; PCW:512-538), so two tool pages with the same
gap show one line; the run log panel renders only when something failed or warned (PCW:1514-1518),
so a clean run shows no record. The only place `blockquote-missing` survives is as JSON inside the
`verdict` event. **Change:** the warning carries `check` and the product through the hub event and
the saved `warnings`; the page shows "tool · Tipalti · blockquote-missing"; the run log shows for
every run.

**F11. Messages and comments say "exactly one" quotation; the code says at most one.**
`one-quotation` refuses with "it carries exactly one" (DG:138) while zero is a gap (DG:123-130) and
the prompt says "at most one block quotation … only when a listed span earns it" (CPB:981-983).
`quotation`'s message opens "does not carry a verifiable block quotation" (DG:119) but fires only
when one is present and invalid. DG:75-77 and TQG:7-31 describe the old rule; TQG:287 says
case-sensitivity "is the point" while both comparisons ignore case (TQG:112, 237); QC:24-27 says
"there is no verbatim check downstream" while TQG:105-119 is one. The UI's angle hint says the
angle "decides what the tool page's block quotation has to answer" (CBP:529-532); the quote
instruction takes product and keyword only and hard-codes the problem/solution framing
(CPB:964-970, 981, 2824); PRB:27-29 says Generate queries at 32 through `BuildNeed`, which is now
the no-brief fallback (Res:679-703). **Change:** one rule stated once; the messages, comments and
hint rewritten from the code.

**F12. Frontend copy that cannot fire, fires on the wrong condition, or contradicts the page it is
on.** The stale-grounding prompt and its "Proceed with stale grounding" button are dead
(PCW:924-944; `AcknowledgeStaleGrounding` is "accepted for the contract's shape; Generate has no
staleness gate that can fire", Ctl:789-790). "Generate is blocked: this project's crawl has no
related pages" is unreachable (`parseSiteSectionJson` returns null when `relatedPages` is empty,
API:61, so `saMissingPages` at PCW:474-477 is always false). "This create has no site crawl…start
the create again" (PCW:948-953) blocks the button on a condition the endpoint now repairs before
the site gate (Ctl:384-396), and says "create". The partner readiness footer says the drafted pages
are "each saved on its own" (PCW:1584-1585); the modal on the same page says "all saved together
at the end" (PCW:1457) and the code agrees with the modal (Coord:400-402). The "already running"
refusal puts a job GUID on screen (Ctl:421-425). NFP:134 still refers to the "Where they fail" box
and NFP:176 to pain points being "added" — neither exists on the form since `4677b69`. The History
section says "append-only" (projects/[id]/page.tsx:36) above a per-entry delete
(ProjectProfilePanel.tsx:228-249). PCW:1117 cites an `AGENTS.md` that is gitignored and absent.
`briefVersion` has no reader in GeekAPI; `latestVersionNumber` is typed (API:216) and unused.
**Change:** delete the dead prompt and the unreachable block; the site-crawl message checks what the
server checks; one sentence for the save rule; `check` and product in the gap line (F10).

**F13. Routes without authorization.** `GccController` carries `[Authorize(ManagePolicy)]` on five
methods only (GccC:354, 372, 384, 405, 437); `versions/{id}` reads, `versions/{id}/revise`,
`approve`, `seo`, `polish`, `repurpose` (GccC:619-731), `brief/partner-quote-readiness` (957) and
`serp/parse` (929) have none, while `GccProjectsController` has it at class level (Ctl:30).
`ValidateAudience = false` (Program.cs:216). A dead private `ToLlm` in the controller still maps
every non-Anthropic value to OpenAI (GccC:1197-1200). **Change:** the policy at class level on
`GccController`; the private `ToLlm` deleted.

**F14. Failures that are swallowed into a different outcome.**

| Where | What is swallowed | What happens instead |
|---|---|---|
| Repo:390-394, 544-551 vs Svc:651-653 | A non-2xx bank read | Treated as a cache miss: paid re-extraction, against the comment "a bank row that cannot be read is a defect to see, not a cache miss" |
| Repo:544-551 → Res:317-330 | `GetProjectAsync` non-2xx | A run without a tool type does no retrieval at all; `PartnerUrlsForAsync` returns no partners (Svc:73-78) |
| Run:176-187 | Completing or failing the job row | Row stays `running`; the next Generate gets 409 "already running" until a restart (Ctl:421-425; GccInterruptedJobsOnStartup.cs:30-49) |
| Coord:985-1012 | Email/social image-prompt failure | Nothing on screen; a log line and a job event only |
| Res:533-537 | A question that returned 0 pages | Its RAG warning is dropped |
| GccCompetitorAnalysisResolver.cs:68-82; GccKnownToolsResolver.cs:75-80; GccPublisherProfileResolver.cs:70-74 | Any failure | Competitor headings, the unlisted-tools list, or the publisher profile silently absent — the pillar is written and guarded as if there were none |
| RAG:450-503, 641-684 | Index/query HTTP errors | "could not be reached" / "RAG query failed ({status})"; the body is only in logs |
| Run:195-222 | Hub pushes | The live event is missing; the page says nothing |

**Change:** each of these is a refusal with its reason, or a warning the operator sees — not a
silent change of path (CLAUDE.md §2 "fail silently" means return the failure state, not substitute a
different success).

**F15. Content types: seven offered, anything accepted.** There is no allow-list: an unknown string
falls through to the generic standalone-blog path (Coord:161-193, 644-653; Svc:964-1018).
`aiTool` is generated as a tool page but saved as `"aitool"`, which `IsToolType` does not match
(Coord:755-756). `social` and `ads` are one type each with no platform, written with "Professional
tone, concise." at 1024 tokens (Svc:2430) — linkedin and facebook have real guidance (Svc:2422-2429)
but are not offered. Eight types have generators and no checkbox; thirteen are disabled on both
sides. The standalone image prompt is offered as a content type. **Change:** the API accepts
exactly what the UI offers; social and ads get a platform or join the disabled thirteen; the image
prompt is a property of a page, not a type (Jeff questioned it as a proof type on 10-06).

**F16. What is recorded about a version.** `metadata_json` holds provider and brief revision only
(Prov:35-43) — no model id, extractor version or bank digest (that is D3, on the unpushed branch).
The run log claims "every model call" (Log:11-13; Ctl:101-105) but extraction calls bypass `GetLlm`
(Ext:104 vs Svc:1862-1866) and cancelled calls are never written (Log:161). The merged research is
never written back (Coord:74). `gcc_version_evidence` has a table, a client and no caller
(Repo:79-100; GeekRepository only deletes from it). The job result is serialized without options,
so the stored JSON is PascalCase while the UI reads camelCase (GccJobsAndSeo.cs:48; the plan named
`JsonSerializerDefaults.Web` and it was not applied). `version_number` is always 1; old content,
evidence and approvals are deleted on save (GccProjectPageRepository.cs:119-121), so "each submit
creates a new version" (PCW:1158) is true only in that the row is replaced. **Change:** D3's
metadata; extraction through the recording provider; cancelled calls recorded as faults; the
evidence table written or dropped; `Web` serializer.

### S4 — dead code, and statements the code contradicts

**F17. Dead on the Generate path.** `ValidateImagePromptRequiresLongForm` (Svc:129-138, no caller);
the `onlyProduct` branch (Svc:758, 767-782) and `GccPartnerToolSlices.ForProduct` (Slices:115-126);
`GccResearchCaps.MaxQuoteables` (GeekApplication GccResearchModels.cs:118, unreferenced); the `section`
parameter at Svc:3672; `AcknowledgeStaleGrounding` (Ctl:789-794, unread); the "declares no partner
URLs" refusal (Svc:792-796, unreachable: Res:581-586 refuses first); the second partner-grounding
refusal (Svc:1416-1426, unreachable from the fan-out); `IsGrounded` (Ext:362); the private
`GccController.ToLlm`; the whole ContentCreatorV2 cluster — 347 `.cs` files, its hub mapped
(Program.cs:375) and workers registered (ServiceRegistration.cs:96, 137) — of which the live path
uses the extraction service and the schema generator (A15, open).

**F18. Comments, log lines and messages that name things that do not exist or do the opposite.**

| Where | Says | Code |
|---|---|---|
| Prov:12 | "`GccGenerationCoordinator.PersistOneAsync`" | No such method; `PersistAllAsync` (Coord:853) |
| Prov:23-25 | `ModelUsed` "only threaded through the dormant v2 writer" | Log:146 records it on every recorded call |
| Svc:1402-1405, 3142 | "GenerateAsync's first catch … answers 400 … 503" | No `GenerateAsync`; the route returns 202 and refusals are job errors (Ctl:439-442; Run:148) |
| Svc:1384-1386 | The resolver refuses "at the controller before generation starts" | It runs in the job, after the delete (Coord:227, 267) |
| Svc:925 | "`RequiredCrawlTypes` already hedged both spellings" | The table is `MustCiteCrawlTypes`, keyed "tool" only (Res:129-133) |
| Res:559-561 | "…so this was written without it" | Pillar/blog are refused for that partner (Coord:125-133); its tool page is refused (Svc:825-831) |
| Res:196-199 | The typed reader "reads at most 32 pages back per run" | 32 per call, one call per question (TPR:51-52; Res:549-552) |
| ProviderOptions.cs:79-82 | Extraction "resolves its provider through `GetDefault()`, so Anthropic only ever serves writing" | Ext:104 uses the operator's provider |
| Notif:22-23 | Type outcomes "sent the moment that type finishes" | Pushed after every type and the save (Coord:278-289, 402-420) |
| Coord:513 / 15-19 / 644-647 | "persists exactly one content type" / two callers / "every type still routed here is disabled" | 656-659 "Nothing is persisted here" / one caller (Run:118) / disabled types are refused at 185-190 |
| GccJobsAndSeo.cs:32-34 | `ResultJson` holds "every artifact body" | Ids, version number and metadata only (Coord:896-913) |
| Svc:526-528 | `<param name="Ledger">` | `ToolPageOutcome` has no such parameter |
| GccC:177-182 vs 205-206 | Wiki/.edu/.gov "parsed as articles" | "removed on 2026-09-29", same method |
| Ext:87-88 | "never throws" | Rethrows cancellations (416-419) |
| Svc:462-465 vs 492-493 | "carries no voice line" | "Write as a Senior IT Consultant advising local SMBs…" |
| Svc:1100-1103 | "`title = tool.Name`" | The title is "{Product}: {keyword}" since `eda5b00`; the code reads `ProductName` first (1105-1107) |
| DG:75-77, DG:119, DG:138, TQG:7-31, TQG:287, QC:24-27, PRB:27-29 | See F11 | |

**Change:** each line rewritten from the code or deleted. CLAUDE.md §1a: a surviving name is read
as a live claim.

### Library (this repo) — the numbers, and what has never been measured

**F19. Retrieval as it runs today** (facts, for the record). Dense `BAAI/bge-small-en-v1.5` + sparse
`Qdrant/bm25`, fused by relative score with `alpha = 0.5` as a hard default (`llama_engine.py:57`);
the keyword half searches `keyword` when sent, else `need` (`dense_query`, since `d8a628e`).
Over-fetch `max(topK·2, 30)` (`query.py:121`, `config.py:132`); pool `max(topK·2, 40)`
(`query.py:164-168`, `config.py:133`); page-diverse selection from the pool (`3092ccd`); exact-text
dedupe **after** the pool cut, with no backfill (`query.py:372-384` — R3's first half, open); every
GeekAPI query sends `minQuality: 0.55` (RAG:621) against a page-level `qualityScore` built from
length, title and blocks (`metadata.py:164-181` — base 0.35, +0.15 title, +0.15 blocks, +0.15 at
500 chars, +0.1 at 2,000, +0.1 at 5,000, −0.2 under 200), so an untitled page without blocks needs
2,000 characters to be searchable at all. Chunks: parent 650/80, child 200/40 tokens in the model's
own tokenizer (`config.py:78-83`); stubs dropped (`b14c200`); repeats collapsed per run
(`1a25095`); `topK ≤ 50` (`models.py:166`). Cohere: `rerank_enabled` defaults true, the client
disables itself without a key (`rerank.py:53`), production has no key — off. Scheduler on by default
(`config.py:123`, compose `:50`).

**F20. Measurement debt.** Every retrieval number before 2026-10-08 is dense-only (`7c47fa1`); the
"10 passages from 10 pages" floor was calibrated on Stampli on that path and never re-derived;
Stage 2's Done-when (Ramp `/products`, `/bill-pay`, `/accounting-automation`; bill.com `/pricing`)
was never read off the re-crawl `4563f7ec`; the near-copy measurement (same paragraph, merchant name
swapped) was never run; whether page-diverse selection lifts the category count needs a readiness
run and has not had one. The acceptance tests of `retrieval-from-the-brief.md` wait on Jeff's
Tipalti re-crawl, re-index and evidence rows.

**F21. Crawler counters that nothing increments** (`Geek-Crawler-v2/tests/KNOWN_GAPS.md`):
`enqueueSuppressedQueue`, `browserRenders`, and `pagesWithoutContent` — the last reads 0 in
`run.json` even when pages without content were saved (`persist.ts` passes a literal `true`). The
plan's "no counter in the report that nothing increments" is not met. Per-directory breakdown of
`refused.directoryCap` (C4) is undecided.

## 2. What the writing model is shown

Sizes are approximate character counts of the literal prompt text, measured from the source
constants; interpolated values are not included. Type codes: **T** tool, **P** pillar, **B** blog,
**E** email, **S** social, **I** standalone image prompt. "Raw" means the enum token is printed as
typed (`commercial_investigation`, `in_market`, `book_now`). **GNF** = GccNicheFraming.cs, **RBB**
= ResearchBriefBuilder.cs, **BC** = brief-catalog.ts.

### 2a. The brief: what the form promises, where each field goes

| Field | The form says | Where it actually goes |
|---|---|---|
| Target keyword | "Target keyword" (CBP:342-357) | Every type, verbatim |
| `taxonomyPath` | "First level is the department, so this is what files the page" (NFP:98-110) | Never in a prompt. Level 1 → URL department (GccContentPath.cs:54-58); levels 2+ unused |
| `coreProblem` | "The one problem the whole niche has… **Every page uses this**" (NFP:112-117) | T body batch 1; B body batch 1; **P never** — it sits only in slot 0's guidance, the lede call renders outline labels only (CPB:884-885, 1775-1781), and the body batches take `Skip(1)` (Svc:2603). No lede of any type sees it. It is the first retrieval question for every partner run (Res:686-689) |
| `painPoints` (derived from the rows' Problem column) | — | T batch 1; B batch 1 **twice** (ToGuidance and FailuresGuidance); P batch 1. Complete, not truncated |
| `automationToPitch` | "The automation to pitch" (NFP:271-280) | T batch 1; B batches 1 and 2; P batch 1; later slots get a pointer with no content (GNF:158-162) |
| `evidence[].solution`, `terms` | "each row searches the partner's crawl on its own. Nothing here is quoted" (NFP:331-336) | Never shown — retrieval need and keyword only (GNF:638-641; Res:691-695), as designed. **But every number in them licenses a figure**: the whole `BriefJson` is number evidence (Svc:3301), so an operator-typed "200+ countries" is a figure the page may print |
| `diagnosisQuestions` | "**The writer uses these as written**, or a subset if the length will not carry them all, and may not invent another" (NFP:137-140) | Never shown to any writer. Code appends them verbatim after "Answer these questions when booking…" (Svc:447-453; GccClosing.cs:44-61). The hint is stale since U5; BC:280-281 still says they reach `ClosingCallToActionInstruction` |
| `perTool[host]` | "Core Problem and Automation replace the category's; pain points are added" (NFP:173-178) | T only (`ForProduct`, Svc:1493-1496). P and B use `ForCategory` (Svc:2500, 2756): **a tool override never reaches the pillar or the blog.** Retrieval uses it per host for every type (Res:683). A per-tool row with the same problem replaces the category's (GNF:377-383) |
| `primaryIntent`, `secondaryIntent`, `buyingStage`, `audienceSegment`, `toneOfVoice`, `eeatSignals` | Intent: no helper. Stage: "Google Ads Full-Funnel objective." Segment: "Google Ads audience segments." (CBP:420-499, 594-596) | Raw tokens, printed up to **five times per tool body call**: BRIEF block twice (Svc:260-265), BuildAudience twice (Svc:2306-2314), BRIEF CONTROLS once (CPB:1479-1486). `consultant_professional` adds the ROLE & METHOD block (Svc:485-501) |
| `audienceNotes` | "these win if they conflict with the segment" (CBP:508) | Verbatim; the conflict rule is stated at Svc:257 and CPB:1419 but not in BRIEF CONTROLS |
| `angle` | "…It also decides what the tool page's block quotation has to answer." (CBP:529-532) | Prose (`DescribeAngle`, CPB:1235-1254) and preferred lede types (CPB:1410-1412) in every lede; **and raw** "Angle: problem_solution" (Svc:258-259, 2305) in the same prompt as "Pick ONE ledeType" (CPB:1426) — the situation CPB:1404-1409 records as having produced a refused blog, since `ParseLedeTypeStrict` refuses an unknown value (LlmResponseJsonParser.cs:250-258). **The quotation instruction is angle-free** (CPB:981-1005, comment 966-970) |
| `ctaType` | "Google Ads call-to-action type." Required for Generate (CBP:559; BC:649) | E and S only, raw "Call to action: book_now" (Svc:2319-2321). T, P and B never see the value (Svc:273-274; CPB:1397-1399); for P and B the field *name* is offered as a licensable heading tag `brief:"ctaType"` (Svc:3583) |
| `ctaLabel` | "Overrides the CTA type's default wording, if set." (CBP:569-571) | E and S only. The long-form closing wording comes from `appsettings.json:18-21` through `GccClosing.cs:47-61` — **on long-form pages it overrides nothing** |
| `paaQuestions` | "One question per line" (CBP:600-615) | P: the FAQ call gets every question verbatim (Svc:2559-2566). P and B bodies: only as `paa:` heading tags, up to 40. **T never; B has no FAQ call** |
| `writingNotes` | "Writing notes (optional)" (CBP:617-631) | T, P, B ledes and bodies, P FAQ, I. Not E or S |
| `lengthBand` | Not shown; saved as "" | Stripped at the controller (Ctl:462, 471-478); the prompt lines that would print it are dead (Svc:275-276; CPB:1400, 1489) |
| `briefVersion` | — | Never read |
| Legacy keys (`hierarchyPlan.recommendedTools`, `competitorUrls`, `audienceDetails`…) | Not on the form | Still read by GeekAPI (GccPartnerUrlResearchService.cs:55, 91-105; Svc:2109-2134) and dropped by `migrateBrief` (BC:494-562), so partner names fall back to host-derived spellings (GccRequiredToolMentions.cs:31-56) |

### 2b. The prompts, by call and by size

- **Shared system prompt** (CPB:675-690), identical on every lede and body call of a contract, so it
  repeats on every batch: persona 170, brand voice 900 (BrandTones.cs), WHO IS TALKING / HOW THIS
  READS 1,800, filler ban 410, HEADINGS 1,250, VARY THE SECTIONS 560, MONEY 460, LINK 430, GROUNDING
  310, CONTENT ONLY 180, OUTPUT 190 — about 6.6k, plus the contract (1.2k lede, 3.1k pillar lede,
  1.0k tool sections, 1.9k sections with provenance). Body and FAQ calls also carry a strict schema
  requiring every `Section` property, `imagePrompt` and `provenance` included
  (ContentSectionJsonSchema.cs:165-174).
- **Tool page, per partner:** extraction (one call per page, banked by digest) + lede + 3 body
  batches (6 slots, 2 per batch, Svc:3111) + FAQ 0–1 (only when the bank has pairs, Svc:1603-1607)
  + image prompts + metadata. The body batch repeats everything except ASSIGNMENT three times:
  THIS PAGE ≈3.8k (positioning interpolated three times, CPB:2718-2758); BRIEF CONTROLS ≈1k; the
  OWN SITE block — which on the tool page holds the BRIEF block, must-mention, BuildAudience and
  `create.Notes` and **no home-page content at all** (no publisher profile is passed, Svc:1460-1477),
  under the line "This is what the publisher already says about themselves, published and live"
  (CPB:636-637); "Tool summary:", which prints notes, brief, must-mention and BuildAudience **a
  second time** (CPB:2766-2770; Svc:1342-1345); **PARTNER DATA: the whole serialized extraction, all
  twenty categories and every provenance field, uncapped** (CPB:2776-2796); the evidence block
  (QUOTEABLE RESEARCH, positions, COMPETITOR RESEARCH ≤5 pages, ALREADY PUBLISHED ≤5, non-USD
  amounts) ≈5.4k plus passages; continuity (the lede, ≤1,800); ASSIGNMENT — batch 1 adds the
  quotation rules (1.8k) and QUOTABLE SPANS (≤40, ≈16k at most); batches 2–3 get "NO QUOTATION IN
  THIS PART" (CPB:2823-2828). The FAQ call gets no brief and no evidence (CPB:2887-2926). The
  metadata call gets the body cut to 2,000 characters.
- **Pillar:** lede + introduction in one call (THIS PAGE ≈14k, CPB:1707-1739, 3252-3279), People
  Also Ask FAQ 0–1 (**no evidence**, CPB:2096-2144), 3 body batches (5 slots, 2/2/1), image prompts,
  then metadata **after** the body (Svc:2624) whose `sectionOutline` is unused. The body evidence
  block is ≈8.1k fixed plus **every partner's pages pooled** (Svc:2493, 2516); continuity is the lede
  only, **not the 500–700-word introduction** (Svc:2601).
- **Blog:** metadata, lede, 3 body batches (6 slots), image prompts; no FAQ. Metadata and body also
  carry a PROJECT SITE block with **a second copy of the brand voice** and a repeat of
  Topic/Notes/BuildAudience (RBB:102-108, 160-206).
- **Email / social:** Topic, Notes, BuildAudience with the raw CTA, must-mention; one image-prompt
  call each. **Standalone image prompt:** the BRIEF block plus the whole research block as
  "Notes / brief:" (Svc:898, 915-920).
- **The research block:** no cap on pages (Svc:389-391; `MaxQuoteables = 3` is never applied); 8
  headings and 6 paragraphs per page — and the six are **the six earliest on the page among the top
  80 by score**, because the client re-sorts the kept chunks into reading order before `Take(6)`
  (RAG:976-982; Svc:405). Each chunk renders as `Section:` / `Target Entity Match:` / `Context:`
  ≤2,000 / `Specific detail:` ≤2,000 / `Linked from this section:` ≤8 (RAG:1143-1181). Continuation
  lines are not indented, while rule 2 says "every passage indented beneath that line belongs to
  it" (Svc:371-372). Quote candidates are a different cut — sentence spans from the page's full typed
  blocks, 50–300 chars, ≤12 per page, ≤40 — shown as `n. "text"  [cite: url]` to tool body batch 1
  only; pillar and blog get none.

### 2c. Prompt lines that contradict each other, a guard, the schema or the post-processing

1. "At most one" quotation (CPB:982-983) vs the refusal "it carries exactly one" (DG:138) vs
   TQG:8 "Every tool page carries a block quotation". The count includes the lede (DG:493-507);
   verification and snapping look at body sections only (DG:114; Svc:1662-1664). — F11.
2. The tool lede is told to write "the opening lede for a schema.org TechnicalArticle **pillar**"
   (CPB:1648); the tool body, FAQ and metadata print "Pillar topic: {product}" (CPB:2767, 2917, 2990).
3. Long-form prompts do not print `ctaType` — refuted — but print "Angle: problem_solution"
   beside "Pick ONE ledeType" (2a). Email and social print "Call to action: book_now"
   (Svc:2319-2321). The `PageBuildsClosing = false` closing still prints "asking for " + CtaType
   (CPB:1087-1089, 1106) for any Workflow caller.
4. The email prompt says "The destination URL is injected by the app" (Svc:2372); no CTA URL exists
   anywhere under Services/ContentCreator — the reply is stored as subject, body, ctaLabel
   (Svc:2394-2406).
5. **Word floors.** The blog slot is "450-600 words — for proportion between sections, not a
   quota" (CPB:2351), the SEO block says "your share is about 720 words" and "the finished page …
   fails outright below" its floor (CPB:482-491), and the batch is measured against 900 — the sum
   of its slots' lower figures (Svc:3484-3496). The tool batch is told "3,500-5,000 words across
   the sections above" (CPB:2848) and measured against 1,100 / 1,200 / 1,000. **Nothing refuses on
   word count** (DG:90-229); a short batch is a warning string (Svc:3409-3415). Scorer floors (pillar
   3,000, blog 1,800, tool 3,000; GccV2LongFormTypes.cs:74-85) differ from the UI bands (blog
   2,000–2,700, tool 3,000–5,000; BC:686-716). Email: "Body must be 150-200 words" (Svc:2370) vs
   the UI band 50–125 (BC:704-709).
6. With `PageBuildsClosing = true` the writer is told to "answer the publisher's questions when
   booking" (GccPublisherPositions.cs:61-63) and "Write no call to action and nothing that asks the
   reader to book … or answer anything" (CPB:1067-1070). The questions themselves reach no prompt in
   either branch — correct, and the opposite of what NFP:137-140 and BC:280-281 say.
7. **Quotes on pillar and blog.** The body is told "Do not quote. No blockquotes" (CPB:363-365)
   and the guard refuses any quote, lede included (DG:185-193, 496-507). Against that, the same
   prompt offers a `quote` paragraph type (CPB:257-261), says "A block quotation is the one
   exception" (CPB:777-778), describes what a blockquote is for (CPB:652-656), says a block
   quotation is "the one place such an amount may stand" (GccCurrencyGrammar.cs:89-90), and offers
   "quote: open with a relevant quotation" as a lede type preferred for `case_study_data`
   (CPB:1270, 1444). No lede prompt says "Do not quote".
8. The tool's OWN SITE text says "a named customer's testimonial" belongs in a blockquote
   (CPB:655); the quotation rule says "NOT a compliment and NOT a testimonial" (CPB:993-995).
9. The tool lede is told "third person" and "Do NOT start with … a question" (CPB:1648, 1652), and
   offered "question" and "directAddress … (you/your)" lede types preferred for some angles and
   audiences (CPB:1268-1269, 1415-1418, 1442-1443). "PAIN BEFORE SOLUTION (required)"
   (CPB:1653-1655) vs `case_study_data`'s "lead with evidence — a number" (CPB:1246-1249). The
   lede "names no partner or tool unless the brief's angle is about that one product"
   (CPB:1132-1133) on a page that is about one product.
10. The prompts allow a figure labelled "hypothetical/illustrative" (CPB:1733, 1867, 2743); the
    numbers guard refuses any figure not in the evidence, with no such exemption (DG:365-387).
11. "CRITICAL: there is no case-study data available" is stated unconditionally on every pillar
    (CPB:1728, 1862), whatever the evidence holds and whatever the angle.
12. The pillar introduction: "every h3 … MUST itself nest 1-3 h4" (CPB:1726-1727) vs "some are
    stronger as continuous prose" (CPB:827-829) and "not a fixed lattice" (CPB:1854).
13. Provenance: the blog suffix refers to "advisory H2s below" that are not rendered
    (CPB:2409-2412 vs 2442-2449); the pillar is told every h3/h4 "needs a real tag"
    (CPB:1888-1890) while the guard licenses `plan` at any depth (GccHeadingProvenanceGuard.cs:166,
    202); "The five forms are" lists six (CPB:570-575).
14. Blog batch 1 shows every pain point twice ("Cover these", "Cover every one of these";
    GNF:89-95, 125-127) and is told each section must cover "something the others … do not"
    (CPB:495-496).
15. Voice: "consultative" and "expert third-person" are fixed (CPB:677, 1710, 2862) whatever the
    brief's tone; the consultant appendix says "Assume peer-level technical knowledge" and names
    "JSONB", "TDD" (Svc:492-498) while the brand voice says "skip heavy tech terms"
    (BrandTones.cs:44) and the system prompt says "Clear, everyday words" (CPB:712-713); the
    appendix's four-phase method is a generic one, against "the operator's, not a generic one"
    (GNF:150-153).
16. Beside "Write no call to action" (CPB:1067-1070): "align examples/CTAs to funnel" (CPB:1482;
    Svc:268), "close on the offer they actually make" (CPB:644-645), and in the tool's last batch
    "Place it after the reader has reason to act" (CPB:2859-2861).
17. The brand voice says "use bullet points for readability, avoid blocks of text"
    (BrandTones.cs:20-21); the system prompt says "some earn a list and most do not" (CPB:827) and
    "Never as decoration" (CPB:532-535).
18. KNOWN TOOLS says "one link per tool per section" (RBB:408-411) and never link the vendor site or
    any absolute URL (RBB:413-415); other lines say link the "first substantive mention"
    (CPB:366-367, 404) and research rule 2 requires the vendor page URL as a run href (Svc:369-377).
19. GROUNDING says "attribute it to the source that published it" (CPB:697-699); the operator
    framing says "do not attribute any of it to a source" (GNF:72-74), as do the publisher's
    positions (GccPublisherPositions.cs:53).
20. The guard refuses "ask yourself", "if these questions", "if your answers" (DG:452-491) and its
    comment says "The instruction names them as forbidden" (DG:449-451). No prompt names them.
21. "Do not reproduce it verbatim" (CPB:2789) vs research rule 3 "Quote verbatim or paraphrase
    closely" (Svc:378).
22. The ROLE & METHOD block and BRIEF CONTROLS legend (CPB:1484) define the non-consultant tones
    differently from the fixed voice lines they sit beside.

### 2d. Asked for and discarded; needed and never asked

- `ledeType` is required by the contract (CPB:1191) and strictly parsed (LlmResponseJsonParser.cs:250-258),
  then thrown away by every caller (Svc:1008, 1552, 2533, 2760); the continuity block's "Hook type"
  line is never filled (CPB:854-857).
- The introduction's heading is discarded (LlmResponseJsonParser.cs:186-190); lede and FAQ tags are
  forced to h2 (:56, 238-239); bold and italic are forced off (:286-292); a quote's runs and cite are
  replaced by the candidate's (TQG:227-231), or a typed fragment by the **whole** candidate span
  (TQG:236-242).
- `imagePrompt` is schema-required on every section, mentioned in no contract text, and overwritten
  by the image-prompt call (Svc:3743-3756); `provenance` is schema-required on tool sections and
  nothing reads it (Svc:1626); pillar and blog `sectionOutline` are requested and unused; tool
  metadata requires nine fields and stores two (Svc:3455-3474, 954-961).
- Titles: the pillar writer sees "Article title: {Topic}" (Svc:2502) but the published title comes
  from the post-body metadata (Svc:2641); the tool writer sees the product name (Svc:1503) but the
  title is "{product}: {keyword}" (Svc:954); `LedeHeadingInstruction` says do not restate a title
  that is not the one published (CPB:1204-1210).
- `ContentGuardrail` rewrites run text and headings on pillar and blog and not on the tool page
  (ContentGuardrail.cs:24-34, 70-90; Svc:2610, 2801).
- Needed and never asked: the quiz-phrasing ban (2c-20); the pillar FAQ is subject to the numbers,
  currency and link guards with no evidence and no grounding line in its prompt; the tool body is
  shown "Public path: /tools/{slug}" (CPB:2770) while the real path is `/tools/{dept}/{descriptor}/{slug}`
  (Svc:1697; GccContentPath.cs:79-89) and any `/tools/` link on a tool page is refused (DG:353-357);
  the pillar metadata prompt says "Derive sectionOutline from keyword SERP…" (CPB:1611) with no SERP
  block rendered on this path (RBB:256-289).

### 2e. Markdown in prompt assembly

CLAUDE.md §1a forbids Markdown "at crawl, ingest, index, extraction, or prompt assembly", and §1b
says the model never emits markup. In the prompts today:

- `block.AppendLine($"  # {heading}")` prints every crawled site heading with a Markdown heading
  marker in the OWN SITE block (CPB:633) — pillar and blog.
- `[{Title}] ({Url})` page headers (Svc:402, 2952), `[Competitor page n: …]` (Svc:3013), `[{label}]`
  (RBB:323) — while research rule 2 forbids exactly the `[title](url)` form (Svc:375-376).
- `- ` bullets in every evidence and brief block (Svc:404-406; RBB:184-198).
- `LeakedMarkupSyntax` catches `**`, `## `, `[x](y)` and `<tag>` (LlmResponseJsonParser.cs:22-24)
  and not a single `# ` — the exact form the prompt shows the model.
- The refusal "use the bold/italic/href fields instead" (LlmResponseJsonParser.cs:362) names fields
  the contract no longer offers; with no retries the model never sees it.

**Change for §2, in one line each:** the prompt is assembled from the `ContentDocument` shapes
and plain labelled lines, never `#`, `[…](…)` or `- `; one statement per rule (quotation, length,
voice, links, attribution, CTA), stated once; the tool page gets its own vocabulary, not the
pillar's; the pillar sees the core problem; a tool override reaches every type that names the tool;
a figure is licensed by retrieved text only, not by the brief; the research block is cut to the
sections that need it (U7b) and PARTNER DATA to the categories the type draws on (Stage 6); the form
hints say what the code does.

## 3. Open plan items, reconciled against git (2026-10-08)

Sixty-three items across the five `fix-*.md` files, the Rag and content-creator-v2 handoffs and the
10-07 plan, each checked against the code and `git log --all`: **done** means the code on `main`
does what the item says, with the commit named — not that it has been proven on a run; **partial**
means part is in code and a named remainder is not; **open** means not on `main`, which includes
the four items that exist only on the unpushed branch and the items only Jeff can do. The count
at the time of the sweep was 21 / 11 / 31; R7 was done the same afternoon, so it stands at
**22 done, 11 partial, 30 open.** The done ones are in their plans' tables. The rest:

| ID | Item | State |
|---|---|---|
| A1 full / R4 | Quotes verified through `/v1/verify`; bank keeps verified only | **Open** — F1 |
| A3 | Gate per type replaces 3-of-20; delete the two counters; measure fill rate first | **Open** — F2 |
| A4 | One Library question per claim, passages kept with the claim | **Partial** — asked per row (`6ea68fb`), not kept (F3) |
| A8 / F11 | Revise is a guarded generate; delete `versions/{id}/revise` | **Open on main** — only on the unpushed branch (F7, F8) |
| A9 → J8 | Brief, generate, versions, approvals under ManagePolicy | **Partial** — projects yes, `GccController` no (F13) |
| A10 | One evidence row per version | **Partial** — table and client, no caller (F16) |
| A13 review | Test: two types in parallel each record only their own models | **Open** |
| A15 | Delete the dormant v2 cluster | **Open** — 347 files (F17) |
| A17 | Crawl discovery ledger readable | **Partial** — returned untyped as `hosts` since `8b291cc`; not typed, not shown |
| A18 | Department from the taxonomy; refuse a generate without a path | **Open** — "marketing" default at Ctl:467, Svc:942, 1449 |
| A19 | Remove the `create.Notes` reads | **Open** — 11 reads |
| F-A19 | Room after thinking; stop_reason named; camelCase job result | **Partial** — stop_reason in the no-block error only (F6, F16) |
| D3 / D4 | metadata_json provenance; data-plane rule test | **Open on main** — unpushed branch (F8) |
| R3 | Collapse before the pool cut with backfill; measure near-copies | **Partial** — selection done; dedupe after the cut; no measurement (F19, F20) |
| C4 | Per-directory breakdown of `directoryCap` | **Open** — needs a decision |
| CR verification | No counter nothing increments | **Open** (F21) |
| Re-crawl #3 | A third declared partner re-crawled | **Open** — Jeff's; lightyear is no longer offered |
| F10 | "Drafted, saving" per type | **Open** — the only per-type push is after the save (Coord:402-420) |
| GR3 | Backfill every project's brief onto the project | **Partial** — one project |
| GR4 | artifacts.project_id; deliverables.artifact_id; jobs by project | **Partial** — jobs still keyed by create_id |
| GR5 / GR6 | `gcc_inputs`; drop `gcc_creates` and `create_id` | **Open** |
| GA3 | versions, revise, approve, SEO, polish, probe, keyword upload under `projects/{id}` | **Partial** — five `/versions` calls remain in the UI |
| GA4 / GA5 / GA6 | Raw inputs + `projects/{id}/serp`; export bundle; remove create routes | **Open** |
| U7b | Pillar/blog calls get only their own sections' passages | **Open** — no selector, no facets; the whole research block on every batch call (§2) |
| §5 Q1, Q3–Q8 (10-07 plan) | Budgets, licensing, facets, paragraph dedupe, PARTNER DATA, reuse discount, valid-values | **Open** — none decided in code |
| §7 Length | Batches come back near 60% of the word floor | **Open** — warnings only |
| §7 Mislabel | The OWN SITE block also carries notes, appendix, must-cover | **Open** — CPB:632; Svc:1907-1910 |
| §7 Tool unlisted | Tool pages have no unlisted-tools check | **Open** — DG:151, 204 vs DG:90 |
| §7 Tool lede | The tool lede prompt calls the page a "pillar" | **Open** — CPB:1648 |
| §7 Short-form | "injected by the app" URL, raw `book_now`, `words: 0` | **Open** — Svc:2372; CPB:2523 |
| §7 Keyword H2 | "The Costs of Manual Automated Approval Workflows" | **Partial** — blog only (`aa7ebee`) |
| Closing flag | `ClosingCallToActionInstruction` survives behind `PageBuildsClosing`, default false | **Open** — GenerationRequest.cs:84; only Svc:1969 sets it |
| Wave-1 proof | One seven-type Generate on "test", read by Jeff | **Open** — the 10-05 runs made three types; the third run is planned |

## 4. Documentation that is false right now

**Corrected today, this repo:** `plans/partner-evidence-reaches-the-writer.md` ("Nothing here is
built" — Stages 1–3 and the verify route landed the morning it was written; Stage 5 shipped 10-08 as
rows; a Status table now leads it); `README.md` R7 (upsert delay 0, API `mem_limit` 7g, no embed
retry, the chunker's tokenizer); `HANDOFF.md` §3 deploy table, §4 R7, §4b, the open-items list.

**Corrected the same day, second pass, at Jeff's instruction** ("Update false documentation"):
every row below except the commit message, which cannot be edited — it is answered by the audit
line (F13) and by `fix-geekapi.md`'s status. The table stays as the record of what was false and
for how long:

| Where | Says | Is |
|---|---|---|
| content-creator-v2 `plans/fix-overview.md:6`, `fix-geekapi.md:3`, `fix-geek-crawler-rag.md:3`, `fix-geekrepository.md:3` | "Nothing in … is built" | Dozens of stage commits on `main` (GB `185df80`…`f420d7b`; Rag `1a25095`…`5e622b6`) |
| `fix-geekapi.md` A8 status | Interim and full rebuild "both built" | Neither on `main` (F7, F8) |
| `fix-geekapi.md:140` (F-A6) | "code requires one or more" quotations | Zero is a gap, more than one refuses (DG:123-140) |
| `fix-geek-crawler-rag.md:66`, R5 | "The scheduler defaults off in the repo" | `config.py:123` and compose `:50` default on |
| `fix-geek-crawler-rag.md` Status | Three R4 defects open | Fixed in `9a0c901` |
| `fix-geek-crawler-rag.md` R2, D15 | The in-process BM25 re-rank "kept" / "rank it" | Deleted (`e2937f0`, `d8a628e`; `bm25_rank.py` gone) |
| `fix-geek-crawler-rag.md:88-89` | `BuildNeed` is "the question GeekAPI sends" | The no-brief fallback only (Res:679-703) |
| `fix-geek-crawler-v2.md` header | Re-post "not yet done" | Removed in CR `4466494` |
| `fix-geek-crawler-v2.md` C3 | No counter nothing increments | Three (F21) |
| content-creator-v2 `HANDOFF.md:41-42` | CC `18a58d8`, GB `e45bd0e` deployed | CC `4677b69`, GB `288dde3` |
| content-creator-v2 `HANDOFF.md` §5 | F1 second half, F7–F9, GF5, R6, A6 review open | F1 (`18fd324`), F7/F9 (`18a58d8`), F8 (`53661db`), GF5 (`53661db`), R6, A6 (`2775051`; retries removed `303d04c`) done |
| GeekBackend `687242a` message | "`ToLlm` refuses one it does not know" | The private `GccController.ToLlm` maps unknown → OpenAI (F13) |
| GeekBackend 10-07 plan §3.3 | The long closing block is replaced | Survives behind `PageBuildsClosing = false` by default |
| GeekBackend A17 | "nothing reads it" | `GET crawls/{runId}` returns `hosts` since `8b291cc` |

## 5. Order of work, and what each step buys

1. **F5, F6 — provider faults** (small): a timeout or a truncation costs one type, not the run; the
   fault is named.
2. **F1 — one verifier**: every extracted item and every quotation checked by `/v1/verify`; the two
   C# comparisons deleted. This is the step that lets any page be called grounded.
3. **F3 then F2 — row ids on passages, then the per-type / per-row gate**: delete 3-of-20; refuse a
   page naming the row that has no evidence. Measure category fill rates before the type lists.
4. **F8 — the branch decision**, then **F7 — Revise** through the guarded path.
5. **F10, F11, F12 — what the operator sees**: gaps with identity, one quotation rule stated once,
   the dead and contradictory copy removed.
6. **F13 — authorization** on `GccController`.
7. **F15 — the type allow-list**; **F14 — the swallowed failures** become refusals or warnings.
8. **§2 — the prompt**: Markdown out of prompt assembly (2e); one statement per rule (2c); the
   pillar sees the core problem and tool overrides reach every type (2a); figures licensed by
   retrieved text only (2a); the research block and PARTNER DATA cut to what the call needs (2b,
   U7b); the form hints rewritten from the code.
9. **F16, F17, F18, §4 — provenance, dead code, the stale lines.**

Measurements that gate decisions, none of which this session can run: Stage 2 on Ramp `4563f7ec`;
near-copies; category fill rates over banked extractions; the Tipalti readiness run after Jeff's
re-crawl, re-index and rows (`retrieval-from-the-brief.md` tests 1–4); the seven-type proof run.

## Rules that bind the fixes

Fail closed, no fallbacks, no auto-repair, no Markdown in the corpus or the prompts (CLAUDE.md §1a,
§1b, §2). A verifier that cannot be reached refuses; a snap that changes words is a refusal; a gap
is reported, never filled. Contract changes land on both sides in paired commits. Nothing here
re-crawls or re-indexes on its own.
