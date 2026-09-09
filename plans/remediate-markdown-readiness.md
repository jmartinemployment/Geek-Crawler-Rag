# Remediate the 17 Markdown-unready crawl runs

Status: **Phase 1 complete — production remediation not started**  
Inventory captured: **2026-09-08**
Phase 1 hardened: **2026-09-09** (`batch-size` default 25, `delay-seconds` default 2,
`complete`/`external` readiness, missing-HTML deleted as `extract_empty`)

## Objective

Convert usable stored HTML to Markdown, delete unusable page rows, and set
`crawl_runs.MarkdownReadyAt` only after each non-empty run passes a complete
readiness check. Zero-page runs must not be marked ready.

This is remediation of historical crawl data. New crawls should arrive with
Markdown and a valid run-level readiness marker.

## Inventory

The 17 terminal, unindexed, Markdown-unready runs split into:

- **13 fully missing:** 73,484 pages lack Markdown; every page still has HTML
  and can be attempted without a re-crawl.
- **1 partial:** n8n has 443 ready pages and 17 missing pages with no HTML. The
  17 unusable rows and their links should be deleted, then the run revalidated.
- **3 zero-page runs:** no backfill is possible. Two are smoke fixtures; the
  QuickBooks run requires a fresh crawl if it is still wanted.

| Pages | Run ID | Site | Classification |
|------:|--------|------|----------------|
| 460 | `a8ee85a3-70e6-4f6a-a475-fd38ffb604db` | https://n8n.io | Partial; 17 missing with no HTML |
| 913 | `1d3963ef-9fc5-453c-97be-8096ed141a74` | https://www.chaserhq.com | Fully missing; HTML retained |
| 956 | `b124ef74-6bbc-4038-ab10-d38e66d35c79` | https://www.zoho.com | Fully missing; HTML retained |
| 1,347 | `9b2c8acf-0d6d-46f8-8652-a19c43cb8d85` | https://tipalti.com | Fully missing; HTML retained |
| 1,751 | `2cc3492c-de60-43e6-8e5d-f4b13920f9f5` | https://contentstudio.io | Fully missing; HTML retained |
| 2,627 | `35f41ddc-ad2f-4397-b108-68b2a641c31e` | https://www.xero.com/us/ | Fully missing; HTML retained |
| 3,407 | `bfa02724-e56e-4472-b6a9-510fbff4105b` | https://www.notion.com | Fully missing; HTML retained |
| 3,639 | `31681465-76a1-4666-b76f-e550e70cd16f` | https://www.rippling.com | Fully missing; HTML retained |
| 3,803 | `1dfaafe1-6e3e-4f00-83aa-5c23fffb187f` | https://turbotax.intuit.com/personal-taxes/online/live/full-service/business-taxes/ | Fully missing; HTML retained |
| 4,002 | `99c6b00b-a395-45ed-8ff8-dfd1d4a47fae` | https://clickup.com | Fully missing; HTML retained |
| 5,337 | `5bda7ddd-3a9b-4520-8b4a-ca08193363de` | https://www.smartsheet.com | Fully missing; HTML retained |
| 6,715 | `a775ac12-1912-4568-8a63-d56c219725b8` | https://stripe.com/tax | Fully missing; HTML retained |
| 10,793 | `14733251-58b1-46c7-a3df-40d38601c7c9` | https://www.avalara.com/us/en/index.html | Fully missing; HTML retained |
| 28,194 | `710bc59c-a734-41c8-8bbf-da8f79270771` | https://www.activecampaign.com | Fully missing; HTML retained |
| 0 | `3e2c8bf0-c6de-4e13-844d-72fd0793f583` | `partner-smoke.example` | Zero-page smoke fixture |
| 0 | `e97bdb65-3ae7-4b0f-b596-b75a0660f3ce` | https://quickbooks.intuit.com | Zero-page production run |
| 0 | `fec37b06-e989-4e35-86ed-ecbe9dacc303` | `rival-smoke.example` | Zero-page smoke fixture |

## Phase 1 — harden the backfill command

Before changing production data:

1. Allow a successful full backfill to mark both `complete` and `external`
   runs ready. This must match the scheduler and reconciliation contract.
2. Remove the top-level `Html` requirement from the page query. Missing-
   Markdown rows without usable HTML must be visited and deleted as
   `extract_empty`; otherwise the n8n run can never become ready.
3. Add a configurable inter-batch delay and keep the existing small,
   cursor-based batches. Default remediation settings should be 25 pages per
   batch and a 1–2 second delay.
4. Add tests covering an `external` run, missing HTML deletion, incomplete
   passes, write errors, and the rule that zero-page runs are never marked
   ready.
5. Run the test suite and a dry run before production writes.

## Phase 2 — remediate one run at a time

Process the 14 non-empty runs in ascending page-count order, using explicit
`--run-id` values. Do not run a global backfill.

For each run:

1. Confirm there is no other backfill for that run.
2. Run a dry pass and save its counts.
3. Run the full write pass with no `--limit`.
4. Require exit code 0, `complete_pass=1`, and `errors=0`.
5. Verify `pages > 0`, missing Markdown count is zero, and
   `MarkdownReadyAt` exists before advancing to the next run.
6. Record updated/deleted/error counts in an operations report.

Pause between runs when Mongo, Qdrant, or VPS CPU is elevated. Do not overlap
the large Avalara or ActiveCampaign passes with an active index job.

Example:

```bash
uv run python scripts/backfill_markdown.py \
  --run-id a8ee85a3-70e6-4f6a-a475-fd38ffb604db \
  --batch-size 25 \
  --delay-seconds 2 \
  --write
```

The established corpus policy applies: locale, crawl-failure, robots-denied,
non-English, and extract-empty pages are deleted with their links rather than
retained or marked indexable.

## Phase 3 — resolve the zero-page runs

- Remove or archive the two `.example` smoke run records from production
  reporting. Never create synthetic pages or readiness markers for them.
- Decide whether QuickBooks remains required. If yes, queue a fresh crawler
  run and let the current crawler produce Markdown. If no, archive/remove the
  empty run from production reporting.
- Keep all three unready until they contain at least one usable page.

## Phase 4 — release to scheduled indexing

After each successful backfill, run targeted readiness reconciliation as a
second check. The scheduler may then select the run naturally, smallest first.
Keep index concurrency at one and the existing two-hour interval.

Do not manually enqueue all remediated runs at once. The two largest runs must
remain subject to the scheduler's page safety cap and normal resource review.

## Acceptance criteria

- All 73,484 retained-HTML pages were attempted exactly once by a full,
  cursor-based pass.
- The n8n run no longer contains its 17 no-HTML/no-Markdown rows.
- Every marked-ready run has at least one page and zero pages with blank
  `Markdown`/`markdown`.
- No run is marked ready after a partial pass or any recorded error.
- The 13 fully missing runs and n8n become eligible, unless extraction leaves
  a documented data-quality blocker.
- The three zero-page runs remain ineligible and have an explicit disposition.
- Scheduler and index-job status explain any remaining exclusion.
- Mongo/Qdrant health and VPS CPU remain stable during the staged work.

## Recovery

Before production writes, confirm a usable Mongo backup or snapshot exists.
Backfilled Markdown can be regenerated from retained HTML. Deleted unusable
rows are intentionally not restored; if classification was wrong, restore
from backup or re-crawl the affected site. Clear `MarkdownReadyAt` immediately
if any post-run verification fails.
