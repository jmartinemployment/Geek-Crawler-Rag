# Establish: No Retries, No Fallbacks, No Crappy Code

Status: **Complete** (policy and documentation rolled out 2026-09-12)

Follow-up for code: [`remove-retry-violations-and-fix-ingest.md`](./remove-retry-violations-and-fix-ingest.md).

## Law

**No Retries. No Fallbacks. No Crappy Code.**

- **No Retries** — Do not add or retain application-level retry loops, exponential backoff, or “try again later” wrappers around failures, including HTTP 5xx responses, timeouts, Mongo failures, OpenAI failures, and GeekAPI `pages/batch` or `links/batch` ingest failures. Fail the operation on its first failure, return the real diagnostic error, and fix the root cause.
- **No Fallbacks** — Do not turn a required-operation failure into apparent success by dropping fields, skipping persistence, swallowing exceptions, or continuing on a best-effort basis. Intentional product processing paths, such as selecting Playwright when static HTML is not viable, are permitted only when explicit, logged, and contract-preserving. They must never hide API or storage failure.
- **No Crappy Code** — Delete incorrect behavior instead of concealing it. Empty error responses, diagnostic truncation, silent catches, and fatal failures without actionable server-side detail are defects. Fix the behavior and preserve enough safe diagnostic context to identify the cause.

## Scope: policy and documentation rollout

This change establishes the rule across the RAG and crawler repositories. It does **not** remove existing violating code or fix the current Avid/Stampli ingest failures.

Known violations, including `LINK_BATCH_ATTEMPTS` in `Geek-Crawler-v2/src/storage/geek-api-client.ts` and `SavePagesWithRetryAsync`, must be removed in a **separately tracked** implementation task. That follow-up must also address the underlying ingest failure rather than adding retries.

## Changes

### 1. Update `plans/rules.md`

Add a hard **Correctness — fail closed** section containing this law and a checklist:

- No application-level retry, backoff, or retry wrapper added or retained.
- Required persistence or ingest failure fails the operation.
- Failure responses retain actionable diagnostic detail.
- No exception is swallowed or converted to success.
- Designed alternate processing paths are explicit and logged.

### 2. Add RAG Cursor rule

Create `.cursor/rules/no-retries-no-fallbacks.mdc` with `alwaysApply: true`.

Require plans and changes to follow the law, reject retries/backoff and silent degradation, and point to `plans/rules.md` as the authoritative source.

### 3. Add crawler Cursor rule

Create `/Users/jeffmartin/development/Geek-Crawler-v2/.cursor/rules/no-retries-no-fallbacks.mdc` with `alwaysApply: true`.

Explicitly prohibit retries for `createPagesBatch` and `createLinksBatch`; require ingest failures to expose the HTTP status and safe, useful response diagnostics; prohibit treating failed persistence as a non-fatal success.

### 4. Align RAG README

In `README.md`: remove wording that promotes bounded application-level retries. Preserve the existing fail-closed embedding configuration, including `OPENAI_EMBEDDING_MAX_RETRIES=0`, and link it to the new rule.

### 5. Align crawler README

In `/Users/jeffmartin/development/Geek-Crawler-v2/README.md`: remove wording that presents application ingest retries as a feature. Clarify that Crawlee request-queue behavior for source-page fetching is distinct from GeekAPI persistence and must not be extended into ingest retry logic. Describe Cheerio and Playwright as explicit processing paths, not silent fallback behavior.

### 6. Track implementation follow-up

Create `plans/remove-retry-violations-and-fix-ingest.md` that lists:

- Remove `LINK_BATCH_ATTEMPTS` / link-batch retry loop in Geek-Crawler-v2.
- Remove `SavePagesWithRetryAsync` (or equivalent) in GeekBackend.
- Fix Avid/Stampli-class `pages/batch` root cause (size limits, empty 500 bodies, etc.) — **not** with retries.

## Verification

- `plans/rules.md` contains the complete law and checklist.
- Both `.cursor/rules/no-retries-no-fallbacks.mdc` files exist and set `alwaysApply: true`.
- Neither README advertises application-level retry behavior.
- The implementation follow-up is explicitly tracked to remove existing retry violations and correct the ingest failure’s root cause.
