# Remove retry violations and fix ingest root cause

Status: **Not started** — separately tracked from the policy rollout in [`no-retries-no-fallbacks.md`](./no-retries-no-fallbacks.md).

## Law

**No Retries. No Fallbacks. No Crappy Code.** Do not “fix” this work by adding retries.

## Known violations to remove

| Location | Violation |
|----------|-----------|
| `Geek-Crawler-v2/src/storage/geek-api-client.ts` | `LINK_BATCH_ATTEMPTS` / link-batch retry loop with backoff |
| GeekBackend `GeekCrawlerPageBatchWriter.SavePagesWithRetryAsync` | Split/retry on retriable batch errors |

## Ingest root cause (Avid / Stampli class)

Runs failed on `POST …/pages/batch` with **502** / empty **500** after large crawls. Investigation pointed at:

- Uncapped raw HTML + capped markdown sent inline per page (Mongo BSON **16 MB**; API/repo request-size mismatch).
- GeekAPI mapping: repo non-2xx → **502**; timeouts / unhandled exceptions → **500** with empty body.

This follow-up must **fix those causes** (size guards, markdown-only ingest if appropriate, explicit JSON error bodies) — not paper over them with retries.

## Out of scope here

Policy and Cursor-rule documentation (already covered by `no-retries-no-fallbacks.md`).
