# Failure findings: `a8ee85a3-70e6-4f6a-a475-fd38ffb604db`

Report date: 2026-09-10 · Source: Hostinger API logs + Mongo + Qdrant + quarantine volume  
Standard: [failure-findings-report-standard.md](failure-findings-report-standard.md)

## 1. Identity

| Field | Value |
|-------|--------|
| runId | `a8ee85a3-70e6-4f6a-a475-fd38ffb604db` |
| Seed | `https://n8n.io` |
| CrawlType / Status | partner / **external** |
| MarkdownReadyAt | `2026-09-09T14:50:35.066588+00:00` |
| mongoPageCount | 443 |

## 2. Primary failure mode

**Indexer cancelled while job was in flight** on attempt **3** (scheduled re-claim), after ~12 minutes of successful embedding/upsert progress.

Not an OpenAI 400/500. Not “operator quarantine.”

## 3. Evidence chain

| UTC | Evidence |
|-----|----------|
| 13:22:17.103 | `Enqueued index job … trigger=scheduled` / scheduler `pages=443` |
| 13:22:17.145 | `Indexing runId=a8ee85a3… status=external mongoPageCount=443` |
| 13:22:17.547 | `Skipping Qdrant delete for retry … attempt=3` (intentional no-wipe at start) |
| ~13:22–13:34 | Progress counters reached **pagesSeen=244**, **pagesEnglish=244**, **chunksUpserted=28086** (frozen on job doc) |
| 13:34:36.312 | `Deleted Qdrant points for runId=a8ee85a3…` |
| 13:35:15.534 | Quarantine file written; nested job snapshot still has `error: Indexer cancelled while job was in flight` |
| 13:36:40.396 | Mongo `state=skipped`, `error` **overwritten** to `Quarantined (no automatic requeue).` |

## 4. Secondary damage

1. **Qdrant wipe after progress** — start correctly skipped delete (`attempt=3`); stop/cleanup later deleted all points for this runId. Current count: **0**.
2. **Error-field overwrite** — failure text replaced by operator park message; current Mongo `error` is not the failure cause.
3. **Lost rebuild progress** — 28 086 chunks upserted in this attempt are gone from Qdrant (counters remain on the job doc only).

## 5. Artifact quality

File: `requeue_quarantine_a8ee85a3-….json`

| Present | Missing / wrong |
|---------|-----------------|
| Snapshot of job at park (`cancelled…`, attempt 3, 28086 chunks) | No embed batch `items`, no HTTP status, no cancel stack |
| Top-level `reason: operator_no_requeue` | That field is an **action**, not a cause; easy to misread as root cause |

## 6. Residual state

| Store | State |
|-------|--------|
| Mongo job | `skipped`, attempt 3, trigger scheduled, progress counters as above, park error text |
| Qdrant | **0** points for this runId |
| Crawl | Still `external`, Markdown ready — source corpus intact |
| Recoverability | Full re-index required; corpus available; do not treat park JSON as diagnostic of embed failure |

## 7. Conclusion

- **Cause:** in-flight **cancel** during attempt 3 scheduled indexing.  
- **Damage:** post-cancel **Qdrant delete** zeroed the run despite 28k upserts; Mongo error text then **park-overwritten**.  
- **Next safe action (this run only):** after no-wipe-on-cancel / empty-embed harden as applicable, deliberate single-run requeue — not covered by this report’s execution.
