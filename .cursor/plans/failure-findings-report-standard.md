# Failure findings report standard

Unacceptable in prior reports:
- Treating `operator_no_requeue` / park stamps as **cause**
- Thin “status tables” without a dated evidence chain
- Overwriting narrative with current Mongo `error` after operator mutation
- Omitting secondary damage (Qdrant wipe) as a separate finding

## Required sections (every quarantined / failed run)

1. **Identity** — runId, seed, crawl Status/Type, ContentReady (`ContentReadyAt`; `MarkdownReadyAt` is legacy — Markdown is forbidden)
2. **Primary failure mode** — one technical sentence (cancel / OpenAI 400 / OpenAI 500 / recovery reclaim). Never “operator quarantine.”
3. **Evidence chain** — timestamped log lines (or job field values *as of failure*, not as of park)
4. **Secondary damage** — wipe, attempt bumps, error-field overwrite, lost points
5. **Artifact quality** — what quarantine JSON actually contains vs what it should
6. **Residual state** — Mongo state/attempt/progress counters · Qdrant point count · recoverability
7. **Conclusion** — cause + damage + next safe action (one run)

Operator park is recorded only under **Secondary damage** or **Artifact quality**, labeled `action`, never `cause`.
