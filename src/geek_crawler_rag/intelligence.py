"""Deterministic, evidence-first query planning and competitor intelligence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Literal

from bs4 import BeautifulSoup

from geek_crawler_rag.diagnostic_models import (
    DiagnosticDocument,
    EvidenceReference,
    QueryOrigin,
    QueryProvenance,
    ReadinessScoreRequest,
)
from geek_crawler_rag.diagnostics import DiagnosticService
from geek_crawler_rag.intelligence_models import (
    ComparativeDimension,
    CompetitorAuditArtifact,
    CompetitorAuditRequest,
    CompetitorDimensionScore,
    CompetitorPageAnalysisArtifact,
    CompetitorPageAnalysisRequest,
    CompetitorPositioningArtifact,
    CompetitorPositioningRequest,
    ContentCompleteness,
    ContentGapArtifact,
    ContentGapRequest,
    DimensionFinding,
    GapRecord,
    IntelligenceDimension,
    IntelligenceProvenance,
    JourneyStage,
    MessagingHypothesis,
    PageReadinessEntry,
    PageSnapshot,
    PlannedQuery,
    PositioningAttribute,
    PositioningGap,
    PrioritizedAction,
    PriorityTier,
    QueryClassification,
    QueryPlanArtifact,
    QueryPlannerRequest,
    QueryPriorityMethodology,
    QueryPrioritySignals,
    ReadinessComparisonArtifact,
    ReadinessComparisonRequest,
    ReadinessDimensionDelta,
    SearchIntent,
)

_TOKEN = re.compile(r"[a-z0-9]+")
_HEADING = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")
_TOPIC_HEADING = re.compile(r"(?m)^#{2,6}\s+(.+?)\s*$")
_QUESTION = re.compile(r"(?m)^(?:#{1,6}\s+)?([^\n?]{3,}\?)\s*$")
_CLUSTER_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "best",
        "buy",
        "for",
        "guide",
        "how",
        "is",
        "pricing",
        "the",
        "to",
        "use",
        "what",
        "why",
        "alternatives",
    }
)
_DIMENSION_PATTERNS: dict[IntelligenceDimension, tuple[re.Pattern[str], ...]] = {
    IntelligenceDimension.TOPICS: (_TOPIC_HEADING,),
    IntelligenceDimension.QUESTIONS: (_QUESTION,),
    IntelligenceDimension.CAPABILITIES: (
        re.compile(
            r"(?im)^.*\b(?:feature|capabilit|supports?|allows?|can help|can [a-z]+).*$"
        ),
    ),
    IntelligenceDimension.PROOF: (
        re.compile(
            r"(?im)^.*\b(?:case stud|customer|used by|trusted by|result|"
            r"\d+(?:\.\d+)?%|testimonial).*$"
        ),
    ),
    IntelligenceDimension.PRICING: (
        re.compile(
            r"(?im)^.*(?:\b(?:price|pricing|plan|per month|per year)\b|"
            r"[$€£]\s?\d).*$"
        ),
    ),
    IntelligenceDimension.PROCESS: (
        re.compile(
            r"(?im)^(?:#{1,6}\s+.*\b(?:how it works|process|steps?)\b.*|"
            r"\s*\d+[.)]\s+.+)$"
        ),
    ),
    IntelligenceDimension.CALLS_TO_ACTION: (
        re.compile(
            r"(?im)^.*\b(?:get started|start (?:a )?(?:free )?trial|"
            r"book (?:a )?demo|contact (?:us|sales)|buy now|sign up).*$"
        ),
    ),
}
_DIMENSIONS = list(IntelligenceDimension)


class IntelligenceService:
    """Pure application service over caller-supplied data; no retrieval or I/O."""

    def __init__(self, diagnostics: DiagnosticService | None = None) -> None:
        self._diagnostics = diagnostics or DiagnosticService()

    def query_plan(self, request: QueryPlannerRequest) -> QueryPlanArtifact:
        rows = list(request.queries)
        seen = {_normalize_query(item.query) for item in rows}
        generated = 0
        if request.max_generated_queries:
            for topic in _unique_clean(request.hypothesis_topics):
                for template in (
                    "what is {topic}",
                    "{topic} alternatives",
                    "{topic} pricing",
                ):
                    query = template.format(topic=topic)
                    normalized = _normalize_query(query)
                    if normalized in seen:
                        continue
                    rows.append(
                        QueryProvenance(
                            query=query,
                            origin=QueryOrigin.GENERATED_HYPOTHESIS,
                            sourceReference="deterministic-template:query-hypotheses.v1",
                        )
                    )
                    seen.add(normalized)
                    generated += 1
                    if generated >= request.max_generated_queries:
                        break
                if generated >= request.max_generated_queries:
                    break

        planned = [_planned_query(row) for row in rows]
        planned.sort(key=lambda item: (-item.priority_score, item.query.casefold()))
        warnings = [
            (
                "Priority scores are explicit deterministic heuristics; no search "
                "volume, traffic, ranking, or demand was inferred."
            )
        ]
        if generated:
            warnings.append(
                f"Generated {generated} query hypothesis(es). They are labeled "
                "generatedHypothesis and are not observed demand."
            )
        if not request.queries:
            warnings.append(
                "No observed or imported query data was supplied; every planned query "
                "is a generated hypothesis."
            )
        return QueryPlanArtifact(
            methodology=QueryPriorityMethodology(),
            queries=planned,
            clusterCount=len({item.cluster_id for item in planned}),
            warnings=warnings,
            sources=request.sources,
        )

    def competitor_page_analysis(
        self, request: CompetitorPageAnalysisRequest
    ) -> CompetitorPageAnalysisArtifact:
        page = request.page
        findings, generated = _analyze_page(page)
        supplied, invalid = _verified_supplied_evidence(page)
        evidence = _dedupe_evidence([*supplied, *generated])
        opportunities = [
            _dimension_opportunity(item.dimension)
            for item in findings
            if item.present is False
        ]
        warnings = [_no_demand_warning()]
        warnings.extend(_digest_warnings([page]))
        if page.content_completeness == ContentCompleteness.PARTIAL:
            warnings.append(
                "Competitor page input is partial; absent signals are reported as unknown "
                "and do not produce absence-based opportunities."
            )
        if invalid:
            warnings.append(
                "Ignored supplied evidence references that failed exact source quote/span "
                f"validation: {', '.join(sorted(invalid))}."
            )
        return CompetitorPageAnalysisArtifact(
            competitorId=page.competitor_id,
            dimensions=findings,
            opportunities=opportunities,
            warnings=warnings,
            provenance=_provenance([page], evidence),
        )

    def content_gap(self, request: ContentGapRequest) -> ContentGapArtifact:
        subject_analyses = [_analyze_page(page) for page in request.subject_pages]
        competitor_analyses = [
            _analyze_page(page) for page in request.competitor_pages
        ]
        all_pages = [*request.subject_pages, *request.competitor_pages]
        supplied: list[EvidenceReference] = []
        invalid: set[str] = set()
        for page in all_pages:
            valid, rejected = _verified_supplied_evidence(page)
            supplied.extend(valid)
            invalid.update(rejected)
        generated = [
            evidence
            for _, evidence_rows in [*subject_analyses, *competitor_analyses]
            for evidence in evidence_rows
        ]
        evidence = _dedupe_evidence([*supplied, *generated])

        subject_partial = any(
            page.content_completeness == ContentCompleteness.PARTIAL
            for page in request.subject_pages
        )
        dimensions: list[ComparativeDimension] = []
        gaps: list[GapRecord] = []
        for dimension in _DIMENSIONS:
            subject_rows = [
                finding
                for findings, _ in subject_analyses
                for finding in findings
                if finding.dimension == dimension
            ]
            competitor_rows = [
                (page, finding)
                for page, (findings, _) in zip(
                    request.competitor_pages, competitor_analyses, strict=True
                )
                for finding in findings
                if finding.dimension == dimension
            ]
            subject_present = any(item.present is True for item in subject_rows)
            subject_coverage: bool | None = (
                True if subject_present else None if subject_partial else False
            )
            covered_competitors = [
                (page, item) for page, item in competitor_rows if item.present is True
            ]
            subject_ids = _unique(
                evidence_id
                for item in subject_rows
                for evidence_id in item.evidence_ids
            )
            competitor_ids = _unique(
                evidence_id
                for _, item in covered_competitors
                for evidence_id in item.evidence_ids
            )
            dimensions.append(
                ComparativeDimension(
                    dimension=dimension,
                    subjectCoverage=subject_coverage,
                    competitorCoverageCount=len(covered_competitors),
                    competitorPageCount=len(request.competitor_pages),
                    subjectEvidenceIds=subject_ids,
                    competitorEvidenceIds=competitor_ids,
                    summary=_comparison_summary(
                        dimension,
                        subject_coverage,
                        len(covered_competitors),
                        len(request.competitor_pages),
                    ),
                )
            )
            if not covered_competitors or subject_coverage is True:
                continue
            source_ids = _unique(
                page.source.source_id for page, _ in covered_competitors
            )
            status = (
                "coverageUnknown"
                if subject_coverage is None
                else "supportedGap"
            )
            confidence = (
                "unknown"
                if subject_coverage is None
                else "high"
                if len(covered_competitors) == len(request.competitor_pages)
                else "medium"
            )
            gaps.append(
                GapRecord(
                    gapId=_stable_id("gap", dimension.value, *source_ids),
                    dimension=dimension,
                    signal=dimension.value,
                    status=status,
                    opportunity=_dimension_opportunity(dimension),
                    competitorSourceIds=source_ids,
                    evidenceIds=competitor_ids,
                    confidence=confidence,
                )
            )

        subject_topics = {
            _normalize_signal(topic)
            for page in request.subject_pages
            for _, _, topic in _dimension_matches(
                _visible_text(page), IntelligenceDimension.TOPICS
            )
        }
        competitor_topics: dict[
            str, tuple[str, list[str], list[str]]
        ] = {}
        evidence_by_id = {item.evidence_id: item for item in evidence}
        for page, (findings, _) in zip(
            request.competitor_pages, competitor_analyses, strict=True
        ):
            topic_finding = next(
                item
                for item in findings
                if item.dimension == IntelligenceDimension.TOPICS
            )
            for topic, evidence_id in zip(
                topic_finding.signals, topic_finding.evidence_ids, strict=True
            ):
                normalized = _normalize_signal(topic)
                if normalized in subject_topics:
                    continue
                label, source_ids, evidence_ids = competitor_topics.setdefault(
                    normalized, (topic, [], [])
                )
                source_ids.append(page.source.source_id)
                evidence_ids.append(evidence_id)
                competitor_topics[normalized] = (label, source_ids, evidence_ids)
        for normalized in sorted(competitor_topics):
            topic, source_ids, evidence_ids = competitor_topics[normalized]
            linked_evidence_ids = [
                evidence_id
                for evidence_id in _unique(evidence_ids)
                if evidence_id in evidence_by_id
            ]
            if not linked_evidence_ids:
                continue
            gaps.append(
                GapRecord(
                    gapId=_stable_id("gap", "topic", normalized, *_unique(source_ids)),
                    dimension=IntelligenceDimension.TOPICS,
                    signal=topic,
                    status="coverageUnknown" if subject_partial else "supportedGap",
                    opportunity=(
                        f'Consider explicit coverage of the supplied competitor topic '
                        f'"{topic}" when relevant to the subject.'
                    ),
                    competitorSourceIds=_unique(source_ids),
                    evidenceIds=linked_evidence_ids,
                    confidence="unknown" if subject_partial else "medium",
                )
            )

        warnings = [_no_demand_warning()]
        warnings.extend(_digest_warnings(all_pages))
        if subject_partial:
            warnings.append(
                "At least one subject page is partial; uncovered subject dimensions are "
                "coverageUnknown rather than asserted gaps."
            )
        partial_competitors = [
            page.source.source_id
            for page in request.competitor_pages
            if page.content_completeness == ContentCompleteness.PARTIAL
        ]
        if partial_competitors:
            warnings.append(
                "Partial competitor inputs may understate competitor coverage: "
                f"{', '.join(partial_competitors)}."
            )
        if invalid:
            warnings.append(
                "Ignored supplied evidence references that failed exact source quote/span "
                f"validation: {', '.join(sorted(invalid))}."
            )
        return ContentGapArtifact(
            dimensions=dimensions,
            gaps=gaps,
            warnings=warnings,
            provenance=_provenance(all_pages, evidence),
        )

    def readiness_comparison(
        self, request: ReadinessComparisonRequest
    ) -> ReadinessComparisonArtifact:
        pages: list[PageSnapshot] = [request.subject_page, *request.competitor_pages]
        subject_entry, subject_evidence = self._page_readiness_entry(
            request.subject_page, role="subject"
        )
        competitor_pairs = [
            self._page_readiness_entry(
                page,
                role="competitor",
                competitor_id=page.competitor_id,
                competitor_name=page.competitor_name,
            )
            for page in request.competitor_pages
        ]
        competitor_entries = [entry for entry, _ in competitor_pairs]
        readiness_evidence = _dedupe_evidence(
            [
                *subject_evidence,
                *[
                    evidence
                    for _, evidence_rows in competitor_pairs
                    for evidence in evidence_rows
                ],
            ]
        )
        deltas: list[ReadinessDimensionDelta] = []
        subject_dimensions = {
            item.dimension: item for item in subject_entry.dimensions
        }
        for dimension in subject_dimensions:
            competitor_scores = [
                CompetitorDimensionScore(
                    sourceId=entry.source_id,
                    competitorId=entry.competitor_id or entry.source_id,
                    score=next(
                        item.score
                        for item in entry.dimensions
                        if item.dimension == dimension
                    ),
                )
                for entry in competitor_entries
            ]
            available = [
                item.score for item in competitor_scores if item.score is not None
            ]
            best = max(available) if available else None
            subject_score = subject_dimensions[dimension].score
            delta = (
                None
                if subject_score is None or best is None
                else round(subject_score - best, 1)
            )
            deltas.append(
                ReadinessDimensionDelta(
                    dimension=dimension,
                    subjectScore=subject_score,
                    competitorScores=competitor_scores,
                    bestCompetitorScore=best,
                    deltaVsBestCompetitor=delta,
                    summary=_readiness_delta_summary(
                        dimension, subject_score, best, delta
                    ),
                )
            )

        prioritized = list(subject_entry.prioritized_fixes)
        for delta in sorted(
            deltas,
            key=lambda item: (
                item.delta_vs_best_competitor
                if item.delta_vs_best_competitor is not None
                else 0,
                item.dimension,
            ),
        ):
            if (
                delta.delta_vs_best_competitor is not None
                and delta.delta_vs_best_competitor < 0
            ):
                fix = (
                    f"Improve subject {delta.dimension}: currently "
                    f"{delta.subject_score} vs best competitor "
                    f"{delta.best_competitor_score} under ai-readiness-heuristic.v1."
                )
                if fix not in prioritized:
                    prioritized.append(fix)

        warnings = [
            (
                "Readiness comparison scores are deterministic heuristics under "
                "ai-readiness-heuristic.v1; they are not traffic, demand, ranking, "
                "or market-perception measurements."
            ),
            _no_demand_warning(),
        ]
        warnings.extend(_digest_warnings(pages))
        for entry in [subject_entry, *competitor_entries]:
            warnings.extend(entry.warnings)
        return ReadinessComparisonArtifact(
            subject=subject_entry,
            competitors=competitor_entries,
            dimensionDeltas=deltas,
            prioritizedFixes=_unique(prioritized),
            warnings=_unique(warnings),
            provenance=_provenance(pages, readiness_evidence),
        )

    def competitor_audit(
        self, request: CompetitorAuditRequest
    ) -> CompetitorAuditArtifact:
        page_analyses = [
            self.competitor_page_analysis(
                CompetitorPageAnalysisRequest(
                    contractVersion="competitorPageAnalysisInput.v1",
                    page=page,
                )
            )
            for page in request.competitor_pages
        ]
        gap_artifact = self.content_gap(
            ContentGapRequest(
                contractVersion="contentGapInput.v1",
                subjectPages=request.subject_pages,
                competitorPages=request.competitor_pages,
            )
        )
        actions = _prioritized_actions(page_analyses, gap_artifact)
        all_pages = [*request.subject_pages, *request.competitor_pages]
        evidence = _dedupe_evidence(
            [
                *gap_artifact.provenance.evidence,
                *[
                    item
                    for analysis in page_analyses
                    for item in analysis.provenance.evidence
                ],
            ]
        )
        warnings = [
            (
                "Competitor audit findings are heuristic comparisons of supplied "
                "documents only; they do not measure traffic, demand, rankings, "
                "citations won, or market perception."
            ),
            *gap_artifact.warnings,
        ]
        for analysis in page_analyses:
            warnings.extend(analysis.warnings)
        return CompetitorAuditArtifact(
            pageAnalyses=page_analyses,
            contentGap=gap_artifact,
            prioritizedActions=actions,
            warnings=_unique(warnings),
            provenance=_provenance(all_pages, evidence),
        )

    def competitor_positioning(
        self, request: CompetitorPositioningRequest
    ) -> CompetitorPositioningArtifact:
        brand_analyses = [_analyze_page(page) for page in request.brand_pages]
        competitor_analyses = [
            _analyze_page(page) for page in request.competitor_pages
        ]
        all_pages: list[PageSnapshot] = [
            *request.brand_pages,
            *request.competitor_pages,
        ]
        supplied: list[EvidenceReference] = []
        invalid: set[str] = set()
        for page in all_pages:
            valid, rejected = _verified_supplied_evidence(page)
            supplied.extend(valid)
            invalid.update(rejected)
        generated = [
            evidence
            for _, evidence_rows in [*brand_analyses, *competitor_analyses]
            for evidence in evidence_rows
        ]
        evidence = _dedupe_evidence([*supplied, *generated])

        attribute_map: list[PositioningAttribute] = []
        brand_attrs: dict[str, list[str]] = {}
        for page, (findings, _) in zip(
            request.brand_pages, brand_analyses, strict=True
        ):
            for finding in findings:
                if finding.present is not True:
                    continue
                for signal, evidence_id in zip(
                    finding.signals, finding.evidence_ids, strict=True
                ):
                    normalized = _normalize_signal(signal)
                    brand_attrs.setdefault(normalized, []).append(evidence_id)
                    attribute_map.append(
                        PositioningAttribute(
                            attributeId=_stable_id(
                                "attr", "brand", finding.dimension.value, normalized
                            ),
                            attribute=signal,
                            party="brand",
                            signals=[signal],
                            evidenceIds=[evidence_id],
                            sourceIds=[page.source.source_id],
                        )
                    )

        competitor_attrs: dict[str, tuple[str, list[str], list[str], list[str]]] = {}
        for page, (findings, _) in zip(
            request.competitor_pages, competitor_analyses, strict=True
        ):
            for finding in findings:
                if finding.present is not True:
                    continue
                for signal, evidence_id in zip(
                    finding.signals, finding.evidence_ids, strict=True
                ):
                    normalized = _normalize_signal(signal)
                    label, competitor_ids, source_ids, evidence_ids = (
                        competitor_attrs.setdefault(
                            normalized, (signal, [], [], [])
                        )
                    )
                    competitor_ids.append(page.competitor_id)
                    source_ids.append(page.source.source_id)
                    evidence_ids.append(evidence_id)
                    competitor_attrs[normalized] = (
                        label,
                        competitor_ids,
                        source_ids,
                        evidence_ids,
                    )
                    attribute_map.append(
                        PositioningAttribute(
                            attributeId=_stable_id(
                                "attr",
                                "competitor",
                                page.competitor_id,
                                finding.dimension.value,
                                normalized,
                            ),
                            attribute=signal,
                            party="competitor",
                            competitorId=page.competitor_id,
                            competitorName=page.competitor_name,
                            signals=[signal],
                            evidenceIds=[evidence_id],
                            sourceIds=[page.source.source_id],
                        )
                    )

        brand_partial = any(
            page.content_completeness == ContentCompleteness.PARTIAL
            for page in request.brand_pages
        )
        perception_gaps: list[PositioningGap] = []
        for normalized in sorted(competitor_attrs):
            if normalized in brand_attrs:
                continue
            label, competitor_ids, _source_ids, evidence_ids = competitor_attrs[
                normalized
            ]
            linked = [
                evidence_id
                for evidence_id in _unique(evidence_ids)
                if evidence_id in {item.evidence_id for item in evidence}
            ]
            if not linked:
                continue
            perception_gaps.append(
                PositioningGap(
                    gapId=_stable_id("pos-gap", normalized, *_unique(competitor_ids)),
                    attribute=label,
                    status="coverageUnknown" if brand_partial else "supportedGap",
                    summary=(
                        f'Competitor pages explicitly signal "{label}" while brand '
                        "pages do not in the supplied content."
                        if not brand_partial
                        else (
                            f'Competitor pages signal "{label}"; brand coverage is '
                            "unknown because at least one brand page is partial."
                        )
                    ),
                    evidenceIds=linked,
                    competitorIds=_unique(competitor_ids),
                )
            )

        for observation in request.ai_answer_observations:
            mentioned = _unique(observation.competitor_ids_mentioned)
            if observation.subject_mentioned is True and not mentioned:
                continue
            if mentioned and observation.subject_mentioned is not True:
                perception_gaps.append(
                    PositioningGap(
                        gapId=_stable_id(
                            "obs-gap",
                            observation.observation_id,
                            *mentioned,
                        ),
                        attribute=f"AI-answer mention for query: {observation.query}",
                        status="observationOnly",
                        summary=(
                            "Supplied AI-answer observation mentions competitor(s) "
                            f"{', '.join(mentioned)} without confirmed subject mention. "
                            "This is an observation record, not measured market perception."
                        ),
                        evidenceIds=[],
                        observationIds=[observation.observation_id],
                        competitorIds=mentioned,
                    )
                )

        messaging: list[MessagingHypothesis] = []
        for gap in perception_gaps:
            if gap.status == "observationOnly" and not gap.evidence_ids:
                related = gap.attribute
                message = (
                    f"Test explicit differentiation against observed competitor mention "
                    f'for "{related}".'
                )
                query = f"why choose brand instead of {gap.competitor_ids[0]}"
            else:
                related = gap.attribute
                message = (
                    f'Consider stating a supportable brand claim covering "{related}" '
                    "when evidence exists in governed sources."
                )
                query = f"{related} for brand"
            messaging.append(
                MessagingHypothesis(
                    hypothesisId=_stable_id("msg", gap.gap_id),
                    message=message,
                    targetQuery=query,
                    relatedAttribute=related,
                    evidenceIds=gap.evidence_ids,
                )
            )

        warnings = [
            (
                "Positioning output mixes supplied page evidence with optional AI-answer "
                "observations. Generated messaging and queries are labeled "
                "generatedHypothesis and must never be treated as measured market "
                "perception, share of voice, or demand."
            ),
            _no_demand_warning(),
        ]
        warnings.extend(_digest_warnings(all_pages))
        if brand_partial:
            warnings.append(
                "At least one brand page is partial; missing brand attributes are "
                "coverageUnknown rather than asserted absences."
            )
        if request.ai_answer_observations:
            warnings.append(
                "AI-answer observations preserve model/engine, query, raw response, and "
                "observation date exactly as supplied; the service does not validate "
                "or fetch those answers."
            )
        else:
            warnings.append(
                "No AI-answer observations were supplied; narrative gaps are derived "
                "only from page content heuristics."
            )
        if invalid:
            warnings.append(
                "Ignored supplied evidence references that failed exact source quote/span "
                f"validation: {', '.join(sorted(invalid))}."
            )
        return CompetitorPositioningArtifact(
            attributeMap=attribute_map,
            perceptionGaps=perception_gaps,
            messagingHypotheses=messaging,
            observations=list(request.ai_answer_observations),
            warnings=_unique(warnings),
            provenance=_provenance(all_pages, evidence),
        )

    def _page_readiness_entry(
        self,
        page: PageSnapshot,
        *,
        role: Literal["subject", "competitor"],
        competitor_id: str | None = None,
        competitor_name: str | None = None,
    ) -> tuple[PageReadinessEntry, list[EvidenceReference]]:
        artifact = self._diagnostics.readiness_score(
            ReadinessScoreRequest(document=_to_diagnostic_document(page))
        )
        entry = PageReadinessEntry(
            role=role,
            sourceId=page.source.source_id,
            competitorId=competitor_id,
            competitorName=competitor_name,
            overallScore=artifact.overall_score,
            dimensions=artifact.dimensions,
            prioritizedFixes=artifact.prioritized_fixes,
            warnings=artifact.warnings,
            evidenceIds=artifact.provenance.evidence_ids,
        )
        return entry, list(artifact.provenance.evidence)


def _to_diagnostic_document(page: PageSnapshot) -> DiagnosticDocument:
    completeness = (
        page.content_completeness.value
        if isinstance(page.content_completeness, ContentCompleteness)
        else page.content_completeness
    )
    return DiagnosticDocument(
        source=page.source,
        visibleContent=page.visible_content,
        mediaType=page.media_type,
        contentCompleteness=completeness,
        evidence=page.evidence,
        existingJsonLd=page.existing_json_ld,
        technical=page.technical,
    )


def _readiness_delta_summary(
    dimension: str,
    subject_score: float | None,
    best_competitor: float | None,
    delta: float | None,
) -> str:
    if subject_score is None and best_competitor is None:
        return (
            f"Subject and competitor {dimension} scores are unknown from supplied input."
        )
    if subject_score is None:
        return (
            f"Subject {dimension} score is unknown; best supplied competitor score is "
            f"{best_competitor}."
        )
    if best_competitor is None:
        return (
            f"Subject {dimension} score is {subject_score}; no competitor score was "
            "available for comparison."
        )
    relation = (
        "ahead of"
        if delta is not None and delta > 0
        else "behind"
        if delta is not None and delta < 0
        else "tied with"
    )
    return (
        f"Subject {dimension} score {subject_score} is {relation} best competitor "
        f"score {best_competitor} (delta {delta})."
    )


def _prioritized_actions(
    page_analyses: list[CompetitorPageAnalysisArtifact],
    gap_artifact: ContentGapArtifact,
) -> list[PrioritizedAction]:
    actions: list[PrioritizedAction] = []
    for gap in gap_artifact.gaps:
        if gap.status == "supportedGap":
            priority = (
                PriorityTier.HIGH
                if gap.confidence == "high"
                else PriorityTier.MEDIUM
            )
            origin: Literal["supportedGap", "coverageUnknown", "pageOpportunity"] = (
                "supportedGap"
            )
        else:
            priority = PriorityTier.LOW
            origin = "coverageUnknown"
        actions.append(
            PrioritizedAction(
                actionId=_stable_id("action", gap.gap_id),
                priority=priority,
                dimension=gap.dimension,
                action=gap.opportunity,
                rationale=(
                    f"Gap status={gap.status}, confidence={gap.confidence}, "
                    f"competitorSourceCount={len(gap.competitor_source_ids)}."
                ),
                evidenceIds=gap.evidence_ids,
                gapId=gap.gap_id,
                origin=origin,
            )
        )
    for analysis in page_analyses:
        for opportunity in analysis.opportunities:
            dimension = next(
                (
                    item.dimension
                    for item in analysis.dimensions
                    if _dimension_opportunity(item.dimension) == opportunity
                ),
                None,
            )
            evidence_ids = next(
                (
                    item.evidence_ids
                    for item in analysis.dimensions
                    if item.dimension == dimension
                ),
                [],
            )
            actions.append(
                PrioritizedAction(
                    actionId=_stable_id(
                        "action",
                        "page",
                        analysis.competitor_id,
                        opportunity,
                    ),
                    priority=PriorityTier.MEDIUM,
                    dimension=dimension,
                    action=(
                        f"Respond to competitor opportunity on "
                        f"{analysis.competitor_id}: {opportunity}"
                    ),
                    rationale=(
                        "Derived from competitor-page absence opportunity in complete "
                        "supplied competitor content."
                    ),
                    evidenceIds=evidence_ids,
                    origin="pageOpportunity",
                )
            )
    order = {PriorityTier.HIGH: 0, PriorityTier.MEDIUM: 1, PriorityTier.LOW: 2}
    actions.sort(key=lambda item: (order[item.priority], item.action_id))
    return actions


def _planned_query(provenance: QueryProvenance) -> PlannedQuery:
    query = " ".join(provenance.query.split())
    intent, stage, rationale = _classify_query(query)
    tokens = _TOKEN.findall(query.casefold())
    provenance_score = {
        QueryOrigin.OBSERVED: 40,
        QueryOrigin.IMPORTED: 25,
        QueryOrigin.GENERATED_HYPOTHESIS: 5,
    }[provenance.origin]
    intent_score = {
        SearchIntent.INFORMATIONAL: 10,
        SearchIntent.COMMERCIAL: 20,
        SearchIntent.TRANSACTIONAL: 25,
        SearchIntent.NAVIGATIONAL: 15,
    }[intent]
    specificity_score = 20 if len(tokens) >= 5 else 10 if len(tokens) >= 3 else 5
    stage_score = {
        JourneyStage.AWARENESS: 5,
        JourneyStage.CONSIDERATION: 10,
        JourneyStage.DECISION: 15,
        JourneyStage.RETENTION: 10,
    }[stage]
    score = provenance_score + intent_score + specificity_score + stage_score
    tier = (
        PriorityTier.HIGH
        if score >= 70
        else PriorityTier.MEDIUM
        if score >= 45
        else PriorityTier.LOW
    )
    cluster_tokens = [token for token in tokens if token not in _CLUSTER_STOP]
    label = " ".join(cluster_tokens[:3]) or "general"
    normalized_provenance = provenance.model_copy(update={"query": query})
    return PlannedQuery(
        queryId=_stable_id("query", provenance.origin.value, query.casefold()),
        query=query,
        provenance=normalized_provenance,
        classification=QueryClassification(
            intent=intent, stage=stage, rationale=rationale
        ),
        clusterId=_stable_id("cluster", label),
        clusterLabel=label,
        priorityScore=score,
        priorityTier=tier,
        prioritySignals=QueryPrioritySignals(
            provenanceScore=provenance_score,
            intentScore=intent_score,
            specificityScore=specificity_score,
            stageScore=stage_score,
        ),
    )


def _classify_query(query: str) -> tuple[SearchIntent, JourneyStage, str]:
    value = query.casefold()
    if re.search(r"\b(?:login|sign in|support|documentation|docs)\b", value):
        return (
            SearchIntent.NAVIGATIONAL,
            JourneyStage.RETENTION,
            "Navigational/support marker matched.",
        )
    if re.search(r"\b(?:buy|price|pricing|trial|demo|coupon|subscribe)\b", value):
        return (
            SearchIntent.TRANSACTIONAL,
            JourneyStage.DECISION,
            "Purchase or conversion marker matched.",
        )
    if re.search(r"\b(?:best|versus|vs|alternative|compare|review)\b", value):
        return (
            SearchIntent.COMMERCIAL,
            JourneyStage.CONSIDERATION,
            "Comparison or evaluation marker matched.",
        )
    return (
        SearchIntent.INFORMATIONAL,
        JourneyStage.AWARENESS,
        "No navigation, conversion, or comparison marker matched.",
    )


def _analyze_page(
    page: PageSnapshot,
) -> tuple[list[DimensionFinding], list[EvidenceReference]]:
    text = _visible_text(page)
    findings: list[DimensionFinding] = []
    evidence: list[EvidenceReference] = []
    for dimension in _DIMENSIONS:
        matches = _dimension_matches(text, dimension)
        dimension_evidence = [
            _match_evidence(page, text, start, end, quote)
            for start, end, quote in matches
        ]
        evidence.extend(dimension_evidence)
        present: bool | None = (
            True
            if dimension_evidence
            else None
            if page.content_completeness == ContentCompleteness.PARTIAL
            else False
        )
        findings.append(
            DimensionFinding(
                dimension=dimension,
                present=present,
                summary=_finding_summary(dimension, present, len(dimension_evidence)),
                signals=_unique(quote for _, _, quote in matches),
                evidenceIds=[item.evidence_id for item in dimension_evidence],
            )
        )
    return findings, _dedupe_evidence(evidence)


def _dimension_matches(
    text: str, dimension: IntelligenceDimension
) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    for pattern in _DIMENSION_PATTERNS[dimension]:
        for match in pattern.finditer(text):
            group = match.group(1) if match.lastindex else match.group(0)
            start = match.start(1) if match.lastindex else match.start()
            end = match.end(1) if match.lastindex else match.end()
            quote = group.strip()
            trim = len(group) - len(group.lstrip())
            start += trim
            end = start + len(quote)
            if quote:
                rows.append((start, end, quote))
    return rows[:20]


def _visible_text(page: PageSnapshot) -> str:
    if page.media_type != "text/html":
        return page.visible_content.replace("\r\n", "\n").strip()
    soup = BeautifulSoup(page.visible_content, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    lines: list[str] = []
    for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        value = node.get_text(" ", strip=True)
        if not value:
            continue
        if node.name and node.name.startswith("h"):
            lines.append(f"{'#' * int(node.name[1])} {value}")
        elif node.name == "li":
            lines.append(f"- {value}")
        else:
            lines.append(value)
    return "\n".join(lines)


def _match_evidence(
    page: PageSnapshot, text: str, start: int, end: int, quote: str
) -> EvidenceReference:
    if len(quote) < 12:
        start = max(0, start - 6)
        end = min(len(text), max(end + 6, start + 12))
        quote = text[start:end]
    return EvidenceReference(
        evidenceId=_stable_id(page.source.source_id, str(start), str(end), quote),
        sourceId=page.source.source_id,
        quote=quote,
        startChar=start,
        endChar=end,
    )


def _verified_supplied_evidence(
    page: PageSnapshot,
) -> tuple[list[EvidenceReference], set[str]]:
    text = _visible_text(page)
    valid: list[EvidenceReference] = []
    invalid: set[str] = set()
    for item in page.evidence:
        quote_start = text.find(item.quote)
        span_valid = (
            item.start_char is None
            or (
                item.end_char is not None
                and text[item.start_char : item.end_char] == item.quote
            )
        )
        if quote_start >= 0 and span_valid:
            valid.append(item)
        else:
            invalid.add(item.evidence_id)
    return valid, invalid


def _provenance(
    pages: list[PageSnapshot], evidence: list[EvidenceReference]
) -> IntelligenceProvenance:
    return IntelligenceProvenance(
        sources=[page.source for page in pages],
        evidenceIds=[item.evidence_id for item in evidence],
        evidence=evidence,
    )


def _finding_summary(
    dimension: IntelligenceDimension, present: bool | None, count: int
) -> str:
    if present is True:
        return f"Found {count} supplied-content signal(s) for {dimension.value}."
    if present is None:
        return (
            f"No {dimension.value} signal occurred in the partial supplied content; "
            "full-page coverage is unknown."
        )
    return f"No {dimension.value} signal occurred in the complete supplied content."


def _comparison_summary(
    dimension: IntelligenceDimension,
    subject: bool | None,
    competitor_count: int,
    competitor_total: int,
) -> str:
    subject_label = (
        "present" if subject is True else "absent" if subject is False else "unknown"
    )
    return (
        f"Subject {dimension.value} coverage is {subject_label}; "
        f"{competitor_count} of {competitor_total} supplied competitor page(s) "
        "contain matching signals."
    )


def _dimension_opportunity(dimension: IntelligenceDimension) -> str:
    opportunities = {
        IntelligenceDimension.TOPICS: "Add clearly headed coverage of relevant competitor-covered topics.",
        IntelligenceDimension.QUESTIONS: "Answer relevant user questions explicitly in visible content.",
        IntelligenceDimension.CAPABILITIES: "Explain concrete capabilities with supportable details.",
        IntelligenceDimension.PROOF: "Add verifiable proof such as sourced results or case studies.",
        IntelligenceDimension.PRICING: "Clarify pricing or plan information when commercially appropriate.",
        IntelligenceDimension.PROCESS: "Explain the process or workflow in explicit steps.",
        IntelligenceDimension.CALLS_TO_ACTION: "Provide a clear next action aligned with the page purpose.",
    }
    return opportunities[dimension]


def _no_demand_warning() -> str:
    return (
        "Analysis uses supplied documents only. It does not infer traffic, search "
        "volume, audience demand, rankings, or competitor performance."
    )


def _digest_warnings(pages: list[PageSnapshot]) -> list[str]:
    mismatches = [
        page.source.source_id
        for page in pages
        if page.source.source_digest
        and page.source.source_digest
        != hashlib.sha256(_visible_text(page).encode()).hexdigest()
    ]
    if not mismatches:
        return []
    return [
        (
            "Declared sourceDigest does not match normalized visible content for: "
            f"{', '.join(mismatches)}. Content was analyzed without replacing provenance."
        )
    ]


def _normalize_query(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalize_signal(value: str) -> str:
    return " ".join(_TOKEN.findall(value.casefold()))


def _unique_clean(values: Iterable[str]) -> list[str]:
    return list(
        dict.fromkeys(" ".join(value.split()) for value in values if value.strip())
    )


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _dedupe_evidence(values: list[EvidenceReference]) -> list[EvidenceReference]:
    return list({item.evidence_id: item for item in values}.values())
