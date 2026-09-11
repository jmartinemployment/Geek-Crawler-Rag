"""Deterministic generated-content services over caller-supplied inputs."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from geek_crawler_rag.content_models import (
    CitableClaim,
    CitableClaimsRequest,
    ClaimContradictionPair,
    ClaimLedgerArtifact,
    ComparisonBriefArtifact,
    ComparisonBriefRequest,
    ComparisonCriterionRow,
    CompetitiveResponseArtifact,
    CompetitiveResponseRequest,
    ContentProvenance,
    FaqCitation,
    FaqGeneratorRequest,
    FaqPair,
    FaqSetArtifact,
    GeneratedHypothesisText,
    PillarArticleArtifact,
    PillarArticleRequest,
    PillarArticleSection,
    PillarOutlineArtifact,
    PillarOutlineRequest,
    PillarOutlineSection,
    RecommendedVerdict,
    RequiredProofPoint,
    ResponseOutlineSection,
    SupportingContentPlanItem,
)
from geek_crawler_rag.diagnostic_models import (
    EvidenceReference,
    QueryOrigin,
    QueryProvenance,
)
from geek_crawler_rag.diagnostics import (
    _SPECIFIC_FACT,
    _faq_pairs,
    _quote_supports_sentence,
    _sentences_with_offsets,
    _visible_text,
)
from geek_crawler_rag.intelligence import (
    _DIMENSIONS,
    _analyze_page,
    _no_demand_warning,
)
from geek_crawler_rag.intelligence import (
    _dedupe_evidence as _dedupe_page_evidence,
)
from geek_crawler_rag.intelligence_models import (
    ContentCompleteness,
    IntelligenceDimension,
    IntelligenceProvenance,
    PageSnapshot,
)

_TOKEN = re.compile(r"[a-z0-9]+")
_HEADING = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_ATTRIBUTION = re.compile(r"(?i)\b(?:according to|per|cited by|reported by)\s+([^,.]+)")
_EXPERT = re.compile(r"(?i)\b(?:expert|researcher|analyst|professor|study|research)\b")
_NUMBER = re.compile(r"\b\d[\d,.]*%?")


class ContentService:
    """Pure application service; methods perform no network or storage operations."""

    def faq_set(self, request: FaqGeneratorRequest) -> FaqSetArtifact:
        document = request.source_document
        text = _visible_text(document) if document else ""
        warnings = [
            (
                "FAQ answers are generated from supplied content and queries only; "
                "they are not verified against live search or citation outcomes."
            )
        ]
        if document and document.content_completeness == "partial":
            warnings.append(
                "Source document is partial; answers may omit context outside the "
                "supplied fragment."
            )

        planned_queries = list(request.queries)
        seen = {_normalize_query(item.query) for item in planned_queries}
        generated = 0
        for topic in _unique_clean([request.topic, *request.hypothesis_topics]):
            for template in ("What is {topic}?", "How does {topic} work?"):
                query = template.format(topic=topic)
                normalized = _normalize_query(query)
                if normalized in seen:
                    continue
                planned_queries.append(
                    QueryProvenance(
                        query=query,
                        origin=QueryOrigin.GENERATED_HYPOTHESIS,
                        sourceReference="deterministic-template:faq-hypotheses.v1",
                    )
                )
                seen.add(normalized)
                generated += 1
                if len(planned_queries) >= request.max_pairs:
                    break
            if len(planned_queries) >= request.max_pairs:
                break
        if generated:
            warnings.append(
                f"Generated {generated} FAQ question hypothesis(es). They are labeled "
                "generatedHypothesis and are not observed demand."
            )

        pairs: list[FaqPair] = []
        evidence: list[EvidenceReference] = []
        if text:
            for question, answer in _faq_pairs(text):
                if len(pairs) >= request.max_pairs:
                    break
                quote_start = text.find(answer)
                if quote_start < 0:
                    continue
                evidence_ref = _evidence(
                    document.source.source_id,
                    text,
                    quote_start,
                    quote_start + len(answer),
                    answer,
                )
                evidence.append(evidence_ref)
                pairs.append(
                    _pair(
                        question,
                        answer,
                        QueryProvenance(
                            query=question,
                            origin=QueryOrigin.IMPORTED,
                            sourceReference="supplied-document-faq",
                            sourceId=document.source.source_id,
                        ),
                        evidence_ref,
                        document.source.url,
                        "supported",
                    )
                )

        for query in planned_queries:
            if len(pairs) >= request.max_pairs:
                break
            if any(
                _normalize_query(query.query) == _normalize_query(item.question)
                for item in pairs
            ):
                continue
            answer, evidence_ref, status = _answer_for_query(
                query.query, document, text
            )
            if evidence_ref:
                evidence.append(evidence_ref)
            pairs.append(
                _pair(
                    query.query,
                    answer,
                    query,
                    evidence_ref,
                    document.source.url if document else None,
                    status,
                )
            )

        evidence = _dedupe_evidence(evidence)
        source = document.source if document else None
        return FaqSetArtifact(
            topic=request.topic.strip(),
            pairs=pairs[: request.max_pairs],
            warnings=warnings,
            provenance=ContentProvenance(
                source=source,
                queries=planned_queries,
                evidenceIds=[item.evidence_id for item in evidence],
                evidence=evidence,
            ),
        )

    def citable_claims(self, request: CitableClaimsRequest) -> ClaimLedgerArtifact:
        document = request.source_document
        text = _visible_text(document)
        partial = document.content_completeness == "partial"
        warnings = [
            (
                "Claims are extracted or rewritten from supplied visible content only; "
                "no statistics were invented to increase claim density."
            )
        ]
        if partial:
            warnings.append(
                "Source document is partial; confidence is unknown and verification "
                "may be unverifiable outside the supplied fragment."
            )

        claims: list[CitableClaim] = []
        evidence: list[EvidenceReference] = []
        seen_texts: set[str] = set()

        for sentence, start, end in _sentences_with_offsets(text):
            if len(claims) >= request.max_claims:
                break
            if not _SPECIFIC_FACT.search(sentence):
                continue
            claim_type = _claim_type_for_sentence(sentence)
            evidence_ref = _evidence(
                document.source.source_id, text, start, end, sentence
            )
            evidence.append(evidence_ref)
            confidence: Literal["high", "medium", "low", "unknown"] = (
                "unknown" if partial else "high"
            )
            status: Literal["supported", "unsupported", "unverifiable"] = (
                "unverifiable" if partial else "supported"
            )
            claim = CitableClaim(
                claimId=_stable_id("claim", sentence.casefold()),
                claimText=sentence.strip(),
                claimType=claim_type,
                attribution=_attribution_for(sentence),
                evidenceIds=[evidence_ref.evidence_id],
                verificationStatus=status,
                confidence=confidence,
                contradictionState="unknown" if partial else "none",
                insertionLocation=request.insertion_target,
                sourceStatement=None,
            )
            key = _normalize_query(claim.claim_text)
            if key in seen_texts:
                continue
            seen_texts.add(key)
            claims.append(claim)

        for statement in request.target_statements:
            if len(claims) >= request.max_claims:
                break
            cleaned = " ".join(statement.split()).strip()
            if not cleaned:
                continue
            support = _best_supporting_sentence(cleaned, text)
            if support is None:
                claims.append(
                    CitableClaim(
                        claimId=_stable_id("claim", "unsupported", cleaned.casefold()),
                        claimText=cleaned,
                        claimType="rewrittenSpecific",
                        attribution=None,
                        evidenceIds=[],
                        verificationStatus="unsupported",
                        confidence="unknown" if partial else "low",
                        contradictionState="unknown",
                        insertionLocation=request.insertion_target,
                        sourceStatement=cleaned,
                    )
                )
                warnings.append(
                    "A target statement lacked exact supporting quotes; it was marked "
                    "unsupported without inventing replacement statistics."
                )
                continue
            sentence, start, end = support
            if not _SPECIFIC_FACT.search(sentence):
                claims.append(
                    CitableClaim(
                        claimId=_stable_id("claim", "unsupported", cleaned.casefold()),
                        claimText=cleaned,
                        claimType="rewrittenSpecific",
                        attribution=None,
                        evidenceIds=[],
                        verificationStatus="unsupported",
                        confidence="unknown" if partial else "low",
                        contradictionState="unknown",
                        insertionLocation=request.insertion_target,
                        sourceStatement=cleaned,
                    )
                )
                warnings.append(
                    "A target statement matched only non-specific prose; no invented "
                    "numbers were added."
                )
                continue
            rewritten = sentence.strip()
            evidence_ref = _evidence(
                document.source.source_id, text, start, end, rewritten
            )
            evidence.append(evidence_ref)
            key = _normalize_query(rewritten)
            if key in seen_texts:
                continue
            seen_texts.add(key)
            claims.append(
                CitableClaim(
                    claimId=_stable_id("claim", "rewrite", cleaned.casefold()),
                    claimText=rewritten,
                    claimType="rewrittenSpecific",
                    attribution=_attribution_for(rewritten),
                    evidenceIds=[evidence_ref.evidence_id],
                    verificationStatus="unverifiable" if partial else "supported",
                    confidence="unknown" if partial else "medium",
                    contradictionState="unknown" if partial else "none",
                    insertionLocation=request.insertion_target,
                    sourceStatement=cleaned,
                )
            )

        evidence = _dedupe_evidence(evidence)
        claims, contradiction_pairs, contradiction_warnings = (
            _annotate_claim_contradictions(claims[: request.max_claims])
        )
        warnings.extend(contradiction_warnings)
        return ClaimLedgerArtifact(
            claims=claims,
            contradictionPairs=contradiction_pairs,
            warnings=_unique(warnings),
            provenance=ContentProvenance(
                source=document.source,
                queries=[],
                evidenceIds=[item.evidence_id for item in evidence],
                evidence=evidence,
            ),
        )

    def comparison_brief(
        self, request: ComparisonBriefRequest
    ) -> ComparisonBriefArtifact:
        subject_analyses = [_analyze_page(page) for page in request.subject_pages]
        competitor_analyses = [_analyze_page(page) for page in request.competitor_pages]
        all_pages: list[PageSnapshot] = [
            *request.subject_pages,
            *request.competitor_pages,
        ]
        evidence = _dedupe_page_evidence(
            [
                item
                for _, rows in [*subject_analyses, *competitor_analyses]
                for item in rows
            ]
        )
        subject_partial = any(
            page.content_completeness == ContentCompleteness.PARTIAL
            for page in request.subject_pages
        )
        competitor_partial = any(
            page.content_completeness == ContentCompleteness.PARTIAL
            for page in request.competitor_pages
        )

        criteria_labels = _unique_clean(request.decision_criteria) or [
            dimension.value for dimension in _DIMENSIONS
        ]
        criteria_rows: list[ComparisonCriterionRow] = []
        extra_evidence: list[EvidenceReference] = []
        for label in criteria_labels:
            dimension = _dimension_for_criterion(label)
            if dimension is None:
                subject_signals, subject_ids, subject_ev = _keyword_signals(
                    request.subject_pages, label
                )
                competitor_signals, competitor_ids, competitor_ev = _keyword_signals(
                    request.competitor_pages, label
                )
                extra_evidence.extend([*subject_ev, *competitor_ev])
                evidence_ids = _unique([*subject_ids, *competitor_ids])
                summary = (
                    f"Custom criterion '{label}': subject has {len(subject_signals)} "
                    f"signal(s); competitor has {len(competitor_signals)} signal(s)."
                )
                if not subject_signals and subject_partial:
                    summary += " Subject coverage is unknown due to partial input."
                if not competitor_signals and competitor_partial:
                    summary += " Competitor coverage is unknown due to partial input."
            else:
                subject_signals, subject_ids = _dimension_signals(
                    subject_analyses, dimension
                )
                competitor_signals, competitor_ids = _dimension_signals(
                    competitor_analyses, dimension
                )
                evidence_ids = _unique([*subject_ids, *competitor_ids])
                subject_state = _coverage_label(bool(subject_signals), subject_partial)
                competitor_state = _coverage_label(
                    bool(competitor_signals), competitor_partial
                )
                summary = (
                    f"{dimension.value}: subject coverage {subject_state}; "
                    f"competitor coverage {competitor_state}."
                )
            criteria_rows.append(
                ComparisonCriterionRow(
                    criterion=label,
                    subjectSignals=subject_signals,
                    competitorSignals=competitor_signals,
                    evidenceIds=evidence_ids,
                    summary=summary,
                )
            )
        evidence = _dedupe_page_evidence([*evidence, *extra_evidence])

        differentiators: list[GeneratedHypothesisText] = []
        proof_requirements: list[GeneratedHypothesisText] = []
        positioning_angles: list[GeneratedHypothesisText] = []
        for row in criteria_rows:
            if row.competitor_signals and not row.subject_signals:
                if subject_partial:
                    continue
                differentiators.append(
                    GeneratedHypothesisText(
                        text=(
                            f"Subject may need explicit coverage of '{row.criterion}' "
                            "where competitor pages already show signals."
                        )
                    )
                )
                proof_requirements.append(
                    GeneratedHypothesisText(
                        text=(
                            f"Supply verifiable proof for '{row.criterion}' before "
                            "asserting parity or superiority."
                        )
                    )
                )
            elif row.subject_signals and not row.competitor_signals:
                if competitor_partial:
                    continue
                differentiators.append(
                    GeneratedHypothesisText(
                        text=(
                            f"Emphasize subject '{row.criterion}' signals that are "
                            "absent from supplied competitor pages."
                        )
                    )
                )
                positioning_angles.append(
                    GeneratedHypothesisText(
                        text=(
                            f"Position {request.subject_name} around documented "
                            f"'{row.criterion}' depth versus {request.competitor_name}."
                        )
                    )
                )

        subject_present = sum(1 for row in criteria_rows if row.subject_signals)
        competitor_present = sum(1 for row in criteria_rows if row.competitor_signals)
        if subject_partial or competitor_partial:
            framing = (
                f"Insufficient complete coverage to assert a winner between "
                f"{request.subject_name} and {request.competitor_name}; treat gaps "
                "as unknown until full pages are supplied."
            )
        elif subject_present > competitor_present:
            framing = (
                f"Heuristic lean toward {request.subject_name} on supplied criteria "
                f"({subject_present} vs {competitor_present} covered dimensions)."
            )
        elif competitor_present > subject_present:
            framing = (
                f"Heuristic lean toward {request.competitor_name} on supplied criteria "
                f"({competitor_present} vs {subject_present} covered dimensions); "
                f"use as a gap brief for {request.subject_name}, not a market ranking."
            )
        else:
            framing = (
                f"Supplied criteria coverage is roughly balanced between "
                f"{request.subject_name} and {request.competitor_name}."
            )

        warnings = [
            "Comparison brief is a structured signal summary, not a full article draft.",
            _no_demand_warning(),
        ]
        if len(request.competitor_pages) > 1:
            warnings.append(
                f"Compared against {len(request.competitor_pages)} competitor pages."
            )
        if subject_partial or competitor_partial:
            warnings.append(
                "Partial page input(s) present; missing signals are unknown, not "
                "asserted absences."
            )
        if differentiators or positioning_angles or proof_requirements:
            warnings.append(
                "Differentiators, proof requirements, and positioning angles are "
                "labeled generatedHypothesis."
            )

        return ComparisonBriefArtifact(
            subjectName=request.subject_name.strip(),
            competitorName=request.competitor_name.strip(),
            criteria=criteria_rows,
            differentiators=differentiators,
            proofRequirements=proof_requirements,
            positioningAngles=positioning_angles,
            recommendedVerdict=RecommendedVerdict(framing=framing),
            warnings=_unique(warnings),
            provenance=IntelligenceProvenance(
                sources=[page.source for page in all_pages],
                evidenceIds=[item.evidence_id for item in evidence],
                evidence=evidence,
            ),
        )

    def competitive_response(
        self, request: CompetitiveResponseRequest
    ) -> CompetitiveResponseArtifact:
        brand_analyses = [_analyze_page(page) for page in request.brand_pages]
        competitor_analyses = [_analyze_page(page) for page in request.competitor_pages]
        all_pages: list[PageSnapshot] = [
            *request.brand_pages,
            *request.competitor_pages,
        ]
        evidence = _dedupe_page_evidence(
            [
                item
                for _, rows in [*brand_analyses, *competitor_analyses]
                for item in rows
            ]
        )
        brand_partial = any(
            page.content_completeness == ContentCompleteness.PARTIAL
            for page in request.brand_pages
        )

        brand_covered = {
            dimension
            for findings, _ in brand_analyses
            for finding in findings
            if finding.present is True
            for dimension in [finding.dimension]
        }
        competitor_covered = {
            dimension
            for findings, _ in competitor_analyses
            for finding in findings
            if finding.present is True
            for dimension in [finding.dimension]
        }
        gaps = [
            dimension
            for dimension in _DIMENSIONS
            if dimension in competitor_covered and dimension not in brand_covered
        ]
        if brand_partial:
            gaps = [
                dimension
                for dimension in gaps
                if dimension
                not in {
                    finding.dimension
                    for findings, _ in brand_analyses
                    for finding in findings
                    if finding.present is None
                }
            ]

        selected = request.response_mode
        if selected == "auto":
            if request.focus_query and gaps:
                selected = "counterNarrative"
            elif len(gaps) >= 3:
                selected = "newPage"
            elif gaps:
                selected = "targetedUpdate"
            else:
                selected = "counterNarrative"

        rationale = {
            "newPage": (
                "Competitor pages cover multiple dimensions absent from complete brand "
                "inputs; a new page can assemble missing proof and answers."
            ),
            "targetedUpdate": (
                "Brand pages already cover some dimensions; targeted updates can close "
                "specific competitor-covered gaps."
            ),
            "counterNarrative": (
                "Response focuses on reframing differentiation and proof without "
                "mirroring competitor structure."
            ),
        }[selected]
        if request.focus_query:
            rationale += f' Focus query supplied: "{request.focus_query.strip()}".'

        content_angles: list[GeneratedHypothesisText] = []
        required_proof: list[RequiredProofPoint] = []
        outline: list[ResponseOutlineSection] = []

        for dimension in gaps or list(competitor_covered)[:3] or _DIMENSIONS[:3]:
            brand_signals, brand_ids = _dimension_signals(brand_analyses, dimension)
            competitor_signals, competitor_ids = _dimension_signals(
                competitor_analyses, dimension
            )
            content_angles.append(
                GeneratedHypothesisText(
                    text=(
                        f"Develop a brand-owned angle on {dimension.value} that answers "
                        "the buyer question without copying competitor wording."
                    )
                )
            )
            if competitor_signals and not brand_signals:
                required_proof.append(
                    RequiredProofPoint(
                        proofId=_stable_id("proof", dimension.value, "gap"),
                        statement=(
                            f"Add brand-owned evidence for {dimension.value} before "
                            "claiming parity with competitor coverage."
                        ),
                        evidenceIds=competitor_ids[:3],
                        origin="suppliedEvidence"
                        if competitor_ids
                        else "generatedHypothesis",
                    )
                )
            elif brand_signals:
                required_proof.append(
                    RequiredProofPoint(
                        proofId=_stable_id("proof", dimension.value, "brand"),
                        statement=(
                            f"Reuse existing brand {dimension.value} evidence in the "
                            "response outline."
                        ),
                        evidenceIds=brand_ids[:3],
                        origin="suppliedEvidence",
                    )
                )
            else:
                required_proof.append(
                    RequiredProofPoint(
                        proofId=_stable_id("proof", dimension.value, "needed"),
                        statement=(
                            f"Gather verifiable brand proof for {dimension.value}; none "
                            "was present in supplied pages."
                        ),
                        evidenceIds=[],
                        origin="generatedHypothesis",
                    )
                )
            outline.append(
                ResponseOutlineSection(
                    heading=_heading_for_dimension(dimension, selected),
                    objective=(
                        f"Address {dimension.value} with answer-first brand messaging "
                        "and exact evidence spans from governed sources."
                    ),
                    evidenceIds=_unique([*brand_ids, *competitor_ids])[:5],
                )
            )

        if request.focus_query:
            outline.insert(
                0,
                ResponseOutlineSection(
                    heading=f"Answer: {request.focus_query.strip()}",
                    objective=(
                        "Lead with a direct brand answer to the focus query, then "
                        "support with proof points."
                    ),
                    evidenceIds=[],
                ),
            )

        warnings = [
            "Does not copy competitor prose into draft body.",
            "Content angles are generatedHypothesis labels, not measured demand.",
            _no_demand_warning(),
        ]
        if len(request.competitor_pages) > 1:
            warnings.append(
                f"Response plan considers {len(request.competitor_pages)} competitor pages."
            )
        if brand_partial:
            warnings.append(
                "At least one brand page is partial; missing brand signals stay "
                "unknown rather than asserted absences."
            )

        return CompetitiveResponseArtifact(
            selectedMode=selected,
            rationale=rationale,
            contentAngles=content_angles,
            requiredProofPoints=required_proof,
            outlineSections=outline,
            warnings=_unique(warnings),
            provenance=IntelligenceProvenance(
                sources=[page.source for page in all_pages],
                evidenceIds=[item.evidence_id for item in evidence],
                evidence=evidence,
            ),
        )

    def pillar_outline(self, request: PillarOutlineRequest) -> PillarOutlineArtifact:
        document = request.source_document
        text = _visible_text(document) if document else ""
        warnings = [
            "Pillar outline is a topic-cluster scaffold, not a full article draft."
        ]
        if document and document.content_completeness == "partial":
            warnings.append(
                "Source document is partial; section evidence may be incomplete."
            )

        evidence: list[EvidenceReference] = []
        sections: list[PillarOutlineSection] = []
        topic = request.topic.strip()

        sections.append(
            PillarOutlineSection(
                sectionId=_stable_id("pillar", topic, "intro"),
                heading=f"What is {topic}?",
                objective=f"Define {topic} with an answer-first opening.",
                answerFirstPrompt=(
                    f"In one or two sentences, state what {topic} is and why it matters."
                ),
                relatedQueries=[
                    item.query
                    for item in request.queries
                    if "what" in item.query.casefold()
                ],
                evidenceIds=[],
            )
        )

        if text:
            for match in _HEADING.finditer(text):
                heading = match.group(1).strip()
                if not heading or heading.casefold() == topic.casefold():
                    continue
                start = match.end()
                following = text[start:].lstrip("\n")
                paragraph = _first_paragraph(following)
                evidence_ids: list[str] = []
                if paragraph:
                    quote_start = text.find(paragraph, start)
                    if quote_start >= 0:
                        evidence_ref = _evidence(
                            document.source.source_id,
                            text,
                            quote_start,
                            quote_start + len(paragraph),
                            paragraph,
                        )
                        evidence.append(evidence_ref)
                        evidence_ids = [evidence_ref.evidence_id]
                related = [
                    item.query
                    for item in request.queries
                    if any(
                        token in heading.casefold()
                        for token in _TOKEN.findall(item.query.casefold())
                        if len(token) > 3
                    )
                ]
                sections.append(
                    PillarOutlineSection(
                        sectionId=_stable_id("pillar", topic, heading),
                        heading=heading,
                        objective=f"Cover '{heading}' with specific, citable detail.",
                        answerFirstPrompt=(
                            f"Open the '{heading}' section with a direct answer before "
                            "expanding."
                        ),
                        relatedQueries=related,
                        evidenceIds=evidence_ids,
                    )
                )

        for query in request.queries:
            if any(
                _normalize_query(query.query) == _normalize_query(section.heading)
                or query.query in section.related_queries
                for section in sections
            ):
                continue
            sections.append(
                PillarOutlineSection(
                    sectionId=_stable_id("pillar", topic, "query", query.query),
                    heading=query.query.rstrip("?"),
                    objective=f"Answer the sourced query: {query.query}",
                    answerFirstPrompt=(
                        f"Answer '{query.query}' in the first sentence, then support "
                        "with evidence."
                    ),
                    relatedQueries=[query.query],
                    evidenceIds=[],
                )
            )

        for template in (
            f"How {topic} works",
            f"{topic} best practices",
            f"{topic} compared to alternatives",
        ):
            if any(
                _normalize_query(template) == _normalize_query(section.heading)
                for section in sections
            ):
                continue
            sections.append(
                PillarOutlineSection(
                    sectionId=_stable_id("pillar", topic, template),
                    heading=template,
                    objective=f"Extend cluster coverage for {template}.",
                    answerFirstPrompt=(
                        f"Lead with the key takeaway for '{template}', then expand."
                    ),
                    relatedQueries=[],
                    evidenceIds=[],
                )
            )

        plan: list[SupportingContentPlanItem] = [
            SupportingContentPlanItem(
                contentType="faq",
                title=f"{topic} FAQ",
                rationale="Capture sourced and hypothesized questions as answer-first pairs.",
            ),
            SupportingContentPlanItem(
                contentType="comparison",
                title=f"{topic} comparison brief",
                rationale="Support evaluation queries with a structured X vs Y brief.",
            ),
            SupportingContentPlanItem(
                contentType="explainer",
                title=f"How {topic} works",
                rationale="Provide a shorter explainer that links back to the pillar.",
            ),
        ]
        for hint in _unique_clean(request.supporting_content_hints):
            content_type: Literal["faq", "comparison", "explainer"] = "explainer"
            lowered = hint.casefold()
            if "faq" in lowered or "?" in hint:
                content_type = "faq"
            elif "vs" in lowered or "compar" in lowered:
                content_type = "comparison"
            plan.append(
                SupportingContentPlanItem(
                    contentType=content_type,
                    title=hint,
                    rationale="Caller-supplied supporting content hint.",
                )
            )

        warnings.append(
            "Supporting content plan items are labeled generatedHypothesis."
        )
        evidence = _dedupe_evidence(evidence)
        return PillarOutlineArtifact(
            topic=topic,
            sections=sections,
            supportingContentPlan=plan,
            warnings=_unique(warnings),
            provenance=ContentProvenance(
                source=document.source if document else None,
                queries=list(request.queries),
                evidenceIds=[item.evidence_id for item in evidence],
                evidence=evidence,
            ),
        )

    def pillar_article(self, request: PillarArticleRequest) -> PillarArticleArtifact:
        outline = self.pillar_outline(
            PillarOutlineRequest.model_validate(
                {
                    "contractVersion": "pillarOutlineInput.v1",
                    "topic": request.topic,
                    "sourceDocument": (
                        request.source_document.model_dump(mode="python", by_alias=True)
                        if request.source_document is not None
                        else None
                    ),
                    "queries": [
                        item.model_dump(mode="python", by_alias=True)
                        for item in request.queries
                    ],
                    "supportingContentHints": list(request.supporting_content_hints),
                }
            )
        )
        document = request.source_document
        text = _visible_text(document) if document else ""
        evidence_by_id = {
            item.evidence_id: item for item in outline.provenance.evidence
        }
        article_sections: list[PillarArticleSection] = []
        markdown_parts = [f"# {outline.topic.strip()}", ""]
        warnings = [
            warning
            for warning in outline.warnings
            if "not a full article" not in warning.casefold()
        ]
        warnings.append(
            "Pillar article bodies prefer multi-paragraph source spans under matching headings; "
            "ungrounded sections stay scaffolds and must not be treated as verified claims."
        )

        for section in outline.sections:
            body_chunks: list[str] = []
            evidence_ids: list[str] = []
            grounded = False

            if document is not None and text:
                grounded_body, grounded_evidence = _ground_section_from_source(
                    section.heading, document, text
                )
                if grounded_body and grounded_evidence:
                    body_chunks = [grounded_body]
                    evidence_ids = [item.evidence_id for item in grounded_evidence]
                    evidence_by_id.update(
                        {item.evidence_id: item for item in grounded_evidence}
                    )
                    grounded = True

            if not body_chunks:
                for evidence_id in section.evidence_ids:
                    ref = evidence_by_id.get(evidence_id)
                    if ref is None or not ref.quote.strip():
                        continue
                    body_chunks.append(ref.quote.strip())
                    evidence_ids.append(evidence_id)
                    grounded = True

            if not body_chunks and text and document is not None:
                heading_l = section.heading.casefold()
                allow_answer_lookup = bool(section.related_queries) or heading_l.endswith(
                    "?"
                ) or heading_l.startswith("what is")
                if allow_answer_lookup:
                    answer, evidence_ref, status = _answer_for_query(
                        section.heading, document, text
                    )
                    if status != "unverifiable" and evidence_ref is not None:
                        body_chunks.append(answer.strip())
                        evidence_ids = [evidence_ref.evidence_id]
                        evidence_by_id[evidence_ref.evidence_id] = evidence_ref
                        grounded = True

            if not body_chunks:
                body_chunks.append(
                    f"{section.answer_first_prompt} "
                    "[Scaffold — no supplied source span grounded this section.]"
                )
                evidence_ids = []

            body = "\n\n".join(body_chunks)
            article_sections.append(
                PillarArticleSection(
                    sectionId=section.section_id,
                    heading=section.heading,
                    bodyMarkdown=body,
                    evidenceIds=evidence_ids,
                    grounded=grounded,
                )
            )
            markdown_parts.append(f"## {section.heading}")
            markdown_parts.append("")
            markdown_parts.append(body)
            markdown_parts.append("")

        if not any(section.grounded for section in article_sections):
            warnings.append(
                "No section was grounded in supplied source evidence; treat the draft as a scaffold only."
            )

        provenance_evidence = _dedupe_evidence(
            [*outline.provenance.evidence, *evidence_by_id.values()]
        )
        return PillarArticleArtifact(
            topic=outline.topic,
            title=outline.topic,
            markdown="\n".join(markdown_parts).strip() + "\n",
            sections=article_sections,
            supportingContentPlan=outline.supporting_content_plan,
            warnings=_unique(warnings),
            provenance=ContentProvenance(
                source=outline.provenance.source,
                queries=list(outline.provenance.queries),
                evidenceIds=[item.evidence_id for item in provenance_evidence],
                evidence=provenance_evidence,
            ),
        )


def _answer_for_query(
    question: str,
    document,
    text: str,
) -> tuple[str, EvidenceReference | None, str]:
    if not text or document is None:
        return (
            "No supplied source document was available to ground this answer.",
            None,
            "unverifiable",
        )
    tokens = [token for token in _TOKEN.findall(question.casefold()) if len(token) > 2]
    best: tuple[int, str] | None = None
    for match in _HEADING.finditer(text):
        heading = match.group(1).strip()
        score = sum(1 for token in tokens if token in heading.casefold())
        if score <= 0:
            continue
        start = match.end()
        following = text[start:].lstrip("\n")
        paragraph = _first_paragraph(following)
        if paragraph and (best is None or score > best[0]):
            best = (score, paragraph)
    if best is None:
        for sentence in _SENTENCE.split(text):
            value = sentence.strip()
            if not value:
                continue
            score = sum(1 for token in tokens if token in value.casefold())
            if score >= max(1, len(tokens) // 2) and (best is None or score > best[0]):
                best = (score, value)
    if best is None:
        return (
            "The supplied source document did not contain an explicit answer span for this question.",
            None,
            "unsupported",
        )
    answer = best[1].strip()
    quote_start = text.find(answer)
    if quote_start < 0:
        return (answer, None, "unsupported")
    evidence_ref = _evidence(
        document.source.source_id,
        text,
        quote_start,
        quote_start + len(answer),
        answer,
    )
    return (answer, evidence_ref, "supported")


def _ground_section_from_source(
    heading: str,
    document,
    text: str,
    *,
    max_paragraphs: int = 3,
) -> tuple[str | None, list[EvidenceReference]]:
    """Pull up to N consecutive body paragraphs under the best matching source heading."""
    if not text or document is None:
        return None, []
    tokens = [token for token in _TOKEN.findall(heading.casefold()) if len(token) > 2]
    best_match: tuple[int, int] | None = None
    for match in _HEADING.finditer(text):
        source_heading = match.group(1).strip()
        if not source_heading:
            continue
        if _normalize_query(source_heading) == _normalize_query(heading):
            best_match = (100, match.end())
            break
        score = sum(1 for token in tokens if token in source_heading.casefold())
        required = max(2, (len(tokens) + 1) // 2) if tokens else 2
        if score < required:
            continue
        if best_match is None or score > best_match[0]:
            best_match = (score, match.end())

    search_from = 0
    if best_match is not None:
        search_from = best_match[1]
    else:
        lowered = heading.casefold()
        if not (lowered.startswith("what is") or "introduction" in lowered):
            return None, []
        # Intro-style sections can ground from the first body block after the title.
        first_heading = _HEADING.search(text)
        search_from = first_heading.end() if first_heading else 0

    following = text[search_from:].lstrip("\n")
    paragraphs = _paragraphs_under(following, max_paragraphs=max_paragraphs)
    if not paragraphs:
        return None, []

    evidence_rows: list[EvidenceReference] = []
    bodies: list[str] = []
    cursor = search_from
    for paragraph in paragraphs:
        quote_start = text.find(paragraph, cursor)
        if quote_start < 0:
            quote_start = text.find(paragraph)
        if quote_start < 0:
            continue
        evidence_rows.append(
            _evidence(
                document.source.source_id,
                text,
                quote_start,
                quote_start + len(paragraph),
                paragraph,
            )
        )
        bodies.append(paragraph)
        cursor = quote_start + len(paragraph)
    if not bodies:
        return None, []
    return "\n\n".join(bodies), evidence_rows


def _paragraphs_under(following: str, *, max_paragraphs: int = 3) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in following.splitlines():
        value = line.strip()
        if value.startswith("#"):
            break
        if not value:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
                if len(paragraphs) >= max_paragraphs:
                    break
            continue
        if value.endswith("?") and not current:
            break
        current.append(value)
    if current and len(paragraphs) < max_paragraphs:
        paragraphs.append(" ".join(current).strip())
    return [item for item in paragraphs if item]


def _first_paragraph(following: str) -> str:
    paragraphs = _paragraphs_under(following, max_paragraphs=1)
    return paragraphs[0] if paragraphs else ""

def _pair(
    question: str,
    answer: str,
    provenance: QueryProvenance,
    evidence_ref: EvidenceReference | None,
    url: str | None,
    status: str,
) -> FaqPair:
    citations: list[FaqCitation] = []
    if evidence_ref:
        citations.append(
            FaqCitation(
                evidenceId=evidence_ref.evidence_id,
                sourceId=evidence_ref.source_id,
                quote=evidence_ref.quote,
                url=url,
            )
        )
    return FaqPair(
        pairId=_stable_id("faq", question.casefold(), answer[:64]),
        question=question,
        answer=answer,
        queryProvenance=provenance,
        citations=citations,
        verificationStatus=status,
    )


def _evidence(
    source_id: str, text: str, start: int, end: int, quote: str
) -> EvidenceReference:
    if len(quote) < 12:
        start = max(0, start - 6)
        end = min(len(text), max(end + 6, start + 12))
        quote = text[start:end]
    return EvidenceReference(
        evidenceId=_stable_id(source_id, str(start), str(end), quote),
        sourceId=source_id,
        quote=quote,
        startChar=start,
        endChar=end,
    )


def _claim_type_for_sentence(
    sentence: str,
) -> Literal[
    "quantifiableFact", "expertPosition", "attributableStatement", "rewrittenSpecific"
]:
    if _NUMBER.search(sentence) or re.search(
        r"\b(?:19|20)\d{2}\b|%|ms|kg|gb", sentence, re.IGNORECASE
    ):
        return "quantifiableFact"
    if _EXPERT.search(sentence) or _ATTRIBUTION.search(sentence):
        return "expertPosition"
    return "attributableStatement"


def _attribution_for(sentence: str) -> str | None:
    match = _ATTRIBUTION.search(sentence)
    if not match:
        return None
    return match.group(1).strip()


def _best_supporting_sentence(statement: str, text: str) -> tuple[str, int, int] | None:
    tokens = [token for token in _TOKEN.findall(statement.casefold()) if len(token) > 2]
    if not tokens:
        return None
    best: tuple[int, str, int, int] | None = None
    for sentence, start, end in _sentences_with_offsets(text):
        score = sum(1 for token in tokens if token in sentence.casefold())
        if _quote_supports_sentence(sentence, statement):
            score += 3
        if score >= max(1, len(tokens) // 2) and (best is None or score > best[0]):
            best = (score, sentence, start, end)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _dimension_for_criterion(label: str) -> IntelligenceDimension | None:
    normalized = label.casefold().replace(" ", "")
    for dimension in _DIMENSIONS:
        if dimension.value.casefold() == normalized:
            return dimension
        if (
            dimension.value.casefold() in normalized
            or normalized in dimension.value.casefold()
        ):
            return dimension
    aliases = {
        "features": IntelligenceDimension.CAPABILITIES,
        "feature": IntelligenceDimension.CAPABILITIES,
        "price": IntelligenceDimension.PRICING,
        "cta": IntelligenceDimension.CALLS_TO_ACTION,
        "workflow": IntelligenceDimension.PROCESS,
        "socialproof": IntelligenceDimension.PROOF,
    }
    return aliases.get(normalized)


def _dimension_signals(
    analyses: list[tuple[list, list[EvidenceReference]]],
    dimension: IntelligenceDimension,
) -> tuple[list[str], list[str]]:
    signals: list[str] = []
    evidence_ids: list[str] = []
    for findings, _ in analyses:
        for finding in findings:
            if finding.dimension != dimension or finding.present is not True:
                continue
            signals.extend(finding.signals)
            evidence_ids.extend(finding.evidence_ids)
    return _unique(signals), _unique(evidence_ids)


def _keyword_signals(
    pages: list[PageSnapshot], label: str
) -> tuple[list[str], list[str], list[EvidenceReference]]:
    tokens = [token for token in _TOKEN.findall(label.casefold()) if len(token) > 2]
    signals: list[str] = []
    evidence_ids: list[str] = []
    evidence: list[EvidenceReference] = []
    for page in pages:
        text = page.visible_content.replace("\r\n", "\n")
        for sentence, start, end in _sentences_with_offsets(text):
            if tokens and sum(
                1 for token in tokens if token in sentence.casefold()
            ) < max(1, len(tokens) // 2):
                continue
            if not tokens and label.casefold() not in sentence.casefold():
                continue
            evidence_ref = _evidence(page.source.source_id, text, start, end, sentence)
            signals.append(sentence.strip())
            evidence_ids.append(evidence_ref.evidence_id)
            evidence.append(evidence_ref)
    return _unique(signals)[:10], _unique(evidence_ids)[:10], _dedupe_evidence(evidence)


def _coverage_label(present: bool, partial: bool) -> str:
    if present:
        return "present"
    if partial:
        return "unknown"
    return "absent"


def _heading_for_dimension(dimension: IntelligenceDimension, mode: str) -> str:
    labels = {
        IntelligenceDimension.TOPICS: "Topic coverage",
        IntelligenceDimension.QUESTIONS: "Direct answers",
        IntelligenceDimension.CAPABILITIES: "Capabilities",
        IntelligenceDimension.PROOF: "Proof and results",
        IntelligenceDimension.PRICING: "Pricing clarity",
        IntelligenceDimension.PROCESS: "How it works",
        IntelligenceDimension.CALLS_TO_ACTION: "Next step",
    }
    base = labels[dimension]
    if mode == "counterNarrative":
        return f"Brand counterpoint: {base}"
    if mode == "targetedUpdate":
        return f"Update: {base}"
    return f"New section: {base}"


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
        "than",
        "over",
        "under",
        "about",
        "into",
        "their",
        "our",
        "your",
        "this",
        "these",
        "those",
        "have",
        "has",
        "had",
        "been",
        "be",
        "can",
        "may",
        "will",
        "would",
        "should",
        "could",
        "not",
        "no",
        "never",
        "none",
        "without",
        "according",
        "per",
        "cited",
        "reported",
    }
)
_NEGATION = re.compile(
    r"(?i)\b(?:not|never|no|none|without|cannot|can't|isn't|aren't|wasn't|weren't|doesn't|don't|didn't)\b"
)
_QUANTITY = re.compile(r"(?i)(?P<num>\d[\d,.]*%?)\s*(?P<unit>[a-z]{0,12})")


def _claim_content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(text.casefold())
        if len(token) > 2 and token not in _STOPWORDS and not token.isdigit()
    }


def _claim_quantities(text: str) -> list[tuple[str, str]]:
    quantities: list[tuple[str, str]] = []
    for match in _QUANTITY.finditer(text):
        num = match.group("num").replace(",", "")
        unit = match.group("unit").casefold()
        quantities.append((num, unit))
    return quantities


def _claims_share_subject(left: set[str], right: set[str]) -> bool:
    overlap = left & right
    if len(overlap) >= 2:
        return True
    return len(overlap) == 1 and min(len(left), len(right)) <= 3


def _quantity_conflict(
    left: list[tuple[str, str]], right: list[tuple[str, str]]
) -> bool:
    if not left or not right:
        return False
    for left_num, left_unit in left:
        for right_num, right_unit in right:
            if left_num == right_num:
                continue
            if left_unit and right_unit and left_unit != right_unit:
                continue
            return True
    return False


def _negation_conflict(left_text: str, right_text: str) -> bool:
    left_neg = bool(_NEGATION.search(left_text))
    right_neg = bool(_NEGATION.search(right_text))
    return left_neg != right_neg


def _annotate_claim_contradictions(
    claims: list[CitableClaim],
) -> tuple[list[CitableClaim], list[ClaimContradictionPair], list[str]]:
    """Mark pairwise quantity/negation conflicts as possible contradictions.

    Partial/unverifiable claims keep contradictionState=unknown and are never
    upgraded to possible. Compatible supported claims stay none.
    """
    if len(claims) < 2:
        return claims, [], []

    states: list[Literal["none", "possible", "unknown"]] = [
        "unknown"
        if claim.contradiction_state == "unknown"
        or claim.verification_status in {"unsupported", "unverifiable"}
        or claim.confidence == "unknown"
        else claim.contradiction_state
        for claim in claims
    ]
    eligible = [
        index
        for index, claim in enumerate(claims)
        if states[index] != "unknown" and claim.verification_status == "supported"
    ]
    pairs: list[ClaimContradictionPair] = []
    for offset, left_index in enumerate(eligible):
        left = claims[left_index]
        left_tokens = _claim_content_tokens(left.claim_text)
        left_qty = _claim_quantities(left.claim_text)
        for right_index in eligible[offset + 1 :]:
            right = claims[right_index]
            right_tokens = _claim_content_tokens(right.claim_text)
            if not _claims_share_subject(left_tokens, right_tokens):
                continue
            right_qty = _claim_quantities(right.claim_text)
            reason: Literal["conflictingQuantities", "negationConflict"] | None = None
            if _quantity_conflict(left_qty, right_qty):
                reason = "conflictingQuantities"
            elif _negation_conflict(left.claim_text, right.claim_text):
                reason = "negationConflict"
            if reason is None:
                continue
            states[left_index] = "possible"
            states[right_index] = "possible"
            pairs.append(
                ClaimContradictionPair(
                    leftClaimId=left.claim_id,
                    rightClaimId=right.claim_id,
                    reason=reason,
                )
            )

    annotated = [
        claim.model_copy(update={"contradiction_state": states[index]})
        for index, claim in enumerate(claims)
    ]
    reason_labels = {
        "conflictingQuantities": "conflicting quantities",
        "negationConflict": "negation conflict",
    }
    warnings = [
        (
            f"Possible contradiction ({reason_labels[pair.reason]}) between claims "
            f"{pair.left_claim_id} and {pair.right_claim_id}."
        )
        for pair in pairs
    ]
    return annotated, pairs, warnings


def _normalize_query(value: str) -> str:
    return " ".join(value.casefold().split())


def _unique_clean(values: list[str]) -> list[str]:
    return list(
        dict.fromkeys(" ".join(value.split()) for value in values if value.strip())
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


def _dedupe_evidence(values: list[EvidenceReference]) -> list[EvidenceReference]:
    return list({item.evidence_id: item for item in values}.values())
