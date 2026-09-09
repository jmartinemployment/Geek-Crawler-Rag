"""Side-effect-free deterministic diagnostics over caller-supplied visible content."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from bs4 import BeautifulSoup

from geek_crawler_rag.diagnostic_models import (
    ArtifactProvenance,
    CanonicalEntity,
    ClaimAssessment,
    DiagnosticDocument,
    DimensionScore,
    EntityMapArtifact,
    EntityMapRequest,
    EntityRelationship,
    EvidenceReference,
    FactDensityArtifact,
    FactDensityRequest,
    ReadinessScoreArtifact,
    ReadinessScoreRequest,
    SchemaMarkupArtifact,
    SchemaMarkupRequest,
    SchemaValidationFinding,
    SectionFactDensity,
)
from geek_crawler_rag.generate import verify_citations
from geek_crawler_rag.models import GenerateCitation, GenerateSource

_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])(?:\s+|$)|(?<=:)\n")
_SPECIFIC_FACT = re.compile(
    r"(?:\b\d[\d,.]*(?:%|ms|s|kg|gb|mb|years?|days?|hours?|\b)"
    r"|\b(?:19|20)\d{2}\b|\baccording to\b|\bresearch\b|\bstudy\b"
    r"|\b[A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+)+\b)"
)
_ENTITY = re.compile(
    r"\b[A-Z][A-Za-z0-9&.-]+"
    r"(?:[ \t]+(?:[A-Z][A-Za-z0-9&.-]+|of|for|and|the)){0,4}\b"
)
_ENTITY_STOP = frozenset(
    {
        "A",
        "An",
        "And",
        "Article",
        "FAQ",
        "How",
        "In",
        "Product",
        "The",
        "This",
        "What",
        "When",
        "Where",
        "Why",
    }
)


@dataclass(frozen=True)
class _Section:
    section_id: str
    heading: str
    text: str
    start: int


class DiagnosticService:
    """Pure application service; methods perform no network or storage operations."""

    def readiness_score(self, request: ReadinessScoreRequest) -> ReadinessScoreArtifact:
        document = request.document
        text = _visible_text(document)
        sections = _sections(text)
        fact_report = self.fact_density(FactDensityRequest(document=document))
        dimensions = [
            _heading_dimension(document, text),
            _answer_first_dimension(document, text),
            _faq_dimension(document, text),
            _schema_dimension(document),
            DimensionScore(
                dimension="factDensity",
                score=fact_report.overall_score,
                explanation=(
                    "Heuristic ratio of specific factual sentences, weighted by "
                    "caller-supplied evidence support."
                ),
                evidenceIds=[
                    evidence_id
                    for claim in fact_report.claims
                    for evidence_id in claim.evidence_ids
                ],
            ),
            _eeat_dimension(document, text),
            _technical_dimension(document),
        ]
        available = [item.score for item in dimensions if item.score is not None]
        warnings = _base_warnings(document)
        overall: float | None = None
        if len(available) == len(dimensions):
            overall = round(sum(available) / len(available), 1)
        else:
            warnings.append(
                "Overall readiness score omitted because one or more rubric dimensions "
                "lack sufficient supplied input."
            )
        fixes = [
            _dimension_fix(item.dimension)
            for item in sorted(
                (
                    item
                    for item in dimensions
                    if item.score is not None and item.score < 80
                ),
                key=lambda item: (item.score or 0, item.dimension),
            )
        ]
        if not sections:
            warnings.append("No analyzable sections were found.")
        return ReadinessScoreArtifact(
            overallScore=overall,
            dimensions=dimensions,
            prioritizedFixes=fixes,
            warnings=_unique(warnings),
            provenance=_provenance(document),
        )

    def fact_density(self, request: FactDensityRequest) -> FactDensityArtifact:
        document = request.document
        text = _visible_text(document)
        verified_evidence, invalid_ids = _verified_evidence(document, text)
        assessments: list[ClaimAssessment] = []
        section_scores: list[SectionFactDensity] = []
        for section in _sections(text):
            sentences = _sentences(section.text)
            facts = 0
            supported = 0
            for index, sentence in enumerate(sentences, 1):
                specific = bool(_SPECIFIC_FACT.search(sentence))
                evidence_ids = [
                    item.evidence_id
                    for item in verified_evidence
                    if _quote_supports_sentence(item.quote, sentence)
                ]
                if specific:
                    facts += 1
                    supported += bool(evidence_ids)
                    status = "supported" if evidence_ids else "unsupported"
                    classification = "specificFact"
                else:
                    status = "notFactual"
                    classification = "generalStatement"
                assessments.append(
                    ClaimAssessment(
                        claimId=f"{section.section_id}-claim-{index}",
                        sectionId=section.section_id,
                        text=sentence,
                        classification=classification,
                        verificationStatus=status,
                        evidenceIds=evidence_ids,
                    )
                )
            score = (
                round(((facts + supported) / (2 * len(sentences))) * 100, 1)
                if sentences
                else None
            )
            section_scores.append(
                SectionFactDensity(
                    sectionId=section.section_id,
                    heading=section.heading,
                    sentenceCount=len(sentences),
                    specificFactCount=facts,
                    supportedFactCount=supported,
                    score=score,
                )
            )
        weighted_numerator = sum(
            (section.score or 0) * section.sentence_count for section in section_scores
        )
        sentence_total = sum(section.sentence_count for section in section_scores)
        overall = (
            round(weighted_numerator / sentence_total, 1) if sentence_total else None
        )
        unsupported = [
            claim.claim_id
            for claim in assessments
            if claim.verification_status == "unsupported"
        ]
        warnings = _base_warnings(document)
        if invalid_ids:
            warnings.append(
                "Ignored evidence references that failed exact source quote/span "
                f"verification: {', '.join(sorted(invalid_ids))}."
            )
        if unsupported:
            warnings.append(
                "Specific claims without verified supplied evidence are marked unsupported; "
                "no replacement facts were generated."
            )
        return FactDensityArtifact(
            overallScore=overall,
            sections=section_scores,
            claims=assessments,
            unsupportedClaimIds=unsupported,
            warnings=_unique(warnings),
            provenance=_provenance(document, evidence=verified_evidence),
        )

    def entity_map(self, request: EntityMapRequest) -> EntityMapArtifact:
        document = request.document
        text = _visible_text(document)
        aliases: dict[str, tuple[str, str, set[str]]] = {}
        for seed in request.seeds:
            names = {seed.canonical_name, *seed.aliases}
            for name in names:
                aliases[name.casefold()] = (
                    seed.canonical_name,
                    seed.entity_type,
                    names - {seed.canonical_name},
                )
        mentions: dict[str, list[tuple[int, int, str, str, set[str]]]] = {}
        candidates = list(_ENTITY.finditer(text))
        for match in candidates:
            value = match.group(0).strip()
            if value in _ENTITY_STOP or len(value) < 3:
                continue
            canonical, entity_type, known_aliases = aliases.get(
                value.casefold(), (value, "other", set())
            )
            mentions.setdefault(canonical.casefold(), []).append(
                (match.start(), match.end(), value, entity_type, known_aliases)
            )
        # Explicit seeds are emitted only if the canonical name or an alias is visible.
        entities: list[CanonicalEntity] = []
        generated_evidence: list[EvidenceReference] = []
        entity_spans: dict[str, list[tuple[int, int]]] = {}
        for key in sorted(mentions):
            rows = mentions[key]
            canonical = aliases.get(key, (rows[0][2], rows[0][3], set()))[0]
            entity_id = _stable_id("entity", canonical.casefold())
            evidence_ids: list[str] = []
            visible_aliases = {row[2] for row in rows if row[2] != canonical}
            known_aliases = set().union(*(row[4] for row in rows))
            entity_type = rows[0][3]
            spans: list[tuple[int, int]] = []
            for start, end, value, _, _ in rows:
                evidence = _span_evidence(document, text, start, end, value)
                if evidence.evidence_id not in evidence_ids:
                    evidence_ids.append(evidence.evidence_id)
                    generated_evidence.append(evidence)
                spans.append((start, end))
            entity_spans[entity_id] = spans
            entities.append(
                CanonicalEntity(
                    entityId=entity_id,
                    canonicalName=canonical,
                    entityType=entity_type,
                    aliases=sorted(visible_aliases | known_aliases),
                    confidence=round(min(0.95, 0.55 + 0.1 * len(rows)), 2),
                    evidenceIds=evidence_ids,
                )
            )
        relationships: list[EntityRelationship] = []
        sentences_with_offsets = _sentences_with_offsets(text)
        for left_index, left in enumerate(entities):
            for right in entities[left_index + 1 :]:
                shared_evidence: list[str] = []
                for sentence, start, end in sentences_with_offsets:
                    if any(
                        start <= span[0] < end for span in entity_spans[left.entity_id]
                    ) and any(
                        start <= span[0] < end for span in entity_spans[right.entity_id]
                    ):
                        evidence = _span_evidence(
                            document, text, start, end, sentence.strip()
                        )
                        generated_evidence.append(evidence)
                        shared_evidence.append(evidence.evidence_id)
                if shared_evidence:
                    relationships.append(
                        EntityRelationship(
                            relationshipId=_stable_id(
                                "relationship", left.entity_id, right.entity_id
                            ),
                            sourceEntityId=left.entity_id,
                            targetEntityId=right.entity_id,
                            relation="coOccursWith",
                            confidence=round(
                                min(0.9, 0.55 + 0.1 * len(shared_evidence)), 2
                            ),
                            evidenceIds=_unique(shared_evidence),
                        )
                    )
        warnings = _base_warnings(document)
        if not entities:
            warnings.append(
                "No canonical entity candidates were present in supplied content."
            )
        return EntityMapArtifact(
            entities=entities,
            relationships=relationships,
            warnings=_unique(warnings),
            provenance=_provenance(
                document, evidence=_dedupe_evidence(generated_evidence)
            ),
        )

    def schema_markup(self, request: SchemaMarkupRequest) -> SchemaMarkupArtifact:
        document = request.document
        text = _visible_text(document)
        requested = set(request.requested_types)
        nodes: list[dict[str, Any]] = []
        warnings = _base_warnings(document)
        if not requested or "Article" in requested:
            article = _article_schema(document, text)
            if article:
                nodes.append(article)
            elif "Article" in requested:
                warnings.append(
                    "Article omitted: visible content has no title/headline."
                )
        faq = _faq_pairs(text)
        if faq and (not requested or "FAQPage" in requested):
            nodes.append(
                {
                    "@context": "https://schema.org",
                    "@type": "FAQPage",
                    "mainEntity": [
                        {
                            "@type": "Question",
                            "name": question,
                            "acceptedAnswer": {"@type": "Answer", "text": answer},
                        }
                        for question, answer in faq
                    ],
                }
            )
        elif "FAQPage" in requested:
            warnings.append(
                "FAQPage omitted: no visible question-and-answer pairs found."
            )
        steps = _howto_steps(text)
        howto_title = _title(document, text)
        if steps and howto_title and (not requested or "HowTo" in requested):
            nodes.append(
                {
                    "@context": "https://schema.org",
                    "@type": "HowTo",
                    "name": howto_title,
                    "step": [
                        {"@type": "HowToStep", "position": index, "text": step}
                        for index, step in enumerate(steps, 1)
                    ],
                }
            )
        elif "HowTo" in requested:
            warnings.append("HowTo omitted: no visible ordered steps found.")
        product = _product_schema(text)
        if product and (not requested or "Product" in requested):
            nodes.append(product)
        elif "Product" in requested:
            warnings.append(
                "Product omitted: visible labeled product name and description were not found."
            )
        validation = _validate_schema_nodes(nodes, text)
        if not nodes:
            warnings.append(
                "No applicable schema could be generated from visible content."
            )
        return SchemaMarkupArtifact(
            jsonLd=nodes,
            validation=validation,
            warnings=_unique(warnings),
            provenance=_provenance(document),
        )


def _visible_text(document: DiagnosticDocument) -> str:
    if document.media_type == "text/html":
        soup = BeautifulSoup(document.visible_content, "lxml")
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
    return document.visible_content.replace("\r\n", "\n").strip()


def _sections(text: str) -> list[_Section]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        return (
            [_Section("section-1", "Document", text.strip(), 0)] if text.strip() else []
        )
    sections: list[_Section] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append(
            _Section("section-1", "Preamble", text[: matches[0].start()].strip(), 0)
        )
    for index, match in enumerate(matches, len(sections) + 1):
        start = match.end()
        end = (
            matches[matches.index(match) + 1].start()
            if match != matches[-1]
            else len(text)
        )
        sections.append(
            _Section(
                f"section-{index}",
                match.group(2).strip(),
                text[start:end].strip(),
                start,
            )
        )
    return sections


def _sentences(text: str) -> list[str]:
    return [
        part.strip(" \n-*")
        for part in _SENTENCE.split(text)
        if len(part.strip(" \n-*")) >= 3
    ]


def _sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    cursor = 0
    for sentence in _sentences(text):
        start = text.find(sentence, cursor)
        if start < 0:
            start = text.find(sentence)
        if start >= 0:
            rows.append((sentence, start, start + len(sentence)))
            cursor = start + len(sentence)
    return rows


def _verified_evidence(
    document: DiagnosticDocument, text: str
) -> tuple[list[EvidenceReference], set[str]]:
    source = GenerateSource(
        pageId=document.source.source_id,
        url=document.source.url or f"source://{document.source.source_id}",
        title=document.source.title,
        sourceDigest=hashlib.sha256(text.encode()).hexdigest(),
    )
    citations = [
        GenerateCitation(
            pageId=item.source_id,
            url=source.url,
            quote=item.quote,
            sourceDigest=source.source_digest,
        )
        for item in document.evidence
        if item.source_id == document.source.source_id
    ]
    kept, _ = verify_citations(
        citations,
        [source],
        [
            {
                "pageId": document.source.source_id,
                "url": source.url,
                "markdown": text,
            }
        ],
    )
    kept_quotes = {(item.page_id, item.quote) for item in kept}
    verified: list[EvidenceReference] = []
    invalid: set[str] = set()
    for item in document.evidence:
        exact_span = (
            item.start_char is None
            or item.end_char is None
            or text[item.start_char : item.end_char] == item.quote
        )
        if (item.source_id, item.quote) in kept_quotes and exact_span:
            verified.append(item)
        else:
            invalid.add(item.evidence_id)
    return verified, invalid


def _quote_supports_sentence(quote: str, sentence: str) -> bool:
    normalized_quote = " ".join(quote.casefold().split())
    normalized_sentence = " ".join(sentence.casefold().split())
    return (
        normalized_quote in normalized_sentence
        or normalized_sentence in normalized_quote
    )


def _provenance(
    document: DiagnosticDocument, *, evidence: list[EvidenceReference] | None = None
) -> ArtifactProvenance:
    selected = (
        evidence
        if evidence is not None
        else _verified_evidence(document, _visible_text(document))[0]
    )
    return ArtifactProvenance(
        source=document.source,
        queries=document.queries,
        evidenceIds=[item.evidence_id for item in selected],
        evidence=selected,
    )


def _base_warnings(document: DiagnosticDocument) -> list[str]:
    warnings = [
        (
            "Scores and classifications are deterministic heuristics, not measured "
            "search-engine or AI-citation outcomes."
        )
    ]
    if document.content_completeness == "partial":
        warnings.append(
            "Input is marked partial; absence-based findings are omitted where completeness "
            "is required."
        )
    actual_digest = hashlib.sha256(_visible_text(document).encode()).hexdigest()
    if document.source.source_digest and document.source.source_digest != actual_digest:
        warnings.append(
            "Declared sourceDigest does not match normalized visible content; computed "
            "content was analyzed without replacing provenance."
        )
    return warnings


def _heading_dimension(document: DiagnosticDocument, text: str) -> DimensionScore:
    levels = [len(marker) for marker, _ in _HEADING.findall(text)]
    if not levels and document.content_completeness == "partial":
        return DimensionScore(
            dimension="headingHierarchy",
            score=None,
            explanation="Partial input does not establish whether headings are absent.",
        )
    skips = sum(right > left + 1 for left, right in pairwise(levels))
    if not levels:
        score = 0.0
    else:
        score = max(0.0, 100.0 - (0 if 1 in levels else 30) - 20 * skips)
    return DimensionScore(
        dimension="headingHierarchy",
        score=score,
        explanation=f"Found {len(levels)} heading(s), {skips} skipped level transition(s).",
    )


def _answer_first_dimension(document: DiagnosticDocument, text: str) -> DimensionScore:
    body = _HEADING.sub("", text).strip()
    first = next((item for item in _sentences(body) if item), "")
    if not first:
        return DimensionScore(
            dimension="answerFirst",
            score=None if document.content_completeness == "partial" else 0,
            explanation="No opening answer sentence was available.",
        )
    words = len(first.split())
    delayed = bool(
        re.match(r"(?i)(in this (article|post)|welcome|have you ever)", first)
    )
    score = 100.0 if words <= 35 and not delayed else 65.0 if words <= 60 else 30.0
    return DimensionScore(
        dimension="answerFirst",
        score=score,
        explanation=f"Opening sentence contains {words} words; delayed-introduction marker={delayed}.",
    )


def _faq_dimension(document: DiagnosticDocument, text: str) -> DimensionScore:
    count = len(_faq_pairs(text))
    score = 100.0 if count >= 3 else 70.0 if count else 0.0
    if not count and document.content_completeness == "partial":
        score = None
    return DimensionScore(
        dimension="faq",
        score=score,
        explanation=f"Found {count} visible question-and-answer pair(s).",
    )


def _schema_dimension(document: DiagnosticDocument) -> DimensionScore:
    markup = document.existing_json_ld
    if markup is None:
        return DimensionScore(
            dimension="schema",
            score=None,
            explanation="No existingJsonLd input was supplied; schema presence is unknown.",
        )
    valid = bool(markup) and all(
        isinstance(item, dict)
        and item.get("@context") == "https://schema.org"
        and item.get("@type")
        for item in markup
    )
    return DimensionScore(
        dimension="schema",
        score=100.0 if valid else 0.0,
        explanation=f"Supplied JSON-LD contains {len(markup)} node(s); basic shape valid={valid}.",
    )


def _eeat_dimension(document: DiagnosticDocument, text: str) -> DimensionScore:
    if document.content_completeness == "partial":
        return DimensionScore(
            dimension="eeat",
            score=None,
            explanation="Partial content cannot establish absence of E-E-A-T indicators.",
        )
    indicators = {
        "byline": bool(re.search(r"(?im)^(?:by|author:)\s+\S+", text)),
        "date": bool(re.search(r"\b(?:19|20)\d{2}\b", text)),
        "sources": bool(re.search(r"(?im)^#{1,6}\s+(?:sources|references)\b", text)),
        "experience": bool(
            re.search(r"(?i)\b(?:we tested|our experience|case study)\b", text)
        ),
        "attribution": bool(re.search(r"(?i)\baccording to\b", text)),
    }
    count = sum(indicators.values())
    return DimensionScore(
        dimension="eeat",
        score=float(count * 20),
        explanation=f"Found {count} of 5 explicit authorship, sourcing, and experience indicators.",
    )


def _technical_dimension(document: DiagnosticDocument) -> DimensionScore:
    signal = document.technical
    if (
        signal is None
        or signal.crawlable is None
        or signal.status_code is None
        or signal.load_time_ms is None
    ):
        return DimensionScore(
            dimension="technicalCrawlabilityPerformance",
            score=None,
            explanation="crawlable, statusCode, and loadTimeMs are all required for this dimension.",
        )
    score = 0.0
    score += 40 if signal.crawlable else 0
    score += 30 if 200 <= signal.status_code < 300 else 0
    score += (
        30 if signal.load_time_ms <= 2500 else 15 if signal.load_time_ms <= 4000 else 0
    )
    return DimensionScore(
        dimension="technicalCrawlabilityPerformance",
        score=score,
        explanation=(
            f"crawlable={signal.crawlable}, status={signal.status_code}, "
            f"loadTimeMs={signal.load_time_ms}."
        ),
    )


def _dimension_fix(dimension: str) -> str:
    fixes = {
        "headingHierarchy": "Use one clear H1 and avoid skipped heading levels.",
        "answerFirst": "Lead with a direct, concise answer before background context.",
        "faq": "Add visible, content-supported question-and-answer pairs where useful.",
        "schema": "Add applicable JSON-LD that exactly matches visible content.",
        "factDensity": "Replace vague prose with verified specifics and attach exact evidence.",
        "eeat": "Add explicit authorship, source attribution, and first-hand evidence.",
        "technicalCrawlabilityPerformance": "Resolve crawl/status issues and improve measured load time.",
    }
    return fixes[dimension]


def _title(document: DiagnosticDocument, text: str) -> str | None:
    if document.source.title and document.source.title in text:
        return document.source.title
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    return match.group(1).strip() if match else None


def _faq_pairs(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    pairs: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        question = re.sub(r"^#{1,6}\s*", "", line).strip()
        question = re.sub(r"(?i)^q:\s*", "", question).strip()
        if not question.endswith("?"):
            continue
        answers: list[str] = []
        for following in lines[index + 1 :]:
            value = following.strip()
            if not value:
                if answers:
                    break
                continue
            if value.startswith("#") or value.endswith("?"):
                break
            answers.append(re.sub(r"(?i)^a:\s*", "", value))
        answer = " ".join(answers).strip()
        if answer:
            pairs.append((question, answer))
    return pairs


def _howto_steps(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"(?m)^\s*\d+[.)]\s+(.+?)\s*$", text)
        if match.group(1).strip()
    ]


def _article_schema(document: DiagnosticDocument, text: str) -> dict[str, Any] | None:
    title = _title(document, text)
    if not title:
        return None
    node: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": title,
        "articleBody": text,
    }
    if document.source.url:
        node["url"] = document.source.url
    return node


def _product_schema(text: str) -> dict[str, Any] | None:
    fields: dict[str, str] = {}
    for key, value in re.findall(
        r"(?im)^(name|product name|description|brand|sku|price|currency):\s*(.+?)\s*$",
        text,
    ):
        fields[key.casefold()] = value.strip()
    name = fields.get("product name") or fields.get("name")
    description = fields.get("description")
    if not name or not description:
        return None
    node: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": name,
        "description": description,
    }
    if fields.get("brand"):
        node["brand"] = {"@type": "Brand", "name": fields["brand"]}
    if fields.get("sku"):
        node["sku"] = fields["sku"]
    if fields.get("price") and fields.get("currency"):
        node["offers"] = {
            "@type": "Offer",
            "price": fields["price"],
            "priceCurrency": fields["currency"],
        }
    return node


def _validate_schema_nodes(
    nodes: list[dict[str, Any]], visible_text: str
) -> list[SchemaValidationFinding]:
    findings: list[SchemaValidationFinding] = []
    required = {
        "Article": ("headline", "articleBody"),
        "FAQPage": ("mainEntity",),
        "HowTo": ("name", "step"),
        "Product": ("name", "description"),
    }
    for node in nodes:
        schema_type = str(node.get("@type") or "unknown")
        syntax_valid = True
        try:
            json.loads(json.dumps(node, ensure_ascii=False))
        except (TypeError, ValueError):
            syntax_valid = False
        findings.append(
            SchemaValidationFinding(
                code="jsonSyntax",
                schemaType=schema_type,
                valid=syntax_valid,
                detail="Node is JSON serializable."
                if syntax_valid
                else "Node is not valid JSON.",
            )
        )
        shape_valid = (
            node.get("@context") == "https://schema.org"
            and schema_type in required
            and all(node.get(field) for field in required.get(schema_type, ()))
        )
        findings.append(
            SchemaValidationFinding(
                code="schemaShape",
                schemaType=schema_type,
                valid=shape_valid,
                detail="Required deterministic shape is present."
                if shape_valid
                else "Required fields are missing.",
            )
        )
        strings = list(_content_strings(node, skip_keys={"@context", "@type", "url"}))
        visible_valid = all(value in visible_text for value in strings)
        findings.append(
            SchemaValidationFinding(
                code="visibleContent",
                schemaType=schema_type,
                valid=visible_valid,
                detail=(
                    "All generated content values occur in supplied visible content."
                    if visible_valid
                    else "At least one generated content value is not visible."
                ),
            )
        )
    return findings


def _content_strings(value: Any, *, skip_keys: set[str]) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in skip_keys or key == "position":
                continue
            yield from _content_strings(child, skip_keys=skip_keys)
    elif isinstance(value, list):
        for child in value:
            yield from _content_strings(child, skip_keys=skip_keys)
    elif isinstance(value, str):
        yield value


def _span_evidence(
    document: DiagnosticDocument, text: str, start: int, end: int, quote: str
) -> EvidenceReference:
    if len(quote) < 12:
        start = max(0, start - (12 - len(quote)))
        end = min(len(text), max(end, start + 12))
        quote = text[start:end]
    return EvidenceReference(
        evidenceId=_stable_id(document.source.source_id, str(start), str(end), quote),
        sourceId=document.source.source_id,
        quote=quote,
        startChar=start,
        endChar=end,
    )


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _dedupe_evidence(values: list[EvidenceReference]) -> list[EvidenceReference]:
    return list({item.evidence_id: item for item in values}.values())
