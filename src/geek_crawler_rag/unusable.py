"""Classify crawl pages that must not stay in the corpus.

Mirrors Geek-Crawler-v2 locale-path rules and reject taxonomy
(locale / challenge-failure / extract-empty).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

KEEP_REGION = frozenset({"us"})
DROP_REGION = frozenset({
    "gb", "uk", "au", "nz", "sg", "ae",
    "ca", "ie", "eu", "in", "za", "jp", "kr", "br", "mx", "de", "fr", "es", "it", "nl",
})
NON_ENGLISH_LOCALE = frozenset({
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az",
    "ba", "be", "bg", "bh", "bi", "bm", "bn", "bo", "br", "bs",
    "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy",
    "da", "de", "dv", "dz",
    "ee", "el", "eo", "es", "et", "eu",
    "fa", "ff", "fi", "fj", "fo", "fr", "fy",
    "ga", "gd", "gl", "gn", "gu", "gv",
    "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz",
    "ia", "id", "ie", "ig", "ii", "ik", "io", "is", "it", "iu",
    "ja", "jv",
    "ka", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku", "kv", "kw", "ky",
    "la", "lb", "lg", "li", "ln", "lo", "lt", "lu", "lv",
    "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
    "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny",
    "oc", "oj", "om", "or", "os",
    "pa", "pi", "pl", "ps", "pt",
    "qu",
    "rm", "rn", "ro", "ru", "rw",
    "sa", "sc", "sd", "se", "sg", "si", "sk", "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw",
    "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts", "tt", "tw", "ty",
    "ug", "uk", "ur", "uz",
    "ve", "vi", "vo",
    "wa", "wo",
    "xh",
    "yi", "yo",
    "za", "zh", "zu",
})

# Backfill skip marks and reject reasons that mean "delete, do not keep".
DELETE_SKIP_REASONS = frozenset({
    "locale",
    "failure",
    "extract_empty",
    "extract_error",
    "fetch_error",
    "no_html",
    "robots",
    "already_markdown",  # only if marked skip without usable body — rare
})

CORPUS_DELETE_REASONS = frozenset({
    "locale",
    "failure",
    "extract_empty",
    "extract_error",
    "non_english",
})


def _first_path_segment(pathname: str) -> str | None:
    parts = [p for p in pathname.split("/") if p]
    return parts[0] if parts else None


def _primary_lang(seg: str) -> str:
    return seg.lower().split("-", 1)[0]


def should_exclude_locale_path(url: str) -> bool:
    try:
        pathname = urlparse(url).path or "/"
        seg = _first_path_segment(pathname)
        if not seg:
            return False
        lower = seg.lower()
        if lower in KEEP_REGION:
            return False
        if lower in DROP_REGION:
            return True
        primary = _primary_lang(seg)
        if primary == "en":
            return False
        return primary in NON_ENGLISH_LOCALE
    except Exception:
        return False


def classify_unusable_page(
    *,
    url: str = "",
    final_url: str = "",
    failure_reason: str | None = None,
    markdown_backfill_skip: str | None = None,
    robots_allowed: bool | None = None,
) -> str | None:
    """Return delete reason or None if the page may stay in the corpus.

    Does not run Readability — use extract_empty when extraction already failed.
    """
    if markdown_backfill_skip in ("locale", "failure", "extract_empty", "extract_error", "fetch_error", "no_html", "robots"):
        return "locale" if markdown_backfill_skip == "locale" else (
            "failure" if markdown_backfill_skip in ("failure", "robots") else "extract_empty"
        )

    page_url = final_url or url
    if should_exclude_locale_path(page_url) or should_exclude_locale_path(url):
        return "locale"

    if robots_allowed is False:
        return "failure"

    if isinstance(failure_reason, str) and failure_reason.strip():
        return "failure"

    return None


def classify_from_mongo_doc(doc: dict[str, Any]) -> str | None:
    return classify_unusable_page(
        url=str(doc.get("Url") or ""),
        final_url=str(doc.get("FinalUrl") or ""),
        failure_reason=doc.get("FailureReason"),
        markdown_backfill_skip=doc.get("MarkdownBackfillSkip"),
        robots_allowed=doc.get("RobotsAllowed"),
    )
