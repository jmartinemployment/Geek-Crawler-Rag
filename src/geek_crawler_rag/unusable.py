"""Classify crawl pages that must not stay in the corpus.

Mirrors Geek-Crawler-v2 locale-path rules and reject taxonomy
(locale / challenge-failure / extract-empty).
"""

from __future__ import annotations

from typing import Any
import logging
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

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

# Reject reasons that mean "delete, do not keep".
DELETE_SKIP_REASONS = frozenset({
    "locale",
    "failure",
    "http_error",
    "extract_empty",
    "extract_error",
    "fetch_error",
    "no_html",
    "robots",
})

# Reasons a page cannot be indexed. Nothing deletes on them any more —
# indexing is read-only over the corpus (see indexer._skip_unusable).
CORPUS_SKIP_REASONS = frozenset({
    "locale",
    "failure",
    "http_error",
    "extract_empty",
    "extract_error",
    "non_english",
    "no_content",
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
    except (ValueError, TypeError, AttributeError):
        # Narrowed from Exception: the only failures reachable here are a URL
        # that will not parse or a value that is not a string. False keeps the
        # page, which is the conservative answer for a locale filter - an
        # unparseable URL is not evidence of a foreign locale - but it is logged
        # so a run that quietly kept malformed URLs leaves a trace.
        logger.debug("Locale check skipped, unparseable url=%r", url)
        return False


#: A sitemap is a link index, not a page. It carries no prose a citation could come from, and it
#: is the one page shape that scales with the size of the whole site rather than with its own
#: content: netsuite.com/portal/sitemap.shtml is 143,292 characters across 3,092 blocks, three
#: times the next largest page in that crawl and the only one over 60k. Chunked and flushed, its
#: upsert body stalled the socket write and killed the run five times -- `httpx.WriteTimeout` inside
#: `_send_request_body`, with Qdrant idle at 0.17% CPU and answering every other request in
#: milliseconds.
#:
#: Matched on the last path segment, so `/portal/sitemap.shtml` and `/sitemap_index.xml` are caught
#: while an article at `/blog/how-to-build-a-sitemap` is not -- that one is about sitemaps and is
#: perfectly citable.
_SITEMAP_SEGMENT = re.compile(r"^sitemap(?:[-_.][\w-]+)*$", re.IGNORECASE)


def is_sitemap_url(url: str) -> bool:
    """True when the URL's own last segment names it a sitemap."""
    try:
        pathname = urlparse(url).path or ""
    except Exception:
        return False
    segments = [seg for seg in pathname.split("/") if seg]
    if not segments:
        return False
    last = segments[-1]
    stem = last.rsplit(".", 1)[0] if "." in last else last
    return bool(_SITEMAP_SEGMENT.match(last) or _SITEMAP_SEGMENT.match(stem))


def classify_unusable_page(
    *,
    url: str = "",
    final_url: str = "",
    failure_reason: str | None = None,
    robots_allowed: bool | None = None,
    status_code: int | None = None,
    blocks: list[Any] | None = None,
) -> str | None:
    """Return delete reason or None if the page may stay in the corpus.

    A page without typed blocks carries no corpus body this service can read.
    The caller counts it and moves on — nothing here deletes.
    """
    page_url = final_url or url
    if should_exclude_locale_path(page_url) or should_exclude_locale_path(url):
        return "locale"

    # A 4xx or 5xx body is the server's error page, not the site's content.
    # This is not this service re-adjudicating what is corpus: the crawler owns
    # that decision and now makes it at the source. What reaches here is rows
    # written before it did, when nothing anywhere looked at the status. A
    # branded 404 carries nav, an apology and suggested links, which clears
    # every prose floor the pipeline has, so it was extracted, chunked and
    # embedded under the URL that did not exist.
    #
    # 0 is kept deliberately. It is the default for a row written without the
    # field, and absence of a status is not evidence of an error.
    if isinstance(status_code, int) and status_code >= 400:
        return "http_error"

    # Before the content checks: a sitemap has plenty of blocks and would otherwise pass every one.
    if is_sitemap_url(page_url) or is_sitemap_url(url):
        return "sitemap"

    if robots_allowed is False:
        return "failure"

    if isinstance(failure_reason, str) and failure_reason.strip():
        return "failure"

    if not (isinstance(blocks, list) and len(blocks) > 0):
        return "no_content"

    return None


def classify_from_mongo_doc(doc: dict[str, Any]) -> str | None:
    raw = doc.get("Blocks")
    if raw is None:
        raw = doc.get("blocks")
    # Same coercions as MongoCorpus._page_from_doc, imported rather than re-implemented.
    # GeekAPI stores these as strings ("404", "t"), so an isinstance check against int/bool
    # is False for every real value -- which is why this branch and the robots branch below
    # were both silently inert. Two readers of one encoding, disagreeing, is the drift
    # CLAUDE.md names.
    from geek_crawler_rag.mongo import _as_bool, _as_int

    return classify_unusable_page(
        url=str(doc.get("Url") or ""),
        final_url=str(doc.get("FinalUrl") or ""),
        failure_reason=doc.get("FailureReason"),
        robots_allowed=_as_bool(doc.get("RobotsAllowed")),
        status_code=_as_int(doc.get("StatusCode")),
        blocks=raw if isinstance(raw, list) else None,
    )
