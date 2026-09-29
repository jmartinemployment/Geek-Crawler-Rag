"""The read boundary has to accept the shape GeekAPI actually writes.

GeekAPI's Mongo class map stores these fields as STRINGS: `LegacyStringInt32Serializer` does
`WriteString(...)` so `StatusCode` is "404", and `LegacyStringBooleanSerializer` writes
"t"/"f" so `RobotsAllowed` is "t". Measured against the live store on 2026-09-29: 4,144 of
4,144 `crawl_pages` rows hold `StatusCode` as a string, and all 4,144 hold `RobotsAllowed` as
"t".

The reads were `status if isinstance(status, int) else None` and
`robots if isinstance(robots, bool) else None`. Both are False for every real value, so both
rejection gates were inert: a 404 page's body was chunked, embedded and quotable under a URL
the server said it did not serve, and `if robots_allowed is False` had never once matched.

Four tests were shipped for the status gate and all passed, because every one handed the
classifier a Python `int` literal. This is the test that was missing — the one the code
review identified as the smallest change that would have caught the whole class:

    classify_from_mongo_doc({"StatusCode": "404", ...}) == "http_error"

`bool` is checked explicitly because it subclasses `int` in Python, so `True` would otherwise
be read as the status code 1.
"""

from __future__ import annotations

import pytest

from geek_crawler_rag.mongo import _as_bool, _as_int
from geek_crawler_rag.unusable import classify_from_mongo_doc

PROSE = [{"kind": "paragraph", "text": "Sorry, we could not find that page. Try our blog."}]


# --- the boundary coercions -------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("404", 404),  # what Mongo actually holds
        ("200", 200),
        (" 503 ", 503),
        (404, 404),  # native, for when the encoding is fixed
        (404.0, 404),
        ("", None),
        ("not-a-number", None),
        (None, None),
        (True, None),  # bool subclasses int; must not read as 1
        (False, None),
    ],
)
def test_as_int_accepts_both_shapes(raw, expected):
    assert _as_int(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("t", True),  # what Mongo actually holds
        ("f", False),
        ("T", True),
        (" f ", False),
        ("true", True),
        ("false", False),
        (True, True),  # native, for when the encoding is fixed
        (False, False),
        ("", None),
        ("maybe", None),
        (None, None),
    ],
)
def test_as_bool_accepts_both_shapes(raw, expected):
    assert _as_bool(raw) == expected


def test_unreadable_values_are_unknown_not_assumed_good():
    # The failure that made this invisible: a value that cannot be read must not be
    # silently treated as 200/allowed. None means "unknown", and the classifier leaves an
    # unknown alone rather than guessing either way.
    assert _as_int("garbage") is None
    assert _as_bool("garbage") is None


# --- the classifier, through the shape Mongo really returns -----------------------------


@pytest.mark.parametrize("code", ["400", "401", "403", "404", "410", "429", "500", "503"])
def test_a_string_status_is_read_as_an_http_error(code):
    """The test whose absence let the whole feature ship inert."""
    doc = {
        "Url": "https://example.com/gone",
        "FinalUrl": "https://example.com/gone",
        "StatusCode": code,
        "RobotsAllowed": "t",
        "Blocks": PROSE,
    }
    assert classify_from_mongo_doc(doc) == "http_error"


@pytest.mark.parametrize("code", ["200", "201", "204", "301", "302", "304"])
def test_a_served_page_is_still_kept(code):
    doc = {
        "Url": "https://example.com/real",
        "FinalUrl": "https://example.com/real",
        "StatusCode": code,
        "RobotsAllowed": "t",
        "Blocks": PROSE,
    }
    assert classify_from_mongo_doc(doc) is None


def test_a_native_int_status_still_works():
    # Both shapes, so fixing the encoding does not invert the bug.
    doc = {
        "Url": "https://example.com/gone",
        "FinalUrl": "https://example.com/gone",
        "StatusCode": 404,
        "RobotsAllowed": True,
        "Blocks": PROSE,
    }
    assert classify_from_mongo_doc(doc) == "http_error"


def test_a_robots_denied_string_is_read_as_a_failure():
    # Zero live rows hold "f" today -- the crawler rejects denied URLs before saving -- but
    # a gate that cannot fire is not a gate.
    doc = {
        "Url": "https://example.com/private",
        "FinalUrl": "https://example.com/private",
        "StatusCode": "200",
        "RobotsAllowed": "f",
        "Blocks": PROSE,
    }
    assert classify_from_mongo_doc(doc) == "failure"


def test_an_absent_status_is_not_an_error():
    # A row with no StatusCode must not be rejected: unknown is not 4xx.
    doc = {
        "Url": "https://example.com/real",
        "FinalUrl": "https://example.com/real",
        "RobotsAllowed": "t",
        "Blocks": PROSE,
    }
    assert classify_from_mongo_doc(doc) is None
