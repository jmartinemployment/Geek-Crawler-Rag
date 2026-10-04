"""F-R10: the C# and Python block→text projections, run over one fixture.

GeekAPI cuts quote candidates from a page's blocks with `GccCorpusBlockMapper` (C#). This service
projects the same blocks to text with `block_text` (Python) and verifies quotes against that. Two
implementations of "join the blocks" are the drift CLAUDE.md §1a names, so they are pinned to one
fixture, `contracts/block-projection/v1.json`:

* this file checks the Python projection against the fixture's `python` field;
* `GeekBackend.Tests/ContentCreator/GccCorpusBlockProjectionContractTests.cs` checks the C#
  projection against `csharp` where the case has one and `python` where it does not;
* the cross-repo workflow requires the two copies of the fixture to be byte-identical.

So wherever `differs` is null the two projections are byte-equal. Where it is set, the difference
is one of the two named here, and `verify_quote` -- what `POST /v1/verify` calls -- normalises
exactly those.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from geek_crawler_rag.block_text import BLOCK_SEPARATOR, render_block_text
from geek_crawler_rag.citation_verify import quote_in_text, verify_quote

FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "contracts"
    / "block-projection"
    / "v1.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]

# The only two differences the route normalises. A new kind here is a new difference between the
# projections, and it is named and justified before the route learns it -- never folded in silently.
NAMED_DIFFERENCES = {"empty-cells", "escaped-markup"}


def _python(blocks: list[dict]) -> list[str]:
    return [t for t in (render_block_text(b) for b in blocks) if t.strip()]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_the_python_projection_matches_the_fixture(case):
    assert _python(case["blocks"]) == case["python"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_a_case_either_agrees_byte_for_byte_or_names_its_difference(case):
    if case["differs"] is None:
        # No `csharp` field: the C# test is held to `python`, byte for byte.
        assert "csharp" not in case
    else:
        assert case["differs"] in NAMED_DIFFERENCES
        assert case["csharp"] != case["python"]


@pytest.mark.parametrize(
    "case", [c for c in CASES if c["differs"]], ids=[c["name"] for c in CASES if c["differs"]]
)
def test_every_csharp_string_is_found_by_the_route_check(case):
    page = BLOCK_SEPARATOR.join(case["python"])
    for quote in case["csharp"]:
        assert verify_quote(quote, page, case["blocks"]), quote


def test_each_named_difference_is_one_the_unnormalised_check_gets_wrong():
    """The normalisation is needed for both differences and for nothing else."""
    failing_raw = {
        case["differs"]
        for case in CASES
        if case["differs"]
        and not all(
            quote_in_text(q, BLOCK_SEPARATOR.join(case["python"]), case["blocks"])
            for q in case["csharp"]
        )
    }
    assert failing_raw == NAMED_DIFFERENCES


def test_there_is_no_fold_table():
    """Both projections leave curly quotes as the page has them, so the route does too: a straight
    quote does not match a curly page. That is a model-output question, not a projection one."""
    page = "Ramp’s “Bill Pay” — one approval queue for every vendor."
    assert verify_quote("Ramp’s “Bill Pay” — one approval queue", page)
    assert not verify_quote('Ramp\'s "Bill Pay" - one approval queue', page)
