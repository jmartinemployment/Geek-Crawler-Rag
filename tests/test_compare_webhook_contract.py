"""The cross-repo contract comparison, including the bug found by testing it.

`scripts/compare_webhook_contract.py` is the only check that the sender's and receiver's copies of
the webhook contract agree. It compares the `fields` map rather than bytes, because a byte diff also
failed on a prose edit -- improving a comment in one repo reddened the other repo's main for no
behavioural reason.

The duplicate-key check was wrong on first write: a document-wide `seen` set flagged `eventType`
because it is a key in both `fields` and `$notes`, so the real contract failed its own comparison.
`object_pairs_hook` fires once per object, so the scope has to be the pairs it is handed.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "compare_webhook_contract.py"
_spec = importlib.util.spec_from_file_location("compare_webhook_contract", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

REAL = (
    pathlib.Path(__file__).resolve().parents[1]
    / "contracts"
    / "rag-index-status"
    / "webhook.v1.json"
)


def _write(tmp_path: pathlib.Path, name: str, doc: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(doc, indent=2))
    return str(path)


def _real() -> dict:
    return json.loads(REAL.read_text())


def test_the_real_contract_agrees_with_itself():
    # The case the first version failed: eventType is a key in `fields` AND in `$notes`.
    assert _module.compare(str(REAL), str(REAL)) == 0


def test_a_prose_only_difference_is_not_a_failure(tmp_path):
    other = _real()
    other["$comment"] = "entirely different prose"
    other["$notes"]["persisted"] = "reworded"
    assert _module.compare(str(REAL), _write(tmp_path, "prose.json", other)) == 0


def test_a_field_missing_on_the_receiver_fails(tmp_path, capsys):
    other = _real()
    del other["fields"]["pagesSkippedUnusable"]
    assert _module.compare(str(REAL), _write(tmp_path, "missing.json", other)) == 1
    out = capsys.readouterr().out
    assert "pagesSkippedUnusable" in out
    assert "::error::" in out


def test_a_field_only_the_receiver_has_fails(tmp_path, capsys):
    other = _real()
    other["fields"]["pagesSkippedRobots"] = "int"
    assert _module.compare(str(REAL), _write(tmp_path, "extra.json", other)) == 1
    assert "pagesSkippedRobots" in capsys.readouterr().out


def test_a_retyped_field_fails_and_names_both_types(tmp_path, capsys):
    other = _real()
    other["fields"]["pagesSkippedUnusable"] = "string"
    assert _module.compare(str(REAL), _write(tmp_path, "retyped.json", other)) == 1
    out = capsys.readouterr().out
    assert "'int'" in out and "'string'" in out


def test_a_duplicate_key_within_one_object_fails(tmp_path, capsys):
    # json.dumps cannot emit one, so this is written as text.
    path = tmp_path / "dupe.json"
    path.write_text(
        REAL.read_text().replace('"attempt": "int",', '"attempt": "int",\n    "attempt": "LIE",', 1)
    )
    assert _module.compare(str(REAL), str(path)) == 1
    assert "repeats the key" in capsys.readouterr().out


def test_a_missing_file_fails_rather_than_passing_vacuously(tmp_path, capsys):
    assert _module.compare(str(REAL), str(tmp_path / "nope.json")) == 1
    assert "cannot read" in capsys.readouterr().out


def test_a_document_without_a_fields_object_fails(tmp_path, capsys):
    assert _module.compare(str(REAL), _write(tmp_path, "empty.json", {"version": 1})) == 1
    assert "no `fields` object" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["x"], ["x", "a"], ["x", "a", "b", "c"]])
def test_wrong_argument_count_fails(argv, capsys):
    assert _module.main(argv) == 1
    assert "usage" in capsys.readouterr().out
