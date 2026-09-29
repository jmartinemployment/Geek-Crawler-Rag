"""The two flags that guard a destructive script have to actually guard it.

`cleanup_unusable_pages.py` is the only script in this repo that issues `delete_many`
against `crawl_pages`, and its own docstring is a memorial to 5,274 pages lost to a filter
nobody expected. Both of its safety inputs were decorative:

* `--dry-run` was declared and read nowhere. `write` came from `args.write` alone, so
  `--write --dry-run` deleted. An operator validating with the script's own documented
  `--dry-run --limit 500` and then re-running with `--write` appended — without removing
  `--dry-run` — got a real delete from a command line that says otherwise.
* `--batch-size` was `type=int` with no validation, and pymongo documents
  `Cursor.limit(0)` as "equivalent to no limit". So `--batch-size 0` turned each bounded
  batch into the entire matching set and bypassed `--limit` with it; a negative value
  reduced every pass to one row.

These are argparse-level, so they cost nothing and fail before a connection is opened.
They are tested through `main(argv)` rather than by inspecting the parser, because the bug
was never in the declaration — `--dry-run` was declared correctly the whole time. It was
that nothing read it.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

# Loaded by path: pytest's pythonpath is ["src"] and scripts/ is not a package, so there is
# no import name for it. No test had ever imported a script before this one -- which is part
# of why a dead CLI flag survived.
_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "cleanup_unusable_pages.py"
_spec = importlib.util.spec_from_file_location("cleanup_unusable_pages", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
# Registered before exec: the script defines a @dataclass, and dataclasses resolves type
# hints through sys.modules[cls.__module__], which is None for a module that is not there yet.
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
main = _module.main


def test_write_and_dry_run_together_are_refused():
    # argparse exits 2 on a usage error. The combination must not reach Mongo at all.
    with pytest.raises(SystemExit) as exc:
        main(["--write", "--dry-run"])
    assert exc.value.code == 2


@pytest.mark.parametrize("bad", ["0", "-1", "-500"])
def test_batch_size_below_one_is_refused(bad):
    # 0 is the dangerous one: pymongo reads limit(0) as no limit, so it would submit the
    # whole matching set as a single delete_many and ignore --limit.
    with pytest.raises(SystemExit) as exc:
        main(["--write", "--batch-size", bad])
    assert exc.value.code == 2


@pytest.mark.parametrize("bad", ["0", "-1", "-500"])
def test_limit_below_one_is_refused(bad):
    """`--limit 0` was accepted, did nothing, and reported itself as unlimited.

    Every budget check reads `counts.budget_used >= limit`, which is true at 0 before any
    step runs -- so the run deleted nothing. Then the banner printed `limit=*`, because
    `args.limit or '*'` renders 0 as the unlimited marker. An operator who typed 0 by mistake
    saw the line that means "no limit, delete everything matching" above a run that touched
    nothing, and the two readings are opposite.
    """
    with pytest.raises(SystemExit) as exc:
        main(["--write", "--limit", bad])
    assert exc.value.code == 2


def test_a_real_limit_is_printed_as_itself(monkeypatch, capsys):
    monkeypatch.setattr(_module, "cleanup", lambda **kw: _module.CleanupCounts())
    assert main(["--dry-run", "--limit", "500"]) == 0
    assert "limit=500" in capsys.readouterr().out


def test_no_limit_is_printed_as_the_unlimited_marker(monkeypatch, capsys):
    monkeypatch.setattr(_module, "cleanup", lambda **kw: _module.CleanupCounts())
    assert main(["--dry-run"]) == 0
    assert "limit=*" in capsys.readouterr().out


def test_limit_of_one_reaches_cleanup(monkeypatch):
    # The boundary, as with --batch-size: 1 is a legitimate "delete exactly one row".
    seen: dict[str, object] = {}

    def fake_cleanup(**kwargs):
        seen.update(kwargs)
        return _module.CleanupCounts()

    monkeypatch.setattr(_module, "cleanup", fake_cleanup)
    assert main(["--write", "--limit", "1"]) == 0
    assert seen["limit"] == 1


def test_batch_size_of_one_is_allowed_past_parsing(monkeypatch):
    # 1 is the boundary and must not be rejected. cleanup() is stubbed so this proves the
    # parse succeeded without opening a connection -- the earlier version of this test let
    # main() reach Mongo and spent 30 seconds per case on socket timeouts the URL cannot
    # shorten, in a suite that otherwise runs in two seconds.
    seen: dict[str, object] = {}

    def fake_cleanup(**kwargs):
        seen.update(kwargs)
        return _module.CleanupCounts()

    monkeypatch.setattr(_module, "cleanup", fake_cleanup)

    assert main(["--write", "--batch-size", "1"]) == 0
    assert seen["batch_size"] == 1
    assert seen["write"] is True


def test_dry_run_alone_is_accepted_and_does_not_write(monkeypatch):
    # The documented invocation. The assertion that matters is write=False reaching cleanup,
    # which is the thing that was broken: --dry-run parsed fine and changed nothing.
    seen: dict[str, object] = {}

    def fake_cleanup(**kwargs):
        seen.update(kwargs)
        return _module.CleanupCounts()

    monkeypatch.setattr(_module, "cleanup", fake_cleanup)

    assert main(["--dry-run"]) == 0
    assert seen["write"] is False


def test_plain_write_still_writes(monkeypatch):
    # The guard must not have broken the real mode.
    seen: dict[str, object] = {}

    def fake_cleanup(**kwargs):
        seen.update(kwargs)
        return _module.CleanupCounts()

    monkeypatch.setattr(_module, "cleanup", fake_cleanup)

    assert main(["--write"]) == 0
    assert seen["write"] is True
