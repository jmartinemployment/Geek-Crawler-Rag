#!/usr/bin/env python3
"""Compare the `fields` map of the two copies of the index-status webhook contract.

The RAG posts this payload and GeekAPI binds it by name over JSON. Neither repo can see the
other's types at compile time, so each keeps a copy of the contract and a test pinning its own
half; this is the only place both copies exist at once, which is what makes it the place the
comparison belongs.

Only `fields` is compared. A byte diff also failed on a prose edit, so improving a comment in one
repo reddened the other's main for no behavioural reason.

Duplicate keys are refused rather than resolved: json.loads keeps the last occurrence while C#'s
JsonDocument.EnumerateObject yields both, so a repeated key makes the two repos read different
documents out of byte-identical files -- the exact drift this contract exists to close.

Usage:
  python3 scripts/compare_webhook_contract.py <sender.json> <receiver.json>
Exit 0 when the maps agree, 1 otherwise, with ::error:: lines for GitHub Actions.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _fields(path: str) -> dict[str, str] | None:
    """The `fields` map, or None with the reason printed."""
    try:
        raw = open(path, encoding="utf-8").read()
    except OSError as exc:
        print(f"::error::cannot read {path}: {exc}")
        return None

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Per-object, not per-document.

        object_pairs_hook fires once per JSON object, so the check has to be scoped to the pairs
        it is handed. A document-wide set flagged `eventType` as duplicated merely because it is
        a key in both `fields` and `$notes` -- caught by exercising this script against the real
        contract before trusting it.
        """
        seen: set[str] = set()
        for key, _value in pairs:
            if key in seen:
                raise ValueError(
                    f"{path} repeats the key {key!r} in one object. Python keeps the last "
                    "occurrence and C# keeps both, so the two repos would read different "
                    "documents out of byte-identical files."
                )
            seen.add(key)
        return dict(pairs)

    try:
        doc = json.loads(raw, object_pairs_hook=no_duplicates)
    except ValueError as exc:
        print(f"::error::{exc}")
        return None

    fields = doc.get("fields")
    if not isinstance(fields, dict):
        print(f"::error::{path} has no `fields` object.")
        return None
    return fields


def compare(sender_path: str, receiver_path: str) -> int:
    sender = _fields(sender_path)
    receiver = _fields(receiver_path)
    if sender is None or receiver is None:
        return 1

    if sender == receiver:
        print(f"Contract `fields` agree: {len(sender)} fields.")
        return 0

    only_sender = sorted(set(sender) - set(receiver))
    only_receiver = sorted(set(receiver) - set(sender))
    retyped = sorted(k for k in set(sender) & set(receiver) if sender[k] != receiver[k])

    if only_sender:
        print(
            f"::error::In the sender's copy only: {only_sender}. Copy the contract to GeekBackend "
            "unchanged and bind each field in RagIndexStatusWebhookRequest -- an unbound field is "
            "discarded on arrival with no error and no log."
        )
    if only_receiver:
        print(
            f"::error::In the receiver's copy only: {only_receiver}. Either the sender stopped "
            "posting them or GeekBackend's copy is ahead of this repo."
        )
    for key in retyped:
        print(f"::error::{key}: sender declares {sender[key]!r}, receiver declares {receiver[key]!r}.")
    return 1


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("::error::usage: compare_webhook_contract.py <sender.json> <receiver.json>")
        return 1
    return compare(argv[1], argv[2])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
