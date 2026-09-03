"""Every flag the guides tell an operator to type must parse.

`--neuron.backend_url` was documented in the validator guide, in the flag table
and in the troubleshooting section, and was never added to the validator's
parser — it lived only on the miner's, under a different name. A validator
following the guide got `unrecognized arguments` and could not run in endpoint
mode at all.

Nothing caught it because the guide and the parser were only ever read by
people. This reads both.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from capability_subnet.common.config import (
    add_common_args,
    add_miner_args,
    add_validator_args,
    add_wallet_args,
)

REPO = Path(__file__).resolve().parents[2]

#: A long option as it appears in a documented command or a flag table.
FLAG = re.compile(r"`(--[a-z][\w.]*)`|^\s*(--[a-z][\w.]*)\s", re.MULTILINE)


def _parser(role: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"capability-subnet-{role}")
    add_wallet_args(parser)
    add_common_args(parser)
    (add_validator_args if role == "validator" else add_miner_args)(parser)
    return parser


def _accepted(parser: argparse.ArgumentParser) -> set[str]:
    return {opt for action in parser._actions for opt in action.option_strings}


def _documented(doc: str) -> set[str]:
    text = (REPO / "docs" / doc).read_text()
    return {m.group(1) or m.group(2) for m in FLAG.finditer(text)} - {None}


@pytest.mark.parametrize("role,doc", [("validator", "validator.md"), ("miner", "miner.md")])
def test_every_documented_flag_is_accepted(role, doc):
    accepted = _accepted(_parser(role))
    # Flags belonging to other tools the guide also shows (btcli, capability-audit).
    foreign = {f for f in _documented(doc) if f.startswith(("--netuid.", "--subtensor.chain"))}
    missing = sorted(f for f in _documented(doc) - accepted - foreign if f.startswith("--neuron."))
    assert missing == [], f"{doc} documents flags {role} does not accept: {missing}"


def test_the_validator_accepts_the_backend_url_the_guide_prints(self=None):
    """The exact flag from the reported failure."""
    parsed = _parser("validator").parse_args(
        [
            "--netuid",
            "103",
            "--neuron.mode",
            "endpoint",
            "--neuron.backend_url",
            "https://engine.example",
        ]
    )
    assert parsed.backend_url == "https://engine.example"


def test_the_older_spelling_still_parses():
    """Validators already running must not break on an upgrade."""
    parsed = _parser("validator").parse_args(["--backend.url", "https://engine.example"])
    assert parsed.backend_url == "https://engine.example"
