"""The guides describe the path that exists, and no longer one that does not.

`capcomp commit` was a preview for one day, and these tests held the guides to
saying so. It is the submission path now: the API is gone, and a miner who
follows a guide still pointing at `capcomp submit` loses a run without being
told why - the chain accepts nothing from them and no service refuses them,
because there is no service.

So the assertions inverted. What they check is the same thing they always did:
that no document describes a route a miner can take and get nothing for.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from capability_subnet.common import constants as C

REPO = Path(__file__).resolve().parents[2]
MINER_DOC = (REPO / "docs" / "miner.md").read_text()
OWNER_DOC = (REPO / "docs" / "owner.md").read_text()
VALIDATOR_DOC = (REPO / "docs" / "validator.md").read_text()


def _cli(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "capability_subnet.miner.cli", *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    return result.stdout + result.stderr


class TestCommitIsDocumentedAsTheWayIn:
    def test_the_miner_guide_leads_with_it(self):
        assert "capcomp commit --recipe" in MINER_DOC

    def test_the_help_epilog_names_commit_rather_than_submit(self):
        text = _cli("--help")
        assert "capcomp commit" in text
        assert "capcomp submit --confirm" not in text

    def test_the_validator_guide_says_the_field_comes_from_the_chain(self):
        assert "chain" in VALIDATOR_DOC.lower()


class TestNothingStillPointsAtTheService:
    """The failure this file exists to prevent, inverted.

    A guide naming the API sends a miner to a route that no longer answers.
    They get no error worth acting on: the chain never heard from them, and the
    service that would have refused them is gone.
    """

    def test_no_guide_tells_a_miner_to_submit_over_http(self):
        for name, doc in (("miner.md", MINER_DOC), ("owner.md", OWNER_DOC)):
            assert "capcomp submit --confirm" not in doc, f"{name} still sends miners to the API"

    def test_the_retired_command_is_not_offered_and_commit_is(self):
        """argparse lists what a miner may run, which is the answer they need.

        The command was briefly kept as a redirect. It is gone outright now:
        `capcomp -h` and the invalid-choice error both name `commit`, so the
        miner who typed the old name is told the new one either way.
        """
        text = _cli("submit", "--recipe", "x.json")

        # Unquoted, because argparse quotes the choice list on 3.10 and 3.11
        # and stops on 3.12. The assertion is about which commands are offered,
        # not about how the interpreter punctuates them.
        assert "invalid choice" in text and "submit" in text
        assert "commit" in text
        assert "commit" in _cli("-h")

    def test_no_guide_calls_the_chain_path_a_preview(self):
        for name, doc in (("miner.md", MINER_DOC), ("owner.md", OWNER_DOC)):
            lowered = doc.lower()
            assert "preview" not in lowered or "capcomp commit" not in lowered, (
                f"{name} still describes the chain path as a preview"
            )


class TestTheDocumentedLimitsAreTheRealOnes:
    def test_the_real_limit_is_documented_as_the_sealed_one(self):
        """The canonical cap is a sanity check; the field is the real limit."""
        assert f"{C.MAX_TIMELOCK_FIELD_BYTES:,} bytes" in MINER_DOC
        assert "how well it compresses" in MINER_DOC

    def test_the_epoch_budget_is_the_constant(self):
        assert f"{C.MAX_EPOCH_COMMIT_BYTES:,} bytes of commitments per epoch" in MINER_DOC
