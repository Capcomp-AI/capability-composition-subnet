"""The chain path exists in the CLI, and the docs must not oversell it.

``capcomp commit`` writes a real commitment to a real chain. Nothing scores it
yet, and that gap is the dangerous one: a miner who reads the command as the
submission path loses a run in silence, because the chain raises no error for a
commitment nobody reads.

So the sync these check is not "the docs mention the command". It is that every
place a miner might look says the same thing about whether it counts - and that
the day it does start counting, these fail until the docs are updated with it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from capability_subnet.common import constants as C

REPO = Path(__file__).resolve().parents[2]
MINER_DOC = (REPO / "docs" / "miner.md").read_text()
OWNER_DOC = (REPO / "docs" / "owner.md").read_text()


def _cli(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "capability_subnet.miner.cli", *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    return result.stdout + result.stderr


class TestTheCommandIsDocumented:
    def test_the_miner_guide_has_a_section_for_it(self):
        assert "Committing on chain (preview)" in MINER_DOC
        assert "capcomp commit --recipe" in MINER_DOC

    def test_the_owner_guide_says_nothing_reads_it(self):
        assert "capcomp commit" in OWNER_DOC
        assert "rehearsal, not a second way in" in OWNER_DOC

    def test_the_help_epilog_names_it_as_unscored(self):
        text = _cli("--help")
        assert "capcomp commit" in text
        assert "not scored yet" in text


class TestNoDocumentClaimsItIsLive:
    """The claim that costs a miner a run is "this is how you submit"."""

    def test_the_miner_guide_still_names_the_api_as_the_way_in(self):
        assert "**The API is the only way in.**" in MINER_DOC

    def test_the_troubleshooting_entry_covers_a_commitment(self):
        assert "the engine does not read commitments" in MINER_DOC
        assert "which is a preview and is not scored" in MINER_DOC

    def test_the_command_says_so_on_every_run(self):
        """Printed unconditionally, not only on --confirm.

        A miner reading a dry run is exactly the one deciding whether this is
        the path to use.
        """
        source = (REPO / "capability_subnet" / "miner" / "cli.py").read_text()
        assert "the chain submission path is not live yet" in source


class TestTheDocumentedLimitsAreTheRealOnes:
    """Numbers a miner would size a recipe against."""

    def test_the_real_limit_is_documented_as_the_sealed_one(self):
        """The canonical cap is a sanity check; the field is the real limit.

        Documenting a canonical byte count as *the* limit is what made the
        earlier number wrong in both directions, so the doc has to name the
        field and the compression, not a single figure to build against.
        """
        assert f"{C.MAX_TIMELOCK_FIELD_BYTES:,} bytes" in MINER_DOC
        assert "how well it compresses" in MINER_DOC

    def test_the_epoch_budget_is_the_constant(self):
        assert f"{C.MAX_EPOCH_COMMIT_BYTES:,} bytes of commitments per epoch" in MINER_DOC

    def test_the_api_limit_still_stands_while_the_api_does(self):
        """RESUBMISSION_LIMIT governs the scored path and has not been removed.

        The chain budget replaces it only when the API retires. Documenting it
        as gone while it is still enforced is the failure this prevents.
        """
        assert C.RESUBMISSION_LIMIT == 3
        assert "`RESUBMISSION_LIMIT`, currently 3" in OWNER_DOC
