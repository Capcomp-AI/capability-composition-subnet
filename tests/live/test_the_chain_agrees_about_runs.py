"""The invariant, against finney rather than against constructed blocks.

The unit tests pin the rule. This checks the rule against what the pallet
actually returns, because every failure today came from the chain reporting
something the code did not expect: a reveal round that disappears when the
commitment opens, a commit block replaced by the reveal's, one slot per hotkey.
Constructed records cannot catch the next one of those.

Skipped unless CAPSUB_LIVE is set, so it never runs in CI: it reads mainnet and
it is a check on reality, not on a change.

    CAPSUB_LIVE=1 pytest tests/live -q
"""

from __future__ import annotations

import collections
import os

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.chain import (
    fetch_metagraph,
    run_for_commitment,
    run_id_for_block,
)
from capability_subnet.common.timelock import reveal_round_for_run

pytestmark = pytest.mark.skipif(not os.environ.get("CAPSUB_LIVE"), reason="CAPSUB_LIVE is unset")

NETUID = 103


@pytest.fixture(scope="module")
def view():
    import bittensor as bt

    return fetch_metagraph(bt.subtensor(network="finney"), NETUID)


@pytest.fixture(scope="module")
def submissions(view):
    """Timelocked records only. The pallet slot also holds the operator's
    archive digest and the pointer-era payloads, which are not submissions."""
    return [
        r
        for r in view.commitment_records
        if getattr(r, "reveal_round", None) or getattr(r, "revealed", None)
    ]


class TestEveryCommitmentResolvesToOneRun:
    def test_there_are_commitments_to_check(self, submissions):
        assert submissions, "no timelocked commitments on chain; nothing to verify"

    def test_every_commitment_resolves(self, submissions, view):
        """A commitment that resolves to no run is one nobody will measure."""
        unresolved = []
        for record in submissions:
            try:
                if getattr(record, "revealed", None):
                    run_for_commitment(revealed_at_block=int(record.revealed[-1][0]))
                else:
                    run_for_commitment(commit_block=int(record.block))
            except Exception as exc:  # noqa: BLE001 - collected and reported
                unresolved.append((record.hotkey[:12], type(exc).__name__, str(exc)[:60]))
        assert not unresolved, f"unresolved commitments: {unresolved}"

    def test_a_sealed_commitment_agrees_with_the_round_it_pins(self, submissions):
        """The round is the miner's own statement of the run they entered, so
        the block-derived answer has to match it while both are readable."""
        mismatched = []
        for record in submissions:
            if getattr(record, "revealed", None):
                continue
            run = run_for_commitment(commit_block=int(record.block))
            if int(getattr(record, "reveal_round", 0) or 0) != reveal_round_for_run(run):
                mismatched.append((record.hotkey[:12], record.block, record.reveal_round))
        assert not mismatched, f"block and round disagree: {mismatched}"

    def test_no_revealed_commitment_resolves_to_its_own_reveal_block_s_run(self, submissions):
        """The precise error that filed run 423 under 424.

        A reveal lands in the run *after* the one whose commitments it opens,
        so an answer equal to the reveal block's own run is the bug returning.
        """
        wrong = []
        for record in submissions:
            if not getattr(record, "revealed", None):
                continue
            reveal_block = int(record.revealed[-1][0])
            run = run_for_commitment(revealed_at_block=reveal_block)
            if run == run_id_for_block(reveal_block, C.DEFAULT_RUN_BLOCKS):
                wrong.append((record.hotkey[:12], reveal_block, run))
        assert not wrong, f"resolved to the reveal block's own run: {wrong}"


class TestTheFieldIsCoherent:
    def test_a_run_s_field_agrees_with_what_a_validator_reads(self, view):
        """field_from_commitments and run_for_commitment must select the same
        set, or the engine and the validators measure different fields."""
        from capability_subnet.common.sealed import field_from_commitments

        registered = set(view.hotkeys)
        for run in (run_id_for_block(view.block, C.DEFAULT_RUN_BLOCKS) - 1,):
            admitted, _ = field_from_commitments(
                view.commitment_records, run, registered=registered
            )
            for entry in admitted:
                assert run_for_commitment(revealed_at_block=entry.block) == run, (
                    f"{entry.hotkey[:12]} admitted to run {run} but resolves elsewhere"
                )

    def test_each_hotkey_holds_at_most_one_commitment(self, submissions):
        """The pallet's own rule, and the reason a re-commit destroys the
        previous one. Asserted so a change to it is noticed here first."""
        counts = collections.Counter(r.hotkey for r in submissions)
        assert not [h for h, n in counts.items() if n > 1]
