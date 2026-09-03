"""Sealing a recipe, and refusing one in terms the miner can act on.

The chain accepts a wrong recipe as readily as a right one - the pallet checks
bytes, not meaning - so a mistake costs the miner a whole run and surfaces a day
later as a line in a report. Every check that can happen before the extrinsic
has to happen before it, and every refusal has to name the limit, the actual
value and the way out.

That makes the refusal text part of the contract rather than decoration, so it
is asserted here like anything else.
"""

from __future__ import annotations

import time
import zlib

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common import timelock as T
from capability_subnet.common.chain import run_opens_block
from capability_subnet.miner import commit as chain
from capability_subnet.miner.recipe import new_recipe

FUTURE_RUN = 900


def _recipe(**kwargs):
    """A valid recipe against whatever the frozen pool currently holds.

    Adapter ids are read from the snapshot rather than written here, so a pool
    change fails at admission where it belongs instead of in this file.
    """
    from capability_subnet.registry.snapshot import load_snapshot

    return new_recipe(list(load_snapshot().adapter_ids)[:2], **kwargs)


@pytest.fixture
def recipe():
    return _recipe()


class TestASealedRecipeFitsOneCommitment:
    """The size story the 1,536-byte cap was chosen to guarantee."""

    def test_a_real_recipe_seals_into_a_single_field(self, recipe):
        sealed = chain.seal(recipe, FUTURE_RUN)
        assert len(sealed.ciphertext) <= C.MAX_TIMELOCK_FIELD_BYTES

    def test_sealing_is_deterministic_in_the_round_it_targets(self, recipe):
        """Two miners sealing for the same run must unseal at the same instant.

        The ciphertext differs every time - the scheme is randomised - but the
        round it is bound to may not.
        """
        first, second = chain.seal(recipe, FUTURE_RUN), chain.seal(recipe, FUTURE_RUN)
        assert first.reveal_round == second.reveal_round
        assert first.recipe_sha256 == second.recipe_sha256

    def test_the_payload_survives_the_round_trip(self, recipe):
        """What is sealed is what the digest was taken over.

        A truncated or reframed payload fails its digest check at ingest and
        reads as a miner who submitted garbage, so the framing is checked here
        rather than discovered there.
        """
        from capability_subnet.miner.submit import canonical_body

        compressed = zlib.compress(canonical_body(recipe), chain.COMPRESSION_LEVEL)
        assert T.unframe_payload(T.frame_payload(compressed)) == compressed
        assert zlib.decompress(compressed) == canonical_body(recipe)

    def test_the_epoch_cost_is_charged_at_the_pallet_floor(self, recipe):
        sealed = chain.seal(recipe, FUTURE_RUN)
        assert sealed.epoch_cost == max(len(sealed.ciphertext), C.MIN_COMMIT_SPACE_BYTES)
        assert sealed.commits_per_epoch >= 3

    def test_room_for_three_commitments_in_an_epoch(self, recipe):
        """The property the cap exists for: resubmission stays cheap.

        Two fields would halve this, which is the whole argument against a
        larger recipe limit.
        """
        sealed = chain.seal(recipe, FUTURE_RUN)
        assert sealed.epoch_cost * 3 <= C.MAX_EPOCH_COMMIT_BYTES


class TestRefusalsNameTheLimitAndTheWayOut:
    """A miner should never have to read the pallet to know why we said no."""

    def test_an_oversized_recipe_names_the_limit_and_a_remedy(self, monkeypatch):
        monkeypatch.setattr(C, "MAX_ONCHAIN_RECIPE_BYTES", 64)
        with pytest.raises(chain.CommitError) as caught:
            chain.seal(_recipe(), FUTURE_RUN)
        message = str(caught.value)
        assert "on-chain limit is 64" in message
        assert "adapter_name" in message, "must say what to shorten"
        assert "capcomp canonicalise" in message, "must say how to check"

    def test_a_past_round_is_refused_before_anything_is_disclosed(self):
        """Sealing to a published round publishes the recipe instead of hiding it."""
        with pytest.raises(chain.CommitError) as caught:
            chain.check_round_is_ahead(T.round_at(time.time()) - 1000)
        message = str(caught.value)
        assert "public the instant it is committed" in message
        assert "Nothing was committed" in message
        assert "--subtensor.network" in message, "must point at the likely cause"

    def test_a_round_still_ahead_is_allowed(self):
        chain.check_round_is_ahead(T.round_at(time.time()) + 1000)

    def test_the_budget_refusal_says_when_to_retry(self):
        budget = chain.Budget(used=2_900, limit=C.MAX_EPOCH_COMMIT_BYTES, blocks_to_reset=145)
        with pytest.raises(chain.CommitError) as caught:
            chain.check_budget(budget, chain.seal(_recipe(), FUTURE_RUN))
        message = str(caught.value)
        assert "2,900 of its 3,100" in message
        assert "29 minutes" in message, "a miner needs to know when, not just that"
        assert "no limit on how many times" in message, "the cap is the chain's, not ours"

    def test_a_budget_with_room_is_not_refused(self):
        budget = chain.Budget(used=0, limit=C.MAX_EPOCH_COMMIT_BYTES, blocks_to_reset=145)
        chain.check_budget(budget, chain.seal(_recipe(), FUTURE_RUN))

    @pytest.mark.parametrize(
        "raised,expected",
        [
            ("SpaceLimitExceeded", "no commitment space left this epoch"),
            ("AccountNotAllowedCommit", "not registered on netuid 103"),
            ("TooManyFieldsInCommitmentInfo", "too many fields"),
            ("Inability to pay some fees", "existential balance"),
        ],
    )
    def test_pallet_errors_are_translated(self, raised, expected):
        assert expected in chain._explain(raised, 103)

    def test_an_unrecognised_error_is_passed_through_verbatim(self):
        """Never swallow what we cannot translate."""
        message = chain._explain("Priority is too low", 103)
        assert "Priority is too low" in message
        assert "Nothing was committed" in message


class TestTheRunIsDerivedNotAskedFor:
    """A recipe sealed for one run and committed in another unseals wrongly."""

    def test_mid_run_a_commitment_joins_the_open_run(self):
        opens = run_opens_block(422, C.DEFAULT_RUN_BLOCKS)
        assert chain.run_for_commit(opens + 3600) == 422

    def test_inside_the_settling_window_it_joins_the_next_run(self):
        """The hour before close belongs to the following run, not this one.

        Sealing has to follow that, or a commitment made in the last hour is
        measured in one run while unsealing on another's schedule.
        """
        opens = run_opens_block(422, C.DEFAULT_RUN_BLOCKS)
        close = run_opens_block(423, C.DEFAULT_RUN_BLOCKS)
        assert chain.run_for_commit(opens + 3600) == 422
        assert chain.run_for_commit(close - 100) == 423

    def test_the_derived_run_always_agrees_with_its_own_check(self):
        opens = run_opens_block(422, C.DEFAULT_RUN_BLOCKS)
        for offset in (1, 100, 3600, C.DEFAULT_RUN_BLOCKS - 1):
            block = opens + offset
            chain.check_window(block, chain.run_for_commit(block))

    def test_naming_a_closed_run_explains_what_it_would_do(self):
        block = run_opens_block(422, C.DEFAULT_RUN_BLOCKS) + 3600
        with pytest.raises(chain.CommitError) as caught:
            chain.check_window(block, 419)
        message = str(caught.value)
        assert "joins run 422's field" in message
        assert "reveal round has passed" in message
        assert "without --run" in message, "must name the fix"

    def test_naming_a_future_run_is_refused_too(self):
        block = run_opens_block(422, C.DEFAULT_RUN_BLOCKS) + 3600
        with pytest.raises(chain.CommitError, match="does not open until block"):
            chain.check_window(block, 425)


class TestNothingFailsQuietly:
    """A chain read that does not answer must stop the command, not guess.

    Every default here would fail in the same direction: toward reporting more
    room than the hotkey has, letting a miner spend a block on a commitment the
    pallet then rejects. The budget is only worth reading if a failure to read
    it is louder than a wrong answer.
    """

    class _Chain:
        """A subtensor whose reads can be made to fail or come back empty."""

        block = 1_000

        def __init__(self, fail=None, empty=None):
            self.fail, self.empty = fail or set(), empty or set()

        def query(self, item, params=None):
            name = getattr(item, "name", str(item))
            if name in self.fail:
                raise RuntimeError(f"{name} unavailable")
            if name in self.empty:
                return None
            return {
                "MaxSpace": 3100,
                "SubnetEpochIndex": 5,
                "Tempo": 360,
                "LastEpochBlock": 900,
                "UsedSpaceOf": {"last_epoch": 5, "used_space": 700},
            }[name]

    REQUIRED = ["MaxSpace", "SubnetEpochIndex", "Tempo", "LastEpochBlock"]

    def test_a_healthy_chain_reads_a_budget(self):
        budget = chain.read_budget(self._Chain(), 103, "5Grw")
        assert (budget.used, budget.limit, budget.blocks_to_reset) == (700, 3100, 260)

    @pytest.mark.parametrize("name", REQUIRED)
    def test_a_failed_read_stops_the_command(self, name):
        with pytest.raises(chain.CommitError, match="Nothing was committed"):
            chain.read_budget(self._Chain(fail={name}), 103, "5Grw")

    @pytest.mark.parametrize("name", REQUIRED)
    def test_an_empty_read_stops_the_command(self, name):
        """None is not zero. Reading nothing is not reading a free budget."""
        with pytest.raises(chain.CommitError, match="returned nothing"):
            chain.read_budget(self._Chain(empty={name}), 103, "5Grw")

    def test_a_hotkey_that_never_committed_has_a_full_budget(self):
        """The one empty answer that is an answer, not a failure."""
        budget = chain.read_budget(self._Chain(empty={"UsedSpaceOf"}), 103, "5Grw")
        assert budget.used == 0
        assert budget.remaining == 3100

    def test_a_stale_epoch_means_a_fresh_budget(self):
        """The pallet zeroes used_space lazily, on the next commitment.

        Reading the stored number without the epoch beside it would refuse a
        miner who actually has their whole allowance.
        """
        chain_ = self._Chain()
        original = chain_.query

        def stale(item, params=None):
            value = original(item, params)
            if getattr(item, "name", "") == "UsedSpaceOf":
                return {"last_epoch": 4, "used_space": 3_000}
            return value

        chain_.query = stale
        assert chain.read_budget(chain_, 103, "5Grw").used == 0
