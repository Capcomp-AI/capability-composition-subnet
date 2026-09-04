"""An unopened field is not an empty one, and must not be paid as one.

A run's commitments unseal ``REVEAL_MARGIN_BLOCKS`` after the run that measures
them opens - an hour. For that hour the chain holds every recipe and a readable
payload for none of them, and the code that reads the field skips a record with
no plaintext. So "nobody entered" and "nothing has opened yet" arrive at the
reader as the same empty list, and they call for opposite responses: the first
burns the run, the second waits an hour and measures it.

The cost of confusing them is a whole run's emission, paid to nobody, for a
field that was fully entered.
"""

from __future__ import annotations

import zlib

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common import timelock as T
from capability_subnet.miner.recipe import new_recipe
from capability_subnet.miner.submit import canonical_body
from capability_subnet.validator.field import FieldError, FieldPending, field_for_run

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
BOB = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


@pytest.fixture(scope="module")
def body():
    from capability_subnet.registry.snapshot import load_snapshot

    return canonical_body(new_recipe(list(load_snapshot().adapter_ids)[:2]))


def _opens(run: int) -> int:
    from capability_subnet.common.chain import run_opens_block

    return run_opens_block(run, C.DEFAULT_RUN_BLOCKS)


class Sealed:
    """A commitment the chain has taken but not yet opened."""

    def __init__(self, hotkey, run_id, *, uid=7):
        self.hotkey, self.uid = hotkey, uid
        self.reveal_round = T.reveal_round_for_run(run_id)
        self.revealed = []


class Opened:
    """A commitment whose plaintext is readable."""

    def __init__(self, hotkey, payload, run_id, *, block, uid=7):
        self.hotkey, self.uid = hotkey, uid
        self.reveal_round = T.reveal_round_for_run(run_id)
        self.revealed = [(block, "0x" + payload.hex())]


class View:
    def __init__(self, records, hotkeys=None):
        self.commitment_records = tuple(records)
        self.hotkeys = list(hotkeys if hotkeys is not None else [r.hotkey for r in records])
        self.block, self.netuid, self.owner_hotkey, self.commitments = 0, 103, "", []


def _patched(monkeypatch, view):
    monkeypatch.setattr("capability_subnet.common.chain.fetch_metagraph", lambda _s, _n: view)


class TestWaitingIsNotBurning:
    def test_a_field_still_sealed_refuses_rather_than_reading_empty(self, monkeypatch):
        """The hour after the measuring run opens. Nothing is readable yet."""
        _patched(monkeypatch, View([Sealed(ALICE, 423)]))

        with pytest.raises(FieldPending, match="not unsealed yet"):
            field_for_run(object(), 424)

    def test_pending_is_not_a_FieldError(self, monkeypatch):
        """The validator catches the two separately and does opposite things.

        As a subclass, ``except FieldError`` would swallow it and burn the run
        - the exact outcome this distinction exists to prevent.
        """
        _patched(monkeypatch, View([Sealed(ALICE, 423)]))

        with pytest.raises(FieldPending):
            field_for_run(object(), 424)
        assert not issubclass(FieldPending, FieldError)

    def test_a_run_nobody_entered_is_still_an_empty_field(self, monkeypatch):
        """No commitments at all. That burns, and should."""
        _patched(monkeypatch, View([]))

        assert field_for_run(object(), 424) == []

    def test_one_still_sealed_holds_the_whole_field(self, monkeypatch, body):
        """Partial opening is not a field either.

        Measuring the half that opened would score a fraction of the run and
        burn the rest, paying the miners whose commitments the chain happened
        to process first.
        """
        opened = Opened(ALICE, zlib.compress(body), 423, block=_opens(423) + 100)
        _patched(monkeypatch, View([opened, Sealed(BOB, 423, uid=8)]))

        with pytest.raises(FieldPending):
            field_for_run(object(), 424)

    def test_once_everything_is_open_the_field_is_measured(self, monkeypatch, body):
        _patched(
            monkeypatch,
            View([Opened(ALICE, zlib.compress(body), 423, block=_opens(423) + 100)]),
        )

        field = field_for_run(object(), 424)
        assert [e.hotkey for e in field] == [ALICE]


class TestOnlyThisRunsReveals:
    def test_a_commitment_sealed_to_another_run_does_not_block_this_one(self, monkeypatch, body):
        """The hostage case, and the reason this keys on the round.

        A miner who seals to the wrong run leaves a commitment that never opens
        on any schedule this run cares about. Waiting on "any sealed record"
        would stall every later run behind it, permanently, on one miner's
        mistake.
        """
        stray = Sealed(BOB, 430, uid=8)  # opens long after run 424 is paid
        opened = Opened(ALICE, zlib.compress(body), 423, block=_opens(423) + 100)
        _patched(monkeypatch, View([opened, stray]))

        field = field_for_run(object(), 424)
        assert [e.hotkey for e in field] == [ALICE]

    def test_a_stray_seal_alone_leaves_the_field_empty_not_pending(self, monkeypatch):
        """Same rule with nothing else present: it burns rather than waits."""
        _patched(monkeypatch, View([Sealed(BOB, 430, uid=8)]))

        assert field_for_run(object(), 424) == []
