"""A validator's field comes off the chain, and nothing else vouches for it.

The submissions service used to hold the bodies and serve them on a schedule it
chose. A validator checked each body against a digest the same service stored
beside it, which is no check at all if the two were written together. The field
now comes from the commitments pallet: every validator reads the same state,
derives the same field, and refuses the same malformed bodies.

So what these protect is that the derivation is complete and unforgiving. The
field has to span both source runs, because the settling rule holds late
commitments over. A body that does not unseal, parse or hash correctly has to
be refused rather than measured. And the chain being unreadable has to be told
apart from the chain holding nothing, because those produce the same weight
vector and mean opposite things.
"""

from __future__ import annotations

import zlib

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common import timelock as T
from capability_subnet.miner.recipe import new_recipe
from capability_subnet.miner.submit import canonical_body
from capability_subnet.validator.field import FieldError, field_for_run

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
BOB = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


@pytest.fixture(scope="module")
def body():
    from capability_subnet.registry.snapshot import load_snapshot

    return canonical_body(new_recipe(list(load_snapshot().adapter_ids)[:2]))


class Record:
    """One pallet record as the SDK hands it over."""

    def __init__(self, hotkey, payload, run_id, *, block, uid=7):
        self.hotkey, self.uid = hotkey, uid
        self.reveal_round = T.reveal_round_for_run(run_id)
        self.revealed = [(block, "0x" + payload.hex())]


class View:
    """A metagraph snapshot carrying whatever records a test needs."""

    def __init__(self, records, hotkeys=None, block=0):
        self.commitment_records = tuple(records)
        self.hotkeys = list(hotkeys if hotkeys is not None else [r.hotkey for r in records])
        self.block, self.netuid, self.owner_hotkey, self.commitments = block, 103, "", []


def _patched(monkeypatch, view_or_error):
    def fetch(_subtensor, _netuid):
        if isinstance(view_or_error, Exception):
            raise view_or_error
        return view_or_error

    monkeypatch.setattr("capability_subnet.common.chain.fetch_metagraph", fetch)


def _opens(run: int) -> int:
    from capability_subnet.common.chain import run_opens_block

    return run_opens_block(run, C.DEFAULT_RUN_BLOCKS)


class TestBodiesAreCheckedBeforeTheyAreMeasured:
    def test_a_good_body_arrives_as_the_exact_bytes_the_miner_sealed(self, monkeypatch, body):
        record = Record(ALICE, zlib.compress(body), 421, block=_opens(421) + 100)
        _patched(monkeypatch, View([record]))

        field = field_for_run(object(), 422)

        assert [e.hotkey for e in field] == [ALICE]
        assert field[0].recipe_raw == body

    def test_a_body_that_is_not_a_recipe_is_refused(self, monkeypatch):
        record = Record(ALICE, zlib.compress(b'{"not":"a recipe"}'), 421, block=_opens(421) + 100)
        _patched(monkeypatch, View([record]))

        assert field_for_run(object(), 422) == []

    def test_one_bad_body_does_not_cost_the_rest_of_the_field(self, monkeypatch, body):
        """Changed deliberately from the service era.

        A malformed body used to fail the whole field, because the service was
        the only thing that could produce one and a wrong body meant the
        service was wrong. A commitment is the miner's own: one miner sealing
        rubbish is that miner's run to lose, and stopping the run over it would
        hand any miner a way to stop every other.
        """
        good = Record(ALICE, zlib.compress(body), 421, block=_opens(421) + 100)
        bad = Record(BOB, b"not compressed at all", 421, block=_opens(421) + 101, uid=8)
        _patched(monkeypatch, View([good, bad]))

        assert [e.hotkey for e in field_for_run(object(), 422)] == [ALICE]

    def test_a_commitment_sealed_to_another_run_is_not_in_this_field(self, monkeypatch, body):
        record = Record(ALICE, zlib.compress(body), 419, block=_opens(421) + 100)
        _patched(monkeypatch, View([record]))

        assert field_for_run(object(), 422) == []

    def test_a_deregistered_hotkey_is_not_measured(self, monkeypatch, body):
        record = Record(ALICE, zlib.compress(body), 421, block=_opens(421) + 100)
        _patched(monkeypatch, View([record], hotkeys=[BOB]))

        assert field_for_run(object(), 422) == []


class TestTheFieldSpansTwoSourceRuns:
    def test_a_settled_commitment_is_measured_by_the_next_run(self, monkeypatch, body):
        record = Record(ALICE, zlib.compress(body), 421, block=_opens(421) + 100)
        _patched(monkeypatch, View([record]))

        assert len(field_for_run(object(), 422)) == 1

    def test_a_late_commitment_is_held_over_one_run(self, monkeypatch, body):
        """Inside the settling window, so 422 does not measure it and 423 does.

        The half a validator reading only N-1 would drop: the miner's row looks
        submitted and never gets scored.

        Which run holds it is decided when the miner commits, not when the
        field is read. `run_for_commit` applies the settling rule to the commit
        block and `capcomp commit` seals to that run's round, so a commitment
        made inside 422's settling window is sealed to 423's round and opens
        with 423's field. It cannot be re-derived at read time: the pallet
        discards the commit block when it opens a commitment, replacing it with
        the block the reveal landed at.
        """
        record = Record(ALICE, zlib.compress(body), 423, block=_opens(423) + 100)
        _patched(monkeypatch, View([record]))

        assert field_for_run(object(), 423) == []
        assert len(field_for_run(object(), 424)) == 1

    def test_a_hotkey_in_both_source_runs_appears_once(self, monkeypatch, body):
        early = Record(ALICE, zlib.compress(body), 420, block=_opens(420) + 50)
        later = Record(ALICE, zlib.compress(body), 421, block=_opens(421) + 50)
        _patched(monkeypatch, View([early, later]))

        field = field_for_run(object(), 422)
        assert len(field) == 1 and field[0].hotkey == ALICE


class TestRefusalsAreExplained:
    def test_an_unreadable_chain_is_not_an_empty_field(self, monkeypatch):
        """They produce the same weight vector and mean opposite things."""
        _patched(monkeypatch, RuntimeError("no node"))

        with pytest.raises(FieldError, match="could not read netuid"):
            field_for_run(object(), 422)

    def test_a_chain_holding_nothing_is_an_empty_field_not_an_error(self, monkeypatch):
        """A run nobody entered is a real answer, and burns rather than raises."""
        _patched(monkeypatch, View([]))

        assert field_for_run(object(), 422) == []

    def test_a_sealed_commitment_is_not_yet_a_submission(self, monkeypatch):
        """Still timelocked: neither measured nor refused, just not open yet."""

        class Pending:
            hotkey, uid, reveal_round, revealed = ALICE, 7, 1, []

        _patched(monkeypatch, View([Pending()], hotkeys=[ALICE]))

        assert field_for_run(object(), 422) == []


class TestItNeedsNoServiceAtAll:
    def test_the_module_reaches_for_no_http_client(self):
        """The point of the change: nothing to ask, nothing to withhold."""
        import capability_subnet.validator.field as field

        source = __import__("inspect").getsource(field)
        assert "httpx" not in source
        assert "api_url" not in source
        assert "http://" not in source and "https://" not in source
