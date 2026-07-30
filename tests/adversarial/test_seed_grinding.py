"""The operator choosing which problems a candidate faces.

Every other check in the audit path asks whether the published results follow
from the published instances. None of them asks where the *instances* came from,
and the instance draw derives from a root only the operator holds.

An operator who could try roots until the draw suited a candidate they had
already evaluated would defeat the whole design while passing every replay: the
seeds would be real, the instances would match them, and the scores would follow
from the traces. What was chosen was the draw.
"""

from __future__ import annotations

from capability_subnet.audit.replay import (
    check_draw_is_bound,
    commitments_agree,
)
from capability_subnet.audit.verify import AuditResult
from capability_subnet.common.schemas import WindowDisclosure
from capability_subnet.scoring.sampler import draw_window, root_commitment

ROOT = 918273645


def disclosure(window_id: int, *, root: int = ROOT, beacon: str = "0xblock") -> WindowDisclosure:
    sample = draw_window(window_id, root=root, hidden_count=4, ood_count=1, beacon=beacon)
    return WindowDisclosure(
        workflow_id="lora_merger_logic_v1",
        window_id=window_id,
        closed_at_block=window_id * 100,
        spec_version=2000,
        hidden_seeds=list(sample.hidden_seeds),
        ood_seeds=list(sample.ood_seeds),
        beacon=sample.beacon,
        root_commitment=sample.root_commitment,
    )


class TestGrindingTheRootIsVisible:
    def test_a_different_root_gives_a_different_draw(self):
        """The attack itself: the operator has a free hand over which problems a
        candidate sees, and every seed it produces is genuine."""
        honest = draw_window(9, root=ROOT, hidden_count=8, ood_count=2, beacon="0xb")
        ground = draw_window(9, root=ROOT + 1, hidden_count=8, ood_count=2, beacon="0xb")
        assert honest.hidden_seeds != ground.hidden_seeds

    def test_re_rooting_between_windows_is_caught(self):
        """What makes it detectable. One root produces every window, so a
        commitment that moves is the operator changing the draw in public."""
        run = [disclosure(1), disclosure(2), disclosure(3, root=ROOT + 1)]
        agreed, detail = commitments_agree(run)
        assert not agreed
        assert "re-rooted" in detail

    def test_an_honest_run_agrees(self):
        agreed, detail = commitments_agree([disclosure(w) for w in (1, 2, 3)])
        assert agreed
        assert "one seed root" in detail

    def test_the_commitment_reveals_nothing_about_the_root(self):
        commitment = root_commitment(ROOT)
        assert str(ROOT) not in commitment
        assert commitment.startswith("sha256:")

    def test_a_window_missing_its_commitment_cannot_be_vouched_for(self):
        run = [disclosure(1), disclosure(2)]
        run[1].root_commitment = ""
        agreed, detail = commitments_agree(run)
        assert not agreed
        assert "no commitment" in detail


class TestTheDrawIsBoundToSomethingTheOperatorDoesNotChoose:
    def test_the_beacon_changes_the_draw(self):
        """Even holding the root fixed, the operator cannot precompute a window:
        the block hash is not theirs and is not known until the window opens."""
        a = draw_window(4, root=ROOT, hidden_count=6, ood_count=1, beacon="0xaaa")
        b = draw_window(4, root=ROOT, hidden_count=6, ood_count=1, beacon="0xbbb")
        assert a.hidden_seeds != b.hidden_seeds
        assert a.probe_seed != b.probe_seed

    def test_the_draw_stays_reproducible_for_a_replay(self):
        """Binding must not cost reproducibility — a disputed window has to
        regenerate exactly."""
        first = draw_window(4, root=ROOT, hidden_count=6, ood_count=1, beacon="0xaaa")
        again = draw_window(4, root=ROOT, hidden_count=6, ood_count=1, beacon="0xaaa")
        assert first == again

    def test_an_unbound_window_is_flagged_rather_than_trusted(self):
        naked = disclosure(1)
        naked.beacon = ""
        naked.root_commitment = ""
        result = AuditResult()
        check_draw_is_bound(naked, result)
        codes = {f.code for f in result.findings}
        assert "unbound_draw" in codes
        assert "unbound_seed_root" in codes

    def test_a_bound_window_raises_nothing(self):
        result = AuditResult()
        check_draw_is_bound(disclosure(1), result)
        assert not [f for f in result.findings if f.code.startswith("unbound")]

    def test_the_probe_seed_is_independent_of_the_instance_seeds(self):
        """Learning the retention probe must not reveal the hidden draw: they
        share a root but not a label."""
        sample = draw_window(4, root=ROOT, hidden_count=6, ood_count=1, beacon="0xaaa")
        assert sample.probe_seed not in sample.hidden_seeds
        assert sample.probe_seed not in sample.ood_seeds


class TestACodeProblemCannotBeAnsweredFromItsOwnStatement:
    """Competitive-programming statements print a worked example, and the corpus
    keeps it as a test case. Harmless when other cases follow — passing needs all
    of them. Not harmless when it is the only case: the expected output is then
    printed in the question, and a program that ignores its input and prints that
    constant passes.

    Roughly a quarter of the admitted pool was in that state, which is free marks
    for every package on the axis whose claim is that execution is the stronger
    signal.
    """

    def test_no_admitted_problem_falls_to_a_constant_program(self):
        from capability_subnet.workflows.lora_merger_logic_v1 import dataset

        exploitable = []
        for item in dataset.load_code():
            expected = {case.expected_stdout.strip() for case in item.cases}
            if len(expected) == 1:
                only = next(iter(expected))
                if only and only in item.question:
                    exploitable.append(item.item_id)
        assert not exploitable, f"{len(exploitable)} problems pass without reading the input"

    def test_a_constant_program_no_longer_passes_a_drawn_instance(self):
        """The attack, run: a package that reads nothing and prints the sample."""
        from capability_subnet.sandbox.python_runner import PythonRunner
        from capability_subnet.workflows import get_workflow
        from capability_subnet.workflows.lora_merger_logic_v1.scoring import run_test_cases

        workflow = get_workflow("lora_merger_logic_v1")
        code_seed = next(s for s in range(200, 600) if workflow.generate_instance(s).is_code)
        instance = workflow.generate_instance(code_seed)

        shown = next(
            (
                c.expected_stdout
                for c in instance.cases
                if c.expected_stdout.strip() in instance.question
            ),
            None,
        )
        if shown is None:
            return  # this draw shows no example at all; nothing to copy

        passed, total, _ = run_test_cases(
            f"print({shown.strip()!r})", instance.cases, PythonRunner()
        )
        assert passed < total, "echoing the statement's example still passes every case"


class TestTheBindingIsCheckedAndNotJustPublished:
    """A field nobody verifies is decoration. These cover the difference between
    publishing a binding and enforcing one."""

    def test_a_fabricated_beacon_is_caught_against_the_chain(self):
        """The check with teeth. Presence proves nothing — an operator can print
        any string — so it has to be compared with the real block."""
        from capability_subnet.audit.replay import verify_beacon_against_chain
        from capability_subnet.audit.verify import AuditResult

        class Chain:
            def block_info(self, block):
                return {"block_hash": "0xtrue-hash-of-the-block"}

        lying = disclosure(3)
        lying.beacon = "0xwhatever-i-like"
        result = AuditResult()
        ok, detail = verify_beacon_against_chain(lying, Chain(), window_blocks=100, result=result)
        assert not ok
        assert "fabricated_beacon" in {f.code for f in result.findings}
        assert "not bound to the block it says" in detail

    def test_an_honest_beacon_passes(self):
        from capability_subnet.audit.replay import verify_beacon_against_chain

        class Chain:
            def block_info(self, block):
                return {"block_hash": "0xagreed"}

        honest = disclosure(3)
        honest.beacon = "0xagreed"
        ok, _ = verify_beacon_against_chain(honest, Chain(), window_blocks=100)
        assert ok

    def test_an_unreachable_chain_is_not_treated_as_fraud(self):
        """Same policy as the rest of the audit path: an outage must not be
        indistinguishable from dishonesty."""
        from capability_subnet.audit.replay import verify_beacon_against_chain

        class Down:
            def block_info(self, block):
                raise ConnectionError("no endpoint")

        ok, detail = verify_beacon_against_chain(disclosure(3), Down(), window_blocks=100)
        assert ok
        assert "not checked" in detail

    def test_the_validator_actually_runs_the_commitment_check(self):
        """It was written and never called, which is the same as not having it."""
        from pathlib import Path

        import capability_subnet.validator.neuron as neuron

        source = Path(neuron.__file__).read_text()
        assert "check_draw_was_not_re_rolled" in source

    def test_the_contract_carries_a_real_commitment(self):
        """Threaded from the engine's configured root, not left empty."""
        from capability_subnet.scoring.sampler import root_commitment
        from capability_subnet.workflows import get_workflow

        contract = get_workflow("lora_merger_logic_v1").build_contract(
            seed_root_commitment=root_commitment(4242)
        )
        assert contract["windows"]["seed_root_commitment"] == root_commitment(4242)
