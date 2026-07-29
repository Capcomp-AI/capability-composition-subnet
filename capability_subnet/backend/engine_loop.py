"""The continuous champion-challenge loop.

There is no round ceremony. A champion holds the throne continuously, challengers
are drawn from the queue in commit order, and each is evaluated one at a time
against the incumbent and the permanent references on that window's hidden
instances.

One pass of the loop does exactly this:

1. **Open the window if it has changed.** Draw fresh hidden instances, then
   re-measure every reference and the incumbent on them. Nobody defends on data
   they have already been measured on.
2. **Admit new commitments.** Cheap checks only; nothing touches a GPU.
3. **Take the queue head as challenger.** The role is assigned mechanically from
   commit order and champion state — nobody chooses it.
4. **Evaluate, gate, compare.** A win rewrites the champion record. A decisive
   loss terminates the challenger permanently.
5. **Publish.** A signed report for the evaluation, and a weight vector for
   validators to fetch.

The failure policy differs between the two kinds of failure, and the difference
matters more than any other rule here: a *miner* failure fails closed — an
invalid submission scores zero — while an *infrastructure* failure fails open —
the queue holds rather than terminating a candidate on flaky hardware. A
one-shot-per-hotkey rule is only defensible if the engine never spends that shot
on its own bad night.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from capability_subnet.backend.baselines import references as ref
from capability_subnet.backend.comparator.comparator import (
    ComparatorConfig,
    compare,
    decisive_loss,
    strongest_reference,
)
from capability_subnet.backend.evaluation import (
    EvaluablePackage,
    EvaluationOutput,
    Evaluator,
    summarise_evaluation,
)
from capability_subnet.backend.reports.publisher import (
    ReportPublisher,
    build_report,
    compatibility_record,
)
from capability_subnet.backend.scorer import gates
from capability_subnet.backend.scorer.aggregate import percentile, valid_rows
from capability_subnet.backend.scorer.contribution import (
    ContributionInputs,
    contribution_score,
    explain,
)
from capability_subnet.backend.scorer.retention import build_probe
from capability_subnet.backend.scorer.sampler import build_instances, draw_window
from capability_subnet.backend.settings import BackendSettings
from capability_subnet.backend.store import Store
from capability_subnet.backend.weights.weight_writer import (
    graded_contribution,
    graded_top3,
    winner_take_all,
)
from capability_subnet.common import constants as C
from capability_subnet.common.chain import window_id_for_block
from capability_subnet.common.schemas import ChampionRecord, QueueEntry, Recipe
from capability_subnet.registry.snapshot import PoolSnapshot
from capability_subnet.workflows import WorkflowModule

log = logging.getLogger(__name__)


@dataclass(slots=True)
class WindowState:
    """The measurements a window is decided on."""

    window_id: int
    hidden_instances: list = field(default_factory=list)
    ood_instances: list = field(default_factory=list)
    #: End-to-end completion of every reference and the incumbent.
    reference_scores: dict[str, float] = field(default_factory=dict)
    #: Sample rows, so the comparator can pair against them.
    reference_results: dict[str, list] = field(default_factory=dict)
    incumbent_latency_seconds: float = 0.0
    #: This window's general-capability probe, and the base model's result on
    #: it. Every package in the window is asked the same items so retention is a
    #: paired comparison like every other comparison here.
    probe_items: list = field(default_factory=list)
    base_probe: object = None
    #: References that could not be measured this window, and why.
    missing_references: dict[str, str] = field(default_factory=dict)
    #: Whether the base model itself was measured. Retention is relative to it,
    #: so without it the retention gate cannot mean anything.
    base_measured: bool = False

    def strongest(self) -> tuple[str, float]:
        """The permanent reference a challenger must clear by the absolute margin.

        The incumbent is excluded on purpose. This bar means "composition beat
        what you get without doing any composition research", and it must stay
        where it is: the incumbent is dealt with separately, by the champion
        margin, which decays.
        """
        return strongest_reference(
            ref.collapse_single_adapters(self.reference_scores, include_incumbent=False)
        )

    def bar_is_complete(self) -> tuple[bool, str]:
        """Whether this window can legitimately crown anyone.

        A challenger has to clear the *strongest* reference. If some references
        went unmeasured, the strongest of those that remain may not be the
        strongest that exists, and the bar is quietly lower than the protocol
        says — the exact circumstance in which a weak package takes the throne
        and nobody can see why.

        Holding the challenger costs it a window. Crowning it on an incomplete
        comparison costs the network its guarantee.
        """
        if not self.base_measured:
            return False, "the base model could not be measured, so base retention is unknown"
        if self.missing_references:
            names = ", ".join(sorted(self.missing_references))
            return False, f"{len(self.missing_references)} reference(s) unmeasured: {names}"
        return True, "every reference scheduled for this window was measured"


class EngineLoop:
    """One deployment of the continuous evaluation engine."""

    def __init__(
        self,
        *,
        settings: BackendSettings,
        store: Store,
        snapshot: PoolSnapshot,
        workflow: WorkflowModule,
        evaluator: Evaluator,
        publisher: ReportPublisher,
    ) -> None:
        self.settings = settings
        self.store = store
        self.snapshot = snapshot
        self.workflow = workflow
        self.evaluator = evaluator
        self.publisher = publisher
        self.window: WindowState | None = None
        #: Latest metagraph view, refreshed by the service each pass. None until
        #: the first chain read, and in dry runs.
        self.metagraph = None

        self.comparator_config = ComparatorConfig(
            axis_margin=settings.axis_margin,
            axis_tolerance=settings.axis_tolerance,
            min_dominant_axes=settings.min_dominant_axes,
            min_axis_samples=settings.min_axis_samples,
            end_to_end_margin=settings.end_to_end_margin,
            champion_margin=settings.champion_margin,
            champion_margin_decay_blocks=settings.champion_margin_decay_blocks,
            strict_pareto=settings.strict_pareto,
        )

    # -- windows ------------------------------------------------------------

    def ensure_window(self, block: int) -> WindowState:
        """Open the window covering ``block``, drawing and measuring if new."""
        window_id = window_id_for_block(block, self.settings.window_blocks)

        if self.window is not None and self.window.window_id == window_id:
            return self.window

        log.info("opening window %d at block %d", window_id, block)

        sample = draw_window(
            window_id,
            root=self.settings.hidden_seed_root,
            hidden_count=self.settings.hidden_instances,
            ood_count=self.settings.ood_instances,
        )
        self.store.record_window(
            window_id, block, list(sample.hidden_seeds), list(sample.ood_seeds)
        )

        hidden, ood = build_instances(sample, self.workflow)
        state = WindowState(
            window_id=window_id,
            hidden_instances=hidden,
            ood_instances=ood,
            # Drawn from the window's own seed, so nobody can tune to a fixed
            # probe and an auditor can regenerate exactly what was asked.
            probe_items=build_probe(sample.probe_seed),
        )

        self._measure_references(state, block)
        self.window = state
        return state

    def _reference_packages(self, state: WindowState) -> list[EvaluablePackage]:
        """Which references to measure on this window's instances.

        The full set is the base model, one entry per capability adapter, three
        standard merges and the operator's own recipe — fifteen packages against
        every instance in the draw, run one at a time, before a single challenger
        is touched. At the latency this subnet gates on, that alone can exceed
        the window it is supposed to fit inside, and a window that cannot finish
        never evaluates anybody.

        Only the strongest contenders are re-measured every window. The rest are
        rotated on a fixed schedule derived from the window id, so each is
        refreshed regularly and which ones are refreshed is not something the
        operator chooses. Any reference not measured this window keeps its most
        recent score for reporting but is excluded from the bar, because a bar
        set from a different instance set is not a paired comparison — which is
        the property the whole design rests on.
        """
        packages: list[EvaluablePackage] = [
            EvaluablePackage(candidate_id=ref.BASE_MODEL, is_base=True)
        ]

        singles = [
            EvaluablePackage(candidate_id=reference.reference_id, adapter_id=reference.adapter_id)
            for reference in ref.single_adapter_references(self.snapshot)
        ]
        merges = [
            EvaluablePackage(candidate_id=reference.reference_id, recipe=reference.recipe)
            for reference in ref.build_references(self.snapshot)
            if reference.kind == "recipe"
        ]

        # The merges and the owner recipe are the bar that actually binds — they
        # are what "composition adds value" is measured against — so they are
        # never rotated out.
        packages += merges

        rotation = self.settings.single_adapter_rotation
        if rotation <= 0 or rotation >= len(singles):
            packages += singles
        else:
            offset = (state.window_id * rotation) % max(1, len(singles))
            packages += [singles[(offset + i) % len(singles)] for i in range(rotation)]

        return packages

    def _measure_references(self, state: WindowState, block: int) -> None:
        """Measure this window's reference set and the incumbent.

        Not optional and not reusable across windows: a challenger compared
        against last window's reference numbers would be compared against a
        different instance set, and the paired statistics would be meaningless.
        """
        packages = self._reference_packages(state)

        for package in packages:
            output = self.evaluator.evaluate(
                package,
                state.hidden_instances,
                state.ood_instances,
                probe_items=state.probe_items,
                base_probe=state.base_probe,
                apply_comparison_gates=False,
            )
            if not output.usable:
                log.error(
                    "reference %s could not be measured this window: %s",
                    package.candidate_id,
                    output.infrastructure_error,
                )
                state.missing_references[package.candidate_id] = (
                    output.infrastructure_error or "unknown"
                )
                continue

            state.reference_scores[package.candidate_id] = output.scores.end_to_end
            state.reference_results[package.candidate_id] = output.hidden_results
            self.store.store_samples(state.window_id, package.candidate_id, output.hidden_results)
            self._retain_traces(state.window_id, package.candidate_id, output)

            if package.candidate_id == ref.BASE_MODEL:
                state.base_probe = output.probe
                state.base_measured = True

            log.info("window %d %s", state.window_id, summarise_evaluation(output))

        self._measure_incumbent(state, block)

    def _measure_incumbent(self, state: WindowState, block: int) -> None:
        """Re-measure the reigning champion on this window's instances."""
        champion = self.store.get_champion()
        if champion is None:
            log.info("no champion holds the throne")
            return

        if not self._champion_still_registered(champion):
            # Its UID now belongs to someone else, so no validator will submit a
            # vector naming it and every window burns. Leaving it enthroned would
            # also keep the dethrone bar pinned to a package nobody can be paid
            # for — a deadlock that resolves only by vacating the throne.
            log.warning(
                "champion %s is no longer registered; vacating the throne",
                champion.hotkey[:12],
            )
            self.store.clear_champion(reason="champion deregistered")
            return

        package = self._package_for_champion(champion)
        if package is None:
            log.error(
                "champion %s cannot be rebuilt; it keeps the throne but cannot be "
                "compared against this window",
                champion.candidate_id,
            )
            return

        output = self.evaluator.evaluate(
            package,
            state.hidden_instances,
            state.ood_instances,
            probe_items=state.probe_items,
            base_probe=state.base_probe,
            apply_comparison_gates=False,
        )
        if not output.usable:
            log.error("could not re-measure the incumbent: %s", output.infrastructure_error)
            return

        state.reference_scores[ref.INCUMBENT] = output.scores.end_to_end
        state.reference_results[ref.INCUMBENT] = output.hidden_results
        self.store.store_samples(state.window_id, ref.INCUMBENT, output.hidden_results)

        durations = sorted(row.wall_seconds for row in valid_rows(output.hidden_results))
        state.incumbent_latency_seconds = percentile(durations, 0.5)

        log.info("window %d incumbent %s", state.window_id, summarise_evaluation(output))

    def _champion_still_registered(self, champion: ChampionRecord) -> bool:
        """Whether the champion's hotkey still holds the UID it was crowned on.

        References are never registered and never paid, so they are exempt. With
        no metagraph read yet the answer is "assume yes": vacating a throne on
        missing information would be worse than defending one window too long.
        """
        if champion.is_reference or ref.is_reference(champion.candidate_id):
            return True
        if self.metagraph is None:
            return True
        return self.metagraph.uid_of(champion.hotkey) is not None

    def _retain_traces(self, window_id: int, candidate_id: str, output) -> None:
        """Keep the sampled traces so this window can be re-scored once closed."""
        if not output.sampled_traces:
            return
        self.store.store_traces(
            window_id,
            candidate_id,
            [
                {
                    "instance_id": result.instance_id,
                    "split": result.split,
                    "instance_seed": result.instance_seed,
                    "trace": trace.to_dict(),
                    "result": result.model_dump_json(),
                }
                for result, trace in output.sampled_traces
            ],
        )

    def build_disclosure(self, window_id: int, block: int):
        """Publish a closed window's instances so anyone can re-score them.

        Only closed windows. Disclosing the window currently being evaluated
        would hand its challenger the test it is sitting.
        """
        from capability_subnet.common.schemas import (
            DisclosedInstance,
            InstanceResult,
            WindowDisclosure,
        )

        current = window_id_for_block(block, self.settings.window_blocks)
        if window_id >= current:
            raise ValueError(
                f"window {window_id} has not closed yet (current is {current}); "
                "disclosing it would publish a test that is still being sat"
            )

        stored = self.store.get_window(window_id)
        if stored is None:
            raise ValueError(f"no record of window {window_id}")

        disclosure = WindowDisclosure(
            workflow_id=self.workflow.workflow_id,
            window_id=window_id,
            closed_at_block=block,
            spec_version=self.workflow_spec_version,
            hidden_seeds=list(stored["hidden_seeds"]),
            ood_seeds=list(stored["ood_seeds"]),
            instances=[
                DisclosedInstance(
                    instance_id=row["instance_id"],
                    instance_seed=row["instance_seed"],
                    split=row["split"],
                    candidate_id=row["candidate_id"],
                    claimed_result=InstanceResult.model_validate_json(row["result"]),
                    trace=row["trace"],
                )
                for row in self.store.load_traces(window_id)
            ],
        )

        if self.publisher.keypair is not None:
            from capability_subnet.common.signing import sign_in_place

            sign_in_place(self.publisher.keypair, disclosure)

        log.info(
            "disclosed window %d: %d seeds, %d re-scorable instances",
            window_id,
            len(disclosure.hidden_seeds) + len(disclosure.ood_seeds),
            len(disclosure.instances),
        )
        return disclosure

    def _package_for_champion(self, champion: ChampionRecord) -> EvaluablePackage | None:
        """Rebuild the champion's package from stored state."""
        if champion.is_reference or ref.is_reference(champion.candidate_id):
            for reference in ref.build_references(self.snapshot):
                if reference.reference_id != champion.candidate_id:
                    continue
                if reference.kind == "base":
                    return EvaluablePackage(candidate_id=ref.INCUMBENT, is_base=True)
                return EvaluablePackage(candidate_id=ref.INCUMBENT, recipe=reference.recipe)
            return None

        recipe = self._load_recipe(champion.recipe_sha256)
        if recipe is None:
            return None
        return EvaluablePackage(
            candidate_id=ref.INCUMBENT,
            recipe=recipe,
            hotkey=champion.hotkey,
            uid=champion.uid,
        )

    def _load_recipe(self, recipe_sha256: str | None) -> Recipe | None:
        """Fetch a stored recipe by digest.

        Champions are re-measured every window, so their recipe has to be
        retrievable long after the miner published it. The engine keeps its own
        copy for exactly this reason: a champion whose pointer went dead must not
        silently stop defending.
        """
        if not recipe_sha256:
            return None

        from capability_subnet.backend.monitor.fetch import FetchError, LocalRecipeSource

        # settings.recipe_path, not a path rebuilt from state_dir. Admission
        # writes to the configured directory; re-deriving it here meant that the
        # moment an operator moved recipe_dir, every challenger was held forever
        # and every champion became unrebuildable — with the two paths agreeing
        # on the default, so nothing showed up in testing.
        source = LocalRecipeSource(self.settings.recipe_path)
        try:
            fetched = source.fetch("local", recipe_sha256)
        except FetchError as exc:
            log.error("could not load recipe %s: %s", recipe_sha256[:19], exc)
            return None

        from capability_subnet.backend.monitor.admission import parse_recipe

        recipe, problems = parse_recipe(fetched.raw)
        if recipe is None:
            log.error("stored recipe %s is invalid: %s", recipe_sha256[:19], problems)
        return recipe

    # -- challenger ---------------------------------------------------------

    def evaluate_next_challenger(self, block: int) -> EvaluationOutput | None:
        """Evaluate the queue head, if there is one."""
        entry = self.store.next_challenger()
        if entry is None:
            return None

        state = self.ensure_window(block)
        recipe = self._load_recipe(entry.recipe_sha256)

        if recipe is None:
            # The engine stored this recipe at admission, so failing to load it
            # is the engine's problem. Hold rather than terminate.
            log.error("holding %s: its recipe could not be loaded", entry.hotkey[:12])
            return None

        self.store.set_status(entry.hotkey, "evaluating")

        # Recipe-digest copy check first, before anything touches a GPU. The
        # artifact check still runs after reconstruction — two differently
        # worded recipes can build identical weights — but catching the
        # verbatim case here means a copier costs a cheap hash lookup instead
        # of a full instance sweep.
        from capability_subnet.backend.monitor.anticopy import check_for_copy

        recipe_copy = check_for_copy(
            self.store,
            hotkey=entry.hotkey,
            recipe_sha256=entry.recipe_sha256,
            first_block=entry.first_block,
        )
        if recipe_copy.is_copy:
            log.info("%s: terminated before evaluation (%s)", entry.hotkey[:12], recipe_copy.detail)
            self.store.set_status(entry.hotkey, "terminated", recipe_copy.detail)
            return None

        reference_id, reference_score = state.strongest()

        package = EvaluablePackage(
            candidate_id=entry.hotkey,
            recipe=recipe,
            hotkey=entry.hotkey,
            uid=entry.uid,
        )

        output = self.evaluator.evaluate(
            package,
            state.hidden_instances,
            state.ood_instances,
            probe_items=state.probe_items,
            base_probe=state.base_probe,
            reference_latency_seconds=state.incumbent_latency_seconds,
            strongest_reference_id=reference_id,
            strongest_reference_score=reference_score,
            end_to_end_margin=self.settings.end_to_end_margin,
            require_beat_reference=self.settings.require_beat_reference,
        )

        log.info("window %d %s", state.window_id, summarise_evaluation(output))

        if not output.usable:
            # Infrastructure failure: return the candidate to the queue untouched.
            self.store.set_status(entry.hotkey, "queued", output.infrastructure_error or "")
            return output

        self.store.store_samples(state.window_id, entry.hotkey, output.hidden_results)
        self._retain_traces(state.window_id, entry.hotkey, output)
        if output.artifact_sha256:
            self.store.set_artifact(entry.hotkey, output.artifact_sha256)

        self._decide(entry, recipe, output, state, block, reference_id, reference_score)
        return output

    def _decide(
        self,
        entry: QueueEntry,
        recipe: Recipe,
        output: EvaluationOutput,
        state: WindowState,
        block: int,
        reference_id: str,
        reference_score: float,
    ) -> None:
        """Apply the gates and the dethrone rule, then publish."""
        from capability_subnet.backend.monitor.anticopy import (
            check_artifact_copy,
            is_champion_artifact,
        )

        if is_champion_artifact(self.store, output.artifact_sha256 or ""):
            # Byte-identical to the reigning champion. It cannot beat itself by
            # a margin, so the outcome is settled; saying *why* beats publishing
            # a statistical tie and leaving the reader to infer it.
            self._finish(
                entry,
                recipe,
                output,
                state,
                block,
                verdict="terminated",
                reason="reconstructs to exactly the reigning champion's artifact",
                reference_id=reference_id,
                reference_score=reference_score,
            )
            return

        copy_verdict = check_artifact_copy(
            self.store,
            hotkey=entry.hotkey,
            artifact_sha256=output.artifact_sha256 or "",
            first_block=entry.first_block,
        )

        if copy_verdict.is_copy:
            self._finish(
                entry,
                recipe,
                output,
                state,
                block,
                verdict="terminated",
                reason=copy_verdict.detail,
                reference_id=reference_id,
                reference_score=reference_score,
            )
            return

        # An engine failure and a candidate failure both block a crowning, and
        # only one of them may end a candidate's run. Infrastructure first: a
        # candidate held for an unreadable memory counter keeps its shot, and a
        # candidate that genuinely failed a limit does not get to hide behind a
        # co-occurring engine fault.
        infrastructure = gates.infrastructure_failures(output.gate_verdicts)
        if infrastructure:
            detail = "; ".join(f"{v.name}: {v.detail}" for v in infrastructure)
            log.error("holding %s: %s", entry.hotkey[:12], detail)
            self.store.set_status(entry.hotkey, "queued", f"engine could not evaluate: {detail}")
            return

        candidate_failed = gates.candidate_failures(output.gate_verdicts)
        if candidate_failed or not output.gates_passed:
            failed = ", ".join(v.name for v in candidate_failed) or "no gates were evaluated"
            self._finish(
                entry,
                recipe,
                output,
                state,
                block,
                verdict="terminated",
                reason=f"failed hard gates: {failed}",
                reference_id=reference_id,
                reference_score=reference_score,
            )
            return

        complete, why = state.bar_is_complete()
        if not complete:
            # Treated as infrastructure, not as a verdict: the challenger has not
            # been shown to be worse than anything, so it keeps its one shot and
            # is re-evaluated once the reference set is whole again.
            log.error("holding %s: %s", entry.hotkey[:12], why)
            self.store.set_status(entry.hotkey, "queued", f"incomplete reference set: {why}")
            return

        champion_results = state.reference_results.get(ref.INCUMBENT, [])
        if not champion_results:
            # An empty throne. The challenger still has to clear every gate,
            # including beating the strongest reference, which it just did.
            champion_results = state.reference_results.get(ref.BASE_MODEL, [])

        champion = self.store.get_champion()
        blocks_held = max(0, block - champion.crowned_at_block) if champion else 0

        outcome = compare(
            output.hidden_results,
            champion_results,
            state.reference_results.get(reference_id, champion_results),
            axes=self.workflow.critical_axes,
            reference_id=reference_id,
            config=self.comparator_config,
            bootstrap_seed=state.window_id,
            champion_blocks_held=blocks_held,
        )

        if outcome.dethrones:
            self._crown(entry, recipe, output, state, block, outcome, reference_id, reference_score)
            return

        verdict = "terminated" if decisive_loss(outcome) else "held"
        self._finish(
            entry,
            recipe,
            output,
            state,
            block,
            verdict=verdict,
            reason=outcome.reason,
            comparator=outcome,
            reference_id=reference_id,
            reference_score=reference_score,
        )

    def _crown(
        self,
        entry: QueueEntry,
        recipe: Recipe,
        output: EvaluationOutput,
        state: WindowState,
        block: int,
        outcome,
        reference_id: str,
        reference_score: float,
    ) -> None:
        report = self._report(
            entry,
            recipe,
            output,
            state,
            block,
            verdict="dethrone",
            reason=outcome.reason,
            comparator=outcome,
            reference_id=reference_id,
            reference_score=reference_score,
        )
        digest = self.publisher.publish(report)

        champion = ChampionRecord(
            candidate_id=entry.hotkey,
            hotkey=entry.hotkey,
            uid=entry.uid,
            recipe_sha256=entry.recipe_sha256,
            artifact_sha256=output.artifact_sha256,
            crowned_at_block=block,
            crowned_at_window=state.window_id,
            last_scores=output.scores,
        )

        # Champion record and justifying report in one transaction: state must
        # never claim a champion that no published report supports.
        self.store.set_champion(champion, report=report)
        self.store.set_status(entry.hotkey, "champion", "dethroned the incumbent")
        self.store.set_meta("champion_report_sha256", digest)
        self.store.store_compatibility(
            state.window_id, entry.hotkey, compatibility_record(output, recipe)
        )

        # The new champion is the incumbent for the rest of this window.
        state.reference_scores[ref.INCUMBENT] = output.scores.end_to_end
        state.reference_results[ref.INCUMBENT] = output.hidden_results

        log.info(
            "%s takes the throne in window %d (%s)",
            entry.hotkey[:12],
            state.window_id,
            outcome.reason,
        )

    def _finish(
        self,
        entry: QueueEntry,
        recipe: Recipe,
        output: EvaluationOutput,
        state: WindowState,
        block: int,
        *,
        verdict: str,
        reason: str,
        comparator=None,
        reference_id: str = "",
        reference_score: float = 0.0,
    ) -> None:
        report = self._report(
            entry,
            recipe,
            output,
            state,
            block,
            verdict=verdict,
            reason=reason,
            comparator=comparator,
            reference_id=reference_id,
            reference_score=reference_score,
        )
        self.publisher.publish(report)
        self.store.store_report(report)
        self.store.set_status(entry.hotkey, verdict if verdict != "held" else "queued", reason)
        self.store.store_compatibility(
            state.window_id, entry.hotkey, compatibility_record(output, recipe)
        )

        log.info("%s: %s (%s)", entry.hotkey[:12], verdict, reason)

    def _report(
        self,
        entry: QueueEntry,
        recipe: Recipe,
        output: EvaluationOutput,
        state: WindowState,
        block: int,
        *,
        verdict: str,
        reason: str,
        comparator=None,
        reference_id: str = "",
        reference_score: float = 0.0,
    ):
        return build_report(
            output,
            window_id=state.window_id,
            block=block,
            workflow_id=self.workflow.workflow_id,
            base_revision=self.snapshot.manifest.revision,
            source_snapshot_sha256=self.snapshot.sha256,
            evaluator_image_digest=self.settings.evaluator_image_digest,
            miner_hotkey=entry.hotkey,
            miner_uid=entry.uid,
            recipe_sha256=entry.recipe_sha256,
            baseline_scores=ref.collapse_single_adapters(state.reference_scores),
            strongest_reference_id=reference_id,
            strongest_reference_score=reference_score,
            comparator=comparator,
            verdict=verdict,
            verdict_reason=reason,
            contribution=explain(
                ContributionInputs(
                    scores=output.scores,
                    reference_e2e=reference_score,
                    champion_e2e=state.reference_scores.get(ref.INCUMBENT),
                )
            )
            if output.gates_passed
            else {},
        )

    # -- weights ------------------------------------------------------------

    def publish_weights(self, block: int) -> None:
        """Compute and store the weight vector validators fetch."""
        window_id = window_id_for_block(block, self.settings.window_blocks)
        champion = self.store.get_champion()
        report_digest = self.store.get_meta("champion_report_sha256")

        if self.settings.incentive_mode == C.MODE_GRADED_CONTRIBUTION:
            vector = graded_contribution(
                champion,
                self._leaderboard(window_id),
                window_id=window_id,
                block=block,
                spec_version=self.workflow_spec_version,
                champion_base_share=self.settings.champion_base_share,
                burn_percentage=self.settings.burn_percentage,
                burn_uid=self._burn_uid(),
                champion_report_sha256=report_digest,
                workflow_id=self.workflow.workflow_id,
                tail=self._survival_tail(),
                tail_share=self.settings.tail_share,
            )
        elif self.settings.incentive_mode == C.MODE_GRADED_TOP3:
            ranked = self._ranked_qualified(window_id)
            vector = graded_top3(
                ranked,
                window_id=window_id,
                block=block,
                spec_version=self.workflow_spec_version,
                burn_percentage=self.settings.burn_percentage,
                burn_uid=self._burn_uid(),
                workflow_id=self.workflow.workflow_id,
            )
        else:
            vector = winner_take_all(
                champion,
                window_id=window_id,
                block=block,
                spec_version=self.workflow_spec_version,
                burn_percentage=self.settings.burn_percentage,
                burn_uid=self._burn_uid(),
                champion_report_sha256=report_digest,
                workflow_id=self.workflow.workflow_id,
                tail=self._survival_tail(),
                tail_share=self.settings.tail_share,
            )

        if self.publisher.keypair is not None:
            from capability_subnet.common.signing import sign_in_place

            sign_in_place(self.publisher.keypair, vector)

        self.store.store_weights(vector)
        log.info(
            "published weights for window %d: %s",
            window_id,
            ", ".join(f"uid {e.uid}={e.weight:.3f}" for e in vector.entries),
        )

    def _burn_uid(self) -> int:
        """Where burned emission goes.

        The subnet owner's UID when the metagraph can resolve it. The configured
        value is a fallback for offline and dry runs only: UID 0 is a neuron
        like any other, and paying it is not burning.
        """
        if self.metagraph is not None:
            owner = self.metagraph.owner_uid()
            if owner is not None:
                return owner
            log.warning(
                "the subnet owner holds no UID; falling back to the configured burn uid %d",
                self.settings.burn_uid,
            )
        return self.settings.burn_uid

    def _survival_tail(self) -> list[tuple[int, str]]:
        """Queued miners, in the order they will be evaluated.

        Front of the queue first, because the taper pays the front most and the
        front is what the engine is about to spend a window on. A hotkey that
        has since deregistered is dropped — its UID belongs to someone else now,
        and the validator would refuse the vector for naming it.
        """
        entries = self.store.list_queue(status="queued")
        tail: list[tuple[int, str]] = []
        for entry in entries:
            if entry.uid is None or not entry.hotkey:
                continue
            if self.metagraph is not None and self.metagraph.uid_of(entry.hotkey) != entry.uid:
                continue
            tail.append((entry.uid, entry.hotkey))
        return tail

    @property
    def workflow_spec_version(self) -> int:
        from capability_subnet import __spec_version__

        return __spec_version__

    def _recent_contributors(self, window_id: int) -> list[tuple[int, str, float]]:
        """Everyone whose recent evaluation cleared every hard gate, with a grade.

        Looks back over a bounded number of windows rather than only the current
        one. A candidate's grade is a statement about the window it was measured
        in, and the engine evaluates roughly one candidate per window — so paying
        only the current window would make a miner's reward depend on the
        accident of when the queue reached it. Paying indefinitely would let an
        early result collect rent after the network moved past it.

        Only the best grade per hotkey counts, so a miner cannot accumulate
        share by being measured repeatedly.
        """
        oldest = max(0, window_id - self.settings.contribution_memory_windows + 1)
        best: dict[str, tuple[int, str, float]] = {}

        for window in range(oldest, window_id + 1):
            for digest, report in self.store.list_reports(window_id=window, limit=200):
                del digest
                if report.miner_uid is None or not report.miner_hotkey:
                    continue
                if ref.is_reference(report.candidate_id) or not report.gates_passed:
                    continue

                grade = contribution_score(
                    ContributionInputs(
                        scores=report.scores,
                        reference_e2e=report.strongest_reference_score,
                        champion_e2e=report.baseline_scores.get(ref.INCUMBENT),
                    )
                )
                existing = best.get(report.miner_hotkey)
                if existing is None or grade > existing[2]:
                    best[report.miner_hotkey] = (report.miner_uid, report.miner_hotkey, grade)

        contributors = list(best.values())
        if self.metagraph is not None:
            # A hotkey that deregistered no longer owns its UID, and a vector
            # naming it would be refused by every validator.
            contributors = [
                item for item in contributors if self.metagraph.uid_of(item[1]) == item[0]
            ]
        return contributors

    def _leaderboard(self, window_id: int) -> list[tuple[int, str, float]]:
        """Gate-clearing submissions, best first, with unresolvable gaps tied.

        Ranking on the raw score alone would let a difference smaller than the
        window can resolve decide who gets paid. Recipes are public, so that is
        not a theoretical concern: a copy of the leader with one coefficient
        nudged scores within noise of it and would take the top slot roughly
        half the time. Ties resolve to the earliest commitment, so a copy — which
        is later by construction — has to be measurably better rather than
        luckier.
        """
        from capability_subnet.backend.comparator.comparator import minimum_detectable_effect
        from capability_subnet.backend.scorer.ranking import Submission, rank

        resolvable = minimum_detectable_effect(self.settings.hidden_instances)
        entries = {entry.hotkey: entry for entry in self.store.list_queue()}

        submissions: list[Submission] = []
        for uid, hotkey, grade in self._recent_contributors(window_id):
            entry = entries.get(hotkey)
            submissions.append(
                Submission(
                    uid=uid,
                    hotkey=hotkey,
                    score=grade,
                    first_block=entry.first_block if entry else 0,
                    resolvable=resolvable,
                )
            )

        return [(s.uid, s.hotkey, s.score) for s in rank(submissions)]

    def _ranked_qualified(self, window_id: int) -> list[tuple[int, str]]:
        """Qualified miners for the graded split, best first.

        Only miners whose report for this window passed every gate are included.
        An unqualified miner is never promoted into a free slot — the share burns
        instead.
        """
        ranked: list[tuple[float, int, str]] = []
        for digest, report in self.store.list_reports(window_id=window_id, limit=200):
            del digest
            if report.miner_uid is None or not report.miner_hotkey:
                continue
            if ref.is_reference(report.candidate_id):
                continue
            if not report.gates_passed:
                continue
            ranked.append((report.scores.qualified_score, report.miner_uid, report.miner_hotkey))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [(uid, hotkey) for _, uid, hotkey in ranked[:3]]
