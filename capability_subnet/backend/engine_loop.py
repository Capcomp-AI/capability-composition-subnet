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
from capability_subnet.backend.scorer.aggregate import percentile, valid_rows
from capability_subnet.backend.scorer.sampler import build_instances, draw_window
from capability_subnet.backend.settings import BackendSettings
from capability_subnet.backend.store import Store
from capability_subnet.backend.weights.weight_writer import graded_top3, winner_take_all
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
    base_e2e: float = 0.0
    incumbent_latency_seconds: float = 0.0
    #: References that could not be measured this window, and why.
    missing_references: dict[str, str] = field(default_factory=dict)
    #: Whether the base model itself was measured. Retention is relative to it,
    #: so without it the retention gate cannot mean anything.
    base_measured: bool = False

    def strongest(self) -> tuple[str, float]:
        return strongest_reference(ref.collapse_single_adapters(self.reference_scores))

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
        return True, "every reference was measured"


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

        self.comparator_config = ComparatorConfig(
            axis_margin=settings.axis_margin,
            axis_tolerance=settings.axis_tolerance,
            min_dominant_axes=settings.min_dominant_axes,
            min_axis_samples=settings.min_axis_samples,
            end_to_end_margin=settings.end_to_end_margin,
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
        state = WindowState(window_id=window_id, hidden_instances=hidden, ood_instances=ood)

        self._measure_references(state, block)
        self.window = state
        return state

    def _measure_references(self, state: WindowState, block: int) -> None:
        """Re-measure every permanent reference and the incumbent.

        This is the expensive part of opening a window, and it is not optional:
        a challenger compared against last window's reference numbers would be
        compared against a different instance set, and the paired statistics
        would be meaningless.
        """
        packages: list[EvaluablePackage] = [
            EvaluablePackage(candidate_id=ref.BASE_MODEL, is_base=True)
        ]
        packages += [
            EvaluablePackage(candidate_id=reference.reference_id, adapter_id=reference.adapter_id)
            for reference in ref.single_adapter_references(self.snapshot)
        ]
        packages += [
            EvaluablePackage(candidate_id=reference.reference_id, recipe=reference.recipe)
            for reference in ref.build_references(self.snapshot)
            if reference.kind == "recipe"
        ]

        for package in packages:
            output = self.evaluator.evaluate(
                package,
                state.hidden_instances,
                state.ood_instances,
                base_e2e=state.base_e2e,
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
                state.base_e2e = output.scores.end_to_end
                state.base_measured = True

            log.info("window %d %s", state.window_id, summarise_evaluation(output))

        self._measure_incumbent(state, block)

    def _measure_incumbent(self, state: WindowState, block: int) -> None:
        """Re-measure the reigning champion on this window's instances."""
        champion = self.store.get_champion()
        if champion is None:
            log.info("no champion holds the throne")
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
            base_e2e=state.base_e2e,
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

        source = LocalRecipeSource(self.settings.state_path / "recipes")
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
            base_e2e=state.base_e2e,
            reference_latency_seconds=state.incumbent_latency_seconds,
            strongest_reference_id=reference_id,
            strongest_reference_score=reference_score,
            end_to_end_margin=self.settings.end_to_end_margin,
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
        from capability_subnet.backend.monitor.anticopy import check_artifact_copy

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

        if not output.gates_passed:
            failed = ", ".join(v.name for v in output.gate_verdicts if not v.passed)
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

        outcome = compare(
            output.hidden_results,
            champion_results,
            state.reference_results.get(reference_id, champion_results),
            axes=self.workflow.critical_axes,
            reference_id=reference_id,
            config=self.comparator_config,
            bootstrap_seed=state.window_id,
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
        )

    # -- weights ------------------------------------------------------------

    def publish_weights(self, block: int) -> None:
        """Compute and store the weight vector validators fetch."""
        window_id = window_id_for_block(block, self.settings.window_blocks)
        champion = self.store.get_champion()
        report_digest = self.store.get_meta("champion_report_sha256")

        if self.settings.incentive_mode == C.MODE_GRADED_TOP3:
            ranked = self._ranked_qualified(window_id)
            vector = graded_top3(
                ranked,
                window_id=window_id,
                block=block,
                spec_version=self.workflow_spec_version,
                burn_percentage=self.settings.burn_percentage,
                burn_uid=self.settings.burn_uid,
                workflow_id=self.workflow.workflow_id,
            )
        else:
            vector = winner_take_all(
                champion,
                window_id=window_id,
                block=block,
                spec_version=self.workflow_spec_version,
                burn_percentage=self.settings.burn_percentage,
                burn_uid=self.settings.burn_uid,
                champion_report_sha256=report_digest,
                workflow_id=self.workflow.workflow_id,
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

    @property
    def workflow_spec_version(self) -> int:
        from capability_subnet import __spec_version__

        return __spec_version__

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
