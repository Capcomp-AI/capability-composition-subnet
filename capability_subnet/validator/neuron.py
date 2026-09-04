"""The validator neuron. Two ways to set weights, chosen by ``--neuron.mode``.

``local`` (the default) measures. It rebuilds every candidate on this host's own
GPUs, serves it through a runtime it starts itself, and scores it against
instances regenerated from a block hash. It needs a CUDA device, an adapter pool
and `capability-subnet[merge]`, and it trusts no one - the whole reason those
requirements are not negotiable. This is the stronger claim: every number comes
from work this host did, so agreeing with the other validators is evidence.

``endpoint`` verifies. It sets weights from a vector an operator's evaluation
engine published, after checking the signature against
``--neuron.trusted_signers``, the arithmetic, that the run matches the chain this
host sees, and that a sampled instance re-scores to the published trace. It needs
no GPU, so a validator that cannot afford cards is still a validator the network
has. It is not a relay: anything that fails verification is burned with this
validator's own stake, not the publisher's.

Validators are not required to agree on artifact bytes. An SVD is not bitwise
reproducible across devices, so two honest validators on different cards build
different weights from the same recipe; they are compared on outcomes over a
shared core of instances instead.

Third parties can still audit without a GPU - the disclosure replay tooling in
:mod:`capability_subnet.audit` is exactly that, and it did not need weight-setting
rights to be useful.
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from capability_subnet.common import constants as C
from capability_subnet.common.chain import (
    MetagraphView,
    current_block,
    fetch_metagraph,
    run_id_for_block,
    run_opens_block,
    submit_weights,
)
from capability_subnet.common.config import build_config
from capability_subnet.common.logging import setup_logging
from capability_subnet.scoring.contribution import ContributionInputs, contribution_score
from capability_subnet.validator.client import safe_fallback

log = logging.getLogger(__name__)


class BurnTargetUnavailable(Exception):
    """Raised when there is no owner UID to route a refused run's share to."""


def _champion_grade(outcome, reigning: float | None) -> float | None:
    """The grade that holds the throne once this run is decided.

    The new champion's if one took it, and the incumbent's otherwise. The bar
    does not move on its own: a run nobody wins burns the miner share and
    leaves the throne exactly where it was, so the next challenger faces the
    same grade rather than a rising one.
    """
    if outcome.weights is None or not outcome.weights.champion_hotkey:
        return reigning

    champion = outcome.weights.champion_hotkey
    for evaluation in outcome.usable:
        if evaluation.candidate_id == champion:
            return contribution_score(
                ContributionInputs(scores=evaluation.scores, reference_e2e=outcome.reference_e2e)
            )

    raise RuntimeError(
        f"run {outcome.run_id} paid {champion[:16]}… as champion but produced no "
        "usable evaluation for it; the throne would be recorded at the wrong grade"
    )


class ValidatorNeuron:
    """Sets weights from its own measurement (local mode) or from a verified
    published vector (endpoint mode), chosen by ``--neuron.mode``."""

    def __init__(self, config=None) -> None:
        import bittensor as bt

        self.config = config or build_config("validator")
        setup_logging(
            getattr(self.config, "log_level", "INFO"),
            log_file=f"{self.config.full_path}/validator.log",
        )

        self.wallet = bt.Wallet(
            self.config.wallet_name,
            self.config.wallet_hotkey,
            path=self.config.wallet_path,
        )
        self.subtensor = bt.Subtensor(self.config.network)
        self.metagraph: MetagraphView = fetch_metagraph(self.subtensor, self.config.netuid)

        # Only local mode measures, so only local mode needs the fleet. A host
        # that cannot measure must find that out here rather than one run later,
        # having scored every miner zero for a dependency it was missing.
        self.mode = getattr(self.config, "mode", "local")
        if self.mode == "local":
            self._preflight_own_evaluation()
        else:
            from capability_subnet.common.config import parse_trusted_signers

            self._trusted = parse_trusted_signers(getattr(self.config, "trusted_signers", "") or "")
            if not self._trusted:
                log.warning(
                    "endpoint mode with no --neuron.trusted_signers: any signature "
                    "will be accepted. Set the allow-list before running for real."
                )

        self.uid = self._resolve_uid()
        self.last_weight_block = 0
        self.should_exit = False
        # Endpoint mode reads scores rather than producing them, so it has no
        # use for a measuring fleet - and building one made a GPU-less host log
        # that it was "measuring on 1 device(s)" before doing nothing of the
        # kind. The queue stays empty, and _slot() is never reached.
        self._slots = self._build_slots() if self.mode == "local" else queue.Queue()

        if self.mode == "local":
            log.info(
                "validator ready: uid %s on netuid %s, measuring on %s",
                self.uid,
                self.config.netuid,
                self.config.device,
            )
        else:
            log.info(
                "validator ready: uid %s on netuid %s, reading scores from %s",
                self.uid,
                self.config.netuid,
                self.config.backend_url,
            )

    def _preflight_own_evaluation(self) -> None:
        """Refuse to start if this host cannot actually measure a candidate.

        Every number this validator submits comes from work it does itself, and
        each thing checked here fails *quietly* at the moment a miner is scored
        rather than loudly at start-up.

        The reconstruction stack is the sharp one. It lives behind an extra, so a
        validator installed from `capability-subnet` without them cannot import
        torch. It cannot then even *parse* a committed recipe, because
        validating a Recipe resolves the merge method. And because
        :func:`~capability_subnet.validator.evaluator.evaluate_candidate` treats
        reconstruction failure as a scored outcome, the result is not a crash: it
        is every miner on the subnet scored zero for this validator's own missing
        dependency, logged once at WARNING, and written to the chain as weights.

        A validator that cannot measure must not vote on who deserves emission.
        """
        problems: list[str] = []

        if not self.config.serve_url:
            problems.append(
                "--neuron.serve_url is unset, so there is nowhere to serve a reconstructed "
                "candidate. This validator measures every candidate itself and needs its own "
                "endpoint."
            )

        try:
            # The same import the recipe parser and the reconstructor reach for.
            from capability_subnet.merge_engine.engine import reconstruct  # noqa: F401
        except ImportError as exc:
            problems.append(
                f"the reconstruction stack cannot be imported ({exc}). A validator rebuilds "
                "every candidate locally, and this is not part of the base install. Install "
                "`capability-subnet[merge]` - there is no mode that skips it."
            )

        device = str(getattr(self.config, "device", "cuda"))
        if not device.startswith("cuda"):
            problems.append(
                f"--neuron.device is {device!r}. A validator rebuilds every submission "
                "locally, and the trimming methods run roughly thirty times slower on a CPU - "
                "a queue that takes minutes per candidate on a GPU takes most of a day, so a "
                "CPU validator would fall behind the run it is meant to decide. Set a CUDA "
                "device; there is no mode that measures without one."
            )
        else:
            try:
                import torch

                if not torch.cuda.is_available():
                    problems.append(
                        f"--neuron.device is {device!r} but torch reports no CUDA device on this "
                        "host, so every reconstruction would fail once the run opened."
                    )
            except ImportError:
                # The merge-stack check below already reports this, and saying it
                # twice would read as two separate problems.
                pass

        pool = Path(self.config.pool_dir)
        if not pool.is_dir():
            problems.append(
                f"the certified adapter pool at {pool} does not exist, so no recipe could be "
                "reconstructed. Materialise it with scripts/import_public_adapters.py."
            )

        if problems:
            for problem in problems:
                log.error("preflight: %s", problem)
            raise SystemExit(
                f"refusing to start: {len(problems)} problem(s) would stop this validator "
                "measuring candidates. Every miner would be scored zero for a failure that "
                "belongs to this host."
            )

    def _build_slots(self) -> queue.Queue[tuple[str, int]]:
        """One measuring slot per device: a CUDA device and a port of its own.

        A served candidate reserves almost the whole card, so a slot is a whole
        device rather than a share of one - two packages on one card would
        contend for memory and each would measure the other's footprint as its
        own, which is precisely what the peak-VRAM gate must not depend on.

        Each slot also gets its own port, because the runtimes run at the same
        time and a shared port would have the second one fail to bind and the
        first one answer its requests.
        """
        bind = urlsplit(self.config.serve_url or "http://127.0.0.1:8000")
        base_port = bind.port or 8000
        configured = str(getattr(self.config, "devices", "") or "").strip()
        if configured:
            devices = [d.strip() for d in configured.split(",") if d.strip()]
        else:
            devices = self._all_cuda_devices()

        slots: queue.Queue[tuple[str, int]] = queue.Queue()
        for index, device in enumerate(devices):
            slots.put((device, base_port + index))
        log.info(
            "measuring on %d device(s): %s",
            len(devices),
            ", ".join(f"{d}@{base_port + i}" for i, d in enumerate(devices)),
        )
        return slots

    @staticmethod
    def _all_cuda_devices() -> list[str]:
        """Every CUDA device on this host, as the default measuring fleet.

        Cards are the unit of parallelism: one candidate per card, so a run's
        throughput is the card count. Defaulting to one device left three
        quarters of a four-card host idle unless the operator knew to pass
        --neuron.devices, and the symptom was not an error - just a validator
        that took four times as long and fell behind the run it is meant to
        decide.

        Raises rather than guessing when there is nothing to enumerate. The
        previous fallback returned ``["cuda"]`` on a host with no torch, no
        driver and no card, so a validator with no GPU at all announced that it
        was measuring on one device and then failed a run later, deep in a
        merge, with an error about the device that never existed. A host that
        cannot measure has to hear so at start-up.

        Raises:
            SystemExit: if no CUDA device can be enumerated.
        """
        try:
            import torch

            count = torch.cuda.device_count()
        except Exception as exc:  # noqa: BLE001 - no torch, no CUDA, no driver
            raise SystemExit(
                f"error: --neuron.mode=local needs CUDA devices and torch could not "
                f"enumerate any: {exc}\n"
                "Local mode measures every candidate on this host's GPUs. Install "
                "torch with CUDA support, or run --neuron.mode=endpoint, which reads "
                "scores from an engine and needs no GPU."
            ) from exc
        if count <= 0:
            raise SystemExit(
                "error: --neuron.mode=local needs at least one CUDA device and torch "
                "reports none.\n"
                "Local mode measures every candidate on this host's GPUs. Check the "
                "driver, or run --neuron.mode=endpoint, which needs no GPU."
            )
        return [f"cuda:{index}" for index in range(count)]

    @contextmanager
    def _slot(self) -> Iterator[tuple[str, int]]:
        """Hold one device for the length of one candidate's measurement."""
        device, port = self._slots.get()
        try:
            yield device, port
        finally:
            self._slots.put((device, port))

    def _resolve_uid(self) -> int:
        hotkey = self.wallet.hotkey.ss58_address
        if hotkey not in self.metagraph.hotkeys:
            raise SystemExit(
                f"{hotkey} is not registered on netuid {self.config.netuid}. "
                "Register it with `btcli subnet register` before running a validator."
            )
        return self.metagraph.hotkeys.index(hotkey)

    # -- loop ---------------------------------------------------------------

    def resync(self) -> None:
        try:
            self.metagraph = fetch_metagraph(self.subtensor, self.config.netuid)
        except Exception:  # noqa: BLE001
            log.warning("metagraph resync failed; continuing with the cached view", exc_info=True)

    def burn_uid(self) -> int:
        """Where a refused run's emission goes.

        The subnet owner's UID, resolved from the metagraph every pass. UID 0 is
        not a burn address - it belongs to whichever neuron registered into the
        first slot - so a compiled-in 0 pays that neuron for nothing. When the
        owner holds no UID there is nothing safe to route to, and the validator
        submits nothing at all rather than paying an arbitrary miner.
        """
        uid = self.metagraph.owner_uid()
        if uid is None:
            raise BurnTargetUnavailable(
                f"the subnet owner hotkey {self.metagraph.owner_hotkey[:12]}… holds no UID on "
                f"netuid {self.config.netuid}, so there is no address to burn to"
            )
        return uid

    def should_set_weights(self, block: int) -> bool:
        if self.config.disable_set_weights:
            return False
        return (block - self.last_weight_block) >= self.config.weight_interval

    def step(self) -> None:
        """One pass: measure this run here, then set weights from it."""
        block = current_block(self.subtensor)
        if not self.should_set_weights(block):
            return

        self.resync()
        if self.mode == "local":
            self._step_own(block)
        else:
            self._step_endpoint(block)

    def _step_own(self, block: int) -> None:
        """Measure this run here, and set weights from what was measured.

        The path where the numbers are this validator's own. The run comes
        from the chain's own block hash and the numbers come from this host's
        GPU; the field is fetched, because miners submit to the submission
        service rather than to the chain, and every recipe in it is checked
        against the digest it was stored under before it is measured.

        Fetching the field is not the same as taking somebody's word for a
        score. What arrives is what the miners wrote, hash-checked; what it is
        worth is decided here.
        """
        from capability_subnet.common.chain import block_beacon
        from capability_subnet.common.schemas import Recipe
        from capability_subnet.validator.field import FieldError, FieldPending, field_for_run
        from capability_subnet.validator.run import Candidate, evaluate_run

        run_blocks = C.DEFAULT_RUN_BLOCKS
        run_id = run_id_for_block(block, run_blocks)
        opened_at = run_opens_block(run_id, run_blocks)

        try:
            beacon = block_beacon(self.subtensor, opened_at)
            if not beacon:
                # It answers "" on its own fallback path rather than raising,
                # so an unreadable hash arrived here looking like a value. The
                # open draw refuses an empty beacon further down, but by then
                # the failure reads as a sampler error rather than as a chain
                # this validator could not read.
                raise ValueError("the chain returned no hash for this block")
        except Exception as exc:
            # Without the beacon there is no draw, and a fabricated one would be
            # this validator choosing the test. Burn rather than invent.
            log.error("no beacon for block %s: %s", opened_at, exc)
            self._burn(block, reason=f"no beacon for block {opened_at}: {exc}")
            return

        try:
            fetched = field_for_run(
                self.subtensor,
                run_id,
                netuid=self.config.netuid,
                run_blocks=run_blocks,
            )
        except FieldPending as exc:
            # The commitments are there and the chain has not opened them yet.
            # Setting no weights at all is the whole point: this returns without
            # touching last_weight_block, so the next poll tries again and the
            # run is measured as soon as its field is readable. Burning here
            # would pay nobody for a run that was, in fact, entered.
            log.info("run %s: %s Waiting; nothing set this pass.", run_id, exc)
            return
        except FieldError as exc:
            # The chain is unreadable. Not a miner's malformed commitment -
            # those are refused one at a time and logged, and the rest of the
            # field is still measured - but the node itself being unreachable,
            # which is the operator's problem and not the subnet's.
            #
            # It burns and says so. An empty field and an unreadable one
            # produce the same weight vector, and reporting the second as the
            # first would describe a subnet nobody entered.
            log.error("run %s: could not read the field: %s", run_id, exc)
            self._burn(block, reason=f"could not read run {run_id}'s field: {exc}")
            return

        # field_for_run has already applied the run-membership and settling
        # rules. This is the separate claim: nothing measured here was written
        # after the draw existed. It should be implied by the above, and it is
        # cheap enough to check rather than assume.
        eligible = [e for e in fetched if e.first_block <= opened_at]
        if len(eligible) != len(fetched):
            log.warning(
                "run %s: %d submission(s) postdate the beacon at block %d; not measured",
                run_id,
                len(fetched) - len(eligible),
                opened_at,
            )

        limit = int(getattr(self.config, "max_candidates_per_run", 0) or 0)
        if 0 < limit < len(eligible):
            # The service returns submission order, so this keeps the earliest
            # and says so. A silent truncation reads as "measured everything".
            log.warning(
                "%d candidates eligible, measuring the first %d by commit order; "
                "%d will not be measured or paid this run",
                len(eligible),
                limit,
                len(eligible) - limit,
            )
            eligible = eligible[:limit]

        candidates: list[Candidate] = []
        for entry in eligible:
            try:
                recipe = Recipe.model_validate_json(entry.recipe_raw)
            except Exception as exc:
                # One unparseable recipe is that miner's problem, not the run's.
                # Dropping it is right; dropping it silently is not, because the
                # miner then reads an unpaid run as a scoring decision.
                log.warning(
                    "run %s: %s… submitted a recipe that does not parse (%s); not measured",
                    run_id,
                    entry.hotkey[:12],
                    exc,
                )
                continue
            candidates.append(
                Candidate(
                    uid=entry.uid,
                    hotkey=entry.hotkey,
                    recipe=recipe,
                    first_block=entry.first_block,
                )
            )

        if not candidates:
            # Genuinely empty, and now it means that: anything merely unopened
            # was refused above as pending, and an unreadable chain burned with
            # its own reason. What is left is a run nobody entered.
            log.warning(
                "run %s: no candidates. The chain holds no readable commitment for "
                "runs %s-%s, so this run burns.",
                run_id,
                run_id - 2,
                run_id - 1,
            )

        reigning = self._reigning_grade(run_id)
        outcome = evaluate_run(
            candidates,
            run_id=run_id,
            beacon=beacon,
            hotkey=self.wallet.hotkey.ss58_address,
            block=block,
            measure=self._measure,
            measure_base=self._measure_base,
            burn_share=C.BURN_SHARE,
            burn_uid=self.burn_uid(),
            champion_grade=reigning,
            workers=self._slots.qsize(),
        )
        for who, why in outcome.flagged_peers.items():
            log.warning("peer %s looks inconsistent: %s", who, why)

        log.info(
            "run %s: measured %s of %s candidates on %s instances",
            run_id,
            len(outcome.usable),
            len(candidates),
            len(outcome.assignment),
        )
        # Measured now, paid next run. The report is written first because it
        # is what the next run submits from.
        self._report(outcome, candidates, run_id, reigning)
        self._submit_measured_earlier(run_id, block)

    def _step_endpoint(self, block: int) -> None:
        """Set weights from the scores the engine measured, re-deriving them here.

        The weaker of the two claims, and it is offered because a validator that
        cannot afford four cards is otherwise a validator the network does not
        have. What it is not is a relay: the engine publishes a signed report for
        every candidate - its scores, its gates and its grade - and this
        validator computes the weight vector from that stream itself. It trusts
        the operator for nothing beyond the measurement, and even that it checks:
        every report must be signed by a hotkey on this validator's own
        allow-list, a sampled instance must re-score to its published trace, and
        the run's draw must match the chain. Anything that fails, it burns
        instead - with its own stake, not the publisher's.
        """
        from capability_subnet.common.chain import run_id_for_block
        from capability_subnet.scoring.weight_vector import vector_from_reports
        from capability_subnet.validator.client import (
            BackendClient,
            BackendUnavailable,
            check_draw_was_not_re_rolled,
            safe_fallback,
            spot_check_run,
            validate_vector,
        )

        client = BackendClient(
            self.config.backend_url,
            trusted_signers=self._trusted,
        )
        run_blocks = client.run_blocks() or C.DEFAULT_RUN_BLOCKS

        # The run being paid now, and the measured run whose reports decide it. A
        # recipe committed in run N is measured in N+1 and paid in N+2, so the
        # field paid in the current run was measured in the run before it.
        paying_run = run_id_for_block(block, run_blocks)
        measured_run = paying_run - 1

        try:
            reports = client.fetch_reports(measured_run)
        except BackendUnavailable as exc:
            log.error("endpoint unreachable (%s); leaving the last weights in force", exc)
            return
        if not reports:
            log.warning(
                "run %d has published no reports yet; leaving the last weights in force",
                measured_run,
            )
            return

        # The validator derives the vector rather than fetching one. champion_grade
        # governs only the throne record, which is not submitted on chain, so the
        # emitted weights are correct whatever it is; the reigning grade is carried
        # for the record.
        vector = vector_from_reports(
            reports,
            run_id=paying_run,
            block=block,
            champion_grade=self._reigning_grade(paying_run),
            burn_uid=self.burn_uid(),
            burn_share=C.BURN_SHARE,
        )

        problems = validate_vector(
            vector,
            metagraph_size=len(self.metagraph.hotkeys),
            hotkeys=self.metagraph.hotkeys,
            current_run=paying_run,
            max_stale_runs=C.WEIGHT_LAG_RUNS + 1,
            burn_uid=self.burn_uid(),
        )
        if not problems:
            # Re-scored from the engine's own published traces, and the seed root
            # checked across recent runs. Neither needs a GPU; both refuse a run
            # that contradicts its own record.
            ok, detail = spot_check_run(client, measured_run)
            if not ok:
                problems = [detail]
            else:
                ok, detail = check_draw_was_not_re_rolled(client, measured_run)
                if not ok:
                    problems = [detail]

        if problems:
            for problem in problems:
                log.error("derived vector refused: %s", problem)
            self._submit(safe_fallback(self.burn_uid(), vector), block)
            return

        log.info(
            "run %d: vector derived from %d report(s) for run %d, %d entries",
            paying_run,
            len(reports),
            measured_run,
            len(vector.entries),
        )
        self._submit(vector, block)

    def _run_report_dir(self) -> Path:
        return Path(self.config.full_path) / "runs"

    def _reigning_grade(self, run_id: int) -> float | None:
        """The grade a challenger in ``run_id`` has to beat.

        The throne is a fact about the subnet's history rather than about one
        run, so it is carried in the run reports: each run records the grade of
        whoever holds it once that run has been decided, and the next run reads
        it back. Nothing is synthesised from the current run's own field -
        crowning this run's leader would mean crowning whoever led a field of
        one, and paying the full champion share every run regardless of whether
        anything was taken.

        ``None`` means the throne is empty and this run fills it. That is the
        state before any run has crowned anybody, and only that state: once a
        report records a grade, an unreadable report is an error rather than an
        empty throne, because treating it as empty would pay a field that
        cleared nothing.
        """
        previous = self._run_report_dir() / f"run-{run_id - 1}.json"
        if not previous.exists():
            log.info(
                "no report for run %d; the throne is empty and run %d fills it",
                run_id - 1,
                run_id,
            )
            return None

        payload = json.loads(previous.read_text())
        grade = payload.get("champion_grade")
        if grade is None:
            log.info("run %d crowned nobody; the throne is still empty", run_id - 1)
            return None
        return float(grade)

    def _submit_measured_earlier(self, run_id: int, block: int) -> None:
        """Submit the vector from the run that closed, not the one just measured.

        A weight vector is a statement about a *closed* run's leaderboard. This
        used to submit the vector it had just computed, which is a leaderboard
        still being written: a candidate measured early in the run competes
        against an empty field, one measured late against a full one, and the
        vector moves under both as the queue is worked through. Two validators
        that reached the queue in a different order therefore submitted
        different vectors from the same evidence.

        So the pipeline is three runs deep - committed in N, measured in N+1,
        paid in N+2 - and this is the last step. Read back from the run report
        rather than kept in memory, so a validator restarted between runs pays
        what it measured instead of starting again with nothing.
        """
        from capability_subnet.common.schemas import WeightEntry, WeightVector

        source_run = run_id - C.WEIGHT_LAG_RUNS
        report = self._run_report_dir() / f"run-{source_run}.json"

        try:
            payload = json.loads(report.read_text())
            entries = payload["weights"]
        except (OSError, ValueError, KeyError) as exc:
            # Never invent a vector. A validator that has not measured run
            # source_run - because it was down, or because this is its first
            # run - has no evidence for paying anyone, and burning is what the
            # rest of this class does when the evidence is missing.
            self._burn(
                block,
                reason=(
                    f"run {source_run} is what run {run_id} pays for, and there is no "
                    f"report for it at {report}: {exc}"
                ),
            )
            return

        if not entries:
            self._burn(
                block,
                reason=f"run {source_run} measured nobody who could be paid",
            )
            return

        vector = WeightVector(
            workflow_id=self.config.workflow_id,
            run_id=source_run,
            computed_at_block=block,
            entries=[
                WeightEntry(
                    uid=entry["uid"],
                    hotkey=entry.get("hotkey", ""),
                    weight=entry["weight"],
                    role=entry.get("role", ""),
                )
                for entry in entries
            ],
        )
        log.info(
            "run %d pays for run %d: %s",
            run_id,
            source_run,
            ", ".join(f"uid {e.uid}={e.weight:.4f}" for e in vector.entries),
        )
        self._submit(vector, block)

    def _report(self, outcome, candidates: list, run_id: int, reigning: float | None) -> None:
        """Record what each candidate scored, and why.

        The weight vector says what a miner was paid; on its own it never says
        what it was paid *for*. These are the numbers the decision was made from,
        and without them a miner asking why it earned nothing - and an operator
        asking whether this host is measuring sanely - both have only the answer
        to look at. They are computed either way; the cost here is writing them
        down before they are thrown away.
        """
        uid_of = {c.hotkey: c.uid for c in candidates}
        rows = []
        for evaluation in outcome.evaluations:
            scores = evaluation.scores
            rows.append(
                {
                    "uid": uid_of.get(evaluation.candidate_id),
                    "hotkey": evaluation.candidate_id,
                    "recipe_sha256": evaluation.recipe_sha256,
                    "artifact_sha256": evaluation.artifact_sha256,
                    "usable": bool(evaluation.usable),
                    "error": evaluation.error or "",
                    "qualified_score": round(scores.qualified_score, 6),
                    "end_to_end": round(scores.end_to_end, 6),
                    "stage_balance": round(scores.stage_balance, 6),
                    "ood": round(scores.ood, 6),
                    "retention": round(scores.retention, 6),
                    "token_efficiency": round(scores.token_efficiency, 6),
                    "artifact_efficiency": round(scores.artifact_efficiency, 6),
                    "valid_samples": scores.valid_samples,
                    "total_samples": scores.total_samples,
                }
            )

        for row in sorted(rows, key=lambda r: -r["qualified_score"]):
            if row["usable"]:
                log.info(
                    "  uid %-3s score %.4f  e2e %.3f ood %.3f ret %.3f tok %.3f  (%s/%s instances)",
                    row["uid"],
                    row["qualified_score"],
                    row["end_to_end"],
                    row["ood"],
                    row["retention"],
                    row["token_efficiency"],
                    row["valid_samples"],
                    row["total_samples"],
                )
            else:
                log.info("  uid %-3s NOT MEASURED: %s", row["uid"], row["error"][:160])

        if outcome.weights is not None:
            paid = {e.uid: e.weight for e in outcome.weights.entries}
            log.info(
                "  weights: %s",
                ", ".join(f"uid {u}={w:.4f}" for u, w in sorted(paid.items(), key=lambda x: -x[1])),
            )

        # Written as well as logged: a log rotates, and this is the evidence a
        # miner would ask for weeks later.
        try:
            directory = Path(self.config.full_path) / "runs"
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "run_id": run_id,
                "netuid": self.config.netuid,
                "validator_hotkey": self.wallet.hotkey.ss58_address,
                "instances": len(outcome.assignment),
                # The grade the *next* run's challengers have to beat. The
                # throne outlives a run, and this file is the only thing that
                # carries it across one - a validator restarted between runs
                # reads it back rather than starting from an empty throne and
                # paying a field that cleared nothing.
                "champion_grade": _champion_grade(outcome, reigning),
                "champion_hotkey": (
                    outcome.weights.champion_hotkey if outcome.weights is not None else None
                ),
                "candidates": rows,
                # Enough to rebuild the vector, because the *next* run submits
                # from this file: a validator restarted between runs has
                # nothing else to pay from.
                "weights": (
                    [
                        {
                            "uid": e.uid,
                            "hotkey": e.hotkey,
                            "weight": e.weight,
                            "role": e.role,
                        }
                        for e in outcome.weights.entries
                    ]
                    if outcome.weights is not None
                    else []
                ),
            }
            (directory / f"run-{run_id}.json").write_text(json.dumps(payload, indent=2))
        except Exception:  # noqa: BLE001 - reporting must never stop the run
            # Louder than it was: this file is no longer only evidence. The
            # next run submits its weights from it, so losing it costs a run's
            # emission - which is burned rather than guessed.
            log.error(
                "could not write the run report; run %d will have nothing to pay from "
                "and will burn",
                run_id,
                exc_info=True,
            )

    def _uid_of(self, hotkey: str) -> int | None:
        try:
            return list(self.metagraph.hotkeys).index(hotkey)
        except ValueError:
            return None

    def _measure_base(self, assignment, sample):
        """Measure the permanent reference on this run's draw.

        The base model with no adapter attached, on the same instances and the
        same probe the candidates face.

        Split across the whole fleet rather than run on one card. The reference
        has to finish before any candidate starts - every candidate's retention
        is scored against its probe and every margin against its score - so it
        is the one sweep no amount of candidate concurrency can overlap, and
        leaving three cards idle through it is the whole fleet running at a
        quarter speed for the length of a full instance draw.
        """
        from concurrent.futures import ThreadPoolExecutor

        from capability_subnet.sandbox.model_client import OpenAICompatibleClient
        from capability_subnet.scoring.aggregate import EfficiencyInputs, aggregate_scores
        from capability_subnet.validator.evaluator import measure_base_model
        from capability_subnet.validator.run import BaseMeasurement
        from capability_subnet.validator.serving import BASE_MODEL, serve_candidate
        from capability_subnet.workflows import get_workflow

        if not self.config.serve_url:
            raise RuntimeError(
                "--neuron.evaluation=own needs --neuron.serve_url: this validator "
                "measures the reference itself and has nowhere to serve it"
            )

        bind = urlsplit(self.config.serve_url)
        host = bind.hostname or "127.0.0.1"
        cards = max(1, self._slots.qsize())
        shards = [tuple(assignment.seeds[i::cards]) for i in range(cards)]
        shards = [shard for shard in shards if shard]

        def run(index: int):
            shard = shards[index]
            with self._slot() as (device, port):
                with serve_candidate(
                    None,  # no adapter: this is the base model
                    base_model_path=self.config.base_model_path,
                    device=device,
                    python_executable=getattr(self.config, "serving_python", ""),
                    host=host,
                    port=port,
                ) as base_url:
                    return measure_base_model(
                        OpenAICompatibleClient(base_url, BASE_MODEL),
                        assignment=assignment,
                        probe_seed=sample.probe_seed,
                        seeds=shard,
                        # One shard asks the probe. It is the same forty
                        # questions on the same model on every card.
                        with_probe=index == 0,
                    )

        log.info(
            "measuring the reference over %d instances across %d card(s)",
            len(assignment.seeds),
            len(shards),
        )
        with ThreadPoolExecutor(max_workers=len(shards)) as pool:
            measured = list(pool.map(run, range(len(shards))))

        results = [row for shard_results, _ in measured for row in shard_results]
        probe = measured[0][1]
        if len(results) != len(assignment.seeds):
            # Aggregating a short sweep would quietly report the reference as
            # weaker than it is, which lowers the bar for every candidate.
            raise RuntimeError(
                f"the reference sweep returned {len(results)} of "
                f"{len(assignment.seeds)} instances; refusing to set a bar from it"
            )

        flow = get_workflow(self.config.workflow_id)
        scores = aggregate_scores(
            results,
            [],
            flow.critical_axes,
            retention=1.0,
            efficiency=EfficiencyInputs(artifact_bytes=0),
        )
        return BaseMeasurement(end_to_end=scores.end_to_end, probe=probe)

    def _measure(self, candidate, inputs):
        """Reconstruct, serve and score one candidate on this host.

        ``inputs`` carries the out-of-distribution draw, the probe seed and the
        measured base model. Every one of them used to be absent here: the OOD
        term scored zero for everybody, retention read 1.0 for everybody, and
        the reference was zero, so three of the six scored terms and two of the
        gates were not measuring anything.
        """
        from capability_subnet.sandbox.model_client import OpenAICompatibleClient
        from capability_subnet.validator.evaluator import evaluate_candidate
        from capability_subnet.validator.serving import CANDIDATE_MODEL, serve_candidate

        if not self.config.serve_url:
            raise RuntimeError(
                "--neuron.evaluation=own needs --neuron.serve_url: this validator "
                "measures candidates itself and has nowhere to serve them"
            )

        bind = urlsplit(self.config.serve_url)

        with self._slot() as (device, port):

            @contextmanager
            def serve(artifact_dir: str):
                """Start a runtime holding *this* candidate, and stop it afterwards.

                Handing the scorer a long-lived endpoint instead would measure
                whatever that endpoint already holds, identically for every
                submission.
                """
                with serve_candidate(
                    artifact_dir,
                    base_model_path=self.config.base_model_path,
                    device=device,
                    python_executable=getattr(self.config, "serving_python", ""),
                    host=bind.hostname or "127.0.0.1",
                    port=port,
                ) as base_url:
                    yield OpenAICompatibleClient(base_url, CANDIDATE_MODEL)

            host = bind.hostname or "127.0.0.1"
            return evaluate_candidate(
                candidate.recipe,
                OpenAICompatibleClient(f"{bind.scheme or 'http'}://{host}:{port}", CANDIDATE_MODEL),
                serve=serve,
                assignment=inputs.assignment,
                ood_seeds=inputs.ood_seeds,
                probe_seed=inputs.probe_seed,
                base_probe=inputs.base.probe,
                reference_e2e=inputs.base.end_to_end,
                reference_id=inputs.base.reference_id,
                pool_dir=self.config.pool_dir,
                artifact_dir=f"{self.config.full_path}/artifacts/{candidate.hotkey[:12]}",
                candidate_id=candidate.hotkey,
                # --neuron.device was declared and read by nothing, so every
                # validator merged on the CPU whatever it configured. The trimming
                # methods are ~30x slower there, paid once per candidate per run.
                device=device,
            )

    def _burn(self, block: int, *, reason: str) -> None:
        """Route the whole share to the subnet owner's UID."""
        from capability_subnet.common.schemas import WeightEntry, WeightVector

        try:
            burn_uid = self.burn_uid()
        except BurnTargetUnavailable as exc:
            log.error("cannot burn: %s. Submitting nothing this pass.", exc)
            return

        vector = WeightVector(
            workflow_id=self.config.workflow_id,
            run_id=run_id_for_block(block, C.DEFAULT_RUN_BLOCKS),
            computed_at_block=block,
            entries=[WeightEntry(uid=burn_uid, hotkey="", weight=1.0, role="burn")],
        )
        log.warning("burning this run's share to uid %d: %s", burn_uid, reason)
        self._submit(safe_fallback(burn_uid, vector), block)

    def _submit(self, vector, block: int) -> None:
        uids, weights = vector.as_uid_weight_lists()

        if self.config.disable_set_weights:
            log.info(
                "weight submission disabled; would have set %s",
                ", ".join(f"uid {u}={w:.4f}" for u, w in zip(uids, weights, strict=True)),
            )
            self.last_weight_block = block
            return

        success, message = submit_weights(
            self.subtensor,
            self.wallet,
            self.config.netuid,
            uids,
            weights,
            # Bittensor compares this against the subnet's WeightsVersionKey and
            # rejects anything below it. netuid 103 sets 0, so 0 is accepted; the
            # cost of passing it is that there is no longer a lever to force
            # validators onto a new ruleset, which has to be coordinated instead.
            version_key=0,
        )

        if success:
            self.last_weight_block = block
            log.info(
                "set weights at block %d: %s",
                block,
                ", ".join(f"uid {u}={w:.4f}" for u, w in zip(uids, weights, strict=True)),
            )
        elif message == "rate limited":
            # The chain's own guard. Advancing the marker stops the validator
            # from retrying every pass until the limit clears.
            self.last_weight_block = block
            log.info("weight submission rate-limited; will retry next interval")
        else:
            log.error("weight submission failed: %s", message)

    def run(self) -> None:
        log.info("entering the validator loop")
        consecutive_failures = 0

        while not self.should_exit:
            try:
                self.step()
                consecutive_failures = 0
            except KeyboardInterrupt:
                break
            except Exception:
                consecutive_failures += 1
                log.exception("validator pass failed (%d in a row)", consecutive_failures)
                time.sleep(min(300.0, self.config.poll_interval * consecutive_failures))
                continue

            time.sleep(self.config.poll_interval)

        log.info("validator stopped")


def main(argv: list[str] | None = None) -> int:
    from capability_subnet.common.banner import show

    del argv  # configuration comes from the shared argument parser

    # Before the config is built, so it is on screen even if the config is
    # refused - a validator reading a start-up error is a validator looking at
    # this terminal. stderr only, and only to a terminal, so nothing that reads
    # a validator's output sees it.
    show()
    setup_logging()
    try:
        ValidatorNeuron().run()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
