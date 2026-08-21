"""The validator neuron. It measures, on its own GPU, or it does not run.

A validator rebuilds every candidate on its own hardware, serves it through a
runtime it starts itself, and scores it against instances it regenerates from a
block hash. It needs a CUDA device, an adapter pool and
`capability-subnet[merge]`. What it does not need is anyone to trust, and that is
the whole reason the requirement is not negotiable.

There used to be a second mode that fetched a signed weight vector from an
operator's evaluation engine and verified it by replaying published traces. It
was cheap — a small VPS, no GPU — and it was genuinely not a relay: it checked
the signature against an allow-list and burned rather than submit anything it
could not verify. It is gone anyway, because "verify what one party measured" is
not the same claim as "measure it", and keeping both let the network describe
itself with the stronger claim while running on the weaker one. Every validator
now does the work, so agreeing with the others is evidence rather than a
formality.

Validators are not required to agree on artifact bytes. An SVD is not bitwise
reproducible across devices, so two honest validators on different cards build
different weights from the same recipe; they are compared on outcomes over a
shared core of instances instead.

Third parties can still audit without a GPU — the disclosure replay tooling in
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
    measured_in_window,
    submit_weights,
    window_id_for_block,
)
from capability_subnet.common.config import build_config
from capability_subnet.common.logging import setup_logging
from capability_subnet.validator.client import safe_fallback

log = logging.getLogger(__name__)


class BurnTargetUnavailable(Exception):
    """Raised when there is no owner UID to route a refused window's share to."""


class ValidatorNeuron:
    """Measures every candidate on this host and sets weights from it."""

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

        # Unconditional. There is no mode this validator can fall back to, so a
        # host that cannot measure a candidate must find that out here rather
        # than discover it one window later having scored every miner zero.
        self._preflight_own_evaluation()

        self.uid = self._resolve_uid()
        self.last_weight_block = 0
        self.should_exit = False
        self._slots = self._build_slots()

        log.info(
            "validator ready: uid %s on netuid %s, measuring on %s",
            self.uid,
            self.config.netuid,
            self.config.device,
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
                "`capability-subnet[merge]` — there is no mode that skips it."
            )

        device = str(getattr(self.config, "device", "cuda"))
        if not device.startswith("cuda"):
            problems.append(
                f"--neuron.device is {device!r}. A validator rebuilds every submission "
                "locally, and the trimming methods run roughly thirty times slower on a CPU — "
                "a queue that takes minutes per candidate on a GPU takes most of a day, so a "
                "CPU validator would fall behind the window it is meant to decide. Set a CUDA "
                "device; there is no mode that measures without one."
            )
        else:
            try:
                import torch

                if not torch.cuda.is_available():
                    problems.append(
                        f"--neuron.device is {device!r} but torch reports no CUDA device on this "
                        "host, so every reconstruction would fail once the window opened."
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
        device rather than a share of one — two packages on one card would
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
        --neuron.devices, and the symptom was not an error — just a validator
        that took four times as long and fell behind the run it is meant to
        decide.

        Falls back to the single configured device if torch cannot enumerate
        them, which is the same behaviour as before rather than a new failure.
        """
        try:
            import torch

            count = torch.cuda.device_count()
        except Exception:  # noqa: BLE001 - no torch, no CUDA, no driver
            count = 0
        if count <= 0:
            return ["cuda"]
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
        """Where a refused window's emission goes.

        The subnet owner's UID, resolved from the metagraph every pass. UID 0 is
        not a burn address — it belongs to whichever neuron registered into the
        first slot — so a compiled-in 0 pays that neuron for nothing. When the
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
        """One pass: measure this window here, then set weights from it."""
        block = current_block(self.subtensor)
        if not self.should_set_weights(block):
            return

        self.resync()
        self._step_own(block)

    def _step_own(self, block: int) -> None:
        """Measure this window here, and set weights from what was measured.

        The operator-free path. Nothing is fetched from anybody: the window comes
        from the chain's own block hash, the candidates come from commitments,
        and the numbers come from this host's GPU.
        """
        from capability_subnet.common.chain import block_beacon, read_commitments
        from capability_subnet.validator.window import Candidate, run_window

        window_blocks = C.DEFAULT_WINDOW_BLOCKS
        window_id = window_id_for_block(block, window_blocks)
        opened_at = window_id * window_blocks

        try:
            beacon = block_beacon(self.subtensor, opened_at)
        except Exception as exc:
            # Without the beacon there is no draw, and a fabricated one would be
            # this validator choosing the test. Burn rather than invent.
            log.error("no beacon for block %s: %s", opened_at, exc)
            self._burn(block, reason=f"no beacon for block {opened_at}: {exc}")
            return

        eligible = []
        for commitment in read_commitments(self.subtensor, self.config.netuid):
            # Only what stood at the moment the window opened. A commitment made
            # after the beacon existed was made by someone who could already see
            # the instances.
            if commitment.block > opened_at:
                continue
            # And only what was committed in the window that just closed.
            if not measured_in_window(commitment.block, window_id, window_blocks):
                continue
            eligible.append(commitment)

        limit = int(getattr(self.config, "max_candidates_per_window", 0) or 0)
        if 0 < limit < len(eligible):
            # read_commitments returns commit order, so this keeps the earliest
            # and says so. A silent truncation reads as "measured everything".
            log.warning(
                "%d candidates eligible, measuring the first %d by commit order; "
                "%d will not be measured or paid this window",
                len(eligible),
                limit,
                len(eligible) - limit,
            )
            eligible = eligible[:limit]

        candidates: list[Candidate] = []
        for commitment in eligible:
            recipe = self._resolve(commitment)
            if recipe is None:
                continue
            uid = self._uid_of(commitment.hotkey)
            if uid is None:
                continue
            candidates.append(
                Candidate(
                    uid=uid,
                    hotkey=commitment.hotkey,
                    recipe=recipe,
                    first_block=commitment.block,
                )
            )

        outcome = run_window(
            candidates,
            window_id=window_id,
            beacon=beacon,
            hotkey=self.wallet.hotkey.ss58_address,
            block=block,
            measure=self._measure,
            measure_base=self._measure_base,
            burn_percentage=self.config.burn_percentage,
            burn_uid=self.burn_uid(),
            incentive_mode=getattr(self.config, "incentive_mode", C.MODE_GRADED_TOP3),
            workers=self._slots.qsize(),
        )
        for who, why in outcome.flagged_peers.items():
            log.warning("peer %s looks inconsistent: %s", who, why)

        log.info(
            "window %s: measured %s of %s candidates on %s instances",
            window_id,
            len(outcome.usable),
            len(candidates),
            len(outcome.assignment),
        )
        self._report(outcome, candidates, window_id)
        self._submit(outcome.weights, block)

    def _report(self, outcome, candidates: list, window_id: int) -> None:
        """Record what each candidate scored, and why.

        The weight vector says what a miner was paid; on its own it never says
        what it was paid *for*. These are the numbers the decision was made from,
        and without them a miner asking why it earned nothing — and an operator
        asking whether this host is measuring sanely — both have only the answer
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
                    "  uid %-3s score %.4f  e2e %.3f ood %.3f ret %.3f tok %.3f  "
                    "(%s/%s instances)",
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
            directory = Path(self.config.full_path) / "windows"
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "window_id": window_id,
                "netuid": self.config.netuid,
                "validator_hotkey": self.wallet.hotkey.ss58_address,
                "instances": len(outcome.assignment),
                "candidates": rows,
                "weights": (
                    [
                        {"uid": e.uid, "weight": e.weight, "role": e.role}
                        for e in outcome.weights.entries
                    ]
                    if outcome.weights is not None
                    else []
                ),
            }
            (directory / f"window-{window_id}.json").write_text(json.dumps(payload, indent=2))
        except Exception:  # noqa: BLE001 - reporting must never stop the window
            log.warning("could not write the window report", exc_info=True)

    def _uid_of(self, hotkey: str) -> int | None:
        try:
            return list(self.metagraph.hotkeys).index(hotkey)
        except ValueError:
            return None

    def _resolve(self, commitment):
        """Fetch a committed recipe and check it against its own digest."""
        from capability_subnet.validator.resolve import ResolutionError, resolve_recipe

        try:
            return resolve_recipe(commitment)
        except ResolutionError as exc:
            log.info("uid %s: %s", commitment.hotkey[:8], exc)
            return None

    def _measure_base(self, assignment, sample):
        """Measure the permanent reference on this window's draw.

        The base model with no adapter attached, on the same instances and the
        same probe the candidates face.

        Split across the whole fleet rather than run on one card. The reference
        has to finish before any candidate starts — every candidate's retention
        is scored against its probe and every margin against its score — so it
        is the one sweep no amount of candidate concurrency can overlap, and
        leaving three cards idle through it is the whole fleet running at a
        quarter speed for the length of a full instance draw.
        """
        from concurrent.futures import ThreadPoolExecutor

        from capability_subnet.sandbox.model_client import OpenAICompatibleClient
        from capability_subnet.scoring.aggregate import EfficiencyInputs, aggregate_scores
        from capability_subnet.validator.evaluator import measure_base_model
        from capability_subnet.validator.serving import BASE_MODEL, serve_candidate
        from capability_subnet.validator.window import BaseMeasurement
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
                # methods are ~30x slower there, paid once per candidate per window.
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
            window_id=window_id_for_block(block, C.DEFAULT_WINDOW_BLOCKS),
            computed_at_block=block,
            entries=[WeightEntry(uid=burn_uid, hotkey="", weight=1.0, role="burn")],
        )
        log.warning("burning this window's share to uid %d: %s", burn_uid, reason)
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
    del argv  # configuration comes from the shared argument parser
    setup_logging()
    try:
        ValidatorNeuron().run()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
