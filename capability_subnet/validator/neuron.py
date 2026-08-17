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

import logging
import sys
import time
from pathlib import Path

from capability_subnet.common import constants as C
from capability_subnet.common.chain import (
    MetagraphView,
    current_block,
    fetch_metagraph,
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

        candidates: list[Candidate] = []
        for commitment in read_commitments(self.subtensor, self.config.netuid):
            # Only what stood at the moment the window opened. A commitment made
            # after the beacon existed was made by someone who could already see
            # the instances.
            if commitment.block > opened_at:
                continue
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
            burn_percentage=self.config.burn_percentage,
            burn_uid=self.burn_uid(),
            incentive_mode=getattr(self.config, "incentive_mode", C.MODE_GRADED_TOP3),
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
        self._submit(outcome.weights, block)

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

    def _measure(self, candidate, assignment):
        """Reconstruct, serve and score one candidate on this host."""
        from contextlib import contextmanager
        from urllib.parse import urlsplit

        from capability_subnet.sandbox.model_client import OpenAICompatibleClient
        from capability_subnet.validator.evaluator import evaluate_candidate
        from capability_subnet.validator.serving import CANDIDATE_MODEL, serve_candidate

        if not self.config.serve_url:
            raise RuntimeError(
                "--neuron.evaluation=own needs --neuron.serve_url: this validator "
                "measures candidates itself and has nowhere to serve them"
            )

        bind = urlsplit(self.config.serve_url)

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
                device=getattr(self.config, "device", "cuda"),
                python_executable=getattr(self.config, "serving_python", ""),
                host=bind.hostname or "127.0.0.1",
                port=bind.port or 8000,
            ) as base_url:
                yield OpenAICompatibleClient(base_url, CANDIDATE_MODEL)

        return evaluate_candidate(
            candidate.recipe,
            OpenAICompatibleClient(self.config.serve_url, CANDIDATE_MODEL),
            serve=serve,
            assignment=assignment,
            pool_dir=self.config.pool_dir,
            artifact_dir=f"{self.config.full_path}/artifacts/{candidate.hotkey[:12]}",
            candidate_id=candidate.hotkey,
            # --neuron.device was declared and read by nothing, so every
            # validator merged on the CPU whatever it configured. The trimming
            # methods are ~30x slower there, paid once per candidate per window.
            device=getattr(self.config, "device", "cpu"),
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
