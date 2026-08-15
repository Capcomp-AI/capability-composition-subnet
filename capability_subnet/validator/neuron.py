"""The validator neuron, in either of its two modes.

``--neuron.evaluation`` decides where a validator's numbers come from, and the
two modes ask for very different machines.

**own** — the default. The validator rebuilds every candidate on its own
hardware, serves it through its own endpoint, and scores it against instances it
regenerates from the pinned corpora. It needs a GPU, an adapter pool and
`capability-subnet[merge]`; what it does *not* need is anyone to trust, which is
the point. Validators are not required to agree on artifact bytes — an SVD is not
bitwise reproducible across devices — so they are compared on outcomes instead.

**delegated** — the thin mode. The validator reconstructs, serves and scores
nothing, and runs on a small VPS: it fetches the signed weight vector an
evaluation engine published, satisfies itself that the vector is trustworthy and
submittable, and sets weights on-chain. What keeps it honest is that it is not a
relay — it verifies the operator signature against an allow-list it controls,
checks the vector against the chain it can see, and burns rather than submitting
anything it cannot verify.

The failure that motivated the start-up preflight below is the seam between them:
the packaging describes the thin mode, the default is the other one, and a
validator that inherited the difference scored every miner zero without ever
saying it could not measure them.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from capability_subnet import __spec_version__
from capability_subnet.common import constants as C
from capability_subnet.common.chain import (
    MetagraphView,
    current_block,
    fetch_metagraph,
    submit_weights,
    window_id_for_block,
)
from capability_subnet.common.config import build_config, parse_trusted_signers
from capability_subnet.common.logging import setup_logging
from capability_subnet.common.signing import SignatureError
from capability_subnet.scoring.weight_vector import apply_validator_burn
from capability_subnet.validator.client import (
    BackendClient,
    BackendUnavailable,
    check_draw_was_not_re_rolled,
    safe_fallback,
    spot_check_window,
    validate_vector,
)

log = logging.getLogger(__name__)


class BurnTargetUnavailable(Exception):
    """Raised when there is no owner UID to route a refused window's share to."""


class ValidatorNeuron:
    """Fetches published weights and sets them on-chain."""

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

        trusted = parse_trusted_signers(self.config.trusted_signers)
        if trusted is None and not getattr(self.config, "allow_unsigned", False):
            raise SystemExit(
                "no operator allow-list configured. Without --backend.trusted_signers "
                "this validator would submit whatever the configured URL returns, so it "
                "refuses to start. Set the operator hotkey(s), or pass "
                "--backend.allow_unsigned to accept that risk deliberately."
            )
        if trusted is None:
            log.warning(
                "running with --backend.allow_unsigned: weight vectors from %s will be "
                "submitted without verifying an operator signature",
                self.config.backend_url,
            )

        self.client = BackendClient(
            self.config.backend_url,
            timeout=self.config.backend_timeout,
            trusted_signers=trusted,
        )

        if self.config.evaluation == "own":
            self._preflight_own_evaluation()

        self.uid = self._resolve_uid()
        self.last_weight_block = 0
        self.should_exit = False

        log.info(
            "validator ready: uid %s on netuid %s, reading %s",
            self.uid,
            self.config.netuid,
            self.config.backend_url,
        )

    def _preflight_own_evaluation(self) -> None:
        """Refuse to start if this host cannot actually measure a candidate.

        In ``own`` mode every number this validator submits comes from work it
        does itself, and each thing checked here fails *quietly* at the moment a
        miner is scored rather than loudly at start-up.

        The reconstruction stack is the sharp one. It is not in the base install
        — the packaging describes a validator as needing no tensor library, which
        was true when ``delegated`` was the only mode and is not true of this one
        — so a validator installed from `capability-subnet` without extras cannot
        import torch. It cannot then even *parse* a committed recipe, because
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
                "candidate. In 'own' mode this validator measures every candidate itself and "
                "needs its own endpoint."
            )

        try:
            # The same import the recipe parser and the reconstructor reach for.
            from capability_subnet.merge_engine.engine import reconstruct  # noqa: F401
        except ImportError as exc:
            problems.append(
                f"the reconstruction stack cannot be imported ({exc}). 'own' mode rebuilds "
                "every candidate locally, and this is not part of the base install. Install "
                "`capability-subnet[merge]`, or run --neuron.evaluation=delegated, which needs "
                "no tensor library."
            )

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
        """One pass: measure or fetch, verify, submit."""
        block = current_block(self.subtensor)
        if not self.should_set_weights(block):
            return

        self.resync()

        if getattr(self.config, "evaluation", "own") == "own":
            self._step_own(block)
            return

        try:
            vector = self.client.fetch_weights()
        except SignatureError as exc:
            # The engine is reachable but the vector cannot be attributed to a
            # trusted operator. Burning is the safe answer: submitting an
            # unverifiable vector would pay whoever produced it.
            log.error("refusing the published vector: %s", exc)
            self._burn(block, reason=str(exc))
            return
        except BackendUnavailable as exc:
            # Burn, do not go quiet. Every other failure here burns; this one
            # returned, which is the one option that is strictly worse than
            # either paying or burning. A validator that stops submitting leaves
            # its previous vector on chain to go stale, so it keeps paying
            # whoever it last named until the chain stops counting it — and then
            # pays nobody while the validator still looks alive.
            #
            # Observed doing exactly that: the engine had published no vector
            # yet, `/weights` answered 404, and three validators sat silent for
            # a day with weights from the previous run still standing.
            log.warning("engine unavailable, burning this pass: %s", exc)
            try:
                self.burn_uid()
            except BurnTargetUnavailable as burn_exc:
                log.error("cannot even burn: %s", burn_exc)
                return
            self._burn(block, reason=str(exc))
            return

        # Staleness is measured against the engine's own window length, not a
        # compiled-in default, so a deployment that tuned the window is judged by
        # the window it actually runs.
        window_blocks = self.client.window_blocks()
        current_window = window_id_for_block(block, window_blocks) if window_blocks else None

        try:
            burn_uid = self.burn_uid()
        except BurnTargetUnavailable as exc:
            log.error("not submitting anything this pass: %s", exc)
            return

        problems = validate_vector(
            vector,
            metagraph_size=self.metagraph.size,
            hotkeys=list(self.metagraph.hotkeys),
            current_window=current_window,
            max_stale_windows=self.config.max_stale_windows,
            burn_uid=burn_uid,
        )

        if problems:
            for problem in problems:
                log.error("weight vector rejected [%s]: %s", problem.code, problem.detail)
            self._burn(block, reason=problems[0].detail)
            return

        # A signature proves the operator produced this vector. It does not
        # prove the evaluation behind it was honest. Before paying, re-score a
        # closed window from the traces the engine itself published: instance
        # generation is a pure function of the seed and the scorer is
        # deterministic, so a score that does not follow from its own trace is
        # caught here — by the party about to pay for it, on a VPS, with no GPU.
        if current_window is not None and self.config.spot_check:
            passed, detail = spot_check_window(self.client, current_window - 1)
            if passed:
                # Replay checks that instances match seeds. It cannot check where the
                # seeds came from, and a root that moves between windows is the
                # operator re-rolling which problems candidates face.
                passed, detail = check_draw_was_not_re_rolled(self.client, current_window - 1)
            if not passed:
                log.error("refusing to pay: %s", detail)
                self._burn(block, reason=detail)
                return
            log.info("spot check %s", detail)

        final = apply_validator_burn(vector, self.config.burn_percentage, burn_uid)
        self._submit(final, block)

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
            spec_version=__spec_version__,
            burn_percentage=self.config.burn_percentage,
            burn_uid=self.burn_uid(),
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
        from capability_subnet.sandbox.model_client import OpenAICompatibleClient
        from capability_subnet.validator.evaluator import evaluate_candidate

        if not self.config.serve_url:
            raise RuntimeError(
                "--neuron.evaluation=own needs --neuron.serve_url: this validator "
                "measures candidates itself and has nowhere to serve them"
            )

        return evaluate_candidate(
            candidate.recipe,
            OpenAICompatibleClient(self.config.serve_url, "candidate"),
            assignment=assignment,
            pool_dir=self.config.pool_dir,
            artifact_dir=f"{self.config.full_path}/artifacts/{candidate.hotkey[:12]}",
            candidate_id=candidate.hotkey,
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
            spec_version=__spec_version__,
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
            version_key=__spec_version__,
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
