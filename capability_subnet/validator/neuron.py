"""The thin validator neuron.

A validator on this subnet does not reconstruct, serve or score anything. It
needs no GPU, no adapter pool and no model. Its job is small and specific: fetch
the signed weight vector the evaluation engine published, satisfy itself that the
vector is trustworthy and submittable, and set weights on-chain.

Keeping validation thin is what makes a centralised evaluation engine workable at
all — running a validator costs a small VPS instead of a GPU host. What keeps it
honest is that the validator is not a relay: it verifies the operator signature
against an allow-list it controls, checks the vector against the chain it can
see, and burns rather than submitting anything it cannot verify.
"""

from __future__ import annotations

import logging
import sys
import time

from capability_subnet import __spec_version__
from capability_subnet.backend.weights.weight_writer import apply_validator_burn
from capability_subnet.common import constants as C
from capability_subnet.common.chain import current_block, submit_weights, window_id_for_block
from capability_subnet.common.config import build_config, parse_trusted_signers
from capability_subnet.common.logging import setup_logging
from capability_subnet.common.signing import SignatureError
from capability_subnet.validator.client import (
    BackendClient,
    BackendUnavailable,
    safe_fallback,
    validate_vector,
)

log = logging.getLogger(__name__)


class ValidatorNeuron:
    """Fetches published weights and sets them on-chain."""

    def __init__(self, config=None) -> None:
        import bittensor as bt

        self.config = config or build_config("validator")
        setup_logging(
            getattr(self.config, "log_level", "INFO"),
            log_file=f"{self.config.neuron.full_path}/validator.log",
        )

        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.Subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)

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

        self.uid = self._resolve_uid()
        self.last_weight_block = 0
        self.should_exit = False

        log.info(
            "validator ready: uid %s on netuid %s, reading %s",
            self.uid,
            self.config.netuid,
            self.config.backend_url,
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
            self.metagraph.sync(subtensor=self.subtensor)
        except Exception:  # noqa: BLE001
            log.warning("metagraph resync failed; continuing with the cached view", exc_info=True)

    def should_set_weights(self, block: int) -> bool:
        if self.config.disable_set_weights:
            return False
        return (block - self.last_weight_block) >= self.config.weight_interval

    def step(self) -> None:
        """One pass: fetch, verify, submit."""
        block = current_block(self.subtensor)
        if not self.should_set_weights(block):
            return

        self.resync()

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
            log.warning("engine unavailable: %s", exc)
            return

        # Staleness is measured against the engine's own window length, not a
        # compiled-in default, so a deployment that tuned the window is judged by
        # the window it actually runs.
        window_blocks = self.client.window_blocks()
        current_window = (
            window_id_for_block(block, window_blocks) if window_blocks else None
        )

        problems = validate_vector(
            vector,
            metagraph_size=int(self.metagraph.n),
            hotkeys=list(self.metagraph.hotkeys),
            current_window=current_window,
            max_stale_windows=self.config.max_stale_windows,
            burn_uid=C.BURN_UID,
        )

        if problems:
            for problem in problems:
                log.error("weight vector rejected [%s]: %s", problem.code, problem.detail)
            self._burn(block, reason=problems[0].detail)
            return

        final = apply_validator_burn(vector, self.config.burn_percentage, C.BURN_UID)
        self._submit(final, block)

    def _burn(self, block: int, *, reason: str) -> None:
        """Route the whole share to the burn UID."""
        from capability_subnet.common.schemas import WeightEntry, WeightVector

        vector = WeightVector(
            workflow_id=self.config.workflow_id,
            window_id=window_id_for_block(block, C.DEFAULT_WINDOW_BLOCKS),
            computed_at_block=block,
            spec_version=__spec_version__,
            entries=[WeightEntry(uid=C.BURN_UID, hotkey="", weight=1.0, role="burn")],
        )
        log.warning("burning this window's share: %s", reason)
        self._submit(safe_fallback(C.BURN_UID, vector), block)

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
