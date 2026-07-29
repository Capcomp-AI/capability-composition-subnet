"""The miner neuron.

A miner's on-chain footprint is one commitment. There is no axon, no inference to
serve, no request to answer, and nothing to keep running: the neuron validates a
recipe, publishes its digest, and exits.

That is the whole protocol contract, and it is deliberately narrow. The engine
judges the artifact a recipe reconstructs to, not the process that produced it —
so a miner is free to search however it likes, on whatever hardware it likes, and
none of that ever touches the network.

One recipe per hotkey is final. The neuron therefore refuses to commit without an
explicit confirmation, and checks everything it can check locally first.
"""

from __future__ import annotations

import logging
import sys

from capability_subnet.common.chain import fetch_metagraph, is_registered, write_commitment
from capability_subnet.common.commitments import CommitmentError, encode_commitment
from capability_subnet.common.config import build_config
from capability_subnet.common.logging import setup_logging
from capability_subnet.miner.recipe import RecipeError, advise, check_recipe, describe, load_recipe
from capability_subnet.registry.snapshot import load_snapshot

log = logging.getLogger(__name__)


class MinerNeuron:
    """Validates and commits one recipe."""

    def __init__(self, config=None) -> None:
        import bittensor as bt

        self.config = config or build_config("miner")
        setup_logging(getattr(self.config, "log_level", "INFO"))
        self._graph = None

        self.snapshot = load_snapshot()
        self.wallet = bt.Wallet(
            self.config.wallet_name,
            self.config.wallet_hotkey,
            path=self.config.wallet_path,
        )
        self.subtensor = bt.Subtensor(self.config.network)

    # -- checks -------------------------------------------------------------

    def check_registered(self) -> bool:
        registered = is_registered(self._metagraph(), self.wallet.hotkey.ss58_address)
        if not registered:
            log.error(
                "%s is not registered on netuid %s. Register it with "
                "`btcli subnet register` before committing.",
                self.wallet.hotkey.ss58_address,
                self.config.netuid,
            )
        return registered

    def _metagraph(self):
        """Read the metagraph once per invocation and reuse it.

        The miner is a one-shot command, so a single read serves both the
        registration check and the existing-commitment check. Two reads could
        disagree, and disagreeing about whether a hotkey is registered is not a
        thing a submission path should do.
        """
        if self._graph is None:
            self._graph = fetch_metagraph(self.subtensor, self.config.netuid)
        return self._graph

    def already_committed(self) -> str | None:
        """This hotkey's existing commitment, if it has one.

        One recipe per hotkey is enforced by the engine at admission, so a second
        commitment is not an error on-chain — it simply will not be evaluated.
        Checking first turns a silently wasted transaction into a clear message.
        """
        try:
            view = self._metagraph()
        except Exception:  # noqa: BLE001
            log.warning("could not read existing commitments", exc_info=True)
            return None

        hotkey = self.wallet.hotkey.ss58_address
        for commitment in view.commitments:
            if commitment.hotkey == hotkey:
                return commitment.payload.recipe_sha256
        return None

    # -- submit -------------------------------------------------------------

    def submit(self) -> int:
        """Validate and, with confirmation, commit.

        Returns:
            A process exit code. Non-zero means nothing was committed.
        """
        if not self.config.recipe:
            log.error("no recipe given. Pass --recipe path/to/recipe.json")
            return 2
        if not self.config.recipe_uri:
            log.error(
                "no recipe URI given. Publish the exact recipe bytes at an immutable "
                "location and pass --recipe_uri"
            )
            return 2

        try:
            recipe = load_recipe(self.config.recipe)
        except RecipeError as exc:
            log.error("%s", exc)
            return 2

        problems = check_recipe(recipe, self.snapshot)
        if problems:
            for problem in problems:
                log.error("recipe problem: %s", problem)
            return 2

        for note in advise(recipe, self.snapshot):
            log.warning("advisory: %s", note)

        digest = recipe.digest()

        try:
            payload = encode_commitment(recipe.workflow_id, digest, self.config.recipe_uri)
        except CommitmentError as exc:
            log.error("%s", exc)
            return 2

        if not self.check_registered():
            return 3

        existing = self.already_committed()
        if existing and existing != digest:
            log.error(
                "this hotkey has already committed recipe %s. One recipe per hotkey is "
                "final; register a new hotkey to submit a different package.",
                existing[:19],
            )
            return 4
        if existing == digest:
            log.info("this recipe is already committed; nothing to do")
            return 0

        print(describe(recipe, self.snapshot))
        print(f"\ncommitment payload ({len(payload.encode())} bytes):\n  {payload}")

        if not self.config.confirm:
            print(
                "\nNothing was committed. Re-run with --confirm to submit.\n"
                "This is final: the hotkey gets one evaluation."
            )
            return 0

        success, message = write_commitment(
            self.subtensor,
            self.wallet,
            self.config.netuid,
            workflow_id=recipe.workflow_id,
            recipe_sha256=digest,
            recipe_uri=self.config.recipe_uri,
        )

        if not success:
            log.error("commitment failed: %s", message)
            return 5

        log.info("committed %s", digest)
        print(
            "\nCommitted. The engine will admit the submission on its next chain read, "
            "then evaluate it when it reaches the head of the queue.\n"
            f"Track it at {self.config.backend_url}/queue/{self.wallet.hotkey.ss58_address}"
        )
        return 0


def main(argv: list[str] | None = None) -> int:
    del argv  # configuration comes from the shared argument parser
    setup_logging()
    try:
        return MinerNeuron().submit()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
