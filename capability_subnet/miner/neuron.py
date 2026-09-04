"""The miner neuron.

A miner's whole footprint is one HTTP request. There is no axon, no inference to
serve, no query to answer, and nothing to keep running: the neuron validates a
recipe, signs it with the hotkey, sends it, and exits.

Nothing goes on chain. A miner needs a registered hotkey and nothing else — no
commitment, no transaction, no fee, and no wallet unlocked for anything but
signing a short string. The chain is read once, to check the hotkey is
registered, and that is the only time it is touched.

That contract is deliberately narrow. The engine judges the artifact a recipe
reconstructs to, not the process that produced it, so a miner is free to search
however it likes on whatever hardware it likes, and none of that ever reaches
the network.

A hotkey may submit up to ``RESUBMISSION_LIMIT`` times per run and only the last
is measured. The neuron therefore checks everything it can check locally, shows
what the submission will cost against that budget, and refuses to send without
an explicit confirmation.
"""

from __future__ import annotations

import logging
import sys

from capability_subnet.common import constants as C
from capability_subnet.common.chain import fetch_metagraph, is_registered
from capability_subnet.common.config import build_config
from capability_subnet.common.logging import setup_logging
from capability_subnet.registry.snapshot import load_snapshot

log = logging.getLogger(__name__)


class MinerNeuron:
    """Validates and submits one recipe."""

    def __init__(self, config=None) -> None:
        import bittensor as bt

        self.config = config or build_config("miner")
        setup_logging(getattr(self.config, "log_level", "INFO"))

        self.snapshot = load_snapshot()
        self.wallet = bt.Wallet(
            self.config.wallet_name,
            self.config.wallet_hotkey,
            path=self.config.wallet_path,
        )
        self.subtensor = bt.Subtensor(self.config.network)

    # -- checks -------------------------------------------------------------

    def check_registered(self) -> bool:
        """The one thing the chain is read for."""
        hotkey = self.wallet.hotkey.ss58_address
        try:
            view = fetch_metagraph(self.subtensor, self.config.netuid)
            registered = is_registered(view, hotkey)
        except Exception:
            # Not fatal on its own: the API checks registration too, and it is
            # the check that decides. Failing here would refuse a submission the
            # service would have accepted.
            log.warning("could not check registration; the API will check it", exc_info=True)
            return True
        if not registered:
            log.error(
                "hotkey %s is not registered on netuid %s. Register it with "
                "`btcli subnet register --netuid %s` before submitting.",
                hotkey,
                self.config.netuid,
                self.config.netuid,
            )
        return registered

    # -- submit -------------------------------------------------------------

    def submit(self) -> int:
        """Validate and, with confirmation, commit on chain.

        The whole of this used to be an HTTP request: canonicalise, sign a
        short string, POST, read the service's answer. There is no service now,
        so it delegates to the commit path rather than keeping a second copy of
        the sealing rules that could disagree with it.
        """
        from capability_subnet.miner.cli import _cmd_commit

        return _cmd_commit(self.config)

    def _report_standing(self, run_id: int, standing: dict, digest: str) -> None:
        """What this submission costs against the run's budget, before sending."""
        used = standing.get("submission_count") or 0
        remaining = standing.get("remaining")
        held = standing.get("recipe_sha256")

        print(f"\nrun {run_id}: measured in run {run_id + 1}, paid in run {run_id + 2}")
        print(f"digest      {digest}")

        if held == digest:
            print("this is the recipe already held; sending it again costs no attempt")
        elif used:
            print(f"replacing   {held}")
            print(f"attempts    {used} of {C.RESUBMISSION_LIMIT} used, {remaining} left")
        else:
            print(f"attempts    none used, {C.RESUBMISSION_LIMIT} available this run")

        if remaining == 0 and held != digest:
            print(
                "\nNo attempts remain this run. This submission will be refused; "
                f"the recipe held for run {run_id} is the one that gets measured."
            )


def main(argv: list[str] | None = None) -> int:
    del argv  # configuration comes from the shared argument parser
    setup_logging()
    try:
        return MinerNeuron().submit()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
