"""The miner neuron.

A miner's whole footprint is one commitment. There is no axon, no inference to
serve, no query to answer, and nothing to keep running: the neuron validates a
recipe, seals it, writes it on chain signed by the hotkey, and exits.

The recipe goes on chain and nowhere else. It is sealed to the drand round its
run closes at, so it is public storage that nobody can read - including the
operator - until the run that measures it has opened. No coldkey is unlocked
and the commitment costs no fee.

That contract is deliberately narrow. The engine judges the artifact a recipe
reconstructs to, not the process that produced it, so a miner is free to search
however it likes on whatever hardware it likes, and none of that ever reaches
the network.

There is no cap on how many times a hotkey may commit in a run; the last
commitment standing before the settling window is the one measured, and the
pallet's per-epoch byte budget is the only limit. The neuron checks everything
it can check locally and refuses to send without an explicit confirmation.
"""

from __future__ import annotations

import logging
import sys

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
            # Not fatal on its own: the pallet refuses a commitment from an
            # unregistered hotkey, and that is the check that decides. Failing
            # here would refuse a commitment the chain would have taken.
            log.warning("could not check registration; the pallet will check it", exc_info=True)
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

        Delegates to the commit path rather than keeping a second copy of the
        sealing rules, which could disagree with it.
        """
        from capability_subnet.miner.cli import _cmd_commit

        return _cmd_commit(self.config)


def main(argv: list[str] | None = None) -> int:
    del argv  # configuration comes from the shared argument parser
    setup_logging()
    try:
        return MinerNeuron().submit()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
