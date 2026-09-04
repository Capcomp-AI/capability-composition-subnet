#!/usr/bin/env python3
"""Miner entry point.

Validates a recipe and, with --confirm, commits it on chain.

Equivalent to `capcomp commit`, which is the documented way in. This file
exists because Bittensor convention puts a neuron entry point here, and a
miner who looks for one should find exactly what the CLI does.

    python neurons/miner.py \
        --netuid 103 \
        --wallet.name <coldkey> --wallet.hotkey <hotkey> \
        --recipe recipe.json \
        --confirm

The recipe goes on chain, sealed: it is written into the commitments pallet
signed by the hotkey, timelocked to the drand round its run closes at, so
nobody can read it - including the operator - until the run that measures it
opens. Run without --confirm first — the miner prints the digest, the sealed
size against the field and epoch limits, which run the commitment would join
and when it would unseal, then exits without sending anything.
"""

import sys

from capability_subnet.miner.neuron import main

if __name__ == "__main__":
    sys.exit(main())
