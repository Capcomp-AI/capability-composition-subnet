#!/usr/bin/env python3
"""Miner entry point.

Validates a recipe and, with --confirm, commits it on chain.

Equivalent to `capcomp commit`, which is the documented way in. This file
exists because Bittensor convention puts a neuron entry point here, and a
miner who looks for one should find the same thing the CLI does rather than
a route that no longer answers.

    python neurons/miner.py \
        --netuid 103 \
        --wallet.name <coldkey> --wallet.hotkey <hotkey> \
        --recipe recipe.json \
        --confirm

Nothing goes on chain and nothing is published anywhere: the recipe travels in
the request body, signed by the hotkey. Run without --confirm first — the miner
prints exactly what it would send, what it would replace, and how many of the
run's attempts it would use, then exits without sending anything.
"""

import sys

from capability_subnet.miner.neuron import main

if __name__ == "__main__":
    sys.exit(main())
