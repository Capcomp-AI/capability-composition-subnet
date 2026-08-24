#!/usr/bin/env python3
"""Miner entry point.

Validates a recipe and, with --confirm, sends it to the submission API.

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
