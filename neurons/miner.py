#!/usr/bin/env python3
"""Miner entry point.

Validates a recipe and, with --confirm, commits its digest on-chain.

    python neurons/miner.py \
        --netuid <netuid> \
        --wallet.name <coldkey> --wallet.hotkey <hotkey> \
        --recipe recipe.json \
        --recipe_uri hf:<owner>/<repo>/recipe.json \
        --confirm

One recipe per hotkey is final. Run without --confirm first: the miner prints
exactly what it would commit and exits without touching the chain.
"""

import sys

from capability_subnet.miner.neuron import main

if __name__ == "__main__":
    sys.exit(main())
