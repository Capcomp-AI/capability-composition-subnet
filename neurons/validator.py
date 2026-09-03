#!/usr/bin/env python3
"""Validator entry point.

Sets weights on-chain, from scores it either measured or read.

    python neurons/validator.py \
        --netuid <netuid> \
        --wallet.name <coldkey> --wallet.hotkey <hotkey> \
        --neuron.mode endpoint \
        --neuron.backend_url https://<engine-host> \
        --neuron.trusted_signers <operator-hotkey>

Endpoint mode reads the engine's signed per-candidate reports, re-derives the
weight vector from them and needs no GPU. Local mode measures every candidate
here instead; see the validator guide for why it cannot be run at present.

Leaving --backend.trusted_signers empty disables signature enforcement and is
only appropriate for local development.
"""

import sys

from capability_subnet.validator.neuron import main

if __name__ == "__main__":
    sys.exit(main())
