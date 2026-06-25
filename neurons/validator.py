#!/usr/bin/env python3
"""Validator entry point.

Fetches the signed weight vector the evaluation engine publishes, verifies it,
and sets weights on-chain. Needs no GPU.

    python neurons/validator.py \
        --netuid <netuid> \
        --wallet.name <coldkey> --wallet.hotkey <hotkey> \
        --backend.url https://<engine-host> \
        --backend.trusted_signers <operator-hotkey>

Leaving --backend.trusted_signers empty disables signature enforcement and is
only appropriate for local development.
"""

import sys

from capability_subnet.validator.neuron import main

if __name__ == "__main__":
    sys.exit(main())
