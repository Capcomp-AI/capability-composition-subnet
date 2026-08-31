"""The validator: measure or verify the field, derive the vector, set weights.

Two modes, chosen by ``--neuron.mode`` (see neuron.py). ``local`` reconstructs,
serves and scores every candidate on its own cards. ``endpoint`` sets weights
from a signed vector an engine published, after verifying it against a trusted
allow-list. Either way the validator answers for the numbers it writes.
"""

from capability_subnet.validator.client import (
    BackendClient,
    BackendUnavailable,
    validate_vector,
)

__all__ = ["BackendClient", "BackendUnavailable", "validate_vector"]
