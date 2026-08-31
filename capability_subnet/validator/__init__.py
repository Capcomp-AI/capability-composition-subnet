"""The validator: measure the field, derive the vector, set weights.

Not a thin one. The mode that fetched a signed vector from somebody else's
engine and verified it by replaying traces is gone — see neuron.py for why —
so a validator reconstructs, serves and scores on its own cards and answers for
the numbers it writes.
"""

from capability_subnet.validator.client import (
    BackendClient,
    BackendUnavailable,
    validate_vector,
)

__all__ = ["BackendClient", "BackendUnavailable", "validate_vector"]
