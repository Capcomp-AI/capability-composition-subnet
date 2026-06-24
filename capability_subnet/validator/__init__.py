"""The thin validator: fetch a signed weight vector, verify it, set weights."""

from capability_subnet.validator.client import (
    BackendClient,
    BackendUnavailable,
    validate_vector,
)

__all__ = ["BackendClient", "BackendUnavailable", "validate_vector"]
