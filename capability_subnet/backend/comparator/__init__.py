"""The champion-challenge dethrone rule and its paired statistics."""

from capability_subnet.backend.comparator.bootstrap import BootstrapResult, paired_bootstrap
from capability_subnet.backend.comparator.comparator import (
    ComparatorConfig,
    compare,
    compare_axis,
    decisive_loss,
    strongest_reference,
)

__all__ = [
    "BootstrapResult",
    "ComparatorConfig",
    "compare",
    "compare_axis",
    "decisive_loss",
    "paired_bootstrap",
    "strongest_reference",
]
