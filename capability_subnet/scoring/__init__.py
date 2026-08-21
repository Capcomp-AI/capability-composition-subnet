"""How a package is judged, ranked and paid.

Everything here decides a published number. It is public and stays public: a
score that cannot be recomputed by the party about to pay for it is a claim, not
a measurement. A validator re-scores a closed window with this code, a miner
scores a candidate locally with the same code, and an auditor checks that a
published verdict follows from the published components.

The evaluation engine that *runs* these rules — its loop, its API, its store, its
serving and its configuration — is operator-only and lives outside this package.
Nothing here imports it.
"""

from capability_subnet.scoring.aggregate import aggregate_scores, valid_rows
from capability_subnet.scoring.bootstrap import BootstrapResult, paired_bootstrap
from capability_subnet.scoring.comparator import (
    ComparatorConfig,
    compare,
    compare_axis,
    decayed_champion_margin,
    decisive_loss,
    minimum_detectable_effect,
    strongest_reference,
)
from capability_subnet.scoring.contribution import (
    ContributionInputs,
    contribution_score,
    explain,
)
from capability_subnet.scoring.ranking import Submission, rank, tied
from capability_subnet.scoring.references import (
    ReferencePackage,
    build_references,
    bar_scores,
    is_reference,
)
from capability_subnet.scoring.sampler import WindowSample, build_instances, draw_window
from capability_subnet.scoring.weight_vector import (
    apply_validator_burn,
    graded_contribution,
    graded_top3,
    winner_take_all,
)

__all__ = [
    "BootstrapResult",
    "ComparatorConfig",
    "ContributionInputs",
    "ReferencePackage",
    "Submission",
    "WindowSample",
    "aggregate_scores",
    "apply_validator_burn",
    "build_instances",
    "build_references",
    "bar_scores",
    "compare",
    "compare_axis",
    "contribution_score",
    "decayed_champion_margin",
    "decisive_loss",
    "draw_window",
    "explain",
    "graded_contribution",
    "graded_top3",
    "is_reference",
    "minimum_detectable_effect",
    "paired_bootstrap",
    "rank",
    "strongest_reference",
    "tied",
    "valid_rows",
    "winner_take_all",
]
