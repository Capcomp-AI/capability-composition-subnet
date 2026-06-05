"""Protocol constants.

Everything in this module is consensus-relevant: miners, the evaluation backend
and validators must all agree on these values. Changing any of them changes how
recipes are reconstructed or scored and therefore requires a spec-version bump
and a coordinated network upgrade.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Protocol identity
# ---------------------------------------------------------------------------

#: Version of the recipe schema miners submit.
RECIPE_SCHEMA_VERSION: Final[int] = 1

#: Prefix written into every on-chain commitment payload so the monitor can tell
#: this subnet's commitments apart from anything else stored under the same key.
COMMITMENT_PREFIX: Final[str] = "capsub"

#: Version tag inside the commitment payload. Bumping it lets the monitor accept
#: old and new payload shapes during a rollout.
COMMITMENT_VERSION: Final[str] = "v1"

#: The single workflow shipped in V1.
DEFAULT_WORKFLOW_ID: Final[str] = "industrial_maintenance_de_v1"

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

BASE_MODEL_REPO: Final[str] = "Qwen/Qwen3-8B"

#: The exact upstream revision is pinned in registry/data/base_manifest.json
#: before genesis. Loading the manifest is the only supported way to read it;
#: this constant exists so error messages can name the file.
BASE_MANIFEST_FILENAME: Final[str] = "base_manifest.json"
ADAPTER_REGISTRY_FILENAME: Final[str] = "adapter_registry.json"

# ---------------------------------------------------------------------------
# Canonical adapter specification
# ---------------------------------------------------------------------------

#: Every certified source adapter is normalised to this rank before admission.
#: One rank keeps linear merging, TIES/DARE behaviour, output-size comparison
#: and resource measurement directly comparable across candidates.
CANONICAL_RANK: Final[int] = 64
CANONICAL_LORA_ALPHA: Final[int] = 64
CANONICAL_LORA_DROPOUT: Final[float] = 0.0
CANONICAL_BIAS: Final[str] = "none"
CANONICAL_DTYPE: Final[str] = "bfloat16"
CANONICAL_PEFT_TYPE: Final[str] = "LORA"

CANONICAL_TARGET_MODULES: Final[tuple[str, ...]] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Decoder depth of the pinned base model. Used to validate layer indices and to
#: build the fixed layer groups a recipe may address. The authoritative value
#: lives in the base manifest; this constant must match it, and the registry
#: raises at load time if it does not.
NUM_HIDDEN_LAYERS: Final[int] = 36

# ---------------------------------------------------------------------------
# Layer groups
# ---------------------------------------------------------------------------
# Recipes address coefficients per fixed group rather than per arbitrary tensor
# name. There are always four contiguous groups, splitting the decoder stack
# into quarters. The group *names* are part of the protocol and never change;
# the layer ranges behind them follow the pinned base model's depth, so
# repinning to a model of different depth does not invalidate existing recipes.

NUM_LAYER_GROUPS: Final[int] = 4

LAYER_GROUP_NAMES: Final[tuple[str, ...]] = tuple(
    f"group_{index}" for index in range(NUM_LAYER_GROUPS)
)


def build_layer_groups(num_hidden_layers: int = NUM_HIDDEN_LAYERS) -> dict[str, tuple[int, int]]:
    """Split ``num_hidden_layers`` into the four named, contiguous groups.

    Remainder layers are distributed to the earliest groups, so group sizes
    differ by at most one and every layer belongs to exactly one group.
    """
    if num_hidden_layers < NUM_LAYER_GROUPS:
        raise ValueError(
            f"a {num_hidden_layers}-layer model cannot be split into "
            f"{NUM_LAYER_GROUPS} layer groups"
        )

    base_size, remainder = divmod(num_hidden_layers, NUM_LAYER_GROUPS)
    groups: dict[str, tuple[int, int]] = {}
    start = 0
    for index, name in enumerate(LAYER_GROUP_NAMES):
        size = base_size + (1 if index < remainder else 0)
        groups[name] = (start, start + size - 1)
        start += size
    return groups


#: Layer ranges for the currently pinned base model.
LAYER_GROUPS: Final[dict[str, tuple[int, int]]] = build_layer_groups()


def layer_group_of(layer_index: int, groups: dict[str, tuple[int, int]] | None = None) -> str:
    """Return the group name that owns ``layer_index``.

    Raises:
        ValueError: if the index falls outside the pinned base model's depth.
    """
    resolved = LAYER_GROUPS if groups is None else groups
    for name, (low, high) in resolved.items():
        if low <= layer_index <= high:
            return name
    highest = max(high for _, high in resolved.values())
    raise ValueError(f"layer index {layer_index} is outside the base model (0..{highest})")


# ---------------------------------------------------------------------------
# Allowed merge methods
# ---------------------------------------------------------------------------
# Names mirror the canonical PEFT combination types so that a recipe can be
# cross-checked against a reference implementation. Every method that produces a
# concatenated or full-rank delta is followed by a mandatory SVD reduction back
# to the declared output rank.

MERGE_LINEAR: Final[str] = "linear"
MERGE_SVD: Final[str] = "svd"
MERGE_CAT_SVD: Final[str] = "cat_svd"
MERGE_TIES_SVD: Final[str] = "ties_svd"
MERGE_DARE_TIES_SVD: Final[str] = "dare_ties_svd"
MERGE_DARE_LINEAR_SVD: Final[str] = "dare_linear_svd"
MERGE_MAGNITUDE_PRUNE_SVD: Final[str] = "magnitude_prune_svd"

ALLOWED_MERGE_METHODS: Final[tuple[str, ...]] = (
    MERGE_LINEAR,
    MERGE_SVD,
    MERGE_CAT_SVD,
    MERGE_TIES_SVD,
    MERGE_DARE_TIES_SVD,
    MERGE_DARE_LINEAR_SVD,
    MERGE_MAGNITUDE_PRUNE_SVD,
)

#: Methods that consume a ``density`` parameter (they drop a fraction of the
#: delta entries before combining).
DENSITY_METHODS: Final[frozenset[str]] = frozenset(
    {
        MERGE_TIES_SVD,
        MERGE_DARE_TIES_SVD,
        MERGE_DARE_LINEAR_SVD,
        MERGE_MAGNITUDE_PRUNE_SVD,
    }
)

#: Methods whose randomness depends on ``random_seed``.
STOCHASTIC_METHODS: Final[frozenset[str]] = frozenset(
    {MERGE_DARE_TIES_SVD, MERGE_DARE_LINEAR_SVD}
)

#: Sign-election strategies available to TIES-family merges.
ALLOWED_SIGN_METHODS: Final[tuple[str, ...]] = ("total", "frequency")

# ---------------------------------------------------------------------------
# Recipe parameter bounds
# ---------------------------------------------------------------------------

ADAPTER_WEIGHT_MIN: Final[float] = -2.0
ADAPTER_WEIGHT_MAX: Final[float] = 2.0

DENSITY_MIN: Final[float] = 0.05
DENSITY_MAX: Final[float] = 1.0

ALLOWED_OUTPUT_RANKS: Final[tuple[int, ...]] = (8, 16, 32, 48, 64, 96, 128)

SVD_CLAMP_QUANTILE_MIN: Final[float] = 0.90
SVD_CLAMP_QUANTILE_MAX: Final[float] = 1.00

RANDOM_SEED_MIN: Final[int] = 0
RANDOM_SEED_MAX: Final[int] = 4_294_967_295

#: A recipe must select at least two adapters — a single-adapter "merge" is one
#: of the reference baselines, not a candidate.
MIN_SELECTED_ADAPTERS: Final[int] = 2
MAX_SELECTED_ADAPTERS: Final[int] = 12

# ---------------------------------------------------------------------------
# Hard deployment gates
# ---------------------------------------------------------------------------

MAX_ARTIFACT_BYTES: Final[int] = 500 * 1024 * 1024  # 500 MB
MAX_PEAK_VRAM_GB: Final[float] = 24.0
MAX_P95_WORKFLOW_SECONDS: Final[float] = 30.0
MAX_AGENT_TURNS: Final[int] = 12
MAX_OUTPUT_TOKENS: Final[int] = 8192

#: Relative retention floor against the unmodified base model on a held-out
#: general-capability probe. A candidate that trades away general ability for
#: workflow score is rejected.
BASE_RETENTION_FLOOR: Final[float] = 0.98

#: A candidate must record zero critical unsafe actions.
MAX_CRITICAL_UNSAFE_ACTIONS: Final[int] = 0

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
# Quality dominates efficiency: a cheap but unreliable artifact cannot win.

WEIGHT_END_TO_END: Final[float] = 0.60
WEIGHT_STAGE_BALANCE: Final[float] = 0.15
WEIGHT_OOD: Final[float] = 0.10
WEIGHT_RETENTION: Final[float] = 0.05
WEIGHT_LATENCY: Final[float] = 0.05
WEIGHT_ARTIFACT_EFFICIENCY: Final[float] = 0.05

QUALIFIED_SCORE_WEIGHTS: Final[dict[str, float]] = {
    "end_to_end": WEIGHT_END_TO_END,
    "stage_balance": WEIGHT_STAGE_BALANCE,
    "ood": WEIGHT_OOD,
    "retention": WEIGHT_RETENTION,
    "latency": WEIGHT_LATENCY,
    "artifact_efficiency": WEIGHT_ARTIFACT_EFFICIENCY,
}

#: Floor applied inside the geometric mean so a single zero stage does not make
#: the whole balance term identically zero and destroy ranking information.
STAGE_BALANCE_EPSILON: Final[float] = 1e-3

# ---------------------------------------------------------------------------
# Champion-challenge comparator
# ---------------------------------------------------------------------------

#: Per-axis absolute margin a challenger must clear to count as dominant.
DEFAULT_AXIS_MARGIN: Final[float] = 0.02

#: Relative band within which a challenger counts as not-worse on an axis.
DEFAULT_AXIS_TOLERANCE: Final[float] = 0.01

#: Number of axes a challenger must dominate under the partial-Pareto rule.
DEFAULT_MIN_DOMINANT_AXES: Final[int] = 1

#: Minimum paired samples on an axis before its verdict is trusted. An axis with
#: fewer valid samples counts as worse.
DEFAULT_MIN_AXIS_SAMPLES: Final[int] = 20

#: Absolute end-to-end completion margin over the strongest reference, in points
#: of completion rate.
DEFAULT_END_TO_END_MARGIN: Final[float] = 0.03

#: One-sided paired bootstrap settings for the end-to-end comparison.
BOOTSTRAP_RESAMPLES: Final[int] = 10_000
BOOTSTRAP_CONFIDENCE: Final[float] = 0.95

# ---------------------------------------------------------------------------
# Continuous loop timing
# ---------------------------------------------------------------------------

#: Blocks per evaluation window. At 12s blocks this is roughly 24 hours. Hidden
#: instances are resampled and the champion re-measured once per window.
DEFAULT_WINDOW_BLOCKS: Final[int] = 7200

#: Hidden instances drawn per window for the canonical comparison.
DEFAULT_HIDDEN_INSTANCES: Final[int] = 100

#: Additional out-of-distribution instances drawn per window.
DEFAULT_OOD_INSTANCES: Final[int] = 30

#: Public pack size shipped to miners for offline search and debugging.
PUBLIC_PACK_INSTANCES: Final[int] = 120

#: In-flight evaluation budget per serving host.
DEFAULT_DISPATCH_BUDGET: Final[int] = 600

# ---------------------------------------------------------------------------
# Incentive
# ---------------------------------------------------------------------------

MODE_WINNER_TAKE_ALL: Final[str] = "winner_take_all"
MODE_GRADED_TOP3: Final[str] = "graded_top3"

#: Emission split used when the graded mode is enabled. Unfilled shares are
#: burned rather than redistributed to unqualified miners.
GRADED_TOP3_SHARES: Final[tuple[float, float, float]] = (0.60, 0.25, 0.15)

#: UID that receives burned emission.
BURN_UID: Final[int] = 0

#: Fraction of emission routed to the burn UID as an operational safety valve.
DEFAULT_BURN_PERCENTAGE: Final[float] = 0.0

# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------

SANDBOX_TEMPERATURE: Final[float] = 0.0
SANDBOX_TOP_P: Final[float] = 1.0

#: Wall-clock ceiling for a single hidden instance, including model generation
#: and every tool call.
SANDBOX_INSTANCE_TIMEOUT_SECONDS: Final[float] = 300.0

#: Ceiling for one generated diagnostic Python execution.
PYTHON_RUNNER_TIMEOUT_SECONDS: Final[float] = 20.0

#: Ceiling for one generated SQL statement against the hidden snapshot.
SQL_TIMEOUT_SECONDS: Final[float] = 10.0
