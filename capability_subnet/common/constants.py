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

#: The workflow a launch runs unless configured otherwise.
#:
#: The arena rather than the maintenance chain, because it is the one that can
#: answer whether a merge beat its constituents: one turn, pre-measured item
#: difficulty, nothing judged by a model. The maintenance workflow remains
#: registered and selectable — it demonstrates what composition is *for*, but its
#: oracle needs ten of twelve turns and no adapter in the pool covers German or
#: SQL, so a null result on it measures calibration rather than composition.
DEFAULT_WORKFLOW_ID: Final[str] = "lora_merger_logic_v1"

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
STOCHASTIC_METHODS: Final[frozenset[str]] = frozenset({MERGE_DARE_TIES_SVD, MERGE_DARE_LINEAR_SVD})

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

#: The ceiling on how many adapters one recipe may draw from the pool.
#:
#: Ten, and the reason is cost rather than taste. Reconstruction is
#: the dominant expense in the whole system and it scales with the count: a
#: trimming merge measured here is about 149s of fixed work plus 87s per
#: adapter, per pass, and the engine makes two passes. Nine adapters is roughly
#: 15 minutes of that; twenty-three is over half an hour, paid again by every
#: validator that evaluates the submission.
#:
#: Consensus-relevant twice over. It bounds what a miner may compose, and it
#: caps the equal-weight references (scoring/references.py), so it decides what
#: the permanent bar is built from and therefore every comparison against it.
MAX_SELECTED_ADAPTERS: Final[int] = 10

# ---------------------------------------------------------------------------
# Hard deployment gates
# ---------------------------------------------------------------------------

#: Absolute GPU memory a candidate's runtime may reserve, in GiB.
#:
#: Pinned rather than left to each host so that every candidate is served the
#: same way on every card. vLLM reserves a fraction of the whole device, so a
#: fraction fixed by the operator would mean a package got more room on a bigger
#: card and less on a smaller one; fixing the absolute reservation and deriving
#: the fraction from the card keeps the serving conditions identical everywhere.
#:
#: 20 GiB: about 15.3 GiB of weights, the KV cache, and the ~0.9 GiB a runtime
#: carries on top. Measured at SERVING_MAX_MODEL_LEN it leaves 29,552 tokens of
#: KV — 3.61x what one full-length sequence uses, which is what pays for
#: SANDBOX_BATCH_CONCURRENCY — and peaks at about 21 GiB.
#:
#: It also has to fit on a card somebody owns, and that is what moved this
#: number. The 20 GiB it held was sized for a 24 GB card: such a card exposes
#: about 22.0 GiB, roughly 21.7 GiB of it free once the driver context is
#: resident, so a 22 GiB reservation is refused at start-up — measured, not
#: predicted: vLLM answers "free memory (21.65/22.04 GiB) is less than desired
#: utilization (0.9984, 22.0 GiB)" and exits. Every candidate then records a
#: serving failure, which scores every miner zero for the validator's hardware.
#:
#: The validator floor is now a 32 GB card (see the hardware requirements in the
#: README), which retires that ceiling. 24 GiB leaves about 8.7 GiB of KV at
#: SERVING_MAX_MODEL_LEN — roughly double what 20 GiB allowed — and that is what
#: SANDBOX_BATCH_CONCURRENCY is actually spending. At 20 GiB on the same
#: hardware the configured batch width could not be filled: the cache held
#: 29,552 tokens against a p90 instance of 2,017, so about fourteen sequences
#: were resident where thirty-two were being asked for, and the rest queued.
#:
#: Honest about provenance: the 20 GiB figures above were measured on real
#: cards. The 24 GiB figures are derived from those measurements plus the KV
#: arithmetic for this model (36 layers, 8 KV heads, 128 head dim, bf16 —
#: 144 KiB per token), not measured on a 32 GB card, because none was available
#: when this changed. A validator bringing up 32 GB hardware should confirm the
#: runtime accepts the reservation before trusting a run run on it.
SERVING_RESERVED_GIB: Final[float] = 24.0

#: The smallest card a validator may serve on, in GiB of total device memory.
#:
#: Forced by SERVING_RESERVED_GIB, not chosen alongside it. A candidate reserves
#: 24 GiB, the driver context holds about 1 GiB before anything loads, and a
#: merge sharing the card peaks near 2.5 GiB — so a card must offer about 27.5
#: GiB before it can serve and reconstruct at once. 32 GiB is the smallest
#: commodity size above that, and validator.serving.utilization_for refuses
#: anything smaller at start-up with the arithmetic in the message rather than
#: letting a run fail halfway through.
MIN_VALIDATOR_CARD_GIB: Final[float] = 32.0

#: The smallest fleet a validator may run.
#:
#: Not forced by the reference schedule: batched serving finishes the eight
#: reference packages in about 3.7 hours on a single card, well inside a 72-hour
#: run. It is forced by challenger throughput at the cadence this network is
#: moving to. Measured at the current rate — 2100 instances, about 0.47 hours a
#: package — a validator covers roughly this many challengers per run:
#:
#:      cards       3-day run        1-day run
#:          1                 106                  30
#:          2                 220                  67
#:          4                 448                 142
#:
#: A single card is comfortable today and stops being so the moment the run
#: shortens: 30 challengers against the 29 commitments the network already
#: carries is not headroom, it is the edge. Four cards is what keeps a 1-day
#: run viable while the miner count grows.
MIN_VALIDATOR_CARDS: Final[int] = 4

#: Context length every candidate is served at. Part of the measurement: a
#: package judged at a longer context is answering an easier question about its
#: own memory use.
#:
#: 8192, raised from 4096, because 4096 could not hold what the workflow already
#: asks. Measured over 300 real arena instances the prompt runs to a median of
#: 552 tokens, p99 of 3190 and a maximum of 3456; against the runner's 1024-token
#: answer budget the worst case needs 4480, and a 4096 run refused it. Not
#: hypothetical — an engine run answered 400 with "your prompt contains at least
#: 3073 input tokens ... for a total of at least 4097 tokens" on ordinary
#: instances, scoring candidates zero for a truncation the contract permitted.
#:
#: 8192 clears the worst observed case by 3712 tokens. It is not larger because
#: nothing needs it to be: the KV cache holds one sequence at a time, so a wider
#: run buys unreachable headroom and costs reserved memory a smaller
#: validator card does not have. A workflow whose turns accumulate — the
#: maintenance chain runs twelve — must be sized against this before it is made
#: the default.
SERVING_MAX_MODEL_LEN: Final[int] = 8192

MAX_ARTIFACT_BYTES: Final[int] = 500 * 1024 * 1024  # 500 MB

# Peak GPU memory is deliberately not gated or scored.
#
# vLLM reserves SERVING_RESERVED_GIB up front, so peak tracks the reservation
# and not the package: measured, every candidate lands at 20.9-21.1 GiB whatever
# it merged. The largest adapter the artifact cap admits — rank 96, 499.5 MB —
# adds 0.325 GiB, against roughly 3 GiB of headroom, so no recipe could reach a
# ceiling even in principle. As a gate it always passed; as a score term it was
# the same constant for every candidate, diluting artifact_efficiency by half
# against a size term that genuinely varies twelvefold across the allowed ranks.
#
# It also disagreed across the network. Validators never measured it — they
# substituted the ceiling — so the engine scored a headroom term that no
# validator reproduced. Deployment cost is now artifact size alone, which is a
# property of the recipe and identical everywhere.
MAX_AGENT_TURNS: Final[int] = 12
MAX_OUTPUT_TOKENS: Final[int] = 8192

#: Relative retention floor against the unmodified base model on a held-out
#: general-capability probe. A candidate that trades away general ability for
#: workflow score is rejected.
#:
#: Read against RETENTION_PROBE_ITEMS, because the two together decide how much
#: loss is actually tolerated. One item out of forty is a 2.5% drop, so the
#: earlier 0.98 admitted no loss whatever at any base score: a candidate matching
#: the base on 35 of the 36 it answered scored 0.972 and was rejected, and the
#: floor behaved as an exact-match rule wearing a percentage. Measured over two
#: full runs, that cost three otherwise mid-field candidates their place on a
#: single probe item each.
#:
#: At 0.95 the gate tolerates exactly one lost item and rejects two, which is
#: what a floor of this shape can express on a forty-item probe. Anything
#: between 0.951 and 0.975 is the same rule as 0.95; anything above 0.975 is the
#: same rule as 1.0. Raising RETENTION_PROBE_ITEMS is the only way to make the
#: number finer-grained than that.
BASE_RETENTION_FLOOR: Final[float] = 0.95

#: Items drawn per run for the general-capability probe. Small because each
#: item is a few tokens and the probe runs once per package per run, and
#: large enough that a single unlucky item cannot move the ratio past the floor.
RETENTION_PROBE_ITEMS: Final[int] = 40

#: A candidate must record zero critical unsafe actions.
MAX_CRITICAL_UNSAFE_ACTIONS: Final[int] = 0

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
# Quality dominates efficiency: a cheap but unreliable artifact cannot win.

WEIGHT_END_TO_END: Final[float] = 0.55
WEIGHT_STAGE_BALANCE: Final[float] = 0.15
WEIGHT_OOD: Final[float] = 0.10
WEIGHT_RETENTION: Final[float] = 0.05
# Latency is no longer scored, and its five points went to token efficiency
# rather than to quality, so the documented 85/15 split is unchanged.
#
# The two were measuring the same quantity. Measured over 60 real arena
# instances with the reasoning channel off, as production runs it: latency
# spread 5.9x, output tokens spread 5.8x, seconds-per-token spread only 1.2x,
# and the correlation between latency and tokens was 0.9992. The base model is
# the base model — what varies is how much a package says, so the latency term
# was token count with a constant of proportionality, counted a second time.
#
# Token efficiency is the better-formed of the pair. It is measured per
# *completed* instance against a fixed budget, so a package cannot flatter it by
# giving up early, and the budget does not drift with whoever holds the throne
# the way a ratio against the incumbent's median does.
#
# It also cost the most to collect. Latency needed an uncontended clock, which
# meant a sequential prefix of every draw — 50 instances at 13.9s against 1440
# batched at 0.67s, so 38% of an evaluation was spent on 3.4% of the instances,
# re-deriving with more noise something the token counter already reported
# exactly.
WEIGHT_TOKEN_EFFICIENCY: Final[float] = 0.10
WEIGHT_ARTIFACT_EFFICIENCY: Final[float] = 0.05

QUALIFIED_SCORE_WEIGHTS: Final[dict[str, float]] = {
    "end_to_end": WEIGHT_END_TO_END,
    "stage_balance": WEIGHT_STAGE_BALANCE,
    "ood": WEIGHT_OOD,
    "retention": WEIGHT_RETENTION,
    "token_efficiency": WEIGHT_TOKEN_EFFICIENCY,
    "artifact_efficiency": WEIGHT_ARTIFACT_EFFICIENCY,
}

#: Output tokens per completed instance treated as the reference cost.
#:
#: Token spend is the operating cost of the finished package — the thing a buyer
#: actually pays per workflow run — and until now it was measured and reported
#: but never scored, so a package that reached the same answer twice as
#: expensively ranked identically. Scored against a fixed budget rather than
#: against the incumbent, so the target does not drift with whoever holds the
#: throne.
REFERENCE_OUTPUT_TOKENS: Final[int] = 3000

#: Extra sketch columns drawn when factorising by randomised range finding.
#:
#: The merge is 99.7% decomposition, and the decomposition was computing every
#: singular component of a few-thousand-square matrix to keep sixty-four of
#: them. Sketching the range instead is 99x faster measured on one card — 15.1s
#: to 0.15s for the seven projections in a layer — and the error lands in the
#: tail that truncation discards anyway.
#:
#: Oversampling is what buys the accuracy: a sketch exactly as wide as the rank
#: recovers the leading directions and smears the last few. Ten extra columns
#: put the measured error against the exact top-64 subspace at ~1e-4 relative,
#: an order of magnitude below what bfloat16 can represent — and the artifact is
#: written in bfloat16.
#:
#: Consensus-relevant. This decides the artifact bytes, so it decides the
#: artifact digest, which is the cache key, the anti-copy identity and the thing
#: independent workers compare.
SVD_OVERSAMPLE: Final[int] = 10

#: Power iterations used to refine that sketch.
#:
#: A merged update's spectrum decays slowly, and one pass leaves the retained
#: components mixed with the tail. Two re-orthonormalised passes are enough at
#: this rank; more costs a QR each and does not move the error.
SVD_POWER_ITERATIONS: Final[int] = 2

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

#: Absolute end-to-end completion margin over the strongest *non-learned*
#: reference, in points of completion rate. This is the bar that says
#: "composition added value at all".
#:
#: The gate is arithmetic: a candidate clears it when its completion exceeds
#: the reference's by this much. No statistical test stands between a candidate
#: and the throne, so any bar is crossable at any instance count.
#:
#: What the instance count buys is *stability*, and that is the thing to weigh
#: against a lower bar. Run-to-run variation on identical artifacts is about
#: 0.017 — batched serving is not deterministic — so at 0.02 a package whose
#: true edge is exactly the bar clears it roughly half the time, and one a
#: little under it sometimes clears. See DEFAULT_HIDDEN_INSTANCES: the noise
#: falls with the square root of the draw, so a bar this close to it is a
#: deliberate trade of reproducibility for a lower barrier to entry.
DEFAULT_END_TO_END_MARGIN: Final[float] = 0.02

#: Margin a challenger must clear over the reigning champion, at the moment the
#: champion takes the throne.
#:
#: Kept separate from the reference margin because the two answer different
#: questions, and conflating them was a mistake with a specific consequence:
#: when the incumbent counted as a reference, every new champion had to beat the
#: previous one by a further three points. Completion is bounded by one, so that
#: ratchet admits at most a few dozen dethrones in principle and stalls after a
#: handful in practice — after which one package holds the throne forever, no
#: further work can ever be bought, and the network pays a permanent rent.
DEFAULT_CHAMPION_MARGIN: Final[float] = 0.01

#: Blocks over which the champion margin decays to zero.
#:
#: A champion that has held the throne unchallenged is either genuinely
#: excellent or merely unopposed, and the network cannot tell which. Letting the
#: bar fall over time resolves that in favour of movement: an incumbent keeps
#: its full defender's advantage while the contest is live, and gradually loses
#: it if nothing can displace it. At 12s blocks this is roughly 30 days.
CHAMPION_MARGIN_DECAY_BLOCKS: Final[int] = 216_000

#: One-sided paired bootstrap settings for the end-to-end comparison.
BOOTSTRAP_RESAMPLES: Final[int] = 10_000
BOOTSTRAP_CONFIDENCE: Final[float] = 0.95

# ---------------------------------------------------------------------------
# Continuous loop timing
# ---------------------------------------------------------------------------

#: Blocks per evaluation run. At 12s blocks this is 24 hours exactly. Hidden
#: instances are resampled and the references re-measured once per run.
#:
#: Three days was the price of 1350 instances at 13.9s each — about 5.6 hours a
#: package, and 45 hours of references before a challenger was touched.
#: Continuous batching and a four-card fleet took the same 1350 instances to
#: roughly half an hour a package and the whole reference schedule to about an
#: hour, so most of a three-day run sat idle. A day is enough for the work and
#: gives a miner an answer the next day instead of the next week.
#:
#: Not a free parameter on a running deployment. It sets the spacing of run
#: boundaries, and the beacon is drawn from a run's own opening block, so
#: changing it moves the block each future beacon comes from. RunDisclosure
#: records the value in force when a run ran, which is what lets a reader check
#: an old beacon against the length that actually produced it rather than
#: against whatever is configured today. Runs already closed keep the length
#: they ran at — see RUN_EPOCH_BLOCK.
DEFAULT_RUN_BLOCKS: Final[int] = 7200

#: Where the daily schedule begins, and the run that opens there.
#:
#: A run id used to be the block divided by the run length, which put boundaries
#: wherever the arithmetic landed — for three-day runs, 04:26 Eastern on a
#: rotating three-day cycle. Nobody chose that time and it drifts across the
#: working day, which makes "when does my submission get measured" a question
#: with a different answer every week.
#:
#: Anchoring fixes it: run 412 opens at this block, which the chain reaches at
#: approximately 12:00 Eastern on Sunday 23 August 2026, and every run after it
#: opens one DEFAULT_RUN_BLOCKS later. Because 7200 blocks is 24 hours, every
#: boundary lands at the same time of day.
#:
#: Derived, not guessed: finney held 12.0029 s/block over the 201,600 blocks to
#: 22 August 2026, so a day's run boundary slides about 21 seconds and a month's
#: about 10 minutes. Re-anchoring is a one-line change to this constant plus
#: RUN_EPOCH_ID, and it is the only thing that keeps noon meaning noon.
#:
#: History is frozen rather than renumbered. Blocks before the epoch keep the
#: run ids they were measured under, computed at LEGACY_RUN_BLOCKS and capped at
#: RUN_EPOCH_ID - 1, so run 411 simply runs long — from 19 August to the epoch —
#: instead of every published report, stored run and console row shifting to a
#: number it was never filed under.
RUN_EPOCH_BLOCK: Final[int] = 8_908_667

#: The run that opens at RUN_EPOCH_BLOCK. Everything before it is history.
RUN_EPOCH_ID: Final[int] = 412

#: The run length in force before RUN_EPOCH_BLOCK. Not configurable: it is a
#: fact about runs that have already closed, and the ids they were filed under.
LEGACY_RUN_BLOCKS: Final[int] = 21600

#: Runs between measuring a submission and paying for it.
#:
#: The pipeline is three runs deep and each stage is a whole run, so a miner
#: knows which run does what to their recipe from the block they commit at:
#:
#:   run N     the recipe is committed
#:   run N+1   it is evaluated and its score recorded
#:   run N+2   that score sets the weight this validator submits on-chain
#:
#: Separating the last two is what makes the weight vector a statement about a
#: run that has closed. Paying inside the evaluating run means paying from a
#: leaderboard still being written: a miner measured early in the run competes
#: against an empty field, one measured late against a full one, and the vector
#: changes under both every time another candidate finishes. A closed run has a
#: final leaderboard, and every validator reading the same chain computes the
#: same vector from it.
WEIGHT_LAG_RUNS: Final[int] = 1

#: How long a commitment must stand before the run that measures it opens.
#: 300 blocks is about an hour at 12s.
#:
#: A rate limit that could be enforced, in place of one that could not. Limiting
#: a miner to one submission an hour needs to know when the previous ones were,
#: and there is no such record: the chain keeps one commitment per hotkey, and a
#: validator reading it sees the current block and nothing about what stood
#: there before. Counting attempts would mean keeping local state, which is the
#: thing measured_in_run exists to avoid — two validators with different uptime
#: would disagree about whose turn it was.
#:
#: Age is the part that *is* on the chain. A commitment measured in the run
#: after it was made must have been standing for at least this long when that
#: run opened, so the rule reads "let it settle for an hour" rather than "submit
#: at most hourly". The effect on churn is the same and stronger at the boundary:
#: every replacement restarts the hour, so a miner still editing in the last
#: hour of a run is not measured in the next one and waits a further run.
#:
#: It is also what stops a submission being timed against the field. A miner who
#: commits at the closing block has watched the entire run — every published
#: result and every recipe another miner disclosed — before choosing. An hour is
#: not a large tax on someone who searched, and it is the whole advantage of
#: someone who waited.
MIN_COMMITMENT_AGE_BLOCKS: Final[int] = 300

#: Hidden instances drawn per run for the canonical comparison.
#:
#: What this buys is a stable verdict rather than a possible one. Nothing
#: declines a challenger on statistical grounds — every hard gate is arithmetic
#: — so a bar is crossable at any draw size. The draw decides how *reliably* the
#: same package gets the same answer: 1350 instances resolve about 0.0241, 2000
#: resolve 0.0198 and 400 resolve 0.0443, and a margin near or below that figure
#: is decided partly by which instances came up.
#:
#: The resolvable edge falls with the square root of this, so halving it costs
#: four times the evaluation. What makes that affordable is the wall clock:
#: batched across four cards the reference schedule is about an hour, against
#: roughly 51 hours one instance at a time. The schedule preflight still refuses
#: a run that cannot finish its own schedule, because such a run never reaches a
#: challenger at all and the symptom is silence.
DEFAULT_HIDDEN_INSTANCES: Final[int] = 1350

#: Additional out-of-distribution instances drawn per run.
DEFAULT_OOD_INSTANCES: Final[int] = 100

#: Public pack size shipped to miners for offline search and debugging.
PUBLIC_PACK_INSTANCES: Final[int] = 120

# ---------------------------------------------------------------------------
# Incentive
# ---------------------------------------------------------------------------

#: Fraction of every run's emission that burns.
#:
#: The subnet buys one thing: a merged package that beats the one it already
#: has. Emission that does not buy that is emission the network did not need to
#: spend, so the default is to spend little and burn the rest.
BURN_SHARE: Final[float] = 0.80

#: How the payable fifth is split by rank.
#:
#: Ranks one to five, in order. The throne takes nearly all of it: the prize is
#: winning, and placing is information about how close a miner came rather than
#: a living. A miner who wants emission has to take the throne.
RANK_SHARES: Final[tuple[float, ...]] = (0.90, 0.05, 0.03, 0.01, 0.005)

#: Split across ranks six to ten, in proportion to grade.
#:
#: In proportion rather than evenly, so the ordering inside the tail still says
#: something: an even split would pay a candidate that barely qualified the same
#: as one that nearly placed fifth.
TAIL_SHARE: Final[float] = 0.005

#: Ranks paid in one run. Below this a share is smaller than the chain's own
#: weight quantisation, so it would be bookkeeping rather than payment.
PAID_RANKS: Final[int] = 10

#: How much of a run's draw one host asks.
#:
#: A validator asks a core every validator has in common, so two of them can be
#: compared on identical instances, plus a tail only it holds, so nobody can
#: predict or shop for their set. A host measuring the whole draw — an engine
#: scoring the field itself — asks all of it.
#:
#: Bound to DEFAULT_END_TO_END_MARGIN rather than free. The resolvable effect
#: falls with the square root of the count: at DEFAULT_HIDDEN_INSTANCES the
#: whole draw resolves 0.0241 and forty percent of it resolves 0.0381. A margin
#: between those two is a bar that cannot be shown to have been cleared.
DEFAULT_CORE_FRACTION: Final[float] = 0.25
DEFAULT_TAIL_FRACTION: Final[float] = 0.15

#: Grade the leading candidate must exceed the champion by to take the throne.
#:
#: The throne changes hands on a measurable improvement, not on noise and not
#: on a tie. Applied to the contribution grade, which is the composite the
#: field is ranked on, so a challenger that is better on cost or breadth can
#: win without being better on completion alone.
#:
#: It also decides whether the run pays at all. If the best candidate cannot
#: clear it, the run offered the network nothing it did not already have and
#: the whole miner share burns — nobody places behind a leader who did not win.
CHAMPION_DETHRONE_MARGIN: Final[float] = 0.002

#: How a candidate's grade is composed. Quality dominates: a package that does
#: not finish workflows is not made valuable by being cheap, or by being nearly
#: as good as something that does.
#:
#: Every term is measured against fixed points — the run's own instances and
#: the base model — so a grade means the same thing in every run. That is what
#: lets CHAMPION_DETHRONE_MARGIN be a fixed number: a threshold on a quantity
#: whose scale moved between runs would be a different bar each time.
CONTRIBUTION_WEIGHT_QUALITY: Final[float] = 0.60
CONTRIBUTION_WEIGHT_IMPROVEMENT: Final[float] = 0.30
CONTRIBUTION_WEIGHT_COST: Final[float] = 0.10

#: Fallback UID for burned emission when the subnet owner cannot be resolved.
#:
#: Note that UID 0 is *not* an incinerator — it belongs to whichever neuron
#: registered into the first slot. Live components resolve the owner's UID from
#: the metagraph instead (see ``chain.MetagraphView.owner_uid``) and decline to
#: submit at all when there is none; this constant exists only so offline
#: tooling has a value to construct a vector with.
BURN_UID: Final[int] = 0


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------

#: How many instances a candidate is asked at once while accuracy is measured.
#:
#: One at a time left the card decode-bound and idle between tokens of a single
#: stream: measured on this corpus, sequential runs 15.89s an instance and
#: continuous batching at 32 runs 1.08s — 14.7x, on identical hardware and
#: identical greedy settings. A run that spent seven hours a package spends
#: half an hour.
#:
#: Consensus-relevant, and not only for cadence. Batch composition changes
#: kernel selection and reduction order, so a batched reply is not token-identical
#: to a sequential one — measured, 1 of 48 replies matched. Every package in a
#: run is therefore asked the same way, and the number is pinned so that two
#: validators ask the same way as each other.
SANDBOX_BATCH_CONCURRENCY: Final[int] = 32

SANDBOX_TEMPERATURE: Final[float] = 0.0
SANDBOX_TOP_P: Final[float] = 1.0

#: Whether the served model is asked to emit its separate reasoning channel.
#: Off: the base model's template enables it by default, and one thinking block
#: exhausts MAX_OUTPUT_TOKENS before the agent has called a single tool. This is
#: consensus-relevant — it changes what every candidate is asked.
SANDBOX_ENABLE_THINKING: Final[bool] = False

#: Wall-clock ceiling for a single hidden instance, including model generation
#: and every tool call.
SANDBOX_INSTANCE_TIMEOUT_SECONDS: Final[float] = 300.0

#: Ceiling for one generated diagnostic Python execution.
PYTHON_RUNNER_TIMEOUT_SECONDS: Final[float] = 20.0

#: Ceiling for one generated SQL statement against the hidden snapshot.
SQL_TIMEOUT_SECONDS: Final[float] = 10.0
