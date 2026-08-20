"""Protocol data contracts.

These models define every structure that crosses a trust boundary:

* :class:`Recipe` — the one declarative document a miner submits.
* :class:`InstanceResult` — one sample row produced by evaluating one hidden
  workflow instance.
* :class:`CandidateScores` — the aggregate scores derived from those rows.
* :class:`EvaluationReport` — the signed, published record of one evaluation.
* :class:`WeightVector` — what validators fetch and set on-chain.

Validation is strict: unknown fields are rejected, because an unknown field in a
miner recipe usually means a miner is trying to smuggle something past the
merge engine.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from capability_subnet.common import constants as C
from capability_subnet.common.hashing import canonical_json_bytes, normalise_digest, sha256_bytes


class StrictModel(BaseModel):
    """Base for every protocol model: no extra fields, validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


# ---------------------------------------------------------------------------
# Recipe — the miner's submission
# ---------------------------------------------------------------------------


class MergeSpec(StrictModel):
    """The merge algorithm and its algorithm-level parameters."""

    combination_type: str = Field(
        description="One of the allowed merge methods.",
    )
    density: float | None = Field(
        default=None,
        description=(
            "Fraction of delta entries kept by the trim/drop step. Required by "
            "the TIES, DARE and magnitude-prune families; rejected otherwise."
        ),
    )
    majority_sign_method: Literal["total", "frequency"] | None = Field(
        default=None,
        description="Sign-election strategy for TIES-family merges.",
    )
    random_seed: int = Field(
        default=0,
        ge=C.RANDOM_SEED_MIN,
        le=C.RANDOM_SEED_MAX,
        description=(
            "Seed for the stochastic drop in the DARE family. Reconstruction is "
            "deterministic given this seed."
        ),
    )

    @field_validator("combination_type")
    @classmethod
    def _known_method(cls, value: str) -> str:
        if value not in C.ALLOWED_MERGE_METHODS:
            raise ValueError(
                f"unknown combination_type {value!r}; allowed: {', '.join(C.ALLOWED_MERGE_METHODS)}"
            )
        # Folded to the canonical spelling on the way in, so a recipe's digest
        # identifies the *package* rather than the wording. Two miners who chose
        # different accepted names for the same merge would otherwise submit
        # distinct recipes that reconstruct to identical bytes, and the anti-copy
        # check would terminate whichever arrived second for copying a package it
        # never saw.
        from capability_subnet.merge_engine.methods import canonical_method

        return canonical_method(value)

    @model_validator(mode="after")
    def _check_parameter_applicability(self) -> MergeSpec:
        needs_density = self.combination_type in C.DENSITY_METHODS
        if needs_density:
            if self.density is None:
                raise ValueError(f"{self.combination_type} requires 'density'")
            if not (C.DENSITY_MIN <= self.density <= C.DENSITY_MAX):
                raise ValueError(
                    f"density {self.density} outside [{C.DENSITY_MIN}, {C.DENSITY_MAX}]"
                )
        elif self.density is not None:
            raise ValueError(f"{self.combination_type} does not accept 'density'")

        is_ties = self.combination_type in (C.MERGE_TIES_SVD, C.MERGE_DARE_TIES_SVD)
        if is_ties:
            if self.majority_sign_method is None:
                raise ValueError(f"{self.combination_type} requires 'majority_sign_method'")
        elif self.majority_sign_method is not None:
            raise ValueError(f"{self.combination_type} does not accept 'majority_sign_method'")

        return self


class CompressionSpec(StrictModel):
    """How the merged delta is factorised back into a LoRA pair."""

    output_rank: int = Field(
        description="Rank of the emitted adapter. Drives artifact size and VRAM.",
    )
    svd_clamp_quantile: float = Field(
        default=1.0,
        ge=C.SVD_CLAMP_QUANTILE_MIN,
        le=C.SVD_CLAMP_QUANTILE_MAX,
        description=(
            "Singular values above this quantile are clamped to the quantile "
            "value, limiting the influence of a few dominant directions."
        ),
    )

    @field_validator("output_rank")
    @classmethod
    def _allowed_rank(cls, value: int) -> int:
        if value not in C.ALLOWED_OUTPUT_RANKS:
            raise ValueError(
                f"output_rank {value} not allowed; choose from {list(C.ALLOWED_OUTPUT_RANKS)}"
            )
        return value


class OutputSpec(StrictModel):
    """Emitted artifact properties. Both fields are fixed in V1 but stated
    explicitly so the recipe fully determines the artifact bytes."""

    dtype: Literal["bfloat16"] = "bfloat16"
    adapter_name: str = Field(default="candidate", min_length=1, max_length=64)

    @field_validator("adapter_name")
    @classmethod
    def _safe_name(cls, value: str) -> str:
        if not all(c.isalnum() or c in "-_" for c in value):
            raise ValueError("adapter_name may contain only letters, digits, '-' and '_'")
        return value


class Recipe(StrictModel):
    """The complete, declarative miner submission.

    A recipe contains no executable content of any kind. Everything it can
    express is a bounded number, a name drawn from a frozen registry, or an
    enum. That is what makes it safe to reconstruct inside the evaluation
    engine without ever running miner code.
    """

    schema_version: int = Field(default=C.RECIPE_SCHEMA_VERSION)
    workflow_id: str = Field(default=C.DEFAULT_WORKFLOW_ID)
    base_revision: str = Field(
        description="The pinned base-model revision this recipe targets.",
        min_length=1,
    )
    source_snapshot_sha256: str = Field(
        description="Digest of the frozen certified adapter snapshot.",
    )
    selected_adapters: list[str] = Field(
        # The bounds are declared on the field, not only in the validator below,
        # so they reach the exported JSON Schema as minItems/maxItems. A field
        # validator does not: the published contract carried the cap in prose
        # while its machine-readable schema left the array unbounded, and a
        # miner generating recipes against that schema was told nothing until
        # the engine refused the commitment it had already paid to make.
        min_length=C.MIN_SELECTED_ADAPTERS,
        max_length=C.MAX_SELECTED_ADAPTERS,
        description=(
            f"Adapter IDs drawn from the certified pool, in any order. "
            f"Between {C.MIN_SELECTED_ADAPTERS} and {C.MAX_SELECTED_ADAPTERS}, "
            f"no duplicates."
        ),
    )
    merge: MergeSpec
    global_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-adapter coefficient applied to the whole delta. Adapters not "
            "listed default to 1.0."
        ),
    )
    layer_group_overrides: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Per-layer-group coefficient overrides, keyed by group name then "
            "adapter ID. An override replaces the global weight for that group."
        ),
    )
    compression: CompressionSpec
    output: OutputSpec = Field(default_factory=OutputSpec)

    # -- validators ---------------------------------------------------------

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value != C.RECIPE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value}; this network accepts "
                f"{C.RECIPE_SCHEMA_VERSION}"
            )
        return value

    @field_validator("source_snapshot_sha256")
    @classmethod
    def _digest_shape(cls, value: str) -> str:
        return normalise_digest(value)

    @field_validator("selected_adapters")
    @classmethod
    def _adapter_selection(cls, value: list[str]) -> list[str]:
        if len(value) < C.MIN_SELECTED_ADAPTERS:
            raise ValueError(
                f"a recipe must select at least {C.MIN_SELECTED_ADAPTERS} adapters; "
                "a single adapter is a reference baseline, not a candidate"
            )
        if len(value) > C.MAX_SELECTED_ADAPTERS:
            raise ValueError(f"at most {C.MAX_SELECTED_ADAPTERS} adapters may be selected")
        if len(set(value)) != len(value):
            duplicates = sorted({a for a in value if value.count(a) > 1})
            raise ValueError(f"duplicate adapters in selection: {duplicates}")
        return value

    @model_validator(mode="after")
    def _coefficients_reference_selected_adapters(self) -> Recipe:
        selected = set(self.selected_adapters)

        unknown = sorted(set(self.global_weights) - selected)
        if unknown:
            raise ValueError(f"global_weights references unselected adapters: {unknown}")

        for group, overrides in self.layer_group_overrides.items():
            if group not in C.LAYER_GROUPS:
                raise ValueError(
                    f"unknown layer group {group!r}; allowed: {list(C.LAYER_GROUP_NAMES)}"
                )
            unknown = sorted(set(overrides) - selected)
            if unknown:
                raise ValueError(
                    f"layer_group_overrides[{group}] references unselected adapters: {unknown}"
                )
            for adapter, weight in overrides.items():
                _check_weight_bounds(weight, f"layer_group_overrides[{group}][{adapter}]")

        for adapter, weight in self.global_weights.items():
            _check_weight_bounds(weight, f"global_weights[{adapter}]")

        return self

    # -- derived ------------------------------------------------------------

    def effective_weight(self, adapter_id: str, layer_group: str) -> float:
        """Coefficient for ``adapter_id`` in ``layer_group``.

        Resolution order: a group override wins over the global weight, which in
        turn wins over the implicit default of 1.0.
        """
        override = self.layer_group_overrides.get(layer_group, {})
        if adapter_id in override:
            return float(override[adapter_id])
        return float(self.global_weights.get(adapter_id, 1.0))

    def canonical_bytes(self) -> bytes:
        """The exact bytes whose digest is committed on-chain."""
        return canonical_json_bytes(self.model_dump(mode="json", exclude_none=True))

    def digest(self) -> str:
        """Content address of this recipe."""
        return sha256_bytes(self.canonical_bytes())

    def sorted_adapters(self) -> list[str]:
        """Adapters in the load order the merge engine uses.

        Reconstruction always loads adapters in sorted ID order so that the
        result never depends on how the miner happened to order the list.
        """
        return sorted(self.selected_adapters)


def _check_weight_bounds(weight: float, where: str) -> None:
    if not (C.ADAPTER_WEIGHT_MIN <= weight <= C.ADAPTER_WEIGHT_MAX):
        raise ValueError(
            f"{where} = {weight} outside [{C.ADAPTER_WEIGHT_MIN}, {C.ADAPTER_WEIGHT_MAX}]"
        )


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------


class StageResult(StrictModel):
    """Deterministic score for one stage of one instance."""

    stage: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    detail: str = ""


class InstanceResult(StrictModel):
    """One sample row: the outcome of running one package on one hidden instance.

    Rows are the atomic unit the comparator works with. Paired statistics
    require that the challenger and the reference both have a row for the same
    ``instance_id``.
    """

    instance_id: str
    instance_seed: int
    split: Literal["hidden", "ood", "public"] = "hidden"

    stages: dict[str, StageResult] = Field(default_factory=dict)
    final_state_correct: bool = False
    end_to_end_success: bool = False

    critical_unsafe_actions: int = 0
    turns_used: int = 0
    output_tokens: int = 0
    input_tokens: int = 0
    wall_seconds: float = 0.0
    error: str | None = Field(
        default=None,
        description=(
            "Set when the harness itself failed rather than the model. Rows with "
            "an infrastructure error are excluded from scoring instead of "
            "counting as a model failure."
        ),
    )

    @property
    def is_valid_sample(self) -> bool:
        """Whether this row may take part in scoring and paired comparison."""
        return self.error is None

    def stage_score(self, stage: str) -> float:
        result = self.stages.get(stage)
        return 0.0 if result is None else result.score


class ResourceMeasurements(StrictModel):
    """Measured deployment cost of one candidate package."""

    adapter_mb: float = 0.0
    mean_workflow_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model_turns: int = 0


class CandidateScores(StrictModel):
    """Aggregate scores over a set of sample rows."""

    end_to_end: float = Field(default=0.0, ge=0.0, le=1.0)
    stage_balance: float = Field(default=0.0, ge=0.0, le=1.0)
    ood: float = Field(default=0.0, ge=0.0, le=1.0)
    retention: float = Field(default=0.0, ge=0.0, le=1.0)
    token_efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    artifact_efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    qualified_score: float = Field(default=0.0, ge=0.0, le=1.0)

    #: Axis -> mean over the rows that scored it. An axis this draw did not
    #: sample is absent rather than zero; read it with the counts below.
    per_stage_means: dict[str, float] = Field(default_factory=dict)
    #: Axis -> how many valid rows scored it, so a reader can tell an axis the
    #: package failed from one the draw never put to it.
    per_stage_samples: dict[str, int] = Field(default_factory=dict)
    valid_samples: int = 0
    total_samples: int = 0


class GateVerdict(StrictModel):
    """Result of one hard gate. Any failure zeroes the candidate."""

    name: str
    passed: bool
    detail: str = ""
    measured: float | int | str | bool | None = None
    limit: float | int | str | bool | None = None


class AxisVerdict(StrictModel):
    """Comparator verdict on one capability axis."""

    axis: str
    challenger_mean: float
    champion_mean: float
    paired_samples: int
    verdict: Literal["dominant", "not_worse", "worse"]
    margin: float
    tolerance: float


class PairedComparison(StrictModel):
    """End-to-end paired statistics against the strongest reference."""

    reference_id: str
    paired_instances: int
    challenger_mean: float
    reference_mean: float
    difference: float
    bootstrap_lcb: float
    bootstrap_resamples: int
    confidence: float
    passed: bool


class ComparatorOutcome(StrictModel):
    """Everything the dethrone decision was based on."""

    per_axis_verdicts: list[AxisVerdict] = Field(default_factory=list)
    dominant_count: int = 0
    min_dominant_required: int = C.DEFAULT_MIN_DOMINANT_AXES
    any_worse_axis: bool = False
    strict_pareto: bool = False
    end_to_end_margin_required: float = C.DEFAULT_END_TO_END_MARGIN
    end_to_end_margin_observed: float = 0.0
    #: The incumbent's remaining defender's advantage, and what the challenger
    #: actually managed against it. Published separately from the reference
    #: margin because they answer different questions and decay differently.
    champion_margin_required: float = C.DEFAULT_CHAMPION_MARGIN
    champion_margin_observed: float = 0.0
    #: The smallest difference this window's sample size could have resolved.
    #: Published so a reader can tell a package that genuinely lost from one the
    #: engine never had the evidence to judge — the two look identical in every
    #: other field of a report.
    minimum_detectable_effect: float = 0.0
    paired: PairedComparison | None = None
    dethrones: bool = False
    reason: str = ""


class EvaluationReport(StrictModel):
    """The published, signed record of one candidate evaluation.

    Anyone can re-derive the weight vector from a stream of these reports, which
    is what keeps a centralised evaluation engine auditable.
    """

    report_version: int = 1
    workflow_id: str = C.DEFAULT_WORKFLOW_ID
    window_id: int
    evaluated_at_block: int

    miner_hotkey: str
    miner_uid: int | None = None
    candidate_id: str = Field(
        description="Stable identifier: the miner hotkey for candidates, or the "
        "reference name for permanent baselines.",
    )

    recipe_sha256: str | None = None
    artifact_sha256: str | None = None
    base_revision: str
    source_snapshot_sha256: str
    evaluator_image_digest: str

    hard_gates: list[GateVerdict] = Field(default_factory=list)
    scores: CandidateScores = Field(default_factory=CandidateScores)
    resources: ResourceMeasurements = Field(default_factory=ResourceMeasurements)

    baseline_scores: dict[str, float] = Field(
        default_factory=dict,
        description="End-to-end completion of every reference on the same instances.",
    )
    strongest_reference_id: str = ""
    strongest_reference_score: float = 0.0

    comparator: ComparatorOutcome | None = None

    verdict: Literal["dethrone", "held", "terminated", "rejected", "reference"] = "held"
    verdict_reason: str = ""
    #: The graded contribution and its terms, when the candidate cleared every
    #: hard gate. Published so a miner that earned a partial share can see which
    #: part of its package earned it — the difference between a signal it can act
    #: on and a number it has to guess at.
    contribution: dict[str, float] = Field(default_factory=dict)

    #: Set by the signer; excluded from the signed payload itself.
    signature: str | None = None
    signer_hotkey: str | None = None

    def signable_bytes(self) -> bytes:
        """Canonical bytes covered by the operator signature."""
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("signature", None)
        payload.pop("signer_hotkey", None)
        return canonical_json_bytes(payload)

    @property
    def gates_passed(self) -> bool:
        """Whether the gates ran and all of them passed.

        A report with no gate verdicts has not passed anything. Returning True
        for it would let a candidate that was never gated count as qualified in
        the graded-emission ranking.
        """
        return bool(self.hard_gates) and all(gate.passed for gate in self.hard_gates)

    def failed_gates(self) -> list[str]:
        return [gate.name for gate in self.hard_gates if not gate.passed]


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


class WeightEntry(StrictModel):
    uid: int = Field(ge=0)
    hotkey: str = ""
    weight: float = Field(ge=0.0, le=1.0)
    role: Literal["champion", "runner_up", "third", "contributor", "queued", "burn"] = "champion"


class WeightVector(StrictModel):
    """The signed weight vector validators fetch and submit on-chain.

    Validators do not recompute scores. They verify the signature, sanity-check
    the vector, apply their own burn floor if configured, and call
    ``set_weights``.
    """

    workflow_id: str = C.DEFAULT_WORKFLOW_ID
    window_id: int
    computed_at_block: int
    mode: Literal["winner_take_all", "graded_top3", "graded_contribution"] = (
        C.MODE_GRADED_CONTRIBUTION
    )
    burn_percentage: float = Field(default=0.0, ge=0.0, le=1.0)

    entries: list[WeightEntry] = Field(default_factory=list)
    champion_hotkey: str | None = None
    champion_report_sha256: str | None = None

    signature: str | None = None
    signer_hotkey: str | None = None

    def signable_bytes(self) -> bytes:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("signature", None)
        payload.pop("signer_hotkey", None)
        return canonical_json_bytes(payload)

    def as_uid_weight_lists(self) -> tuple[list[int], list[float]]:
        """Split into the ``uids`` / ``weights`` lists the chain call expects."""
        return [e.uid for e in self.entries], [e.weight for e in self.entries]

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> WeightVector:
        if not self.entries:
            return self
        total = sum(e.weight for e in self.entries)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total}")
        uids = [e.uid for e in self.entries]
        if len(set(uids)) != len(uids):
            raise ValueError("duplicate uid in weight vector")
        return self


# ---------------------------------------------------------------------------
# Champion state
# ---------------------------------------------------------------------------


class DisclosedInstance(StrictModel):
    """One instance from a closed window, published so it can be re-scored."""

    instance_id: str
    instance_seed: int
    split: str
    candidate_id: str
    #: The engine's own scored row. An auditor regenerates the instance from the
    #: seed, re-scores the trace, and compares against this.
    claimed_result: InstanceResult
    #: The immutable record the scorer read. Everything needed to reproduce the
    #: scoring arithmetic without running a model.
    trace: dict[str, Any] = Field(default_factory=dict)


class WindowDisclosure(StrictModel):
    """What a closed window makes public.

    Hidden instances are drawn fresh every window and never reused, so a closed
    window's instances have no further value as a secret — and considerable
    value as evidence. Publishing them lets anyone regenerate the exact problems
    a candidate faced and re-run the deterministic scorer over its stored trace,
    which turns "the engine says this candidate scored 0.8" into something that
    can be checked without a GPU.

    Only closed windows are ever disclosed. Publishing the current window would
    hand its challenger the test it is sitting.
    """

    #: Required, with no default. A disclosure is an audit record and has to name
    #: its own subject: defaulting to whichever workflow is *currently* configured
    #: means replaying a disclosure written before the operator switched workflows
    #: would regenerate the wrong instances and score them against the wrong
    #: scorer — silently, since both would succeed.
    workflow_id: str
    window_id: int
    closed_at_block: int

    #: Blocks per window at the time this window ran. Recorded because the window
    #: id is the block divided by it and the beacon is the hash of the block the
    #: window opened at, so without it a reader cannot work out which block a past
    #: beacon should have come from — and therefore cannot check that it did.
    #: Defaulting to the *current* setting would silently re-derive old windows
    #: against a new length, which is exactly the check this is meant to support.
    #: 0 means a disclosure written before this field existed.
    window_blocks: int = 0

    #: Every hidden and out-of-distribution seed the window drew. Disclosing all
    #: of them rather than a subset is deliberate: a subset the engine chose
    #: could be the subset it scored honestly.
    hidden_seeds: list[int] = Field(default_factory=list)
    ood_seeds: list[int] = Field(default_factory=list)

    #: The public value this window's draw was bound to — the hash of the block
    #: it opened at. Published so the draw can be checked against the chain
    #: instead of taken on trust: the seeds derive from it, and the operator does
    #: not choose it.
    beacon: str = ""
    #: Hash of the operator's seed root, identical in every window of a
    #: deployment. It reveals nothing about the root and binds the operator to
    #: one: a value that moves between windows is the draw being changed, in
    #: public.
    root_commitment: str = ""

    #: Traces for a bounded, deterministically chosen sample.
    instances: list[DisclosedInstance] = Field(default_factory=list)

    signature: str | None = None
    signer_hotkey: str | None = None

    def signable_bytes(self) -> bytes:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("signature", None)
        payload.pop("signer_hotkey", None)
        return canonical_json_bytes(payload)


class ChampionRecord(StrictModel):
    """The reigning champion. Rewritten only by a successful dethrone."""

    candidate_id: str
    hotkey: str | None = None
    uid: int | None = None
    recipe_sha256: str | None = None
    artifact_sha256: str | None = None
    crowned_at_block: int = 0
    crowned_at_window: int = 0
    successful_defenses: int = 0
    is_reference: bool = Field(
        default=False,
        description=(
            "True for the seeded reference package that holds the throne before "
            "any miner has taken it. A reference champion earns no emission; if "
            "it holds the throne the workflow share is burned."
        ),
    )
    last_scores: CandidateScores | None = None


class QueueEntry(StrictModel):
    """One admitted challenger awaiting evaluation."""

    hotkey: str
    uid: int
    recipe_sha256: str
    recipe_uri: str
    first_block: int
    admitted_at_block: int
    status: Literal["queued", "evaluating", "champion", "terminated", "rejected"] = "queued"
    status_reason: str = ""


def recipe_json_schema() -> dict[str, Any]:
    """JSON Schema for the recipe, for publication alongside the workflow contract."""
    return Recipe.model_json_schema()
