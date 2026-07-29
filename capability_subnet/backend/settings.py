"""Engine configuration.

Loaded from a YAML file with environment-variable overrides, so an operator can
keep the durable configuration in version control and inject per-host details
(GPU assignment, database credentials, wallet name) without editing it.

Anything that changes how a candidate is *scored* belongs in the protocol
constants, not here. This file only describes how this particular deployment of
the engine is wired up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from capability_subnet.common import constants as C


def _env(name: str, default: Any) -> Any:
    return os.environ.get(f"CAPSUB_{name.upper()}", default)


def _coerce(value: Any, target: Any) -> Any:
    if isinstance(target, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(target, int) and not isinstance(target, bool):
        return int(value)
    if isinstance(target, float):
        return float(value)
    return value


@dataclass(slots=True)
class BackendSettings:
    """Everything the engine needs to run one deployment."""

    # -- chain ---------------------------------------------------------------
    netuid: int = 1
    network: str = "finney"
    chain_endpoint: str = ""
    wallet_name: str = "capability"
    wallet_hotkey: str = "default"

    #: Commitments made at or before this block are ignored. Set it to the block
    #: the arena opened at, so commitments left over from a previous arena
    #: version cannot enter the queue.
    min_commit_block: int = 0

    # -- storage -------------------------------------------------------------
    state_dir: str = "state"
    artifact_cache_dir: str = "state/artifacts"
    adapter_pool_dir: str = "pool"
    report_dir: str = "state/reports"

    #: The engine's own verified copy of every admitted recipe. Champions are
    #: re-measured every window, so this must outlive the miner's own pointer.
    #:
    #: Empty means "under state_dir", which is what almost every deployment
    #: wants. Spelling the default as a second literal path was how admission and
    #: the champion loader came to disagree the moment an operator moved
    #: state_dir: both looked correct, and they only pointed at the same place
    #: for the default value.
    recipe_dir: str = ""

    # -- windows -------------------------------------------------------------
    window_blocks: int = C.DEFAULT_WINDOW_BLOCKS
    hidden_instances: int = C.DEFAULT_HIDDEN_INSTANCES
    ood_instances: int = C.DEFAULT_OOD_INSTANCES

    #: Single-adapter references measured per window, rotated by window id.
    #:
    #: Zero or a value at least as large as the pool measures all of them, which
    #: is correct and, at eleven adapters times a full instance draw, is most of
    #: a window's GPU budget spent before any challenger is looked at. Rotating
    #: keeps the "beat the best specialist" bar honest over time while leaving
    #: room in the window to actually evaluate somebody.
    single_adapter_rotation: int = 3

    #: Seed the per-window hidden instance draw is derived from. It must stay
    #: secret: publishing it would publish every future hidden instance.
    hidden_seed_root: int = 0

    # -- evaluation ----------------------------------------------------------
    workflow_id: str = C.DEFAULT_WORKFLOW_ID

    #: How the candidate gets served. ``managed`` starts a vLLM process per
    #: candidate with that candidate's adapter applied — the only mode in which
    #: the engine actually measures the package it built. ``external`` points at
    #: a runtime the operator manages and cannot apply an adapter, so it is for
    #: development against the base model only.
    serving_mode: str = "managed"
    serving_base_url: str = "http://127.0.0.1:8000"
    serving_model_name: str = "candidate"

    #: Local path to the materialised base model. Required in managed mode.
    base_model_path: str = ""
    serving_host: str = "127.0.0.1"
    serving_port: int = 8000
    serving_gpu_index: int = 0
    serving_startup_timeout: float = 900.0
    serving_max_model_len: int = 16384
    serving_gpu_memory_utilization: float = 0.90

    #: Interpreter vLLM runs under. Empty uses the engine's own. Set it when
    #: vLLM lives in its own virtualenv, which is common because it pins torch
    #: versions tightly.
    serving_python: str = ""

    #: vLLM's parser for the base model's tool-call syntax. Qwen3 emits Hermes-style
    #: calls. Without a parser vLLM never populates ``message.tool_calls`` and
    #: every instance fails for a reason unrelated to the candidate.
    tool_call_parser: str = "hermes"

    #: Reasoning parser, when the served model emits a separate thinking channel.
    #: Empty disables it. See ``sandbox_enable_thinking`` — this subnet turns
    #: thinking off, so the default is empty.
    reasoning_parser: str = ""
    postgres_dsn: str = ""
    evaluator_image_digest: str = "unpinned"

    #: Independent workers that reconstruct each recipe. Their artifact digests
    #: must agree before the candidate is evaluated. One worker disables the
    #: cross-check, which is acceptable only for local development.
    reconstruction_workers: int = 2

    #: Where the merge arithmetic runs. ``cuda`` is ~30x faster on the trimming
    #: methods, which must decompose a materialised update per projection.
    #:
    #: Consensus-relevant: cuSOLVER and LAPACK do not agree bit-for-bit, so an
    #: artifact digest reproduces only on the same device class. Every published
    #: report records the device it was built on, and every worker in one
    #: deployment must use the same one — which the cross-worker digest check
    #: enforces automatically.
    merge_device: str = "cuda"
    merge_gpu_index: int = 0

    #: Whether an unmeasurable peak-memory reading fails the resource gate. True
    #: on any host that is supposed to have a GPU: a broken counter must not let
    #: every candidate through the memory limit unchecked.
    require_vram_measurement: bool = True

    #: Traces retained per candidate per window for later disclosure. Chosen by
    #: instance identifier rather than by outcome, so the engine cannot retain
    #: only the runs that flatter it. Zero disables disclosure and removes the
    #: strongest check available to anyone outside the operator.
    disclosure_traces: int = 10

    # -- comparator ----------------------------------------------------------
    axis_margin: float = C.DEFAULT_AXIS_MARGIN
    axis_tolerance: float = C.DEFAULT_AXIS_TOLERANCE
    min_dominant_axes: int = C.DEFAULT_MIN_DOMINANT_AXES
    min_axis_samples: int = C.DEFAULT_MIN_AXIS_SAMPLES
    end_to_end_margin: float = C.DEFAULT_END_TO_END_MARGIN
    champion_margin: float = C.DEFAULT_CHAMPION_MARGIN
    champion_margin_decay_blocks: int = C.CHAMPION_MARGIN_DECAY_BLOCKS
    strict_pareto: bool = False

    # -- incentive -----------------------------------------------------------
    incentive_mode: str = C.MODE_GRADED_CONTRIBUTION
    champion_base_share: float = C.CHAMPION_BASE_SHARE
    contribution_memory_windows: int = C.CONTRIBUTION_MEMORY_WINDOWS
    burn_percentage: float = C.DEFAULT_BURN_PERCENTAGE
    burn_uid: int = C.BURN_UID

    #: Share of payable emission spread across queued and former champions, so a
    #: miner waiting its turn is not pruned before it is ever evaluated.
    tail_share: float = C.DEFAULT_TAIL_SHARE

    # -- api -----------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # -- operations ----------------------------------------------------------
    poll_seconds: float = 30.0
    log_level: str = "INFO"
    dry_run: bool = False

    extra: dict[str, Any] = field(default_factory=dict)

    # -- derived -------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir)

    @property
    def database_path(self) -> Path:
        return self.state_path / "engine.sqlite"

    @property
    def recipe_path(self) -> Path:
        return Path(self.recipe_dir) if self.recipe_dir else self.state_path / "recipes"

    def ensure_directories(self) -> None:
        for path in (
            self.state_path,
            Path(self.artifact_cache_dir),
            Path(self.report_dir),
            self.recipe_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """Check for settings that would make a live run unsafe or meaningless.

        Returns:
            Problems, empty when the deployment is sound. The service refuses to
            start on a non-empty list unless it is running in dry-run mode.
        """
        problems: list[str] = []

        if self.hidden_seed_root == 0:
            problems.append(
                "hidden_seed_root is 0. Every deployment must set its own secret root; "
                "leaving the default makes the hidden instances predictable."
            )
        if self.reconstruction_workers < 2:
            problems.append(
                "reconstruction_workers is below 2, which disables the cross-worker "
                "artifact-hash check. A reconstruction bug would then go unnoticed."
            )
        if self.evaluator_image_digest == "unpinned":
            problems.append(
                "evaluator_image_digest is unpinned. Published reports would not "
                "identify the software that produced them."
            )
        if self.min_commit_block <= 0:
            problems.append(
                "min_commit_block is unset, so commitments from before this arena "
                "opened would be admitted."
            )
        if self.disclosure_traces <= 0:
            problems.append(
                "disclosure_traces is 0, so closed windows publish no re-scorable "
                "instances. That removes the only check on the engine's scoring "
                "that does not require trusting the operator."
            )
        if self.merge_device not in ("cpu", "cuda"):
            problems.append(f"unknown merge_device {self.merge_device!r}; expected 'cpu' or 'cuda'")
        if self.serving_mode not in ("managed", "external"):
            problems.append(
                f"unknown serving_mode {self.serving_mode!r}; expected 'managed' or 'external'"
            )
        if self.serving_mode == "managed" and not self.base_model_path:
            problems.append(
                "serving_mode is 'managed' but base_model_path is unset, so there is no "
                "model for the engine to start a runtime from."
            )
        if self.serving_mode == "external":
            problems.append(
                "serving_mode is 'external'. That mode cannot apply a candidate's adapter, "
                "so every candidate and every reference would be measured against the same "
                "static endpoint and no challenger could ever be distinguished. Use "
                "'managed' for any deployment that pays emission."
            )
        if self.incentive_mode not in C.ALLOWED_INCENTIVE_MODES:
            problems.append(
                f"unknown incentive_mode {self.incentive_mode!r}; "
                f"expected one of {list(C.ALLOWED_INCENTIVE_MODES)}"
            )
        if not (0.0 <= self.burn_percentage <= 1.0):
            problems.append("burn_percentage must be between 0 and 1")
        if not (0.0 <= self.tail_share < 1.0):
            problems.append("tail_share must be at least 0 and below 1")

        return problems


def load_settings(path: str | Path | None = None) -> BackendSettings:
    """Load settings from YAML, then apply environment overrides.

    Every field can be overridden with ``CAPSUB_<FIELD_NAME>``. The override is
    coerced to the field's declared type, so ``CAPSUB_WINDOW_BLOCKS=600`` yields
    an integer rather than a string that later fails an arithmetic comparison.
    """
    data: dict[str, Any] = {}

    config_path = Path(path) if path else Path(os.environ.get("CAPSUB_CONFIG", "backend.yaml"))
    if config_path.is_file():
        import yaml

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must contain a mapping at the top level")
        data.update(loaded)

    settings = BackendSettings()
    known = {f.name for f in fields(BackendSettings)}

    for key, value in data.items():
        if key in known:
            setattr(settings, key, _coerce(value, getattr(settings, key)))
        else:
            settings.extra[key] = value

    for f in fields(BackendSettings):
        if f.name == "extra":
            continue
        override = os.environ.get(f"CAPSUB_{f.name.upper()}")
        if override is not None:
            setattr(settings, f.name, _coerce(override, getattr(settings, f.name)))

    return settings
