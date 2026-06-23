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
    recipe_dir: str = "state/recipes"

    # -- windows -------------------------------------------------------------
    window_blocks: int = C.DEFAULT_WINDOW_BLOCKS
    hidden_instances: int = C.DEFAULT_HIDDEN_INSTANCES
    ood_instances: int = C.DEFAULT_OOD_INSTANCES

    #: Seed the per-window hidden instance draw is derived from. It must stay
    #: secret: publishing it would publish every future hidden instance.
    hidden_seed_root: int = 0

    # -- evaluation ----------------------------------------------------------
    workflow_id: str = C.DEFAULT_WORKFLOW_ID
    serving_base_url: str = "http://127.0.0.1:8000"
    serving_model_name: str = "candidate"
    dispatch_budget: int = C.DEFAULT_DISPATCH_BUDGET
    postgres_dsn: str = ""
    evaluator_image_digest: str = "unpinned"

    #: Independent workers that reconstruct each recipe. Their artifact digests
    #: must agree before the candidate is evaluated. One worker disables the
    #: cross-check, which is acceptable only for local development.
    reconstruction_workers: int = 2

    #: Whether an unmeasurable peak-memory reading fails the resource gate. True
    #: on any host that is supposed to have a GPU: a broken counter must not let
    #: every candidate through the memory limit unchecked.
    require_vram_measurement: bool = True

    # -- comparator ----------------------------------------------------------
    axis_margin: float = C.DEFAULT_AXIS_MARGIN
    axis_tolerance: float = C.DEFAULT_AXIS_TOLERANCE
    min_dominant_axes: int = C.DEFAULT_MIN_DOMINANT_AXES
    min_axis_samples: int = C.DEFAULT_MIN_AXIS_SAMPLES
    end_to_end_margin: float = C.DEFAULT_END_TO_END_MARGIN
    strict_pareto: bool = False

    # -- incentive -----------------------------------------------------------
    incentive_mode: str = C.MODE_WINNER_TAKE_ALL
    burn_percentage: float = C.DEFAULT_BURN_PERCENTAGE
    burn_uid: int = C.BURN_UID

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
        return Path(self.recipe_dir)

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
        if self.incentive_mode not in (C.MODE_WINNER_TAKE_ALL, C.MODE_GRADED_TOP3):
            problems.append(f"unknown incentive_mode {self.incentive_mode!r}")
        if not (0.0 <= self.burn_percentage <= 1.0):
            problems.append("burn_percentage must be between 0 and 1")

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
