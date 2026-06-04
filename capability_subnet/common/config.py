"""Neuron configuration.

Follows the standard Bittensor neuron pattern: every role composes wallet,
subtensor and logging arguments with its own, and the resulting config object is
what the neuron reads at runtime. Environment variables provide defaults so the
same command line works under a process manager without a long argument list.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import bittensor as bt

from capability_subnet.common import constants as C


def _env(name: str, default):
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments every role shares."""
    parser.add_argument(
        "--netuid",
        type=int,
        default=_env_int("CAPSUB_NETUID", 1),
        help="Subnet netuid.",
    )
    parser.add_argument(
        "--workflow_id",
        type=str,
        default=_env("CAPSUB_WORKFLOW_ID", C.DEFAULT_WORKFLOW_ID),
        help="Workflow this neuron participates in.",
    )
    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        default=_env_int("CAPSUB_EPOCH_LENGTH", 100),
        help="Blocks between metagraph resyncs.",
    )
    parser.add_argument(
        "--neuron.device",
        type=str,
        default=_env("CAPSUB_DEVICE", "cpu"),
        help="Torch device for any local computation.",
    )
    parser.add_argument(
        "--logging.level",
        dest="log_level",
        type=str,
        default=_env("CAPSUB_LOG_LEVEL", "INFO"),
        help="Python logging level for this subnet's own loggers.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=_env_bool("CAPSUB_MOCK"),
        help="Run against in-process mocks instead of a live chain.",
    )


def add_validator_args(parser: argparse.ArgumentParser) -> None:
    """Thin validator arguments.

    The validator does no evaluation. It polls the engine's read-only API,
    verifies the signature on the published weight vector, and sets weights.
    """
    parser.add_argument(
        "--neuron.name",
        type=str,
        default="validator",
        help="Name used for this neuron's state directory.",
    )
    parser.add_argument(
        "--backend.url",
        dest="backend_url",
        type=str,
        default=_env("CAPSUB_BACKEND_URL", "http://127.0.0.1:8080"),
        help="Base URL of the evaluation engine's read-only API.",
    )
    parser.add_argument(
        "--backend.timeout",
        dest="backend_timeout",
        type=float,
        default=_env_float("CAPSUB_BACKEND_TIMEOUT", 30.0),
        help="HTTP timeout in seconds when polling the engine.",
    )
    parser.add_argument(
        "--backend.trusted_signers",
        dest="trusted_signers",
        type=str,
        default=_env("CAPSUB_TRUSTED_SIGNERS", ""),
        help=(
            "Comma-separated operator hotkeys whose signatures this validator "
            "accepts. Leave empty ONLY for local development."
        ),
    )
    parser.add_argument(
        "--backend.allow_unsigned",
        dest="allow_unsigned",
        action="store_true",
        default=_env_bool("CAPSUB_ALLOW_UNSIGNED"),
        help=(
            "Accept weight vectors that carry no verifiable operator signature. "
            "Without this flag an empty trusted-signer list is a startup error, "
            "because silently trusting whatever answers at the configured URL is "
            "not a default anyone should get by omission."
        ),
    )
    parser.add_argument(
        "--neuron.weight_interval",
        dest="weight_interval",
        type=int,
        default=_env_int("CAPSUB_WEIGHT_INTERVAL", 300),
        help="Minimum blocks between weight submissions.",
    )
    parser.add_argument(
        "--neuron.poll_interval",
        dest="poll_interval",
        type=float,
        default=_env_float("CAPSUB_POLL_INTERVAL", 60.0),
        help="Seconds between polls of the engine API.",
    )
    parser.add_argument(
        "--neuron.burn_percentage",
        dest="burn_percentage",
        type=float,
        default=_env_float("CAPSUB_BURN_PERCENTAGE", C.DEFAULT_BURN_PERCENTAGE),
        help=(
            "Additional fraction of emission this validator routes to the burn "
            "UID on top of whatever the published vector already burns."
        ),
    )
    parser.add_argument(
        "--neuron.disable_set_weights",
        dest="disable_set_weights",
        action="store_true",
        default=_env_bool("CAPSUB_DISABLE_SET_WEIGHTS"),
        help="Compute and log the weight vector without submitting it.",
    )
    parser.add_argument(
        "--neuron.max_stale_windows",
        dest="max_stale_windows",
        type=int,
        default=_env_int("CAPSUB_MAX_STALE_WINDOWS", 3),
        help=(
            "Stop resubmitting a published vector once it is this many windows "
            "behind the chain head. Prevents a dead engine from pinning "
            "emission to a stale champion forever."
        ),
    )


def add_miner_args(parser: argparse.ArgumentParser) -> None:
    """Miner arguments.

    A miner's on-chain action is a single commitment. Everything else — the
    composition search itself — happens privately, on whatever hardware the
    miner chooses, and is not configured here.
    """
    parser.add_argument(
        "--neuron.name",
        type=str,
        default="miner",
        help="Name used for this neuron's state directory.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=_env("CAPSUB_RECIPE", ""),
        help="Path to the recipe JSON file to submit.",
    )
    parser.add_argument(
        "--recipe_uri",
        type=str,
        default=_env("CAPSUB_RECIPE_URI", ""),
        help=(
            "Immutable, content-addressed location where the exact recipe bytes "
            "are published (hf:, ipfs: or https:)."
        ),
    )
    parser.add_argument(
        "--backend.url",
        dest="backend_url",
        type=str,
        default=_env("CAPSUB_BACKEND_URL", "http://127.0.0.1:8080"),
        help="Engine API, used to read the contract and check submission status.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help=(
            "Actually submit. Without it the miner performs every check and "
            "prints what it would commit. One recipe per hotkey is final."
        ),
    )


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    """Evaluation engine arguments (owner-operated)."""
    parser.add_argument(
        "--state_dir",
        type=str,
        default=_env("CAPSUB_STATE_DIR", "state"),
        help="Directory holding the engine database, artifact cache and reports.",
    )
    parser.add_argument(
        "--window_blocks",
        type=int,
        default=_env_int("CAPSUB_WINDOW_BLOCKS", C.DEFAULT_WINDOW_BLOCKS),
        help="Blocks per evaluation window.",
    )
    parser.add_argument(
        "--hidden_instances",
        type=int,
        default=_env_int("CAPSUB_HIDDEN_INSTANCES", C.DEFAULT_HIDDEN_INSTANCES),
        help="Hidden workflow instances sampled per window.",
    )
    parser.add_argument(
        "--ood_instances",
        type=int,
        default=_env_int("CAPSUB_OOD_INSTANCES", C.DEFAULT_OOD_INSTANCES),
        help="Out-of-distribution instances sampled per window.",
    )
    parser.add_argument(
        "--incentive_mode",
        type=str,
        choices=[C.MODE_WINNER_TAKE_ALL, C.MODE_GRADED_TOP3],
        default=_env("CAPSUB_INCENTIVE_MODE", C.MODE_WINNER_TAKE_ALL),
        help="Emission split policy.",
    )
    parser.add_argument(
        "--api_host",
        type=str,
        default=_env("CAPSUB_API_HOST", "0.0.0.0"),
        help="Bind address for the read-only API.",
    )
    parser.add_argument(
        "--api_port",
        type=int,
        default=_env_int("CAPSUB_API_PORT", 8080),
        help="Port for the read-only API.",
    )


def build_config(role: str) -> bt.Config:
    """Assemble the config object for ``role`` (``miner``/``validator``/``backend``)."""
    parser = argparse.ArgumentParser(
        prog=f"capability-subnet-{role}",
        description=f"Capability Composition Subnet — {role}",
    )
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    add_common_args(parser)
    if role == "validator":
        add_validator_args(parser)
    elif role == "miner":
        add_miner_args(parser)
    elif role == "backend":
        add_backend_args(parser)
    else:
        raise ValueError(f"unknown role {role!r}")

    config = bt.config(parser)
    config.role = role
    _finalise(config)
    return config


def _finalise(config: bt.Config) -> None:
    """Derive the state directory and make sure it exists."""
    neuron = getattr(config, "neuron", None)
    name = getattr(neuron, "name", None) if neuron is not None else None
    if name is None:
        return

    full_path = Path(
        os.path.expanduser(
            f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/{name}"
        )
    )
    full_path.mkdir(parents=True, exist_ok=True)
    config.neuron.full_path = str(full_path)


def parse_trusted_signers(raw: str) -> set[str] | None:
    """Turn the comma-separated allow-list into a set.

    An empty value returns ``None``, which disables signature enforcement. That
    is intentional for local development and dangerous anywhere else, so callers
    log loudly when they see it.
    """
    entries = {item.strip() for item in (raw or "").split(",") if item.strip()}
    return entries or None
