"""Neuron configuration.

Every role composes the shared wallet/network arguments with its own, and the
resulting config object is what the neuron reads at runtime. Environment
variables provide defaults so the same command line works under a process
manager without a long argument list.

The argument surface is defined here rather than borrowed from the SDK. Through
the 10.x line a neuron called ``bt.wallet.add_args(parser)`` and ``bt.config(parser)``;
neither exists in 11.x, where ``bittensor.config`` is a module and wallets are
constructed directly. Owning the parser means an SDK that reorganises its
helpers changes one import in this file instead of breaking every entry point,
and it keeps the flags miners and validators type stable across upgrades.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from capability_subnet.common import constants as C


class Config(SimpleNamespace):
    """Parsed neuron configuration.

    A namespace rather than a dataclass because the roles genuinely carry
    different fields, and every consumer reads it with ``getattr`` defaults.
    """

    def __repr__(self) -> str:  # pragma: no cover - operator convenience
        fields = ", ".join(f"{k}={v!r}" for k, v in sorted(vars(self).items()) if k != "wallet")
        return f"Config({fields})"


def add_wallet_args(parser: argparse.ArgumentParser) -> None:
    """Wallet and network selection, matching the names btcli uses."""
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        type=str,
        default=_env("BT_WALLET_NAME", "default"),
        help="Coldkey wallet name.",
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        default=_env("BT_WALLET_HOTKEY", "default"),
        help="Hotkey name within the wallet.",
    )
    parser.add_argument(
        "--wallet.path",
        dest="wallet_path",
        type=str,
        default=_env("BT_WALLET_PATH", "~/.bittensor/wallets"),
        help="Directory holding the wallets.",
    )
    parser.add_argument(
        "--subtensor.network",
        dest="network",
        type=str,
        default=_env("BT_NETWORK", "finney"),
        help="Named network, or a ws:// / wss:// endpoint.",
    )
    parser.add_argument(
        "--logging.logging_dir",
        dest="logging_dir",
        type=str,
        default=_env("BT_LOGGING_DIR", "~/.bittensor/miners"),
        help="Root directory for neuron state and logs.",
    )


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
        dest="epoch_length",
        type=int,
        default=_env_int("CAPSUB_EPOCH_LENGTH", 100),
        help="Blocks between metagraph resyncs.",
    )
    parser.add_argument(
        "--neuron.device",
        dest="device",
        type=str,
        default=_env("CAPSUB_DEVICE", "cuda"),
        help=(
            "Torch device the merge runs on. 'own' evaluation requires a CUDA "
            "device and refuses to start without one: the trimming methods "
            "decompose a materialised update per projection and are ~30x slower "
            "on a CPU, which is the difference between measuring a queue and "
            "never finishing one."
        ),
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
        dest="neuron_name",
        type=str,
        default="validator",
        help="Name used for this neuron's state directory.",
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
        "--neuron.serve_url",
        dest="serve_url",
        default=_env("CAPSUB_SERVE_URL", ""),
        help=(
            "OpenAI-compatible endpoint this validator serves reconstructed "
            "candidates through. Required by --neuron.evaluation=own; standing "
            "the server up is the validator's own business, exactly as it is the "
            "miner's."
        ),
    )
    parser.add_argument(
        "--neuron.base_model_path",
        dest="base_model_path",
        default=_env("CAPSUB_BASE_MODEL_PATH", "base-model/Qwen3-8B"),
        help=(
            "Local copy of the pinned base model. Required by "
            "--neuron.evaluation=own: a candidate is the base model with its "
            "merged adapter applied, and evaluation runs offline."
        ),
    )
    parser.add_argument(
        "--neuron.serving_python",
        dest="serving_python",
        default=_env("CAPSUB_SERVING_PYTHON", ""),
        help=(
            "Interpreter the serving runtime is started under. Empty uses this "
            "one; set it when vLLM has its own virtualenv, which is usual "
            "because it pins torch tightly."
        ),
    )
    parser.add_argument(
        "--neuron.pool_dir",
        dest="pool_dir",
        default=_env("CAPSUB_POOL_DIR", "pool"),
        help="Certified adapter pool on disk, for reconstructing candidates.",
    )
    parser.add_argument(
        "--neuron.devices",
        dest="devices",
        default=_env("CAPSUB_DEVICES", ""),
        help=(
            "Comma-separated CUDA devices to measure on, e.g. "
            "'cuda:0,cuda:1,cuda:2,cuda:3'. One candidate is measured per device "
            "at a time, because a served package reserves almost the whole card. "
            "Empty measures on every CUDA device this host has, which is almost "
            "always what you want: cards are the unit of parallelism and a run's "
            "throughput is the card count."
        ),
    )
    parser.add_argument(
        "--neuron.max_candidates_per_run",
        dest="max_candidates_per_run",
        type=int,
        default=_env_int("CAPSUB_MAX_CANDIDATES_PER_RUN", 0),
        help=(
            "Stop after this many candidates in one run, taken in commit "
            "order. 0 measures everything eligible. A run that cannot finish "
            "sets no weights at all, so a host slower than its queue should "
            "bound this rather than fall behind silently."
        ),
    )
    parser.add_argument(
        "--incentive_mode",
        dest="incentive_mode",
        type=str,
        choices=list(C.ALLOWED_INCENTIVE_MODES),
        default=_env("CAPSUB_INCENTIVE_MODE", C.MODE_GRADED_CONTRIBUTION),
        help=(
            "How this validator turns the field it measured into weights. "
            "'graded_top3' splits 60/25/15 across the top three; "
            f"'graded_contribution' burns {C.NO_CHAMPION_BURN_SHARE:.0%} and pays "
            f"the leader {C.CHAMPION_BASE_SHARE:.0%} of the rest, with up to "
            f"{C.MAX_GRADED_CONTRIBUTORS - 1} graded runners-up sharing the "
            "remainder."
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
        dest="neuron_name",
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
        "--run_blocks",
        type=int,
        default=_env_int("CAPSUB_RUN_BLOCKS", C.DEFAULT_RUN_BLOCKS),
        help="Blocks per evaluation run.",
    )
    parser.add_argument(
        "--hidden_instances",
        type=int,
        default=_env_int("CAPSUB_HIDDEN_INSTANCES", C.DEFAULT_HIDDEN_INSTANCES),
        help="Hidden workflow instances sampled per run.",
    )
    parser.add_argument(
        "--ood_instances",
        type=int,
        default=_env_int("CAPSUB_OOD_INSTANCES", C.DEFAULT_OOD_INSTANCES),
        help="Out-of-distribution instances sampled per run.",
    )
    parser.add_argument(
        "--incentive_mode",
        type=str,
        choices=list(C.ALLOWED_INCENTIVE_MODES),
        default=_env("CAPSUB_INCENTIVE_MODE", C.MODE_GRADED_CONTRIBUTION),
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


def build_config(role: str) -> Config:
    """Assemble the config object for ``role`` (``miner``/``validator``/``backend``)."""
    parser = argparse.ArgumentParser(
        prog=f"capability-subnet-{role}",
        description=f"Capability Composition Subnet — {role}",
    )
    add_wallet_args(parser)
    add_common_args(parser)
    if role == "validator":
        add_validator_args(parser)
    elif role == "miner":
        add_miner_args(parser)
    elif role == "backend":
        add_backend_args(parser)
    else:
        raise ValueError(f"unknown role {role!r}")

    namespace = parser.parse_args()
    config = Config(**vars(namespace))
    config.role = role
    _finalise(config)
    return config


def _finalise(config: Config) -> None:
    """Derive the state directory and make sure it exists."""
    name = getattr(config, "neuron_name", None)
    if not name:
        return

    full_path = Path(
        os.path.expanduser(
            f"{config.logging_dir}/{config.wallet_name}/{config.wallet_hotkey}"
            f"/netuid{config.netuid}/{name}"
        )
    )
    full_path.mkdir(parents=True, exist_ok=True)
    config.full_path = str(full_path)


def parse_trusted_signers(raw: str) -> set[str] | None:
    """Turn the comma-separated allow-list into a set.

    An empty value returns ``None``, which disables signature enforcement. That
    is intentional for local development and dangerous anywhere else, so callers
    log loudly when they see it.
    """
    entries = {item.strip() for item in (raw or "").split(",") if item.strip()}
    return entries or None
