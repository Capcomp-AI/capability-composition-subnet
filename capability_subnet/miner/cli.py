"""Miner command line.

Everything a miner needs that does not touch the chain: inspect the pool, build a
recipe, validate it, predict its artifact size, and see exactly what would be
committed. Committing itself lives in the neuron, behind an explicit confirmation,
because it is the one irreversible step.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from capability_subnet.common import constants as C
from capability_subnet.common.hashing import canonical_json_str
from capability_subnet.common.logging import setup_logging
from capability_subnet.miner.recipe import (
    RecipeError,
    advise,
    check_recipe,
    describe,
    estimate_artifact_bytes,
    load_recipe,
    new_recipe,
    write_recipe,
)
from capability_subnet.registry.snapshot import load_snapshot


def _cmd_pool(args: argparse.Namespace) -> int:
    snapshot = load_snapshot()
    if args.json:
        sys.stdout.write(canonical_json_str(snapshot.document()))
        return 0

    print(f"source snapshot: {snapshot.sha256}")
    print(f"base model     : {snapshot.manifest.model_repo} @ {snapshot.manifest.revision}")
    print(f"canonical rank : {snapshot.registry.canonical_rank}")
    print("\nadapters:")
    for adapter_id in snapshot.adapter_ids:
        entry = snapshot.registry.get(adapter_id)
        tag = "  [controlled distractor]" if entry.is_distractor else ""
        print(f"  {adapter_id:<34} {entry.capability}{tag}")
        print(f"      {entry.description}")
    print("\nlayer groups:")
    for name, (low, high) in sorted(snapshot.manifest.layer_groups.items()):
        print(f"  {name}: layers {low}–{high}")
    return 0


def _starting_selection(snapshot) -> list[str]:
    """Adapters `init` starts from when none are named.

    The whole pool is not a valid recipe — a selection is bounded above, and the
    bound is well below the number of certified adapters. Taking every one of
    them produced a recipe the schema then rejected, which made a miner's first
    command fail for a reason that looked like the pool's fault.

    The first few in sorted order, deliberately: this is a starting point to
    edit, not a suggestion about which adapters compose well. Choosing that is
    the miner's whole job.
    """
    return sorted(snapshot.registry.capability_adapters())[: C.MAX_SELECTED_ADAPTERS]


def _cmd_init(args: argparse.Namespace) -> int:
    snapshot = load_snapshot()

    try:
        if args.random:
            from capability_subnet.miner.baseline import random_recipe

            recipe = random_recipe(seed=args.seed, snapshot=snapshot)
        else:
            recipe = new_recipe(
                args.adapters or _starting_selection(snapshot),
                combination_type=args.method,
                density=args.density,
                majority_sign_method=args.sign_method,
                random_seed=args.seed,
                output_rank=args.output_rank,
                svd_clamp_quantile=args.clamp,
                snapshot=snapshot,
            )
    except RecipeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    digest = write_recipe(recipe, args.out)
    print(f"wrote {args.out}")
    print(f"recipe digest: {digest}")
    print("\nEdit the coefficients, re-run `validate`, and evaluate locally before committing.")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        recipe = load_recipe(args.recipe)
    except RecipeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    snapshot = load_snapshot()
    problems = check_recipe(recipe, snapshot)
    notes = advise(recipe, snapshot)

    print(describe(recipe, snapshot))

    if notes:
        print("\nadvisories:")
        for note in notes:
            print(f"  - {note}")

    if problems:
        print("\nproblems (the engine would reject this):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("\nThis recipe would be admitted.")
    return 0


def _cmd_digest(args: argparse.Namespace) -> int:
    try:
        recipe = load_recipe(args.recipe)
    except RecipeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(recipe.digest())
    return 0


def _cmd_canonicalise(args: argparse.Namespace) -> int:
    """Rewrite a recipe in the exact form whose digest goes on-chain."""
    try:
        recipe = load_recipe(args.recipe)
    except RecipeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    target = args.out or args.recipe
    digest = write_recipe(recipe, target)
    print(f"wrote {target}")
    print(f"recipe digest: {digest}")
    return 0


def _cmd_size(args: argparse.Namespace) -> int:
    try:
        recipe = load_recipe(args.recipe)
    except RecipeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from capability_subnet.common import constants as C

    snapshot = load_snapshot()
    size = estimate_artifact_bytes(recipe, snapshot)
    limit = C.MAX_ARTIFACT_BYTES

    print(f"output rank : {recipe.compression.output_rank}")
    print(f"artifact    : {size / (1024 * 1024):.1f} MB")
    print(f"gate limit  : {limit / (1024 * 1024):.0f} MB")
    if size > limit:
        print("\nThis exceeds the artifact-size gate. Choose a lower output rank.", file=sys.stderr)
        return 1
    return 0


def _cmd_contract(args: argparse.Namespace) -> int:
    from capability_subnet.workflows import get_workflow

    snapshot = load_snapshot()
    workflow = get_workflow(snapshot.registry.workflow_id)
    contract = workflow.build_contract(snapshot)

    if args.section:
        if args.section not in contract:
            print(
                f"no section {args.section!r}; available: {sorted(contract)}",
                file=sys.stderr,
            )
            return 2
        sys.stdout.write(canonical_json_str(contract[args.section]))
        return 0

    sys.stdout.write(canonical_json_str(contract))
    return 0


def _cmd_pack(args: argparse.Namespace) -> int:
    from capability_subnet.workflows.industrial_maintenance_de_v1.public_pack import generate_pack

    manifest = generate_pack(
        args.out,
        count=args.count,
        ood_count=args.ood_count,
        write_databases=not args.no_databases,
    )
    print(f"wrote {manifest['instance_count']} public and {manifest['ood_count']} OOD instances")
    print(f"tree digest: {manifest['tree_sha256']}")
    return 0


def _cmd_timing(args: argparse.Namespace) -> int:
    """Which run a submission made now would be measured and paid in.

    Submitting inside the run's closing window is not refused — but it costs a
    run, and it is almost never what a miner means, so it is said out loud here
    rather than discovered a run later when nothing was scored.
    """
    from capability_subnet.common import constants as C
    from capability_subnet.common.chain import (
        measuring_run_for,
        run_position,
        weighting_run_for,
    )

    position = run_position(args.block, C.DEFAULT_RUN_BLOCKS)
    minutes = C.MIN_COMMITMENT_AGE_BLOCKS * 12 / 60
    # Asked, not re-derived. The rule has two halves and a copy that keeps only
    # the obvious one is wrong exactly at the boundary this is about.
    measured = measuring_run_for(args.block, C.DEFAULT_RUN_BLOCKS)
    paid = weighting_run_for(args.block, C.DEFAULT_RUN_BLOCKS)

    if position.in_settling_window:
        print(
            f"run {position.run_id} closes in {position.blocks_remaining} blocks, inside the "
            f"{minutes:.0f}-minute settling window.\n"
            f"A submission made now is refused: it has to have been in for "
            f"{C.MIN_COMMITMENT_AGE_BLOCKS} blocks when a run opens to be measured by it, "
            f"and it cannot be.\n"
            f"Nothing is stored and no attempt is spent. Wait for run "
            f"{position.run_id + 1} to open, then send it — it is measured in run "
            f"{position.run_id + 2}."
        )
        return 3 if args.strict_timing else 0

    print(
        f"run {position.run_id}: measured in run {measured}, paid in run {paid}. "
        f"{position.blocks_until_settling_window} blocks "
        f"(~{position.blocks_until_settling_window * 12 / 3600:.1f}h) left to change your mind."
    )
    return 0


def _wallet(args: argparse.Namespace):
    """The hotkey, loaded only when a command actually needs to sign."""
    import bittensor as bt

    return bt.Wallet(args.wallet_name, args.wallet_hotkey, path=args.wallet_path)


def _cmd_check(args: argparse.Namespace) -> int:
    """Ask the service whether a recipe would be admitted. Costs nothing.

    Local `validate` already answers most of this from the shipped pool. This
    asks the *running* engine, which is the one that decides, and needs no
    wallet — so it can be run on any machine, as often as you like.
    """
    import httpx

    try:
        raw = Path(args.recipe).read_text()
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        response = httpx.post(
            f"{args.api_url.rstrip('/')}/check", json={"recipe": raw}, timeout=45.0
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"error: could not reach the submission API: {exc}", file=sys.stderr)
        return 4

    if payload.get("ok"):
        print("ok — this recipe would be admitted")
        return 0
    for problem in payload.get("problems") or []:
        print(f"  {problem}", file=sys.stderr)
    print("\nThis recipe would be refused. Nothing was submitted.", file=sys.stderr)
    return 1


def _cmd_status(args: argparse.Namespace) -> int:
    """What this hotkey has submitted in the current run."""
    import httpx

    hotkey = args.hotkey or _wallet(args).hotkey.ss58_address
    try:
        response = httpx.get(f"{args.api_url.rstrip('/')}/status/{hotkey}", timeout=45.0)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"error: could not reach the submission API: {exc}", file=sys.stderr)
        return 4

    run = payload["run_id"]
    if not payload.get("submitted"):
        print(f"run {run}: nothing submitted. {payload['remaining']} attempts available.")
        return 0

    print(f"run {run}: measured in run {run + 1}, paid in run {run + 2}")
    print(f"  holding   {payload['recipe_sha256']}")
    print(f"  attempts  {payload['submission_count']} used, {payload['remaining']} left")
    for superseded in payload.get("superseded") or []:
        print(f"  replaced  {superseded}")

    admission = payload.get("admission")
    if not admission:
        print("  state     accepted; the engine has not reached it yet")
        return 0

    state = admission.get("state")
    if state == "rejected":
        # Non-zero, so a script that submits and checks notices. This is the
        # one outcome a miner has to act on, and acting on it means submitting
        # a corrected recipe while the run is still open.
        print(f"\n  REJECTED: {admission['reason']}")
        print(
            f"  Nothing will be measured for run {run} unless you submit a "
            f"corrected recipe. {payload['remaining']} attempts left."
        )
        return 1

    if state == "measured":
        print(
            f"  state     measured in run {admission['measured_in_run']}; "
            f"the result opens in run {run + 2}"
        )
    else:
        print(f"  state     {state}, to be measured in run {admission['measured_in_run']}")
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    """Validate, sign and send. Without --confirm, sends nothing."""
    from capability_subnet.miner.neuron import MinerNeuron

    return MinerNeuron(config=args).submit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capcomp",
        description=(
            "Build, check and submit composition recipes for the Capability Composition Subnet."
        ),
        epilog=(
            "A submission is one signed request: nothing goes on chain and "
            "nothing is published anywhere. Start with `capcomp init`, then "
            "`capcomp check`, then `capcomp submit --confirm`."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pool = subparsers.add_parser("pool", help="Show the frozen certified adapter pool.")
    pool.add_argument("--json", action="store_true")
    pool.set_defaults(func=_cmd_pool)

    init = subparsers.add_parser("init", help="Write a starting recipe.")
    init.add_argument("--out", default="recipe.json")
    init.add_argument("--adapters", nargs="*", default=None)
    init.add_argument("--method", default="ties_svd")
    init.add_argument("--density", type=float, default=0.45)
    init.add_argument("--sign-method", dest="sign_method", default="total")
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--output-rank", dest="output_rank", type=int, default=64)
    init.add_argument("--clamp", type=float, default=1.0)
    init.add_argument(
        "--random",
        action="store_true",
        help="Draw a random valid recipe instead of using the arguments above.",
    )
    init.set_defaults(func=_cmd_init)

    validate = subparsers.add_parser(
        "validate", help="Run every check the engine runs at admission."
    )
    validate.add_argument("--recipe", required=True)
    validate.set_defaults(func=_cmd_validate)

    digest = subparsers.add_parser("digest", help="Print a recipe's canonical digest.")
    digest.add_argument("--recipe", required=True)
    digest.set_defaults(func=_cmd_digest)

    canonicalise = subparsers.add_parser(
        "canonicalise",
        help="Rewrite a recipe in its canonical form.",
    )
    canonicalise.add_argument("--recipe", required=True)
    canonicalise.add_argument("--out", default=None)
    canonicalise.set_defaults(func=_cmd_canonicalise)

    size = subparsers.add_parser("size", help="Predict the artifact size without building it.")
    size.add_argument("--recipe", required=True)
    size.set_defaults(func=_cmd_size)

    contract = subparsers.add_parser("contract", help="Print the published workflow contract.")
    contract.add_argument("--section", default=None)
    contract.set_defaults(func=_cmd_contract)

    pack = subparsers.add_parser("pack", help="Generate the public development pack.")
    pack.add_argument("--out", default="data/public_pack")
    pack.add_argument("--count", type=int, default=120)
    pack.add_argument("--ood-count", dest="ood_count", type=int, default=30)
    pack.add_argument("--no-databases", action="store_true")
    pack.set_defaults(func=_cmd_pack)

    def _add_api(sub):
        sub.add_argument(
            "--api.url",
            dest="api_url",
            default=os.environ.get("CAPSUB_API_URL", "https://api.capcomp.ai"),
            help="Submission API. This is where a recipe goes; nothing is put on chain.",
        )

    def _add_wallet(sub):
        sub.add_argument(
            "--wallet.name",
            dest="wallet_name",
            default=os.environ.get("CAPSUB_WALLET_NAME", "default"),
        )
        sub.add_argument(
            "--wallet.hotkey",
            dest="wallet_hotkey",
            default=os.environ.get("CAPSUB_WALLET_HOTKEY", "default"),
        )
        sub.add_argument(
            "--wallet.path",
            dest="wallet_path",
            default=os.environ.get("CAPSUB_WALLET_PATH", "~/.bittensor/wallets"),
        )

    check = subparsers.add_parser(
        "check",
        help="Ask the engine whether a recipe would be admitted. Costs no attempt.",
    )
    check.add_argument("--recipe", required=True)
    _add_api(check)
    check.set_defaults(func=_cmd_check)

    status = subparsers.add_parser(
        "status", help="What this hotkey has submitted in the current run."
    )
    status.add_argument(
        "--hotkey",
        default="",
        help="An ss58 address. Defaults to the wallet's hotkey.",
    )
    _add_api(status)
    _add_wallet(status)
    status.set_defaults(func=_cmd_status)

    submit = subparsers.add_parser(
        "submit", help="Sign and send a recipe. Without --confirm, sends nothing."
    )
    submit.add_argument("--recipe", required=True)
    submit.add_argument("--netuid", type=int, default=int(os.environ.get("CAPSUB_NETUID", "103")))
    submit.add_argument(
        "--subtensor.network", dest="network", default=os.environ.get("CAPSUB_NETWORK", "finney")
    )
    submit.add_argument(
        "--confirm",
        action="store_true",
        help="Actually send. Omit for a dry run that checks everything.",
    )
    submit.add_argument(
        "--log.level", dest="log_level", default=os.environ.get("CAPSUB_LOG_LEVEL", "INFO")
    )
    _add_api(submit)
    _add_wallet(submit)
    submit.set_defaults(func=_cmd_submit)

    timing = subparsers.add_parser(
        "timing", help="Which run a submission made now is measured and paid in."
    )
    timing.add_argument(
        "--block",
        type=int,
        required=True,
        help=(
            "Current chain block. Reports which run would measure a submission "
            "made now, and warns when the run is close enough to closing that it "
            "would be held over to the run after."
        ),
    )
    timing.add_argument(
        "--strict-timing",
        action="store_true",
        help=(
            "Exit non-zero when --block falls inside the settling window, so a "
            "script does not submit into a run that will not measure it."
        ),
    )
    timing.set_defaults(func=_cmd_timing)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging("WARNING")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
