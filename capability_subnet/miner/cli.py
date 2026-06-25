"""Miner command line.

Everything a miner needs that does not touch the chain: inspect the pool, build a
recipe, validate it, predict its artifact size, and see exactly what would be
committed. Committing itself lives in the neuron, behind an explicit confirmation,
because it is the one irreversible step.
"""

from __future__ import annotations

import argparse
import sys

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


def _cmd_init(args: argparse.Namespace) -> int:
    snapshot = load_snapshot()
    adapters = args.adapters or list(snapshot.registry.capability_adapters())

    try:
        recipe = new_recipe(
            adapters,
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


def _cmd_commitment(args: argparse.Namespace) -> int:
    """Show the exact payload that would be written on-chain."""
    from capability_subnet.common.commitments import CommitmentError, encode_commitment

    try:
        recipe = load_recipe(args.recipe)
        payload = encode_commitment(recipe.workflow_id, recipe.digest(), args.recipe_uri)
    except (RecipeError, CommitmentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(payload)
    print(f"\n{len(payload.encode('utf-8'))} bytes", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capability-miner",
        description="Build, validate and inspect composition recipes.",
        epilog="Committing a recipe is done with `python neurons/miner.py --confirm`.",
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
        help="Rewrite a recipe in the exact form whose digest is committed.",
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

    commitment = subparsers.add_parser(
        "commitment", help="Show the exact on-chain payload for a recipe."
    )
    commitment.add_argument("--recipe", required=True)
    commitment.add_argument("--recipe-uri", dest="recipe_uri", required=True)
    commitment.set_defaults(func=_cmd_commitment)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging("WARNING")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
