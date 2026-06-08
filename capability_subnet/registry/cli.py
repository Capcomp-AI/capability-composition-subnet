"""Registry command line.

Operator-facing tooling for inspecting the frozen pool and running admission
gates against a candidate adapter directory.
"""

from __future__ import annotations

import argparse
import json
import sys

from capability_subnet.common.hashing import canonical_json_str
from capability_subnet.common.logging import setup_logging
from capability_subnet.registry.certification import certify_adapter
from capability_subnet.registry.snapshot import load_snapshot


def _cmd_snapshot(args: argparse.Namespace) -> int:
    snapshot = load_snapshot()
    document = snapshot.document()
    if args.json:
        sys.stdout.write(canonical_json_str(document))
        return 0

    print(f"source snapshot : {document['source_snapshot_sha256']}")
    print(f"base model      : {document['base_model_repo']} @ {document['base_revision']}")
    print(f"canonical rank  : {document['canonical_rank']}")
    print(f"adapters        : {document['adapter_count']}")
    for adapter_id in document["adapters"]:
        entry = snapshot.registry.get(adapter_id)
        tag = " [distractor]" if entry.is_distractor else ""
        state = "certified" if entry.certified else "PENDING CERTIFICATION"
        print(f"  - {adapter_id:<34} {entry.capability:<30} {state}{tag}")
    if not document["fully_certified"]:
        print(
            "\nThis pool is not fully certified and must not be used on a live network.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    snapshot = load_snapshot()
    payload = [
        {
            "adapter_id": entry.adapter_id,
            "capability": entry.capability,
            "license": entry.license,
            "rank": entry.rank,
            "lora_alpha": entry.lora_alpha,
            "is_distractor": entry.is_distractor,
            "certified": entry.certified,
            "artifact_sha256": entry.artifact_sha256,
            "known_overlaps": list(entry.known_overlaps),
            "description": entry.description,
        }
        for entry in sorted(snapshot.registry.adapters, key=lambda e: e.adapter_id)
    ]
    sys.stdout.write(canonical_json_str(payload))
    return 0


def _cmd_certify(args: argparse.Namespace) -> int:
    snapshot = load_snapshot()
    entry = None
    if snapshot.registry.contains(args.adapter_id):
        entry = snapshot.registry.get(args.adapter_id)

    outcome = certify_adapter(
        args.directory,
        adapter_id=args.adapter_id,
        manifest=snapshot.manifest,
        license_id=args.license or (entry.license if entry else "unknown"),
        declared_allows_derivatives=(
            args.allows_derivatives
            if args.allows_derivatives is not None
            else (entry.license_allows_derivatives if entry else False)
        ),
        capability_score=args.capability_score,
        base_retention=args.base_retention,
        converted_from_rank=args.converted_from_rank,
        recertified_after_conversion=args.recertified,
        min_capability_score=args.min_capability_score,
    )

    print(outcome.summary())
    for gate in outcome.gates:
        marker = "PASS" if gate.passed else "FAIL"
        print(f"  [{marker}] {gate.name:<14} {gate.detail}")

    if outcome.admitted:
        print(f"\nartifact_sha256: {outcome.artifact_sha256}")
        print("Record this digest in the registry entry, then regenerate the snapshot digest.")
        return 0
    return 1


def _cmd_verify(args: argparse.Namespace) -> int:
    """Check that a recipe's declared scope matches the frozen pool."""
    snapshot = load_snapshot()
    with open(args.recipe, encoding="utf-8") as handle:
        recipe = json.load(handle)

    problems = snapshot.validate_recipe_scope(
        recipe.get("base_revision", ""), recipe.get("source_snapshot_sha256", "")
    )
    unknown = snapshot.registry.unknown_ids(recipe.get("selected_adapters", []))
    if unknown:
        problems.append(f"recipe selects adapters that are not in the pool: {unknown}")

    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("recipe scope matches the frozen pool")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capability-registry",
        description="Inspect the certified adapter pool and run admission gates.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser(
        "snapshot", help="Print the frozen snapshot digest and pool contents."
    )
    snapshot_parser.add_argument("--json", action="store_true", help="Emit JSON.")
    snapshot_parser.set_defaults(func=_cmd_snapshot)

    list_parser = subparsers.add_parser("list", help="List every certified adapter as JSON.")
    list_parser.set_defaults(func=_cmd_list)

    certify_parser = subparsers.add_parser(
        "certify", help="Run every admission gate against an adapter directory."
    )
    certify_parser.add_argument("directory", help="Directory holding the adapter files.")
    certify_parser.add_argument("--adapter-id", required=True)
    certify_parser.add_argument("--license", default=None)
    certify_parser.add_argument(
        "--allows-derivatives",
        dest="allows_derivatives",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Assert whether the license permits redistributing derivative weights.",
    )
    certify_parser.add_argument("--capability-score", type=float, default=None)
    certify_parser.add_argument("--base-retention", type=float, default=None)
    certify_parser.add_argument("--converted-from-rank", type=int, default=None)
    certify_parser.add_argument("--recertified", action="store_true")
    certify_parser.add_argument("--min-capability-score", type=float, default=0.60)
    certify_parser.set_defaults(func=_cmd_certify)

    verify_parser = subparsers.add_parser(
        "verify-recipe", help="Check a recipe's declared scope against the frozen pool."
    )
    verify_parser.add_argument("--recipe", required=True)
    verify_parser.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
