"""Workflow command line.

Generate the public pack, inspect a single instance, print the contract, and run
the reference solver against generated instances to confirm the environment is
solvable.

That last command is the one worth knowing about. If the reference solver cannot
complete an instance, the generator, the tools or the scorer is broken, and every
candidate score from that instance is meaningless. Running it after any change to
the workflow is the cheapest way to find that out.
"""

from __future__ import annotations

import argparse
import sys

from capability_subnet.common.hashing import canonical_json_str
from capability_subnet.common.logging import setup_logging
from capability_subnet.workflows import available_workflows, get_workflow


def _cmd_generate_pack(args: argparse.Namespace) -> int:
    from capability_subnet.workflows.industrial_maintenance_de_v1.public_pack import generate_pack

    manifest = generate_pack(
        args.out,
        count=args.count,
        ood_count=args.ood_count,
        seed=args.seed,
        include_truth=not args.without_truth,
        write_databases=not args.no_databases,
    )
    print(f"wrote {manifest['instance_count']} public and {manifest['ood_count']} OOD instances")
    print(f"pack seed  : {manifest['pack_seed']}")
    print(f"tree digest: {manifest['tree_sha256']}")
    print(
        "\nRegenerating from the same seed reproduces this pack exactly. "
        "Compare tree digests to confirm your copy matches the published one."
    )
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    workflow = get_workflow(args.workflow)
    instance = workflow.generate_instance(args.seed, split=args.split)

    # The rendering below reads the maintenance workflow's instance fields, which
    # only that workflow has. `--workflow` selects freely, so anything else gets
    # the generic rendering rather than an AttributeError.
    if not hasattr(instance, "machine_model_de"):
        return _show_generic(instance, args)

    if args.json:
        from capability_subnet.workflows.industrial_maintenance_de_v1.public_pack import (
            instance_to_dict,
        )

        sys.stdout.write(
            canonical_json_str(instance_to_dict(instance, include_truth=args.with_truth))
        )
        return 0

    print(f"instance   : {instance.instance_id}")
    print(f"machine    : {instance.machine_model_de} ({instance.machine_id})")
    print(f"location   : {instance.machine_location_de}")
    print(f"sensor     : {instance.sensor_label_de} [{instance.sensor_unit}]")
    if instance.ood_mutations:
        print(f"mutations  : {', '.join(instance.ood_mutations)}")

    print("\n--- task ---")
    print(instance.task_prompt_de)

    print("\n--- manual sections ---")
    for section in instance.manual:
        print(f"  {section.section_id}  {section.title_de}")

    print("\n--- sensor log (first 12 lines) ---")
    print("\n".join(instance.sensor_log.splitlines()[:12]))

    print("\n--- database ---")
    print(instance.database.schema_description_de)

    if args.with_truth:
        truth = instance.truth
        print("\n--- ground truth (never shown to a candidate) ---")
        print(f"  fault code        : {truth.fault_code}")
        print(f"  component         : {truth.component_label_de}")
        print(f"  manual reference  : {truth.manual_reference}")
        print(f"  SQL answer        : {truth.sql_expected_rows}")
        print(f"  diagnostic        : {truth.diagnostic_expected}")
        print(f"  part              : {truth.required_part_number} × {truth.required_quantity}")
        print(f"  safety steps      : {', '.join(truth.required_safety_steps)}")
        print(f"  recommendation    : {truth.recommendation}")

    return 0


def _show_generic(instance: object, args: argparse.Namespace) -> int:
    """Render any workflow's instance from the fields every instance has.

    Deliberately field-driven rather than workflow-specific: a third workflow
    should be inspectable the day it is registered, without editing this file.
    """
    from dataclasses import asdict, is_dataclass

    fields = asdict(instance) if is_dataclass(instance) else dict(vars(instance))

    if args.json:
        sys.stdout.write(canonical_json_str(fields))
        return 0

    question = fields.pop("question", "")
    cases = fields.pop("cases", ()) or ()
    answer = fields.pop("answer", "")

    for key, value in fields.items():
        print(f"{key:11s}: {value}")

    print("\n--- task ---")
    print(question)

    if cases:
        print(f"\n--- test cases ({len(cases)}) ---")
        for number, case in enumerate(cases[:3], start=1):
            stdin = case["stdin"] if isinstance(case, dict) else case.stdin
            expected = case["expected_stdout"] if isinstance(case, dict) else case.expected_stdout
            print(f"  {number}. stdin={stdin[:60]!r} -> {expected[:60]!r}")
        if len(cases) > 3:
            print(f"  ... {len(cases) - 3} more")

    if args.with_truth and answer:
        print("\n--- ground truth (never shown to a candidate) ---")
        print(f"  {answer}")
    elif answer:
        print("\n(pass --with-truth to show the expected answer)")

    return 0


def _cmd_contract(args: argparse.Namespace) -> int:
    workflow = get_workflow(args.workflow)
    sys.stdout.write(canonical_json_str(workflow.build_contract()))
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    """Confirm the environment is solvable.

    Runs the scripted reference solver — which knows the answers, and is
    therefore not a candidate — against freshly generated instances. Every one
    should reach end-to-end success. Any failure is a bug in the workflow, not a
    hard instance.
    """
    from capability_subnet.sandbox.orchestrator import run_instance
    from capability_subnet.sandbox.reference_solver import ReferenceSolverClient

    workflow = get_workflow(args.workflow)
    failures = 0

    for offset in range(args.count):
        seed = args.seed + offset
        instance = workflow.generate_instance(seed, split=args.split)
        outcome = run_instance(instance, ReferenceSolverClient(instance))
        result = outcome.result

        if result.end_to_end_success:
            print(f"  {instance.instance_id}  ok ({result.turns_used} turns)")
            continue

        failures += 1
        failed = [name for name, stage in result.stages.items() if not stage.passed]
        print(f"  {instance.instance_id}  FAILED: {failed or result.error}", file=sys.stderr)
        for name in failed:
            print(f"      {name}: {result.stages[name].detail}", file=sys.stderr)

    print(f"\n{args.count - failures} of {args.count} instances solved")

    if failures:
        print(
            "\nThe reference solver knows every answer, so a failure here means the "
            "environment itself is broken. Candidate scores from these instances "
            "would not mean anything.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    del args
    for workflow_id in available_workflows():
        workflow = get_workflow(workflow_id)
        print(f"{workflow_id}  {workflow.title}")
        print(f"  stages: {', '.join(workflow.stages)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capability-workflow",
        description="Generate, inspect and self-test workflow instances.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser("generate-public-pack", help="Write the public development pack.")
    pack.add_argument("--out", default="data/public_pack")
    pack.add_argument("--count", type=int, default=120)
    pack.add_argument("--ood-count", dest="ood_count", type=int, default=30)
    pack.add_argument("--seed", type=int, default=20260701)
    pack.add_argument("--without-truth", action="store_true")
    pack.add_argument("--no-databases", action="store_true")
    pack.set_defaults(func=_cmd_generate_pack)

    show = subparsers.add_parser("show", help="Print one generated instance.")
    show.add_argument("--seed", type=int, required=True)
    show.add_argument("--split", default="public", choices=["public", "hidden", "ood"])
    show.add_argument("--workflow", default="industrial_maintenance_de_v1")
    show.add_argument("--with-truth", action="store_true")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=_cmd_show)

    contract = subparsers.add_parser("contract", help="Print the published contract.")
    contract.add_argument("--workflow", default="industrial_maintenance_de_v1")
    contract.set_defaults(func=_cmd_contract)

    selftest = subparsers.add_parser("selftest", help="Confirm generated instances are solvable.")
    selftest.add_argument("--count", type=int, default=10)
    selftest.add_argument("--seed", type=int, default=1)
    selftest.add_argument("--split", default="hidden", choices=["public", "hidden", "ood"])
    selftest.add_argument("--workflow", default="industrial_maintenance_de_v1")
    selftest.set_defaults(func=_cmd_selftest)

    listing = subparsers.add_parser("list", help="List available workflows.")
    listing.set_defaults(func=_cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging("WARNING")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
