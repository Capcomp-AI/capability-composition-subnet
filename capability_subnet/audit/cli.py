"""Audit command line.

Point it at a running engine, or at a directory of reports somebody handed you,
and it will tell you whether what was published is self-consistent.

Nothing here needs the operator's cooperation beyond a readable API, and nothing
needs a GPU. That is the point: the checks are only worth having if a sceptic
can run them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from capability_subnet.audit.verify import AuditResult, audit_window, verify_report
from capability_subnet.common.logging import setup_logging
from capability_subnet.common.schemas import (
    EvaluationReport,
    WeightVector,
    WindowDisclosure,
)


def _signers(raw: str | None) -> set[str] | None:
    entries = {item.strip() for item in (raw or "").split(",") if item.strip()}
    return entries or None


def _report_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _load_local(directory: Path) -> list[EvaluationReport]:
    reports: list[EvaluationReport] = []
    for path in sorted(directory.rglob("*.json")):
        try:
            reports.append(EvaluationReport.model_validate_json(path.read_text("utf-8")))
        except Exception:  # noqa: BLE001 - a directory may hold unrelated files
            continue
    return reports


def _fetch(base_url: str, path: str, timeout: float):
    import httpx

    response = httpx.get(f"{base_url.rstrip('/')}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def _emit(result: AuditResult, as_json: bool) -> int:
    if as_json:
        json.dump(
            {
                "ok": result.ok,
                "reports_checked": result.reports_checked,
                "findings": [
                    {
                        "code": f.code,
                        "severity": f.severity,
                        "subject": f.subject,
                        "detail": f.detail,
                    }
                    for f in result.findings
                ],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        for finding in result.findings:
            print(f"  {finding}")
        print(f"\n{result.summary()}")
        if result.ok and not result.warnings:
            print("Everything published is consistent with everything else published.")
        elif result.ok:
            print("No contradictions found. See the warnings above.")
        else:
            print(
                "\nAt least one published claim contradicts another. That is not a\n"
                "difference of opinion about a candidate — it means the record itself\n"
                "does not hold together."
            )

    return 0 if result.ok else 1


def _cmd_window(args: argparse.Namespace) -> int:
    trusted = _signers(args.trusted_signers)
    if trusted is None:
        print(
            "warning: no --trusted-signers given, so signatures are checked for\n"
            "         validity but not for who produced them.\n",
            file=sys.stderr,
        )

    try:
        if args.reports_dir:
            reports = _load_local(Path(args.reports_dir))
            if args.window is not None:
                reports = [r for r in reports if r.window_id == args.window]
            vector = None
            if args.weights:
                vector = WeightVector.model_validate_json(
                    Path(args.weights).read_text("utf-8")
                )
        else:
            path = (
                f"/reports?window_id={args.window}&limit=200"
                if args.window is not None
                else "/reports?limit=200"
            )
            payload = _fetch(args.backend_url, path, args.timeout)
            reports = [EvaluationReport.model_validate(r) for r in payload["reports"]]
            weights_path = (
                f"/weights/{args.window}" if args.window is not None else "/weights"
            )
            try:
                vector = WeightVector.model_validate(
                    _fetch(args.backend_url, weights_path, args.timeout)
                )
            except Exception:  # noqa: BLE001 - a window may predate any vector
                vector = None
    except Exception as exc:  # noqa: BLE001
        return _report_error(str(exc))

    if not reports:
        return _report_error("no reports found to audit")

    result = audit_window(
        reports,
        vector,
        trusted_signers=trusted,
        expected_snapshot=args.snapshot or None,
        expected_base_revision=args.base_revision or None,
    )
    return _emit(result, args.json)


def _cmd_report(args: argparse.Namespace) -> int:
    try:
        if args.file:
            report = EvaluationReport.model_validate_json(
                Path(args.file).read_text("utf-8")
            )
        else:
            report = EvaluationReport.model_validate(
                _fetch(args.backend_url, f"/reports/{args.digest}", args.timeout)
            )
    except Exception as exc:  # noqa: BLE001
        return _report_error(str(exc))

    result = verify_report(
        report,
        trusted_signers=_signers(args.trusted_signers),
        expected_snapshot=args.snapshot or None,
        expected_base_revision=args.base_revision or None,
    )
    return _emit(result, args.json)


def _cmd_recipe(args: argparse.Namespace) -> int:
    """Confirm a published recipe reconstructs to the digest a report claims.

    This is the one check that needs the certified pool materialised locally,
    and it is the strongest: it turns "the operator says this artifact came from
    that recipe" into something anyone with the pool can confirm.
    """
    from capability_subnet.merge_engine.engine import reconstruct
    from capability_subnet.merge_engine.loader import SafetensorsAdapterSource
    from capability_subnet.miner.recipe import load_recipe
    from capability_subnet.registry.snapshot import load_snapshot

    try:
        recipe = load_recipe(args.recipe)
        snapshot = load_snapshot()
        source = SafetensorsAdapterSource(args.pool)
        built = reconstruct(recipe, snapshot, source)
    except Exception as exc:  # noqa: BLE001
        return _report_error(str(exc))

    print(f"  recipe digest  : {recipe.digest()}")
    print(f"  rebuilt artifact: {built.artifact_sha256}")

    if args.expect:
        matches = built.artifact_sha256 == args.expect
        print(f"  claimed artifact: {args.expect}")
        print(f"\n{'MATCH' if matches else 'MISMATCH'}")
        if not matches:
            print(
                "\nThe recipe does not reconstruct to the artifact the report claims.\n"
                "Either the software differs from the evaluator image, or the report\n"
                "does not describe what was actually evaluated.",
                file=sys.stderr,
            )
            return 1
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    """Regenerate a closed window's instances and re-score its published traces.

    The strongest check available without the operator's cooperation, and it
    needs no GPU: instance generation is a pure function of the seed, and the
    scorer reads nothing but the trace.
    """
    from capability_subnet.audit.replay import verify_disclosure

    try:
        if args.file:
            disclosure = WindowDisclosure.model_validate_json(
                Path(args.file).read_text("utf-8")
            )
        else:
            disclosure = WindowDisclosure.model_validate(
                _fetch(
                    args.backend_url,
                    f"/windows/{args.window}/disclosure",
                    args.timeout,
                )
            )
    except Exception as exc:  # noqa: BLE001
        return _report_error(str(exc))

    outcome, result = verify_disclosure(
        disclosure, trusted_signers=_signers(args.trusted_signers)
    )

    if not args.json:
        print(f"  window {disclosure.window_id}")
        print(
            f"  seeds disclosed : "
            f"{len(disclosure.hidden_seeds) + len(disclosure.ood_seeds)}"
        )
        print(f"  re-scored       : {outcome.summary()}\n")

    code = _emit(result, args.json)
    return code if code else (0 if outcome.ok else 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capability-audit",
        description=(
            "Independently verify what the evaluation engine published. Needs no "
            "GPU and no cooperation from the operator beyond a readable API."
        ),
    )
    parser.add_argument("--backend-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--trusted-signers",
        default="",
        help="Comma-separated operator hotkeys whose signatures you accept.",
    )
    parser.add_argument("--snapshot", default="", help="Expected pool snapshot digest.")
    parser.add_argument("--base-revision", default="", help="Expected base revision.")
    parser.add_argument("--json", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    window = subparsers.add_parser(
        "window", help="Audit a window's reports and the weight vector derived from them."
    )
    window.add_argument("--window", type=int, default=None)
    window.add_argument("--reports-dir", default=None, help="Audit local files instead.")
    window.add_argument("--weights", default=None, help="Local weight vector JSON.")
    window.set_defaults(func=_cmd_window)

    report = subparsers.add_parser("report", help="Audit a single report.")
    report.add_argument("--digest", default=None)
    report.add_argument("--file", default=None)
    report.set_defaults(func=_cmd_report)

    recipe = subparsers.add_parser(
        "recipe", help="Rebuild a published recipe and compare its artifact digest."
    )
    recipe.add_argument("--recipe", required=True)
    recipe.add_argument("--pool", default="pool")
    recipe.add_argument("--expect", default=None, help="Artifact digest the report claims.")
    recipe.set_defaults(func=_cmd_recipe)

    replay = subparsers.add_parser(
        "replay",
        help="Re-score a closed window from its published disclosure. No GPU needed.",
    )
    replay.add_argument("--window", type=int, default=None)
    replay.add_argument("--file", default=None, help="Local disclosure JSON.")
    replay.set_defaults(func=_cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging("WARNING")
    args = build_parser().parse_args(argv)

    if args.command == "report" and not (args.digest or args.file):
        return _report_error("pass --digest or --file")
    if args.command == "replay" and args.window is None and not args.file:
        return _report_error("pass --window or --file")

    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
