#!/usr/bin/env python3
"""Materialise the certified pool from the public adapters the registry names.

Run once per arena version, by the operator, before genesis:

    python scripts/import_public_adapters.py --out pool

For every entry in ``registry/data/adapter_registry.json`` this fetches the two
files it needs at the pinned upstream revision, refuses anything whose config
has drifted from what the registry recorded, re-factorises the update to the
canonical rank, writes the normalised artifact, and hashes it.

An adapter already on disk whose bytes still hash to the digest the registry
recorded is verified and skipped, so re-running is cheap and does not touch
artifacts that are already correct. A digest that does not match is re-imported
rather than trusted, and the reason is logged. ``--force`` re-imports everything
regardless, which is what to use after a change to how normalisation computes.

The digest is then written back into the registry. That is the point of the
exercise: until an adapter has been imported, its ``artifact_sha256`` is a
placeholder and the engine's preflight refuses to start.

Certification is deliberately *not* automatic. ``--certify`` records the
structural gates, but ``capability_score`` and ``base_retention`` are
measurements, not defaults, and have to be supplied by the operator from an
actual evaluation run:

    capability-registry certify --adapter code-generation-v1 \
        --capability-score 0.71 --base-retention 0.994
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from capability_subnet.common.logging import setup_logging
from capability_subnet.registry.adapters import load_registry
from capability_subnet.registry.base_model import load_base_manifest
from capability_subnet.registry.importer import ImportError_, SourceSpec, import_adapter

log = logging.getLogger("import_public_adapters")

REGISTRY_PATH = Path(__file__).resolve().parents[1] / (
    "capability_subnet/registry/data/adapter_registry.json"
)


def _already_verified(out_dir: Path, adapter_id: str, recorded: str | None) -> tuple[bool, str]:
    """Whether this adapter is on disk and still matches its recorded digest.

    Returns:
        ``(verified, detail)``. ``detail`` is empty when there was nothing to
        check — no artifact, or no digest recorded yet — and otherwise says what
        was found, so a re-import is never silent about why.
    """
    from capability_subnet.common.hashing import sha256_file

    artifact = out_dir / adapter_id / "adapter_model.safetensors"
    if not artifact.exists():
        return False, ""
    if not recorded or recorded.endswith(":"):
        # Nothing to compare against: the registry has not recorded a digest for
        # this adapter yet, which is the state a first import starts from.
        return False, "present but the registry records no digest yet"

    actual = sha256_file(artifact)
    if actual == recorded:
        return True, f"already materialised and matches {recorded[:19]}…"
    return False, f"on disk as {actual[:19]}… but the registry records {recorded[:19]}…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="pool", help="Directory the normalised pool is written to."
    )
    parser.add_argument(
        "--cache", default=".hf-cache", help="Scratch directory for the fetched upstream files."
    )
    parser.add_argument("--only", nargs="*", default=None, help="Import only these adapter ids.")
    parser.add_argument(
        "--write-registry",
        action="store_true",
        help="Write each artifact digest back into the registry once the import succeeds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-import adapters that are already materialised and whose artifact matches "
            "the registry. Without this they are verified and skipped."
        ),
    )
    args = parser.parse_args(argv)

    setup_logging("INFO")

    manifest = load_base_manifest()
    if not manifest.is_pinned:
        log.error("the base manifest is not pinned; pin it before importing a pool")
        return 2

    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry = load_registry()

    out_dir = Path(args.out)
    cache_dir = Path(args.cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.only) if args.only else None
    records: dict[str, dict] = {}
    failures: list[str] = []
    skipped: list[str] = []

    for entry in raw["adapters"]:
        adapter_id = entry["adapter_id"]
        if wanted is not None and adapter_id not in wanted:
            continue

        # An adapter already materialised, whose bytes still hash to the digest
        # the registry recorded, has nothing left to prove. Re-importing it means
        # re-downloading the upstream files and re-running the decomposition to
        # arrive at bytes that are already on disk — minutes per adapter, and for
        # a full pool the difference between a re-run that takes seconds and one
        # that takes the better part of an hour.
        #
        # The digest is checked rather than assumed from the file existing: a
        # truncated or half-written artifact is exactly the case where skipping
        # would be wrong, and it is indistinguishable from a good one by presence
        # alone. A mismatch falls through and re-imports.
        if not args.force:
            verified, detail = _already_verified(out_dir, adapter_id, entry.get("artifact_sha256"))
            if verified:
                log.info("%s: %s, skipping", adapter_id, detail)
                skipped.append(adapter_id)
                continue
            if detail:
                log.info("%s: %s, re-importing", adapter_id, detail)

        spec = SourceSpec(
            adapter_id=adapter_id,
            repo=entry["source_repo"],
            revision=entry["source_revision"],
            rank=int(entry["source_rank"]),
            lora_alpha=int(entry["source_lora_alpha"]),
            base_repo=manifest.model_repo,
        )
        try:
            records[adapter_id] = import_adapter(
                spec, manifest=manifest, out_dir=out_dir, cache_dir=cache_dir
            )
        except (ImportError_, Exception) as exc:  # noqa: BLE001 - reported per adapter
            log.error("%s: import failed — %s", adapter_id, exc)
            failures.append(adapter_id)

    print()
    print(f"imported {len(records)} of {len(registry.ids)} adapters into {out_dir}")
    # Stated rather than implied. "imported 0 of 30" on a pool that is entirely
    # present and correct would otherwise read as a total failure.
    if skipped:
        print(f"  already verified, not re-imported: {len(skipped)} (--force to re-import)")
    lossy = [rid for rid, r in records.items() if not r["lossless"]]
    if lossy:
        print(f"  lossy conversions needing recertification: {', '.join(sorted(lossy))}")
    if failures:
        print(f"  failed: {', '.join(failures)}")

    if args.write_registry and records:
        for entry in raw["adapters"]:
            record = records.get(entry["adapter_id"])
            if record is not None:
                entry["artifact_sha256"] = record["artifact_sha256"]
        REGISTRY_PATH.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {len(records)} digests into {REGISTRY_PATH.name}")
        print(
            "\nNext: measure each adapter and record it with `capability-registry certify`. "
            "The engine refuses to start until every adapter is certified."
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
