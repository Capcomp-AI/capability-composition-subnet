#!/usr/bin/env python3
"""Materialise the certified pool from the public adapters the registry names.

Run once per arena version, by the operator, before genesis:

    python scripts/import_public_adapters.py --out pool

For every entry in ``registry/data/adapter_registry.json`` this fetches the two
files it needs at the pinned upstream revision, refuses anything whose config
has drifted from what the registry recorded, re-factorises the update to the
canonical rank, writes the normalised artifact, and hashes it.

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

    for entry in raw["adapters"]:
        adapter_id = entry["adapter_id"]
        if wanted is not None and adapter_id not in wanted:
            continue

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
