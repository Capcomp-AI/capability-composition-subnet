"""Verify a published run archive against the chain.

Everything here runs for somebody who trusts nobody: it reads the commitment
from the chain, fetches the archive from wherever the commitment says it is,
and checks the two against each other. No operator endpoint is involved and no
credential is needed.

Four things have to agree, and each is checked against something fixed by the
step before it rather than against a claim in the same document:

1. every file matches the digest the manifest records for it;
2. the root recomputed from the manifest matches the root stored in it;
3. the signature over that root verifies against the hotkey it names;
4. the root matches the digest reached from the chain — directly for the run
   the commitment names, or through the ``previous_root`` chain for an older
   one.

The commitment names the newest archived run only, because the Commitments
pallet holds one value per hotkey and each write replaces the last. Older runs
are reached by walking back: each manifest carries the previous run and its
root, so the walk is itself verified at every step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from capability_subnet.common import archive
from capability_subnet.common.signing import verify_payload

REPO_TEMPLATE = "capcomp/sn103-run-{run}"
REPO_TYPE = "dataset"


class AuditError(Exception):
    """The archive could not be reached, or does not check out."""


class _Root:
    """A payload whose signable bytes are a bundle root, for verify_payload."""

    def __init__(self, run_id: int, root: str) -> None:
        self._message = archive.signing_message(run_id, root)

    def signable_bytes(self) -> bytes:
        return self._message


def read_commitment(netuid: int = 103, network: str = "finney") -> tuple[Any, int]:
    """The archive commitment on chain, and the block it landed in."""
    import bittensor as bt

    graph = bt.subtensor(network=network).subnets.metagraph(netuid=netuid)
    for record in graph.commitments.values():
        value = getattr(record, "value", None)
        if value and archive.looks_like_archive(value):
            return archive.decode(value), int(record.block)
    raise AuditError(
        f"no archive commitment on netuid {netuid}. Nothing has been published, "
        f"or it was published by a hotkey that is no longer registered."
    )


def fetch(run_id: int, revision: str = "") -> Path:
    """Download one run's archive. Unauthenticated, as any reader would."""
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                repo_id=REPO_TEMPLATE.format(run=run_id),
                repo_type=REPO_TYPE,
                revision=revision or None,
                token=False,
            )
        )
    except Exception as exc:
        raise AuditError(
            f"could not fetch run {run_id} from "
            f"{REPO_TEMPLATE.format(run=run_id)}: {type(exc).__name__}: {exc}"
        ) from exc


def check(directory: Path, expected_root: str = "") -> dict[str, Any]:
    """Verify one fetched bundle. Raises rather than returning a soft failure."""
    document = json.loads((directory / "manifest.json").read_text())

    ok, problems = archive.verify_bundle(directory, expected_root=expected_root)
    if not ok:
        raise AuditError("; ".join(problems))

    signer, signature = document.get("signer", ""), document.get("signature", "")
    if not signer or not signature:
        raise AuditError(f"run {document['submitted_run']} carries no signature")
    if not verify_payload(_Root(document["submitted_run"], document["root"]), signature, signer):
        raise AuditError(
            f"the signature on run {document['submitted_run']} does not verify against {signer}"
        )
    return document


def recompute_grades(directory: Path) -> tuple[int, int]:
    """Re-derive every grade from the published axes. Returns (checked, matched)."""
    from capability_subnet.common import constants as C

    rows = json.loads((directory / "scores.json").read_text())
    scored = [r for r in rows if r.get("grade") is not None]
    matched = 0
    for row in scored:
        quality = min(
            1.0,
            sum(w * (row.get(axis) or 0.0) for axis, w in C.QUALIFIED_SCORE_WEIGHTS.items()),
        )
        grade = (
            0.50 * row["term_quality"] + 0.40 * row["term_improvement"] + 0.10 * row["term_cost"]
        )
        if abs(quality - row["qualified_score"]) < 1e-6 and abs(grade - row["grade"]) < 1e-6:
            matched += 1
    return len(scored), matched


def audit(run_id: int | None, *, netuid: int = 103, network: str = "finney") -> dict[str, Any]:
    """Verify one run, reached from the chain.

    ``run_id`` of None audits whichever run the commitment names.
    """
    published, block = read_commitment(netuid=netuid, network=network)
    revision = published.location.split("@", 1)[1] if "@" in published.location else ""

    target = published.run_id if run_id is None else run_id
    if target > published.run_id:
        raise AuditError(
            f"run {target} is not archived; the chain names run {published.run_id} as the newest"
        )

    steps: list[dict[str, Any]] = []
    current_run, expected, current_revision = published.run_id, published.root_sha256, revision

    while True:
        directory = fetch(current_run, current_revision)
        document = check(directory, expected_root=expected)
        checked, matched = recompute_grades(directory)
        steps.append(
            {
                "run": current_run,
                "root": document["root"],
                "signer": document["signer"],
                "recipes": sum(1 for f in document["files"] if f["path"].startswith("recipes/")),
                "graded": checked,
                "grades_recomputed": matched,
                "directory": str(directory),
            }
        )
        if current_run == target:
            break
        previous = document.get("previous_run")
        if previous is None:
            raise AuditError(
                f"reached run {current_run}, the first archived, without finding run {target}"
            )
        current_run, expected, current_revision = previous, document["previous_root"], ""

    return {
        "netuid": netuid,
        "commitment_block": block,
        "chain_names_run": published.run_id,
        "audited_run": target,
        "walked": len(steps),
        "steps": steps,
    }
