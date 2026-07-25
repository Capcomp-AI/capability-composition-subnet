"""Registry admission gates.

An adapter joins the certified pool only after passing every gate below. These
run once, offline, when the operator admits an adapter — not per evaluation.
They exist because the merge engine loads these tensors into a process that also
holds hidden evaluation material, so an adapter file is untrusted input until it
has been proven to be nothing but finite numbers of the expected shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capability_subnet.common import constants as C
from capability_subnet.common.hashing import sha256_file
from capability_subnet.registry import licenses
from capability_subnet.registry.base_model import BaseManifest

#: File extensions that may appear in an adapter directory. Anything else is
#: rejected outright — in particular pickle-backed formats, which can execute
#: code on load, and any kind of script or binary.
ALLOWED_FILENAMES = frozenset(
    {
        "adapter_model.safetensors",
        "adapter_config.json",
        "README.md",
        "LICENSE",
    }
)

BANNED_SUFFIXES = frozenset(
    {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".py", ".so", ".sh", ".exe", ".dll"}
)


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    detail: str = ""

    def __bool__(self) -> bool:
        return self.passed


@dataclass(frozen=True, slots=True)
class CertificationOutcome:
    adapter_id: str
    admitted: bool
    artifact_sha256: str | None
    gates: tuple[GateResult, ...]

    def failures(self) -> list[GateResult]:
        return [gate for gate in self.gates if not gate.passed]

    def summary(self) -> str:
        if self.admitted:
            return f"{self.adapter_id}: admitted ({len(self.gates)} gates passed)"
        names = ", ".join(gate.name for gate in self.failures())
        return f"{self.adapter_id}: rejected (failed: {names})"


def check_no_executable_content(adapter_dir: Path) -> GateResult:
    """Reject any file that is not a safetensors weight file or plain metadata."""
    offenders: list[str] = []
    for path in sorted(adapter_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(adapter_dir).as_posix()
        if path.suffix.lower() in BANNED_SUFFIXES:
            offenders.append(f"{rel} (banned extension {path.suffix})")
        elif rel not in ALLOWED_FILENAMES and path.suffix.lower() != ".safetensors":
            offenders.append(f"{rel} (not on the allow-list)")

    if offenders:
        return GateResult(
            "security",
            False,
            "adapter directory contains disallowed files: " + "; ".join(offenders),
        )
    return GateResult("security", True, "safetensors and metadata only")


def check_adapter_config(
    adapter_dir: Path, manifest: BaseManifest
) -> tuple[GateResult, dict[str, Any] | None]:
    """Validate ``adapter_config.json`` against the canonical adapter spec."""
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        return GateResult("config", False, "adapter_config.json is missing"), None

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return GateResult("config", False, f"adapter_config.json is not valid JSON: {exc}"), None

    problems: list[str] = []

    if config.get("peft_type") != C.CANONICAL_PEFT_TYPE:
        problems.append(
            f"peft_type={config.get('peft_type')!r}, expected {C.CANONICAL_PEFT_TYPE!r}"
        )
    if int(config.get("r", -1)) != C.CANONICAL_RANK:
        problems.append(f"r={config.get('r')}, expected {C.CANONICAL_RANK}")
    if int(config.get("lora_alpha", -1)) != C.CANONICAL_LORA_ALPHA:
        problems.append(f"lora_alpha={config.get('lora_alpha')}, expected {C.CANONICAL_LORA_ALPHA}")
    if float(config.get("lora_dropout", -1.0)) != C.CANONICAL_LORA_DROPOUT:
        problems.append(
            f"lora_dropout={config.get('lora_dropout')}, expected {C.CANONICAL_LORA_DROPOUT}"
        )
    if config.get("bias") != C.CANONICAL_BIAS:
        problems.append(f"bias={config.get('bias')!r}, expected {C.CANONICAL_BIAS!r}")

    declared_modules = set(config.get("target_modules") or ())
    if declared_modules != set(C.CANONICAL_TARGET_MODULES):
        problems.append(
            f"target_modules={sorted(declared_modules)}, expected {sorted(C.CANONICAL_TARGET_MODULES)}"
        )

    # Remote code is how an adapter config smuggles execution into model loading.
    for forbidden in ("auto_map", "custom_module_mapping", "modules_to_save"):
        if config.get(forbidden):
            problems.append(f"{forbidden} is set; remote or custom code is not permitted")

    if problems:
        return GateResult("config", False, "; ".join(problems)), config
    return GateResult("config", True, "matches the canonical adapter spec"), config


def check_base_revision(config: dict[str, Any] | None, manifest: BaseManifest) -> GateResult:
    """The adapter must declare the exact pinned base revision."""
    if config is None:
        return GateResult("base_revision", False, "no adapter config to check")

    declared = str(config.get("base_model_revision") or config.get("revision") or "")
    if not declared:
        return GateResult(
            "base_revision",
            False,
            "adapter config declares no base_model_revision",
        )
    if declared != manifest.revision:
        return GateResult(
            "base_revision",
            False,
            f"adapter targets base revision {declared!r}, pool is pinned to {manifest.revision!r}",
        )
    return GateResult("base_revision", True, f"targets pinned revision {declared}")


def check_tensor_shapes(adapter_dir: Path, manifest: BaseManifest) -> GateResult:
    """Every expected tensor is present, with the right shape, and nothing else is."""
    from safetensors import safe_open  # imported lazily so CPU-only tooling stays light

    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        return GateResult("shapes", False, "adapter_model.safetensors is missing")

    expected_keys = set(manifest.all_tensor_keys())
    problems: list[str] = []

    try:
        with safe_open(weights_path, framework="pt") as handle:
            present = set(handle.keys())
            missing = sorted(expected_keys - present)
            extra = sorted(present - expected_keys)
            if missing:
                problems.append(f"{len(missing)} missing tensors, first: {missing[:3]}")
            if extra:
                problems.append(f"{len(extra)} unexpected tensors, first: {extra[:3]}")

            for key in sorted(expected_keys & present):
                shape = tuple(handle.get_slice(key).get_shape())
                module = _module_from_key(key)
                matrix = "lora_A" if key.endswith("lora_A.weight") else "lora_B"
                if matrix == "lora_A":
                    want = manifest.expected_lora_a_shape(module, C.CANONICAL_RANK)
                else:
                    want = manifest.expected_lora_b_shape(module, C.CANONICAL_RANK)
                if shape != want:
                    problems.append(f"{key}: shape {shape}, expected {want}")
                    if len(problems) > 8:
                        break
    except Exception as exc:  # noqa: BLE001 - any read failure is a rejection
        return GateResult("shapes", False, f"could not read safetensors file: {exc}")

    if problems:
        return GateResult("shapes", False, "; ".join(problems[:8]))
    return GateResult("shapes", True, f"{len(expected_keys)} tensors match the base model")


def check_finite_values(adapter_dir: Path) -> GateResult:
    """Scan every tensor for NaN and infinity.

    A single non-finite entry propagates through the merge and poisons every
    downstream candidate that selects this adapter, so the scan is exhaustive
    rather than sampled.
    """
    import torch
    from safetensors import safe_open

    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        return GateResult("numerical", False, "adapter_model.safetensors is missing")

    offenders: list[str] = []
    try:
        with safe_open(weights_path, framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if not torch.isfinite(tensor.to(torch.float32)).all():
                    offenders.append(key)
                    if len(offenders) >= 5:
                        break
    except Exception as exc:  # noqa: BLE001
        return GateResult("numerical", False, f"could not scan tensors: {exc}")

    if offenders:
        return GateResult("numerical", False, f"non-finite values in: {offenders}")
    return GateResult("numerical", True, "all values finite")


def check_license(license_id: str, *, declared_allows_derivatives: bool) -> GateResult:
    verdict = licenses.review(license_id, declared_allows_derivatives=declared_allows_derivatives)
    return GateResult("license", verdict.admissible, verdict.reason)


def check_capability_certification(
    capability_score: float | None,
    base_retention: float | None,
    *,
    min_capability_score: float,
) -> GateResult:
    """The adapter must actually be good at its declared capability, and must
    not have destroyed the base model's general ability to get there."""
    if capability_score is None:
        return GateResult("capability", False, "no capability certification run has been recorded")
    if capability_score < min_capability_score:
        return GateResult(
            "capability",
            False,
            f"capability score {capability_score:.3f} below the {min_capability_score:.3f} floor",
        )
    if base_retention is None:
        return GateResult("capability", False, "no base-retention measurement recorded")
    if base_retention < C.BASE_RETENTION_FLOOR:
        return GateResult(
            "capability",
            False,
            f"base retention {base_retention:.3f} below the {C.BASE_RETENTION_FLOOR:.3f} floor",
        )
    return GateResult(
        "capability",
        True,
        f"capability {capability_score:.3f}, base retention {base_retention:.3f}",
    )


def check_conversion_recertified(converted_from_rank: int | None, recertified: bool) -> GateResult:
    """An adapter normalised to the canonical rank must be recertified afterwards.

    Rank conversion is lossy. Certifying the original and admitting the converted
    file would put weights in the pool that nobody has measured.
    """
    if converted_from_rank is None:
        return GateResult("conversion", True, "native canonical rank, no conversion")
    if not recertified:
        return GateResult(
            "conversion",
            False,
            f"converted from rank {converted_from_rank} but not recertified afterwards",
        )
    return GateResult(
        "conversion", True, f"converted from rank {converted_from_rank} and recertified"
    )


def _module_from_key(key: str) -> str:
    """Extract the projection name from a canonical tensor key."""
    parts = key.split(".")
    for candidate in C.CANONICAL_TARGET_MODULES:
        if candidate in parts:
            return candidate
    raise ValueError(f"cannot determine module from tensor key {key!r}")


def certify_adapter(
    adapter_dir: str | Path,
    *,
    adapter_id: str,
    manifest: BaseManifest,
    license_id: str,
    declared_allows_derivatives: bool,
    capability_score: float | None = None,
    base_retention: float | None = None,
    converted_from_rank: int | None = None,
    recertified_after_conversion: bool = False,
    min_capability_score: float = 0.60,
) -> CertificationOutcome:
    """Run every admission gate against an adapter directory.

    The artifact digest is computed only if the file passes structural checks —
    hashing a file the operator is about to reject wastes nothing but time, but
    recording a digest for a rejected adapter invites someone to paste it into
    the registry later.
    """
    adapter_path = Path(adapter_dir)
    gates: list[GateResult] = []

    if not adapter_path.is_dir():
        return CertificationOutcome(
            adapter_id,
            False,
            None,
            (GateResult("exists", False, f"{adapter_path} is not a directory"),),
        )

    gates.append(check_no_executable_content(adapter_path))
    config_gate, config = check_adapter_config(adapter_path, manifest)
    gates.append(config_gate)
    gates.append(check_base_revision(config, manifest))
    gates.append(check_tensor_shapes(adapter_path, manifest))
    gates.append(check_finite_values(adapter_path))
    gates.append(check_license(license_id, declared_allows_derivatives=declared_allows_derivatives))
    gates.append(
        check_capability_certification(
            capability_score, base_retention, min_capability_score=min_capability_score
        )
    )
    gates.append(check_conversion_recertified(converted_from_rank, recertified_after_conversion))

    admitted = all(gate.passed for gate in gates)
    digest = sha256_file(adapter_path / "adapter_model.safetensors") if admitted else None

    return CertificationOutcome(adapter_id, admitted, digest, tuple(gates))
