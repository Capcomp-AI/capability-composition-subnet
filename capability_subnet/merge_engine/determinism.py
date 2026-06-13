"""Determinism controls for reconstruction.

Reconstruction is the one part of this protocol where two machines must agree
byte-for-byte. Everything that could introduce variation is pinned here:

* global RNG state and per-tensor derived seeds,
* deterministic kernel selection,
* single-threaded reduction order,
* a canonical sign convention for singular vectors.

The last one matters more than it looks. A singular value decomposition is only
unique up to a joint sign flip of each left/right vector pair, and different
LAPACK builds make different choices. Without a canonical convention two correct
implementations produce different — but equally valid — factorisations, and the
artifact hashes diverge.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import random
from collections.abc import Iterator

import numpy as np
import torch

log = logging.getLogger(__name__)

#: Working precision for every merge and decomposition. Inputs arrive as
#: bfloat16 and are widened before any arithmetic; the result is narrowed back
#: only at write time.
WORK_DTYPE = torch.float32

#: Emitted artifact precision.
OUTPUT_DTYPE = torch.bfloat16


def derive_seed(*parts: object) -> int:
    """Derive a stable 63-bit seed from arbitrary components.

    Every stochastic step keys its generator on the recipe seed *plus* the
    identity of the tensor it is about to touch. That makes the drop mask for a
    given tensor independent of how many tensors were processed before it, so
    the result cannot depend on iteration order, sharding or parallelism.
    """
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def generator_for(*parts: object) -> torch.Generator:
    """A CPU generator seeded from ``parts``.

    Always CPU: CUDA generators produce different streams across driver and
    architecture versions, which would make the artifact hash depend on the GPU
    the worker happened to be assigned.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(*parts))
    return generator


def seed_everything(seed: int) -> None:
    """Pin every global RNG a merge could touch."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on host
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def deterministic_context(seed: int = 0, *, threads: int = 1) -> Iterator[None]:
    """Run a merge under fully pinned numerics.

    Restores the previous global settings on exit so importing the merge engine
    never silently changes how the rest of a host process computes.
    """
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_tf32_cudnn = torch.backends.cudnn.allow_tf32
    previous_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    try:
        # Reduction order in a multi-threaded sum is not fixed; one thread makes
        # floating-point accumulation reproducible.
        torch.set_num_threads(max(1, threads))
        # TF32 silently truncates float32 mantissas on Ampere and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            # warn_only keeps a merge running on a build that lacks a
            # deterministic kernel for some operation, rather than aborting; the
            # cross-worker hash check is the backstop.
            torch.use_deterministic_algorithms(True, warn_only=True)
        except RuntimeError as exc:
            # Loud. Reconstruction is the one place two machines must agree
            # byte-for-byte, and a build that cannot enable deterministic kernels
            # will produce artifacts that differ from every other worker's. A
            # silent failure here surfaces later as an unexplained hash mismatch
            # that halts the candidate.
            log.error(
                "could not enable deterministic algorithms on this build (%s). "
                "Artifact hashes from this host may not match other workers.",
                exc,
            )
        seed_everything(seed)
        yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32_matmul
        torch.backends.cudnn.allow_tf32 = previous_tf32_cudnn
        # Restoring the previous setting is best-effort and genuinely
        # uninteresting: the merge is over, and the value is being put back to
        # whatever the host had before.
        with contextlib.suppress(RuntimeError):
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=True)
        if previous_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas


def canonicalize_svd_signs(u: torch.Tensor, vh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a fixed sign convention to a decomposition.

    For each component, the entry of largest magnitude in the left singular
    vector is forced positive, and the corresponding right vector is flipped
    with it so the product is unchanged. Ties in magnitude resolve to the lowest
    index because ``argmax`` returns the first maximum.

    Args:
        u: left singular vectors, shape ``(m, k)``.
        vh: right singular vectors, shape ``(k, n)``.

    Returns:
        The sign-canonical ``(u, vh)`` pair. ``u @ diag(s) @ vh`` is unchanged.
    """
    if u.numel() == 0 or vh.numel() == 0:
        return u, vh

    dominant_row = torch.argmax(u.abs(), dim=0)
    dominant_values = u[dominant_row, torch.arange(u.shape[1], device=u.device)]
    signs = torch.where(
        dominant_values < 0,
        torch.tensor(-1.0, dtype=u.dtype, device=u.device),
        torch.tensor(1.0, dtype=u.dtype, device=u.device),
    )
    return u * signs.unsqueeze(0), vh * signs.unsqueeze(1)
