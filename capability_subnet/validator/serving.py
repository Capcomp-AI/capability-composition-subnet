"""Serving one candidate on the validator's own hardware.

A validator in ``own`` evaluation rebuilds each submission and then has to
measure *that* package. Pointing a scorer at an endpoint which happens to be
running is not measuring it: every candidate would be scored against whatever
that endpoint already holds, two different recipes would produce the same
numbers, and the ranking would have nothing to rank. The evaluation engine
refuses to run in the mode that does this, and it is right to.

So the adapter is applied by starting a runtime for it and stopping the runtime
afterwards. One candidate, one process, no shared state between them.

Reserving memory in absolute terms rather than as a fraction of the card is what
keeps the serving conditions identical across validators: a fraction would give
a package more room on a bigger card, and a candidate answering with a larger KV
cache than its peers is not being asked the same question.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from capability_subnet.common import constants as C

log = logging.getLogger(__name__)

#: How long to wait for the endpoint to answer before giving up. A first start
#: on a host with cold kernel caches compiles before it serves.
STARTUP_TIMEOUT_S = 900.0

#: Name the adapter is served under. The base model is served under a different
#: one deliberately: registering both under the same id leaves two entries with
#: the same name in /v1/models, and a request naming it is then ambiguous
#: between the merged package and the bare base model.
CANDIDATE_MODEL = "candidate"
BASE_MODEL = "base"

#: Device memory that is gone before a runtime asks for any — the CUDA context
#: and the driver's own allocations. Measured at roughly 0.39 GiB on a 24 GB
#: card; carried at 1.0 GiB so the check refuses a card that would only just
#: fail, rather than one that fails by a margin too small to see.
DRIVER_CONTEXT_GIB = 1.0


class ServingError(RuntimeError):
    """The candidate's runtime could not be started."""


class _StartupLog:
    """Drains the runtime's output so it can never block on a full pipe.

    A pipe holds 64 KB. A runtime that fails after printing more than that fills
    it, blocks in ``write``, and so never exits — leaving ``poll()`` reporting
    nothing while the caller waits out its whole timeout holding a GPU, and then
    reports a timeout for what was an error sitting unread in the pipe.
    """

    MAX_RETAINED_BYTES = 8 * 1024 * 1024

    def __init__(self, stream) -> None:
        self._chunks: list[bytes] = []
        self._retained = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._drain, args=(stream,), name="candidate-serving-log", daemon=True
        )
        self._thread.start()

    def _drain(self, stream) -> None:
        if stream is None:
            return
        try:
            for line in iter(stream.readline, b""):
                with self._lock:
                    if self._retained < self.MAX_RETAINED_BYTES:
                        self._chunks.append(line)
                        self._retained += len(line)
        except (ValueError, OSError):
            pass

    def tail(self, limit: int = 40) -> str:
        with self._lock:
            text = b"".join(self._chunks).decode("utf-8", "replace")
        lines = [line for line in text.splitlines() if line.strip()]
        return "\n".join(lines[-limit:])

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout)


def utilization_for(total_gib: float, reserved_gib: float = C.SERVING_RESERVED_GIB) -> float:
    """The fraction of a card that reserves ``reserved_gib``.

    vLLM takes a fraction of the whole device, so the fraction that holds a
    package steady is different on every card size. Deriving it from the card
    keeps what is reserved — and therefore the KV cache a candidate answers
    with — the same everywhere.
    """
    if total_gib <= 0.0:
        raise ValueError("total_gib must be positive")
    if reserved_gib <= 0.0:
        raise ValueError("reserved_gib must be positive")
    # Against *usable* memory, not total. A device never offers all of it: the
    # driver context is resident before anything is loaded, so a card whose
    # total merely exceeds the reservation can still be unable to hold it. The
    # runtime discovers that at start-up and refuses, once per candidate, and
    # the window records it as a reconstruction failure — which reads as every
    # miner scoring zero for a fault that belongs to this host.
    usable_gib = total_gib - DRIVER_CONTEXT_GIB
    if reserved_gib >= usable_gib:
        raise ServingError(
            f"a candidate needs {reserved_gib:.1f} GiB and this device has "
            f"{total_gib:.1f} GiB, of which about {usable_gib:.1f} GiB is free once "
            f"the driver context is resident. Evaluation on this host would measure "
            "the card rather than the package."
        )
    return reserved_gib / total_gib


def _device_total_gib(device: str) -> float:
    import torch

    index = 0
    if ":" in device:
        index = int(device.split(":", 1)[1])
    return torch.cuda.get_device_properties(index).total_memory / (1024**3)


def build_command(
    artifact_dir: str | Path,
    *,
    base_model_path: str | Path,
    python_executable: str,
    host: str,
    port: int,
    gpu_memory_utilization: float,
) -> list[str]:
    """The runtime invocation for one candidate."""
    return [
        python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(base_model_path),
        "--served-model-name",
        BASE_MODEL,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(C.SERVING_MAX_MODEL_LEN),
        "--seed",
        "0",
        "--gpu-memory-utilization",
        f"{gpu_memory_utilization:.4f}",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--enable-lora",
        "--max-lora-rank",
        str(max(C.ALLOWED_OUTPUT_RANKS)),
        "--lora-modules",
        f"{CANDIDATE_MODEL}={artifact_dir}",
        "--max-loras",
        "1",
        "--kv-cache-dtype",
        "fp8",
        "--max-num-seqs",
        # The server must admit at least as many sequences as the evaluator puts
        # in flight, or continuous batching is throttled back to one at the
        # runtime and the client just queues.
        str(C.SANDBOX_BATCH_CONCURRENCY),
    ]


def _environment(python_executable: str, device: str) -> dict[str, str]:
    """Inherited, with only what must differ overridden.

    The interpreter's own bin/ goes first on PATH: a runtime shells out to build
    tools that live beside it, and when it has its own virtualenv those tools are
    not on the inherited PATH. The failure is a bare FileNotFoundError from deep
    inside start-up, which reads like anything except a PATH problem.
    """
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(Path(python_executable).absolute().parent), env.get("PATH", "")]
    ).rstrip(os.pathsep)
    env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1] if ":" in device else "0"
    # Nothing may swap the package under measurement while it is being measured.
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "0"
    # The pool and the base model are materialised ahead of time; an evaluation
    # must never reach the network.
    env.setdefault("HF_HUB_OFFLINE", "1")
    return env


@contextmanager
def serve_candidate(
    artifact_dir: str | Path,
    *,
    base_model_path: str | Path,
    device: str = "cuda",
    python_executable: str = "",
    host: str = "127.0.0.1",
    port: int = 8000,
    startup_timeout: float = STARTUP_TIMEOUT_S,
) -> Iterator[str]:
    """Serve one candidate, yielding the base URL it answers on.

    The runtime is stopped on the way out whether or not scoring succeeded, so
    the next candidate starts from an empty card.
    """
    import httpx

    interpreter = python_executable or sys.executable
    if not Path(str(base_model_path)).is_dir():
        raise ServingError(
            f"the base model is not materialised at {base_model_path}. Evaluation runs "
            "offline, so it must be on disk before a candidate can be served."
        )
    if shutil.which("nvidia-smi") is None:
        log.warning("nvidia-smi is not on PATH; peak memory may not be readable")

    utilization = utilization_for(_device_total_gib(device))
    command = build_command(
        artifact_dir,
        base_model_path=base_model_path,
        python_executable=interpreter,
        host=host,
        port=port,
        gpu_memory_utilization=utilization,
    )

    log.info(
        "serving candidate from %s on %s (reserving %.1f GiB, utilization %.3f)",
        artifact_dir,
        device,
        C.SERVING_RESERVED_GIB,
        utilization,
    )
    process = subprocess.Popen(
        command,
        env=_environment(interpreter, device),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    startup_log = _StartupLog(process.stdout)
    base_url = f"http://{host}:{port}"

    try:
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                startup_log.join()
                raise ServingError(
                    f"the candidate's runtime exited during start-up with code "
                    f"{process.returncode}:\n{startup_log.tail()}"
                )
            try:
                if httpx.get(f"{base_url}/health", timeout=5.0).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2.0)
        else:
            raise ServingError(
                f"the candidate's runtime did not answer within {startup_timeout:.0f}s:"
                f"\n{startup_log.tail()}"
            )

        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            log.warning("the candidate's runtime did not exit; killing it")
            process.kill()
            process.wait(timeout=30)


__all__ = [
    "CANDIDATE_MODEL",
    "ServingError",
    "build_command",
    "serve_candidate",
    "utilization_for",
]
