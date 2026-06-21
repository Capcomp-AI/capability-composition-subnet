"""Serving a candidate.

Each candidate is served behind an OpenAI-compatible endpoint with its adapter
already applied, on one assigned GPU, in bfloat16. Three properties of that
setup are load-bearing:

* **The adapter is baked in at start-up and cannot be changed.** Dynamic adapter
  loading is switched off, so nothing inside the sandbox can swap the package
  being measured for a different one.
* **The endpoint reaches nothing but the agent container.** It has no route to
  the scorer, the hidden instances or the network.
* **bfloat16 for every canonical score.** Quantised runs are useful and are
  reported, but never mixed into a comparison — a package measured at a
  different precision is a different package.

The in-flight dispatch budget bounds how many evaluations one host accepts at
once. It exists because latency is a scored quantity: a host running twice as
many sandboxes as it has capacity for would report latencies that say more about
the operator's provisioning than about the candidate.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from capability_subnet.common import constants as C
from capability_subnet.sandbox.model_client import ModelClient, OpenAICompatibleClient

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ServingHandle:
    """A running endpoint serving one candidate."""

    base_url: str
    model_name: str
    adapter_path: str | None
    #: ``None`` when the GPU counter is unavailable, so the gate can distinguish
    #: "used no memory" from "we do not know".
    peak_vram_gb: float | None = None

    def client(self, timeout: float = 120.0) -> ModelClient:
        return OpenAICompatibleClient(self.base_url, self.model_name, timeout=timeout)


class ServingError(Exception):
    """Raised when a candidate cannot be served.

    Always an infrastructure failure, never a candidate failure: a package that
    passed every structural gate and then could not be loaded points at the
    engine, and the scheduler holds the candidate rather than terminating it.
    """


class DispatchBudget:
    """Bounds concurrent evaluations on one serving host."""

    def __init__(self, capacity: int = C.DEFAULT_DISPATCH_BUDGET) -> None:
        if capacity <= 0:
            raise ValueError("dispatch capacity must be positive")
        self.capacity = capacity
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def available(self) -> int:
        return max(0, self.capacity - self._in_flight)

    @contextmanager
    def reserve(self, count: int = 1) -> Iterator[None]:
        if count > self.available:
            raise ServingError(
                f"dispatch budget exhausted: {count} requested, {self.available} of "
                f"{self.capacity} available"
            )
        self._in_flight += count
        try:
            yield
        finally:
            self._in_flight -= count


class ExternalServer:
    """An endpoint an operator manages outside the engine.

    The engine points at a URL and assumes the operator restarted the runtime
    with the right adapter. This is the simplest deployment and the one used for
    development; it trades automation for the operator being able to run whatever
    serving stack they already have.
    """

    def __init__(self, base_url: str, model_name: str) -> None:
        self.base_url = base_url
        self.model_name = model_name

    @contextmanager
    def serve(self, adapter_path: str | Path | None) -> Iterator[ServingHandle]:
        log.info(
            "using the externally managed endpoint at %s for adapter %s",
            self.base_url,
            adapter_path or "<base model>",
        )
        yield ServingHandle(
            base_url=self.base_url,
            model_name=self.model_name,
            adapter_path=str(adapter_path) if adapter_path else None,
        )


class ManagedVllmServer:
    """Starts and stops a vLLM process per candidate.

    Restarting per candidate rather than hot-swapping adapters is deliberate.
    Hot-swapping is faster, but it leaves the previous candidate's state in the
    runtime's caches, and a measurement that depends on which package was
    evaluated before it is not a measurement.
    """

    def __init__(
        self,
        *,
        base_model_path: str,
        model_name: str = "candidate",
        host: str = "127.0.0.1",
        port: int = 8000,
        gpu_index: int = 0,
        startup_timeout: float = 600.0,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.base_model_path = base_model_path
        self.model_name = model_name
        self.host = host
        self.port = port
        self.gpu_index = gpu_index
        self.startup_timeout = startup_timeout
        self.extra_args = extra_args

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _command(self, adapter_path: str | None) -> list[str]:
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.base_model_path,
            "--served-model-name",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "16384",
            # Dynamic adapter loading is the one feature that would let something
            # inside the sandbox change the package under measurement.
            "--disable-log-requests",
        ]
        if adapter_path:
            command += [
                "--enable-lora",
                "--max-lora-rank",
                str(max(C.ALLOWED_OUTPUT_RANKS)),
                "--lora-modules",
                f"{self.model_name}={adapter_path}",
            ]
        return command + list(self.extra_args)

    @contextmanager
    def serve(self, adapter_path: str | Path | None) -> Iterator[ServingHandle]:
        adapter = str(adapter_path) if adapter_path else None
        command = self._command(adapter)

        log.info("starting the candidate endpoint on GPU %d", self.gpu_index)
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command,
            env={
                "CUDA_VISIBLE_DEVICES": str(self.gpu_index),
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "0",
                "HF_HUB_OFFLINE": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        try:
            self._wait_until_ready(process)
            yield ServingHandle(
                base_url=self.base_url,
                model_name=self.model_name,
                adapter_path=adapter,
                peak_vram_gb=read_peak_vram_gb(self.gpu_index),
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                log.warning("candidate endpoint did not exit; killing it")
                process.kill()
                process.wait(timeout=30)

    def _wait_until_ready(self, process: subprocess.Popen) -> None:
        import httpx

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = (process.stdout.read() if process.stdout else b"") or b""
                raise ServingError(
                    f"the endpoint exited during start-up with code {process.returncode}: "
                    f"{output.decode('utf-8', 'replace')[-2000:]}"
                )
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    return
            except Exception:  # noqa: BLE001 - still starting
                pass
            time.sleep(2.0)

        raise ServingError(f"the endpoint did not become ready within {self.startup_timeout:.0f}s")


def read_peak_vram_gb(gpu_index: int = 0) -> float | None:
    """Peak memory on the assigned GPU, in gigabytes.

    Returns:
        The measurement, or ``None`` when there is no GPU or the counter could
        not be read.

    ``None`` rather than ``0.0``. A zero would sail through the peak-memory gate,
    so a host whose GPU counter is broken would silently pass every candidate —
    including one that needs far more memory than the gate allows. Reporting the
    absence lets the gate say "unmeasured" and lets the operator decide whether
    that is acceptable, which is a decision they can only make if they are told.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            log.debug("no CUDA device present; peak memory is unmeasured")
            return None
        return torch.cuda.max_memory_allocated(gpu_index) / (1024**3)
    except Exception as exc:  # noqa: BLE001
        # The message carries the cause; a full traceback per candidate would
        # bury the evaluation log without adding anything.
        log.warning(
            "could not read GPU memory on device %s (%s); peak memory will be "
            "reported as unmeasured",
            gpu_index,
            exc,
        )
        return None


def reset_vram_peak(gpu_index: int = 0) -> None:
    """Clear the peak counter before serving a new candidate.

    A failure here is worth a warning rather than a shrug: without a reset, the
    next candidate inherits the previous one's peak and is measured against
    memory it never allocated.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(gpu_index)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "could not reset the GPU memory counter on device %s (%s); the next "
            "candidate's peak may include this one's",
            gpu_index,
            exc,
        )
