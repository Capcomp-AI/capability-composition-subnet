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

Evaluation is sequential, one instance at a time against one endpoint, and that
is a measurement decision rather than a simplification: latency is a scored
quantity, so a host running more sandboxes than it has capacity for would report
numbers that say more about the operator's provisioning than about the candidate.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
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


class ExternalServer:
    """An endpoint an operator manages outside the engine.

    The engine points at a URL and assumes the operator restarted the runtime
    with the right adapter. This is the simplest deployment and the one used for
    development; it trades automation for the operator being able to run whatever
    serving stack they already have.
    """

    def __init__(self, base_url: str, model_name: str, *, allow_adapters: bool = False) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.allow_adapters = allow_adapters

    @contextmanager
    def serve(self, adapter_path: str | Path | None) -> Iterator[ServingHandle]:
        if adapter_path and not self.allow_adapters:
            # This class cannot load an adapter — it points at a process it does
            # not own. Yielding anyway would score every candidate against
            # whatever the operator last started, so every package would post
            # identical numbers, no challenger could ever clear the margin, and
            # the network would burn its emission for reasons no report would
            # explain. Refusing makes that a loud infrastructure failure.
            raise ServingError(
                f"the external server cannot apply {adapter_path}. It serves whatever "
                "process is already running at its URL, so a candidate would be measured "
                "against the wrong weights. Set serving_mode: managed to have the engine "
                "start a runtime per candidate, or set allow_adapters when you have "
                "genuinely restarted the runtime with this artifact yourself."
            )

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
        startup_timeout: float = 900.0,
        max_model_len: int = 16384,
        tool_call_parser: str = "hermes",
        reasoning_parser: str = "",
        gpu_memory_utilization: float = 0.90,
        python_executable: str = "",
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.base_model_path = base_model_path
        self.model_name = model_name
        self.host = host
        self.port = port
        self.gpu_index = gpu_index
        self.startup_timeout = startup_timeout
        self.max_model_len = max_model_len
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.gpu_memory_utilization = gpu_memory_utilization
        # The interpreter vLLM runs under. Defaults to the engine's own, which is
        # right when they share an environment. Separating them is common in
        # practice — vLLM pins torch versions tightly enough that operators
        # routinely keep it in its own virtualenv — and hardcoding sys.executable
        # made that impossible without patching the engine.
        self.python_executable = python_executable or sys.executable
        self.extra_args = extra_args

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _command(self, adapter_path: str | None) -> list[str]:
        command = [
            # An explicit interpreter, never whatever "python" resolves to on
            # PATH: a venv or conda deployment has vLLM installed for a specific
            # interpreter and generally not for /usr/bin/python.
            self.python_executable,
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
            str(self.max_model_len),
            "--seed",
            "0",
            "--gpu-memory-utilization",
            f"{self.gpu_memory_utilization:.2f}",
            # Without these two the server ignores the `tools` parameter
            # entirely: it renders them into the prompt but never parses the
            # reply, so `message.tool_calls` comes back empty and `tool_choice:
            # "auto"` is rejected outright. Every instance would then fail for a
            # reason that has nothing to do with the candidate.
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            self.tool_call_parser,
        ]
        if self.reasoning_parser:
            command += ["--reasoning-parser", self.reasoning_parser]

        # Request logging is off by default in current vLLM, and the flag that
        # used to turn it off was removed rather than deprecated — passing it
        # makes the server exit with an argparse error before it loads
        # anything, which surfaces as every candidate failing to be served. The
        # engine does not control which vLLM an operator installed, so it asks
        # rather than assumes.
        if self._accepts("--disable-log-requests"):
            command.append("--disable-log-requests")
        if adapter_path:
            command += [
                "--enable-lora",
                "--max-lora-rank",
                str(max(C.ALLOWED_OUTPUT_RANKS)),
                "--lora-modules",
                f"{self.model_name}={adapter_path}",
                # The served name maps to the adapter, so a request naming the
                # model gets the candidate rather than the untouched base.
                "--max-loras",
                "1",
            ]
        return command + list(self.extra_args)

    @lru_cache(maxsize=8)  # noqa: B019 - keyed per server instance, bounded by flag count
    def _accepts(self, flag: str) -> bool:
        """Whether this vLLM build accepts ``flag``.

        Asked once per flag by running the server's own ``--help``. vLLM removes
        options between releases without a deprecation window, and an unknown
        option is an immediate argparse exit — so a flag the engine merely
        *prefers* must never be the reason nothing can be served.

        Flags the protocol genuinely depends on are not routed through here.
        `--enable-auto-tool-choice` and `--tool-call-parser` are load-bearing: if
        a vLLM build lacks them it cannot run this workflow at all, and failing
        loudly at start-up is the correct outcome.
        """
        try:
            probe = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [self.python_executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
                capture_output=True,
                text=True,
                timeout=180,
                env=self._environment(),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not probe vLLM options (%s); omitting %s", exc, flag)
            return False
        return flag in probe.stdout

    def _environment(self) -> dict[str, str]:
        """The subprocess environment.

        Inherited rather than replaced. The previous build passed a four-entry
        dict, which dropped HOME, HF_HOME, LD_LIBRARY_PATH and the virtualenv —
        enough for vLLM to fail at import on any deployment that is not a bare
        system-Python install. Only the variables that must differ are
        overridden.
        """
        env = dict(os.environ)

        # Put the serving interpreter's own bin/ first on PATH. A runtime shells
        # out to build tools that live beside it — vLLM JIT-compiles kernels with
        # `ninja` — and when the interpreter belongs to a different virtualenv
        # than the engine, those tools are not on the inherited PATH. The failure
        # is a bare `FileNotFoundError: 'ninja'` from deep inside engine
        # start-up, which reads like anything except a PATH problem.
        # Deliberately not resolved: a virtualenv's `bin/python` is a symlink to
        # the system interpreter, so resolving it yields /usr/bin and adds the
        # one directory that was already on PATH. The venv's own bin/ — where
        # its console scripts live — is the literal parent.
        interpreter_bin = str(Path(self.python_executable).absolute().parent)
        env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")]).rstrip(os.pathsep)

        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        # The one feature that would let something inside the sandbox swap the
        # package under measurement for a different one.
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "0"
        # The pool and the base model are materialised by the operator ahead of
        # time; a candidate evaluation must never reach the network.
        env.setdefault("HF_HUB_OFFLINE", "1")
        return env

    @contextmanager
    def serve(self, adapter_path: str | Path | None) -> Iterator[ServingHandle]:
        adapter = str(adapter_path) if adapter_path else None
        command = self._command(adapter)

        log.info(
            "starting the candidate endpoint on GPU %d for %s",
            self.gpu_index,
            adapter or "<base model>",
        )
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command,
            env=self._environment(),
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
                    f"the endpoint exited during start-up with code {process.returncode}. "
                    + summarise_startup_failure(output.decode("utf-8", "replace"))
                )
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    return
            except Exception:  # noqa: BLE001 - still starting
                pass
            time.sleep(2.0)

        raise ServingError(f"the endpoint did not become ready within {self.startup_timeout:.0f}s")


#: Lines worth surfacing from a failed start-up, in priority order. vLLM prints
#: thousands of lines before it dies and the cause is rarely in the last of them.
_FAILURE_MARKERS: tuple[str, ...] = (
    "ValueError:",
    "RuntimeError:",
    "AssertionError:",
    "OutOfMemoryError",
    "torch.OutOfMemoryError",
    "ImportError:",
    "ModuleNotFoundError:",
    "OSError:",
    "error: unrecognized arguments",
    "error: argument",
    "No such file or directory",
)


def summarise_startup_failure(output: str, *, limit: int = 6) -> str:
    """Pull the cause out of a serving runtime's start-up log.

    The previous behaviour was to attach the last 2000 characters. That reliably
    hid the reason: a runtime prints thousands of lines of banner, config dump
    and shutdown noise after the exception that killed it, so the tail is
    whatever happened to be flushed last and the cause is somewhere in the
    middle. Every start-up failure looked the same, and looked like nothing.

    Matching on exception markers instead means the operator gets the line that
    explains it — an out-of-memory, an unrecognised option, a missing model
    path — rather than a fragment of a file path.
    """
    lines = [line.strip() for line in output.splitlines() if line.strip()]

    interesting = [line for line in lines if any(marker in line for marker in _FAILURE_MARKERS)]
    # Deduplicated: a failure in a worker subprocess is typically re-raised in
    # the parent, so the same sentence arrives two or three times.
    seen: set[str] = set()
    unique = [line for line in interesting if not (line in seen or seen.add(line))]

    if unique:
        return " | ".join(unique[:limit])

    # Nothing matched a known marker. The tail is a poor answer but it is better
    # than an empty one, and saying that it is a fallback stops the reader from
    # assuming it is the cause.
    tail = " ".join(lines[-limit:])
    return f"no recognised error line; last output was: {tail[-600:]}"


def read_peak_vram_gb(gpu_index: int = 0) -> float | None:
    """Memory in use on the assigned GPU, in gigabytes.

    Read through NVML, which reports what the *device* is holding across every
    process on it. That is the only reading that can see the candidate at all:
    the model runs in a separate vLLM process, so ``torch.cuda.max_memory_allocated``
    called here would report this process's own allocations — essentially zero —
    and every candidate would sail through a gate that never looked at it.

    NVML also indexes devices globally, while the child sees its assigned GPU
    remapped to 0 by ``CUDA_VISIBLE_DEVICES``. ``gpu_index`` is therefore the
    global index, matching what the operator configured.

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
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.used) / (1024**3)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001
        # The message carries the cause; a full traceback per candidate would
        # bury the evaluation log without adding anything.
        log.warning(
            "could not read GPU memory on device %s via NVML (%s); peak memory will "
            "be reported as unmeasured",
            gpu_index,
            exc,
        )
        return None


class VramSampler:
    """Tracks the high-water mark on one GPU while a candidate is being run.

    A single reading taken after start-up measures the model's resident weights
    and nothing else — not the KV cache under load, which is where a candidate
    with a large adapter actually costs memory. Sampling across the run is what
    makes the reported peak a peak.
    """

    def __init__(self, gpu_index: int = 0, *, interval: float = 5.0) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self._peak: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def peak_gb(self) -> float | None:
        return self._peak

    def sample(self) -> None:
        reading = read_peak_vram_gb(self.gpu_index)
        if reading is None:
            return
        self._peak = reading if self._peak is None else max(self._peak, reading)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> VramSampler:
        self.sample()
        self._thread = threading.Thread(target=self._run, name="vram-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)
        self.sample()
