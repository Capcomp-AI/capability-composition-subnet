"""The dependency layering, enforced rather than documented.

A validator on this subnet is advertised as a small VPS with no GPU. That claim
is only true if the validator's install is actually small, and nothing in a code
review reliably catches the moment it stops being — someone adds a convenience
import at the top of a shared module, every role inherits it, and the install
grows by a tensor library that most roles never call.

So the boundary is a test. The layers below import nothing heavier than the base
dependency set, and if that stops being true this fails with the name of the
module that broke it.

The same boundary is what makes a repository split *possible* later without
being necessary now: these packages already have no path to the tensor stack, so
they could be extracted whenever there is a reason to. Until then the seam is
cheaper to hold here, where one test guards it, than across repositories, where
it would be held by two version numbers that can drift apart.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "capability_subnet"

#: Distributions that are *not* in the base install. Importing one of these from
#: a light layer means that layer's role now has to download it.
HEAVY = {
    "torch",
    "numpy",
    "safetensors",
    "transformers",
    "accelerate",
    "peft",
    "vllm",
    "huggingface_hub",
    "psycopg",
    "docker",
    "fastapi",
    "uvicorn",
    "pynvml",
}

#: Packages every role installs, which therefore must stay light.
LIGHT_LAYERS = ("common", "workflows", "miner", "validator", "audit", "platform")


def _module_level_imports(path: pathlib.Path) -> set[str]:
    """Top-level distributions a module imports *at import time*.

    Only module-level statements count. An import inside a function body is a
    deliberate deferral — the cost is paid by the caller that reaches that code
    path, not by everyone who imports the module — and this codebase uses that
    pattern throughout precisely so the light layers stay light.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])

        # Class bodies are executed at import time too, so their imports count.
        if isinstance(node, ast.ClassDef):
            for inner in node.body:
                if isinstance(inner, ast.Import):
                    found.update(alias.name.split(".")[0] for alias in inner.names)
                elif isinstance(inner, ast.ImportFrom) and inner.level == 0 and inner.module:
                    found.add(inner.module.split(".")[0])

    return found


def _modules_in(layer: str) -> list[pathlib.Path]:
    return sorted((PACKAGE_ROOT / layer).rglob("*.py"))


@pytest.mark.parametrize("layer", LIGHT_LAYERS)
def test_the_light_layers_import_nothing_heavy(layer: str):
    """A validator must not download a tensor library it never calls."""
    offenders: list[str] = []

    for path in _modules_in(layer):
        heavy = _module_level_imports(path) & HEAVY
        if heavy:
            rel = path.relative_to(PACKAGE_ROOT.parent)
            offenders.append(f"{rel} imports {sorted(heavy)}")

    assert not offenders, (
        f"the {layer!r} layer is installed by every role and must stay light, but:\n  "
        + "\n  ".join(offenders)
        + "\n\nMove the import inside the function that needs it, or move the code "
        "into a layer whose role already installs that dependency."
    )


def _heavy_modules_after_importing(target: str) -> set[str]:
    """Which heavy libraries a fresh interpreter loads to import ``target``.

    A subprocess, because the test session has already imported the tensor
    stack for the reconstruction suites and nothing short of a new interpreter
    unloads it. Popping entries from ``sys.modules`` would test the popping.
    """
    import json
    import subprocess
    import sys

    code = (
        f"import sys, json; import {target}; "
        "print(json.dumps(sorted({n.split('.')[0] for n in sys.modules})))"
    )
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return set(json.loads(completed.stdout.strip().splitlines()[-1])) & HEAVY


def test_the_validator_reaches_the_chain_without_the_tensor_stack():
    """The end-to-end version of the rule above, on the module that matters most."""
    heavy = _heavy_modules_after_importing("capability_subnet.validator.neuron")
    assert not (heavy & {"torch", "transformers", "safetensors", "vllm"}), heavy


def test_the_auditor_can_replay_without_the_tensor_stack():
    """Replay re-runs the deterministic scorer over published traces.

    It reconstructs nothing, so anyone with a laptop can check a closed window.
    That is the property that makes the published record meaningful to someone
    who is not the operator, and it disappears the moment replay needs a GPU.
    """
    heavy = _heavy_modules_after_importing("capability_subnet.audit.replay")
    assert "torch" not in heavy, heavy


def test_the_merge_engine_is_allowed_to_be_heavy():
    """The boundary is a boundary, not a ban — reconstruction genuinely needs it."""
    heavy = set()
    for path in _modules_in("merge_engine"):
        heavy |= _module_level_imports(path) & HEAVY
    assert "torch" in heavy


class TestTheServingCommandIsCorrect:
    """The vLLM invocation decides whether any candidate can be measured at all.

    Every flag below was missing at some point, and each absence produced the
    same symptom — every instance failing for a reason unrelated to the
    candidate — so they are pinned rather than trusted to review.
    """

    @staticmethod
    def _server(**kwargs):
        from capability_subnet.backend.executor.serving import ManagedVllmServer

        defaults = dict(base_model_path="/models/base", python_executable="/venv/bin/python")
        defaults.update(kwargs)
        return ManagedVllmServer(**defaults)

    def test_tool_calling_is_enabled_and_parsed(self):
        """Without both, `message.tool_calls` is never populated and
        `tool_choice: "auto"` is rejected outright."""
        command = self._server()._command(None)
        assert "--enable-auto-tool-choice" in command
        assert command[command.index("--tool-call-parser") + 1] == "hermes"

    def test_the_adapter_is_applied_when_one_is_given(self):
        """The whole point: the engine must serve the artifact it just built."""
        command = self._server()._command("/artifacts/candidate")
        assert "--enable-lora" in command
        assert "candidate=/artifacts/candidate" in command

    def test_no_adapter_means_the_bare_base_model(self):
        command = self._server()._command(None)
        assert "--enable-lora" not in command

    def test_the_interpreter_is_explicit(self):
        """Never "python" from PATH: vLLM lives in a specific environment."""
        assert self._server()._command(None)[0] == "/venv/bin/python"

    def test_the_environment_is_inherited_not_replaced(self):
        """A four-entry env dropped HOME, HF_HOME and LD_LIBRARY_PATH, which is
        enough for vLLM to fail at import on most real deployments."""
        import os

        env = self._server()._environment()
        assert len(env) > 4
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "0"
        assert env["HF_HUB_OFFLINE"] == "1"
        if "PATH" in os.environ:
            assert env["PATH"] == os.environ["PATH"]


class TestAStartUpFailureExplainsItself:
    """The engine's own diagnostics decide how long an outage lasts.

    These cases are transcribed from real failures encountered bringing the
    engine up on a live host. Each one was originally reported as a fragment of
    a file path, because the message was the last 2000 characters of a log whose
    final lines are shutdown noise — so every distinct failure looked identical
    and looked like nothing.
    """

    def test_an_out_of_memory_failure_names_the_memory(self):
        from capability_subnet.backend.executor.serving import summarise_startup_failure

        log = "\n".join(
            ["INFO banner"] * 500
            + [
                "ValueError: Free memory on device cuda:0 (5.15/23.52 GiB) on startup is "
                "less than desired GPU memory utilization (0.85, 19.99 GiB)."
            ]
            + ["INFO shutting down"] * 500
        )
        summary = summarise_startup_failure(log)
        assert "Free memory on device cuda:0" in summary

    def test_a_dead_gpu_names_the_device_count(self):
        from capability_subnet.backend.executor.serving import summarise_startup_failure

        log = "\n".join(
            ["INFO banner"] * 300
            + ["AssertionError: DP adjusted local rank 0 is out of bounds for 0 devices."]
            + ["INFO teardown"] * 300
        )
        assert "out of bounds for 0 devices" in summarise_startup_failure(log)

    def test_an_unknown_flag_names_the_flag(self):
        from capability_subnet.backend.executor.serving import summarise_startup_failure

        log = "usage: api_server [-h] ...\napi_server: error: unrecognized arguments: --disable-log-requests"
        assert "--disable-log-requests" in summarise_startup_failure(log)

    def test_the_same_error_repeated_by_workers_is_reported_once(self):
        from capability_subnet.backend.executor.serving import summarise_startup_failure

        line = "RuntimeError: Engine core initialization failed."
        summary = summarise_startup_failure("\n".join([line] * 5))
        assert summary.count("Engine core initialization failed") == 1

    def test_an_unrecognised_failure_says_so_rather_than_guessing(self):
        """A fallback presented as a cause is worse than an admitted fallback."""
        from capability_subnet.backend.executor.serving import summarise_startup_failure

        summary = summarise_startup_failure("\n".join(f"INFO step {i}" for i in range(100)))
        assert "no recognised error line" in summary
