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

    It reconstructs nothing, so anyone with a laptop can check a closed run.
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


class TestNothingIsKeyedByWhicheverWorkflowIsDefault:
    """Every workflow identity is stated, never read from the current default.

    A workflow read from `DEFAULT_WORKFLOW_ID` stops existing when the default
    moves: it unregisters, calls itself by the new default's name, loses its
    on-chain commitment code, and lets a disclosure claim a workflow it is not.
    """

    def test_every_workflow_registers_under_its_own_id(self):
        from capability_subnet.workflows import available_workflows, get_workflow

        for workflow_id in available_workflows():
            assert get_workflow(workflow_id).workflow_id == workflow_id

    def test_both_shipped_workflows_survive_a_change_of_default(self, monkeypatch):
        from capability_subnet.common import constants as C
        from capability_subnet.workflows import available_workflows

        registered = set(available_workflows())
        assert {"industrial_maintenance_de_v1", "lora_merger_logic_v1"} <= registered

        monkeypatch.setattr(C, "DEFAULT_WORKFLOW_ID", "something-else-entirely")
        assert set(available_workflows()) == registered

    def test_every_workflow_has_a_commitment_code(self):
        from capability_subnet.common.commitments import WORKFLOW_CODES, decode_commitment
        from capability_subnet.common.hashing import sha256_bytes
        from capability_subnet.workflows import available_workflows

        uri = "hf:alice/recipes/final.json"
        for workflow_id in available_workflows():
            assert workflow_id in WORKFLOW_CODES, f"{workflow_id} could never be committed"
            from capability_subnet.common.commitments import encode_commitment

            payload = encode_commitment(workflow_id, sha256_bytes(b"x"), uri)
            assert decode_commitment(payload).workflow_id == workflow_id

    def test_a_disclosure_must_name_its_own_workflow(self):
        """Otherwise an old disclosure is scored against whatever workflow is
        configured when it is replayed."""
        import pytest

        from capability_subnet.common.schemas import RunDisclosure

        with pytest.raises(Exception):  # noqa: B017 - pydantic's own validation error
            RunDisclosure(run_id=1, closed_at_block=1)

    def test_every_workflow_publishes_the_protocol_facts_a_miner_needs(self):
        """A contract without the base model or the pool is not one a miner can
        build against."""
        from capability_subnet.workflows import available_workflows, get_workflow

        required = {
            "workflow_id",
            "base_model",
            "source_pool",
            "recipe",
            "stages",
            "hard_gates",
            "champion_challenge",
            "runs",
            "incentive",
        }
        for workflow_id in available_workflows():
            contract = get_workflow(workflow_id).build_contract()
            assert required <= set(contract), f"{workflow_id} omits {required - set(contract)}"
            assert contract["base_model"]["repo"]
            assert contract["source_pool"]["adapters"]
            assert contract["stages"]["order"] == list(get_workflow(workflow_id).stages)


class TestTheOperatorEngineIsSeparable:
    """`backend/` ships in a private repo, so nothing public may depend on it.

    The rules that decide a published number live in `scoring/` and stay public: a
    score a validator cannot recompute is a claim, not a measurement. The engine
    that runs those rules is the operator's. If a public module reaches into
    `backend/`, the two can no longer be shipped apart.
    """

    PUBLIC_PACKAGES = (
        "common",
        "registry",
        "merge_engine",
        "workflows",
        "sandbox",
        "scoring",
        "miner",
        "validator",
        "audit",
    )

    @staticmethod
    def _imports(package: str) -> set[str]:
        import ast
        import pathlib

        found: set[str] = set()
        for path in pathlib.Path("capability_subnet", package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    found.add(node.module)
                elif isinstance(node, ast.Import):
                    found.update(alias.name for alias in node.names)
        return found

    def test_no_public_package_imports_the_engine(self):
        offenders = {
            package: sorted(
                name
                for name in self._imports(package)
                if name.startswith("capability_subnet.backend")
            )
            for package in self.PUBLIC_PACKAGES
        }
        offenders = {k: v for k, v in offenders.items() if v}
        assert not offenders, f"public packages reaching into the operator engine: {offenders}"

    def test_the_scoring_rules_stand_alone(self):
        """A validator installs the public package and nothing else, so the rules
        cannot need the engine's store, settings or serving to run."""
        import capability_subnet.scoring as scoring

        assert scoring.__all__
        assert all(hasattr(scoring, name) for name in scoring.__all__)

    def test_every_score_deciding_module_is_public(self):
        """Anything that turns evidence into a number a miner is paid on belongs
        in `scoring/`, where it can be re-run by whoever is about to pay."""
        import pathlib

        expected = {
            "aggregate",
            "bootstrap",
            "comparator",
            "contribution",
            "gates",
            "ranking",
            "references",
            "retention",
            "sampler",
            "weight_vector",
        }
        present = {
            p.stem
            for p in pathlib.Path("capability_subnet/scoring").glob("*.py")
            if p.stem != "__init__"
        }
        assert expected == present, f"scoring/ drifted: {expected ^ present}"


class TestTheBaseInstallNeedsNoTensorStack:
    """A validator installs the base package and no torch, on a documented 20 GB
    VPS. Anything it does must work without the tensor stack."""

    @staticmethod
    def _without_torch():
        import builtins
        from contextlib import contextmanager

        @contextmanager
        def _guard():
            real = builtins.__import__

            def blocked(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch."):
                    raise ImportError("torch is not installed")
                return real(name, *args, **kwargs)

            builtins.__import__ = blocked
            try:
                yield
            finally:
                builtins.__import__ = real

        return _guard()

    def test_a_contract_builds_without_torch(self):
        """The published contract names every merge method and its stages. That
        is description, not arithmetic, so reading it must not require the
        implementation — a validator checks contracts and merges nothing."""
        import importlib
        import sys

        for module in [m for m in sys.modules if m.startswith("capability_subnet.workflows")]:
            del sys.modules[module]

        with self._without_torch():
            workflows = importlib.import_module("capability_subnet.workflows")
            contract = workflows.get_workflow("lora_merger_logic_v1").build_contract()

        assert contract["recipe"]["merge_methods"]
        assert contract["base_model"]["repo"]

    def test_the_method_descriptions_live_apart_from_the_arithmetic(self):
        import capability_subnet.common.merge_methods as descriptions
        from capability_subnet.common import constants as constants_module

        source = open(descriptions.__file__, encoding="utf-8").read()
        assert "import torch" not in source
        assert set(descriptions.PIPELINES) == set(constants_module.ALLOWED_MERGE_METHODS)


class TestTheDeclaredDependenciesAreEnough:
    """A dependency that is installed here but undeclared works for us and fails
    for everyone else — the same class of bug as a file missing from git."""

    @staticmethod
    def _declared() -> set[str]:
        """Base dependency names, read without a TOML parser.

        `tomllib` is 3.11+ and this package supports 3.10, so importing it here
        failed the entire suite on the oldest version the project claims to run
        on — and passed locally, because local is 3.12. The block being read is a
        flat array of strings; it does not need a parser, and a test *about*
        dependencies should not add one.
        """
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent.parent
        text = (root / "pyproject.toml").read_text()
        block = re.search(r"(?ms)^dependencies\s*=\s*\[(.*?)^\]", text)
        assert block, "could not find the base dependencies array in pyproject.toml"

        names = set()
        for line in block.group(1).splitlines():
            quoted = re.match(r"""\s*["'](.+?)["']""", line)
            if quoted:
                names.add(re.split(r"[<>=!\[; ]", quoted.group(1))[0].lower().replace("-", "_"))
        return names

    def test_the_default_workflow_can_draw_an_instance_on_a_base_install(self):
        """A validator installs the base package and re-scores a closed run,
        which regenerates every instance. Anything that path imports has to be a
        base dependency, not an extra."""
        declared = self._declared()
        for third_party in ("pandas", "huggingface_hub"):
            assert third_party in declared, (
                f"{third_party} is imported when generating an instance for the "
                "default workflow but is not a base dependency"
            )

    def test_replay_regenerates_instances_and_so_needs_them(self):
        """Pins why: if replay stops calling generate_instance, this rule can be
        revisited. While it does, the dependency is not optional."""
        from pathlib import Path

        import capability_subnet.audit.replay as replay

        source = Path(replay.__file__).read_text()
        assert "generate_instance" in source


class TestTheDeclaredPythonFloorIsHonoured:
    """`requires-python = ">=3.10"` is a promise, and CI runs 3.10 to keep it.

    Nothing was checking it, so a 3.11-only import reached main and failed the
    whole suite on the oldest supported version while passing on every developer
    machine — which is 3.12.
    """

    #: Standard-library modules that do not exist on 3.10.
    TOO_NEW = {"tomllib": "3.11"}

    #: Syntax and names introduced after 3.10.
    TOO_NEW_NAMES = {
        "datetime.UTC": "3.11",
        "StrEnum": "3.11",
        "TaskGroup": "3.11",
        "ExceptionGroup": "3.11",
        "typing.Self": "3.11",
        "itertools.batched": "3.12",
        "typing.override": "3.12",
    }

    @staticmethod
    def _sources():
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent.parent
        for package in ("capability_subnet", "tests", "neurons", "scripts"):
            directory = root / package
            if not directory.is_dir():
                continue
            for path in directory.rglob("*.py"):
                # This file names every construct it forbids, so scanning it
                # would only ever find its own lookup table.
                if "__pycache__" in path.parts or path.name == Path(__file__).name:
                    continue
                yield path, path.read_text(encoding="utf-8")

    def test_no_module_newer_than_the_floor_is_imported(self):
        import re

        offences = []
        for path, text in self._sources():
            body = re.sub(r'"""(?:.|\n)*?"""', "", text)  # docstrings may name them
            for module, version in self.TOO_NEW.items():
                if re.search(rf"^\s*(?:import {module}\b|from {module}\b)", body, re.M):
                    offences.append(f"{path.name} imports {module} ({version}+)")
        assert not offences, "; ".join(offences)

    def test_no_name_newer_than_the_floor_is_used(self):
        import re

        offences = []
        for path, text in self._sources():
            body = re.sub(r'"""(?:.|\n)*?"""', "", text)
            for name, version in self.TOO_NEW_NAMES.items():
                if re.search(rf"\b{re.escape(name)}\b", body):
                    offences.append(f"{path.name} uses {name} ({version}+)")
        assert not offences, "; ".join(offences)

    def test_the_floor_and_the_classifiers_agree(self):
        import re
        from pathlib import Path

        text = (Path(__file__).resolve().parent.parent.parent / "pyproject.toml").read_text()
        floor = re.search(r'requires-python\s*=\s*">=(\d+\.\d+)"', text)
        assert floor, "requires-python is not declared"
        classified = set(re.findall(r"Programming Language :: Python :: (\d+\.\d+)", text))
        assert floor.group(1) in classified, (
            f"requires-python says >={floor.group(1)} but no classifier claims it"
        )
