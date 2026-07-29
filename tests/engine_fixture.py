"""A fully wired evaluation engine over the miniature pool.

Shared by the end-to-end suite and the disclosure tests, which both need a real
engine but exercise different parts of it. Defined once so the two cannot drift
into testing subtly different systems.
"""

from __future__ import annotations

import pytest

from capability_subnet.backend.engine_loop import EngineLoop
from capability_subnet.backend.evaluation import Evaluator
from capability_subnet.backend.executor.reconstruction import ArtifactCache, Reconstructor
from capability_subnet.backend.reports.publisher import ReportPublisher
from capability_subnet.backend.settings import BackendSettings
from capability_subnet.merge_engine.loader import SafetensorsAdapterSource
from capability_subnet.sandbox.reference_solver import ReferenceSolverClient
from capability_subnet.workflows import get_workflow
from tests.conftest import MAINTENANCE_WORKFLOW_ID


class ScriptedServer:
    """Serves a scripted solver whose competence depends on the package.

    The mapping from artifact to impairment is the test's way of saying "this
    package is worse at that capability". A real deployment gets the same signal
    from a real model; the engine cannot tell the difference, which is the point.
    """

    def __init__(
        self,
        impairments_by_artifact: dict[str | None, frozenset[str]] | None = None,
        *,
        default_impairments: frozenset[str] = frozenset(),
    ) -> None:
        self.impairments_by_artifact = impairments_by_artifact or {}
        # A default rather than an exhaustive map: the reference set includes one
        # entry per capability adapter plus several merges plus the bare base
        # model, and a test that has to enumerate all of them is a test that
        # silently stops covering anything the reference set gains later.
        self.default_impairments = default_impairments
        self.served: list[str | None] = []

    def impairments_for(self, adapter_path) -> frozenset[str]:
        key = str(adapter_path.name) if adapter_path is not None else None
        return self.impairments_by_artifact.get(key, self.default_impairments)

    class _Handle:
        def __init__(self, outer, adapter_path) -> None:
            self.outer = outer
            self.adapter_path = adapter_path
            self.peak_vram_gb = 8.0

        def client(self, timeout: float = 120.0):
            return _PackageAwareClient(self.outer.impairments_for(self.adapter_path))

    def serve(self, adapter_path):
        from contextlib import contextmanager

        @contextmanager
        def _serve():
            self.served.append(str(adapter_path) if adapter_path else None)
            yield ScriptedServer._Handle(self, adapter_path)

        return _serve()


class _PackageAwareClient:
    """A model client that builds a solver per instance.

    The engine hands the same client to every instance in a batch, but the
    reference solver is constructed from one instance. This wrapper closes that
    gap by rebuilding the solver whenever it sees a new opening conversation.
    """

    def __init__(self, impairments: frozenset[str]) -> None:
        self.impairments = impairments
        self._solver = None
        self._instance_key: str | None = None

    def complete(self, messages, tools, *, seed: int, max_tokens: int):
        # The general-capability probe is a single user turn with no tools. It
        # is answered here rather than by the workflow solver because it is
        # deliberately *not* a workflow question — that separation is the whole
        # reason the retention gate can see anything the completion score cannot.
        if len(messages) == 1 and messages[0].role == "user":
            return self._answer_probe(messages[0].content or "")

        key = messages[1].content if len(messages) > 1 else ""
        assert key is not None

        if self._instance_key != key:
            self._instance_key = key
            self._solver = ReferenceSolverClient(
                _instance_for_conversation(key), impairments=self.impairments
            )
        return self._solver.complete(messages, tools, seed=seed, max_tokens=max_tokens)

    def _answer_probe(self, prompt: str):
        """Answer a probe item, correctly unless this package lost retention.

        ``general_retention`` in the impairment set stands for a merge that
        traded away general instruction following: the package still answers,
        but pads the reply, which is exactly the failure the probe's exact-match
        comparison is built to catch.
        """
        from capability_subnet.sandbox.model_client import ModelReply

        expected = _PROBE_ANSWERS.get(prompt)
        if expected is None:
            return ModelReply(content=None, tool_calls=[], error="no probe answer registered")
        if "general_retention" in self.impairments:
            return ModelReply(content=f"Sure! The answer is {expected}.", tool_calls=[])
        return ModelReply(content=expected, tool_calls=[])


_INSTANCE_INDEX: dict[str, object] = {}

#: Probe prompt -> expected answer, registered as each window's probe is drawn.
#: The harness equivalent of the model simply knowing the answer.
_PROBE_ANSWERS: dict[str, str] = {}


def register_probe(items) -> None:
    for item in items:
        _PROBE_ANSWERS[item.prompt] = item.expected


def _instance_for_conversation(observation: str):
    """Recover which instance a conversation belongs to.

    The scripted solver needs the instance object; the engine only hands the
    client an opening message. Registering instances as they are generated is a
    test-harness shortcut, not something a real client could do — a served model
    has no such index and no need for one.
    """
    for instance in _INSTANCE_INDEX.values():
        if instance.machine_id in observation and instance.task_prompt_de in observation:
            return instance
    raise AssertionError("conversation does not match any registered instance")


@pytest.fixture
def engine(tmp_path, tiny_snapshot, tiny_pool_dir):
    """A fully wired engine over the miniature pool."""
    import dataclasses

    base = get_workflow(MAINTENANCE_WORKFLOW_ID)

    def _tracking_generate(seed, *, split="hidden"):
        instance = base.generate_instance(seed, split=split)
        _INSTANCE_INDEX[instance.instance_id] = instance
        return instance

    # The workflow module is frozen, so a tracking variant is built rather than
    # patched. That also keeps the registry's cached module untouched, which
    # matters because it is shared across the session.
    workflow = dataclasses.replace(base, generate_instance=_tracking_generate)

    # The probe is drawn inside the loop, so the harness learns its answers by
    # wrapping the draw rather than by reaching into the engine.
    import capability_subnet.backend.engine_loop as engine_loop_module

    original_build_probe = engine_loop_module.build_probe

    def _tracking_build_probe(seed, *args, **kwargs):
        items = original_build_probe(seed, *args, **kwargs)
        register_probe(items)
        return items

    engine_loop_module.build_probe = _tracking_build_probe

    settings = BackendSettings(
        state_dir=str(tmp_path / "state"),
        artifact_cache_dir=str(tmp_path / "cache"),
        adapter_pool_dir=str(tiny_pool_dir),
        report_dir=str(tmp_path / "reports"),
        window_blocks=100,
        hidden_instances=4,
        ood_instances=1,
        hidden_seed_root=987654321,
        reconstruction_workers=2,
        min_axis_samples=3,
        end_to_end_margin=0.10,
        dry_run=True,
    )
    settings.ensure_directories()

    from capability_subnet.backend.store import Store

    store = Store(settings.database_path)
    server = ScriptedServer({})

    evaluator = Evaluator(
        reconstructor=Reconstructor(
            tiny_snapshot,
            SafetensorsAdapterSource(tiny_pool_dir),
            ArtifactCache(settings.artifact_cache_dir),
            workers=settings.reconstruction_workers,
        ),
        server=server,
        adapter_pool_dir=tiny_pool_dir,
        stages=workflow.critical_axes,
        min_valid_samples=3,
    )

    loop = EngineLoop(
        settings=settings,
        store=store,
        snapshot=tiny_snapshot,
        workflow=workflow,
        evaluator=evaluator,
        publisher=ReportPublisher(settings.report_dir),
    )

    try:
        yield loop, store, server, settings
    finally:
        store.close()
        _INSTANCE_INDEX.clear()


def _store_recipe(settings: BackendSettings, recipe) -> None:
    """Persist a recipe where the engine looks for it."""
    directory = settings.state_path / "recipes"
    directory.mkdir(parents=True, exist_ok=True)
    digest = recipe.digest().split(":", 1)[1]
    (directory / f"{digest}.json").write_bytes(recipe.canonical_bytes())


def _artifact_name_for(loop: EngineLoop, recipe) -> str:
    """The cache directory name a recipe reconstructs into.

    Used to tell the scripted server which served package is which, since it
    identifies a candidate by the artifact directory it is asked to serve.
    """
    from capability_subnet.merge_engine.engine import reconstruct

    result = reconstruct(recipe, loop.snapshot, loop.evaluator.reconstructor.source)
    return result.artifact_sha256.split(":", 1)[1]
