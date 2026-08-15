"""Reconstruction determinism.

The protocol's strongest claim is that a recipe reconstructs to the same bytes on
every machine running the same software. Everything downstream depends on it: the
artifact digest is the anti-copy identity, the cache key, and the thing two
workers must agree on before a candidate is scored.

These tests cannot prove cross-machine agreement — only running on two machines
can do that, which is what the engine's cross-worker check is for. What they can
prove is that the result does not depend on anything *within* a machine that is
allowed to vary: thread count, iteration order, adapter list order, repeated
invocation, or having merged something else first.
"""

from __future__ import annotations

import pytest
import torch

from capability_subnet.common import constants as C
from capability_subnet.merge_engine.canonical_writer import artifact_digest_of_tensors
from capability_subnet.merge_engine.engine import reconstruct

ALL_METHODS = [
    (C.MERGE_LINEAR, None, None),
    (C.MERGE_SVD, None, None),
    (C.MERGE_CAT_SVD, None, None),
    (C.MERGE_TIES_SVD, 0.4, "total"),
    (C.MERGE_DARE_TIES_SVD, 0.35, "frequency"),
    (C.MERGE_DARE_LINEAR_SVD, 0.6, None),
    (C.MERGE_MAGNITUDE_PRUNE_SVD, 0.25, None),
]


@pytest.mark.parametrize("method,density,sign_method", ALL_METHODS)
class TestPerMethodDeterminism:
    def test_repeated_reconstruction_is_identical(
        self, tiny_snapshot, tiny_source, recipe_factory, method, density, sign_method
    ):
        recipe = recipe_factory(
            combination_type=method, density=density, sign_method=sign_method, seed=4242
        )
        digests = {
            reconstruct(recipe, tiny_snapshot, tiny_source).artifact_sha256 for _ in range(3)
        }
        assert len(digests) == 1

    def test_thread_count_does_not_change_the_artifact(
        self, tiny_snapshot, tiny_source, recipe_factory, method, density, sign_method
    ):
        # Multi-threaded accumulation reorders floating-point additions. If that
        # leaked through, two workers on differently-sized hosts would disagree.
        recipe = recipe_factory(
            combination_type=method, density=density, sign_method=sign_method, seed=99
        )
        digests = {
            reconstruct(recipe, tiny_snapshot, tiny_source, threads=threads).artifact_sha256
            for threads in (1, 2, 4)
        }
        assert len(digests) == 1

    def test_a_previous_merge_does_not_influence_the_next(
        self, tiny_snapshot, tiny_source, recipe_factory, method, density, sign_method
    ):
        recipe = recipe_factory(
            combination_type=method, density=density, sign_method=sign_method, seed=7
        )
        clean = reconstruct(recipe, tiny_snapshot, tiny_source).artifact_sha256

        # Consume global RNG state and merge something else first.
        torch.rand(1000)
        reconstruct(
            recipe_factory(
                combination_type=C.MERGE_DARE_TIES_SVD,
                density=0.2,
                sign_method="total",
                seed=1,
            ),
            tiny_snapshot,
            tiny_source,
        )
        torch.rand(1000)

        after = reconstruct(recipe, tiny_snapshot, tiny_source).artifact_sha256
        assert clean == after


class TestWrittenArtifactDeterminism:
    def test_the_written_file_matches_the_in_memory_digest(
        self, tiny_snapshot, tiny_source, recipe_factory, tmp_path
    ):
        # The cache consults the digest before deciding where to write, so a
        # divergence here would silently store an artifact under the wrong key.
        recipe = recipe_factory(combination_type=C.MERGE_TIES_SVD, density=0.5)

        dry = reconstruct(recipe, tiny_snapshot, tiny_source)
        written = reconstruct(recipe, tiny_snapshot, tiny_source, output_dir=tmp_path / "a")

        assert dry.artifact_sha256 == written.artifact_sha256

    def test_two_writes_produce_identical_bytes(
        self, tiny_snapshot, tiny_source, recipe_factory, tmp_path
    ):
        recipe = recipe_factory(combination_type=C.MERGE_DARE_TIES_SVD, density=0.4, seed=5)

        first = reconstruct(recipe, tiny_snapshot, tiny_source, output_dir=tmp_path / "one")
        second = reconstruct(recipe, tiny_snapshot, tiny_source, output_dir=tmp_path / "two")

        assert first.artifact_sha256 == second.artifact_sha256
        assert (tmp_path / "one" / "adapter_model.safetensors").read_bytes() == (
            tmp_path / "two" / "adapter_model.safetensors"
        ).read_bytes()

    def test_the_config_written_beside_the_weights_is_stable(
        self, tiny_snapshot, tiny_source, recipe_factory, tmp_path
    ):
        recipe = recipe_factory()
        reconstruct(recipe, tiny_snapshot, tiny_source, output_dir=tmp_path / "one")
        reconstruct(recipe, tiny_snapshot, tiny_source, output_dir=tmp_path / "two")

        assert (tmp_path / "one" / "adapter_config.json").read_text() == (
            tmp_path / "two" / "adapter_config.json"
        ).read_text()

    def test_tensor_key_order_does_not_change_the_bytes(self):
        # Serialisation sorts keys, so a dictionary built in a different order
        # must still produce the same file.
        generator = torch.Generator().manual_seed(21)
        tensors = {f"key_{index}": torch.randn(4, 4, generator=generator) for index in range(8)}
        shuffled = dict(reversed(list(tensors.items())))

        assert artifact_digest_of_tensors(tensors) == artifact_digest_of_tensors(shuffled)

    def test_a_non_contiguous_tensor_serialises_like_its_contiguous_twin(self):
        # A transposed view holds the same numbers in a different layout, and
        # safetensors records the layout.
        base = torch.randn(6, 4, generator=torch.Generator().manual_seed(22))
        view = base.t().t()

        assert artifact_digest_of_tensors({"w": base}) == artifact_digest_of_tensors({"w": view})


class TestInstanceDeterminism:
    def test_the_same_seed_gives_the_same_instance(self, workflow):
        first = workflow.generate_instance(987654, split="hidden")
        again = workflow.generate_instance(987654, split="hidden")

        assert first.sensor_log == again.sensor_log
        assert first.truth == again.truth
        assert first.database.rows == again.database.rows

    def test_splits_are_disjoint_for_the_same_seed(self, workflow):
        # Otherwise a miner could infer hidden instances from the public pack by
        # guessing seeds.
        public = workflow.generate_instance(555, split="public")
        hidden = workflow.generate_instance(555, split="hidden")

        assert public.sensor_log != hidden.sensor_log
        assert public.instance_id != hidden.instance_id

    def test_hidden_case_expectations_match_the_reference_implementation(self, workflow):
        from capability_subnet.workflows.industrial_maintenance_de_v1.diagnostics import (
            reference_analyze,
        )

        instance = workflow.generate_instance(31337, split="hidden")
        for case in instance.diagnostic.hidden_cases:
            assert case["expected"] == reference_analyze(case["readings"], case["threshold"])

    def test_the_truth_is_derived_from_the_values_the_candidate_can_see(self, workflow):
        # If the truth were computed from a higher-precision series than the log
        # shows, a correct candidate would be scored wrong.
        instance = workflow.generate_instance(24680, split="hidden")
        truth = instance.truth
        readings = list(instance.diagnostic.public_readings)

        assert max(readings) == pytest.approx(truth.peak_value)
        assert (
            sum(1 for value in readings if value > truth.warn_threshold) == truth.exceedance_count
        )


class TestTheMergeRunsOnEitherDevice:
    """Every method must survive the device the engine is configured for.

    These were CPU-only until a real GPU build failed on the first projection:
    the DARE family draws its drop mask on the CPU *by design* — CUDA generators
    differ across drivers and architectures, so a GPU-drawn mask would make an
    artifact depend on which card the worker was assigned — and the mask was
    then applied to a CUDA delta without being moved. Nothing in a CPU-only
    suite can see that.
    """

    @staticmethod
    def _devices():
        import torch

        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    @pytest.mark.parametrize(
        "method",
        [
            C.MERGE_LINEAR,
            C.MERGE_SVD,
            C.MERGE_TIES_SVD,
            C.MERGE_DARE_TIES_SVD,
            C.MERGE_DARE_LINEAR_SVD,
            C.MERGE_MAGNITUDE_PRUNE_SVD,
        ],
    )
    def test_every_method_reconstructs_on_every_available_device(
        self, method, tiny_snapshot, tiny_source, recipe_factory
    ):
        from capability_subnet.merge_engine.engine import reconstruct

        needs_density = method in C.DENSITY_METHODS
        recipe = recipe_factory(
            combination_type=method,
            density=0.5 if needs_density else None,
            sign_method="total" if method in (C.MERGE_TIES_SVD, C.MERGE_DARE_TIES_SVD) else None,
        )

        for device in self._devices():
            result = reconstruct(recipe, tiny_snapshot, tiny_source, device=device)
            assert result.artifact_sha256.startswith("sha256:")
            assert result.stats.merge_device == device

    def test_the_stochastic_mask_is_drawn_on_the_cpu_whatever_the_device(self):
        """The property the device fix must not break.

        A mask drawn on the GPU would be a different mask on a different card,
        and the artifact digest would stop being reproducible across a
        deployment's own workers.
        """
        import torch

        from capability_subnet.merge_engine.methods import random_drop_rescale

        cpu_delta = torch.ones(64, 64)
        on_cpu = random_drop_rescale(cpu_delta, 0.5, seed_parts=(1, "a", "site"))

        if torch.cuda.is_available():
            on_gpu = random_drop_rescale(cpu_delta.cuda(), 0.5, seed_parts=(1, "a", "site"))
            # Same mask, same survivors, same rescaling — only the device differs.
            assert torch.equal(on_cpu, on_gpu.cpu())


class TestImportDeterminism:
    """Materialising the pool is reconstruction's twin, and was not held to it.

    Every digest in the registry comes from this path, and every later
    reconstruction is checked against those digests. It re-factorises with the
    same float32 decomposition the merge engine uses, but ran outside the
    determinism controls entirely — so the artifact, and therefore the recorded
    ``artifact_sha256``, followed the importing host's core count. Two operators
    importing the same upstream adapter at the same pinned revision got different
    pools, each perfectly reproducible on its own machine.
    """

    @staticmethod
    def _upstream(manifest, rank):
        """Upstream-shaped LoRA factors for every canonical projection."""
        from capability_subnet.registry.importer import _source_key

        attention = set(manifest.raw["attention_modules"])
        template = manifest.layer_module_template
        generator = torch.Generator().manual_seed(11)

        source = {}
        for layer in range(manifest.num_hidden_layers):
            for module in C.CANONICAL_TARGET_MODULES:
                block = "self_attn" if module in attention else "mlp"
                shape = manifest.shape_of(module)
                source[_source_key(template, layer, block, module, "lora_A")] = torch.randn(
                    rank, shape.in_features, generator=generator
                )
                source[_source_key(template, layer, block, module, "lora_B")] = torch.randn(
                    shape.out_features, rank, generator=generator
                )
        return source

    def test_the_decomposition_runs_under_pinned_numerics(self, tiny_manifest, monkeypatch):
        """Asserted at the decomposition itself rather than by comparing digests.

        Comparing digests across thread counts is the property that matters, but
        it cannot be demonstrated on a 16-wide fixture: BLAS does not parallelise
        a matrix that small, so the divergence this guards against does not
        reproduce at test scale and such a test would pass whether or not the
        controls were in place. What *is* checkable here, and what fails the
        moment the context is removed again, is that the arithmetic runs
        single-threaded — which is the property the real divergence came from.
        """
        # By module, not `from ... import factorize`: the package exports a
        # function of that name, so the attribute is the function rather than the
        # module the importer resolves its symbol from.
        from importlib import import_module

        from capability_subnet.registry.importer import SourceSpec, normalise_tensors

        factorize = import_module("capability_subnet.merge_engine.factorize")
        observed: list[int] = []
        real = factorize.factorize_product

        def recording(*args, **kwargs):
            observed.append(torch.get_num_threads())
            return real(*args, **kwargs)

        monkeypatch.setattr(factorize, "factorize_product", recording)

        source = self._upstream(tiny_manifest, C.CANONICAL_RANK)
        spec = SourceSpec(
            adapter_id="thread-probe",
            repo="upstream/probe",
            revision="0" * 40,
            rank=C.CANONICAL_RANK,
            lora_alpha=C.CANONICAL_RANK,
            base_repo=tiny_manifest.model_repo,
        )

        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(max(2, previous))
            normalise_tensors(source, spec=spec, manifest=tiny_manifest)
        finally:
            torch.set_num_threads(previous)

        assert observed, "the decomposition never ran"
        assert set(observed) == {1}, (
            f"the decomposition ran with {sorted(set(observed))} threads; float32 "
            "reduction order would then follow the host's core count"
        )
        # And the caller's setting is put back, so importing does not silently
        # reconfigure the process that did it.
        assert torch.get_num_threads() == previous

    def test_repeated_import_is_identical(self, tiny_manifest):
        from capability_subnet.registry.importer import SourceSpec, normalise_tensors

        source = self._upstream(tiny_manifest, C.CANONICAL_RANK)
        spec = SourceSpec(
            adapter_id="repeat-probe",
            repo="upstream/probe",
            revision="0" * 40,
            rank=C.CANONICAL_RANK,
            lora_alpha=C.CANONICAL_RANK,
            base_repo=tiny_manifest.model_repo,
        )

        digests = set()
        for _ in range(3):
            # Consume global RNG state between runs; a seeded context must not
            # inherit whatever the process happened to be doing beforehand.
            torch.rand(1000)
            tensors, _ = normalise_tensors(source, spec=spec, manifest=tiny_manifest)
            digests.add(artifact_digest_of_tensors(tensors))

        assert len(digests) == 1
