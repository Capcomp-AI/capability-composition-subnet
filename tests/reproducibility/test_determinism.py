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
from capability_subnet.merge_engine.engine import artifact_hashes_agree, reconstruct

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


class TestWorkerAgreement:
    def test_agreement_is_reported_when_digests_match(
        self, tiny_snapshot, tiny_source, recipe_factory
    ):
        recipe = recipe_factory()
        results = [reconstruct(recipe, tiny_snapshot, tiny_source) for _ in range(3)]

        agreed, detail = artifact_hashes_agree(results)
        assert agreed
        assert "agree" in detail

    def test_disagreement_is_reported_rather_than_silently_resolved(
        self, tiny_snapshot, tiny_source, recipe_factory
    ):
        # Scoring one of two disagreeing artifacts would mean paying for a result
        # nobody can reproduce.
        first = reconstruct(
            recipe_factory(seed=1, combination_type=C.MERGE_DARE_TIES_SVD, density=0.3),
            tiny_snapshot,
            tiny_source,
        )
        second = reconstruct(
            recipe_factory(seed=2, combination_type=C.MERGE_DARE_TIES_SVD, density=0.3),
            tiny_snapshot,
            tiny_source,
        )

        agreed, detail = artifact_hashes_agree([first, second])
        assert not agreed
        assert "distinct artifacts" in detail

    def test_no_reconstructions_is_not_agreement(self):
        agreed, _ = artifact_hashes_agree([])
        assert not agreed

    def test_the_reconstructor_cross_checks_before_caching(
        self, tiny_snapshot, tiny_pool_dir, recipe_factory, tmp_path
    ):
        from capability_subnet.backend.executor.reconstruction import ArtifactCache, Reconstructor
        from capability_subnet.merge_engine.loader import SafetensorsAdapterSource

        cache = ArtifactCache(tmp_path / "cache")
        reconstructor = Reconstructor(
            tiny_snapshot, SafetensorsAdapterSource(tiny_pool_dir), cache, workers=3
        )

        outcome = reconstructor.build(recipe_factory())
        assert outcome.workers_agreed
        assert not outcome.from_cache
        assert cache.contains(outcome.artifact_sha256)

        # The second build is served from the cache and never rebuilt.
        again = reconstructor.build(recipe_factory())
        assert again.from_cache
        assert again.artifact_sha256 == outcome.artifact_sha256


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
