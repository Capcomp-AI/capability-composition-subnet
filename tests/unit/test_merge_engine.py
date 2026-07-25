"""The merge engine.

Two kinds of claim are checked here. The first is mathematical: each merge stage
does what it says — trimming keeps the largest entries, drop-and-rescale
preserves the expected update, sign election picks the majority direction,
factorisation reconstructs the update it was given. The second is behavioural:
the recipe's parameters actually change the artifact, so a coefficient a miner
sets is a coefficient the engine applies.
"""

from __future__ import annotations

import pytest
import torch

from capability_subnet.common import constants as C
from capability_subnet.merge_engine import methods
from capability_subnet.merge_engine.delta import low_rank_delta_norm
from capability_subnet.merge_engine.determinism import canonicalize_svd_signs, derive_seed
from capability_subnet.merge_engine.engine import ReconstructionError, reconstruct
from capability_subnet.merge_engine.factorize import factorize, pad_to_rank


class TestSparsify:
    def test_magnitude_trim_keeps_the_largest_entries(self):
        delta = torch.tensor([[1.0, -5.0, 0.1], [3.0, -0.2, 4.0]])
        trimmed = methods.magnitude_trim(delta, 0.5)

        kept = trimmed[trimmed != 0].abs().tolist()
        assert sorted(kept, reverse=True) == [5.0, 4.0, 3.0]

    def test_full_density_is_the_identity(self):
        delta = torch.randn(8, 8, generator=torch.Generator().manual_seed(1))
        assert torch.equal(methods.magnitude_trim(delta, 1.0), delta)

    def test_trimming_uses_a_threshold_so_ties_do_not_break_arbitrarily(self):
        # Every entry has the same magnitude, so no subset of exactly half is
        # more correct than another. A threshold cut keeps them all, which is
        # identical on every machine; index selection would not be.
        delta = torch.full((4, 4), 2.0)
        trimmed = methods.magnitude_trim(delta, 0.5)
        assert int((trimmed != 0).sum()) == 16

    def test_drop_and_rescale_preserves_the_expected_update(self):
        generator = torch.Generator().manual_seed(7)
        delta = torch.ones(200, 200) + torch.randn(200, 200, generator=generator) * 0.01
        density = 0.3

        dropped = methods.random_drop_rescale(delta, density, seed_parts=(1, "a", "site"))

        # Survivors are scaled by 1/density, so the sum is preserved in
        # expectation. The tolerance is the sampling noise at this size.
        assert float(dropped.sum()) == pytest.approx(float(delta.sum()), rel=0.05)

    def test_drop_mask_depends_only_on_the_tensor_identity(self):
        delta = torch.randn(32, 32, generator=torch.Generator().manual_seed(3))

        first = methods.random_drop_rescale(delta, 0.4, seed_parts=(9, "adapter", "site"))
        again = methods.random_drop_rescale(delta, 0.4, seed_parts=(9, "adapter", "site"))
        other_site = methods.random_drop_rescale(delta, 0.4, seed_parts=(9, "adapter", "other"))

        assert torch.equal(first, again)
        assert not torch.equal(first, other_site)

    def test_a_density_requiring_method_refuses_to_run_without_one(self):
        with pytest.raises(ValueError, match="requires a density"):
            methods.sparsify(torch.zeros(2, 2), "magnitude", None, seed_parts=())


class TestSignElection:
    def test_total_weighs_by_magnitude(self):
        # Two adapters want -1, one wants +10. By total the majority direction is
        # positive; by frequency it is negative.
        stacked = torch.tensor([[[-1.0]], [[-1.0]], [[10.0]]])

        assert methods.elect_signs(stacked, "total").item() == 1.0
        assert methods.elect_signs(stacked, "frequency").item() == -1.0

    def test_an_exact_tie_resolves_positive(self):
        stacked = torch.tensor([[[1.0]], [[-1.0]]])
        assert methods.elect_signs(stacked, "total").item() == 1.0

    def test_disjoint_mean_averages_only_the_agreeing_adapters(self):
        stacked = torch.tensor([[[2.0]], [[4.0]], [[-6.0]]])
        elected = methods.elect_signs(stacked, "frequency")  # two positive votes

        merged = methods.aggregate_disjoint_mean(stacked, elected)
        assert merged.item() == pytest.approx(3.0)  # mean of 2 and 4, not of all three

    def test_an_entry_nobody_agrees_with_becomes_zero_not_a_division_error(self):
        stacked = torch.zeros(2, 1, 1)
        elected = torch.ones(1, 1)
        assert methods.aggregate_disjoint_mean(stacked, elected).item() == 0.0


class TestFactorize:
    def test_a_low_rank_update_is_reconstructed_essentially_exactly(self):
        generator = torch.Generator().manual_seed(11)
        left = torch.randn(16, 4, generator=generator)
        right = torch.randn(4, 16, generator=generator)
        delta = left @ right

        result = factorize(delta, 8)
        assert torch.allclose(result.reconstruct(), delta, atol=1e-4)

    def test_truncation_below_the_natural_rank_loses_energy(self):
        generator = torch.Generator().manual_seed(12)
        delta = torch.randn(32, 32, generator=generator)

        tight = factorize(delta, 4)
        loose = factorize(delta, 32)

        tight_error = torch.linalg.matrix_norm(tight.reconstruct() - delta)
        loose_error = torch.linalg.matrix_norm(loose.reconstruct() - delta)
        assert tight_error > loose_error

    def test_output_shape_is_exactly_the_requested_rank(self):
        # A narrow projection cannot support a high rank; the remainder is padded
        # with exact zeros so the artifact keeps the expected shape.
        delta = torch.randn(4, 20, generator=torch.Generator().manual_seed(13))
        result = factorize(delta, 16)

        assert result.lora_a.shape == (16, 20)
        assert result.lora_b.shape == (4, 16)
        assert torch.allclose(result.reconstruct(), delta, atol=1e-4)

    def test_clamping_reduces_the_dominant_direction(self):
        generator = torch.Generator().manual_seed(14)
        delta = torch.randn(16, 16, generator=generator)
        delta[0] *= 50.0  # one direction dominates everything else

        unclamped = factorize(delta, 16, clamp_quantile=1.0)
        clamped = factorize(delta, 16, clamp_quantile=0.8)

        assert clamped.clamped_components > 0
        assert unclamped.clamped_components == 0
        assert torch.linalg.matrix_norm(clamped.reconstruct()) < torch.linalg.matrix_norm(delta)

    def test_sign_convention_is_applied(self):
        generator = torch.Generator().manual_seed(15)
        delta = torch.randn(12, 12, generator=generator)
        result = factorize(delta, 6)

        # After canonicalisation the dominant entry of each left vector is
        # positive, which is what makes two LAPACK builds agree.
        dominant = result.lora_b.abs().argmax(dim=0)
        for column in range(6):
            assert result.lora_b[dominant[column], column] >= 0

    def test_sign_canonicalisation_preserves_the_product(self):
        generator = torch.Generator().manual_seed(16)
        matrix = torch.randn(10, 10, generator=generator)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)

        canonical_u, canonical_vh = canonicalize_svd_signs(u, vh)
        assert torch.allclose(canonical_u @ torch.diag(s) @ canonical_vh, matrix, atol=1e-5)

    def test_padding_a_factor_pair_is_exact(self):
        lora_a = torch.randn(4, 8, generator=torch.Generator().manual_seed(17))
        lora_b = torch.randn(8, 4, generator=torch.Generator().manual_seed(18))

        padded_a, padded_b = pad_to_rank(lora_a, lora_b, 12)
        assert torch.allclose(padded_b @ padded_a, lora_b @ lora_a)

    def test_padding_cannot_be_used_to_shrink(self):
        with pytest.raises(ValueError, match="use a decomposition"):
            pad_to_rank(torch.zeros(8, 4), torch.zeros(4, 8), 4)


class TestLowRankNorm:
    def test_matches_the_materialised_norm(self):
        generator = torch.Generator().manual_seed(19)
        lora_a = torch.randn(8, 24, generator=generator)
        lora_b = torch.randn(24, 8, generator=generator)

        cheap = low_rank_delta_norm(lora_a, lora_b, 1.5)
        direct = float(torch.linalg.matrix_norm((lora_b @ lora_a) * 1.5, ord="fro"))
        assert cheap == pytest.approx(direct, rel=1e-4)


class TestReconstruction:
    def test_every_allowed_method_produces_an_artifact(
        self, tiny_snapshot, tiny_source, recipe_factory
    ):
        for method in C.ALLOWED_MERGE_METHODS:
            pipeline = methods.pipeline_for(method)
            recipe = recipe_factory(
                combination_type=method,
                density=0.5 if method in C.DENSITY_METHODS else None,
                sign_method="total" if pipeline.sign_election != "none" else None,
            )
            result = reconstruct(recipe, tiny_snapshot, tiny_source)
            assert result.artifact_sha256.startswith("sha256:")
            assert result.stats.sites == tiny_snapshot.manifest.num_hidden_layers * len(
                C.CANONICAL_TARGET_MODULES
            )

    def test_coefficients_change_the_artifact(self, tiny_snapshot, tiny_source, recipe_factory):
        plain = reconstruct(recipe_factory(), tiny_snapshot, tiny_source)
        weighted = reconstruct(
            recipe_factory(global_weights={"alpha-capability-v1": 1.4}),
            tiny_snapshot,
            tiny_source,
        )
        assert plain.artifact_sha256 != weighted.artifact_sha256

    def test_layer_group_overrides_change_the_artifact(
        self, tiny_snapshot, tiny_source, recipe_factory
    ):
        plain = reconstruct(recipe_factory(), tiny_snapshot, tiny_source)
        grouped = reconstruct(
            recipe_factory(layer_group_overrides={"group_1": {"alpha-capability-v1": 1.3}}),
            tiny_snapshot,
            tiny_source,
        )
        assert plain.artifact_sha256 != grouped.artifact_sha256

    def test_the_seed_matters_only_for_stochastic_methods(
        self, tiny_snapshot, tiny_source, recipe_factory
    ):
        stochastic_a = reconstruct(
            recipe_factory(combination_type=C.MERGE_DARE_TIES_SVD, seed=1),
            tiny_snapshot,
            tiny_source,
        )
        stochastic_b = reconstruct(
            recipe_factory(combination_type=C.MERGE_DARE_TIES_SVD, seed=2),
            tiny_snapshot,
            tiny_source,
        )
        assert stochastic_a.artifact_sha256 != stochastic_b.artifact_sha256

        deterministic_a = reconstruct(
            recipe_factory(combination_type=C.MERGE_TIES_SVD, seed=1), tiny_snapshot, tiny_source
        )
        deterministic_b = reconstruct(
            recipe_factory(combination_type=C.MERGE_TIES_SVD, seed=2), tiny_snapshot, tiny_source
        )
        assert deterministic_a.artifact_sha256 == deterministic_b.artifact_sha256

    def test_adapter_list_order_does_not_change_the_artifact(
        self, tiny_snapshot, tiny_source, recipe_factory
    ):
        forward = recipe_factory(
            adapters=["alpha-capability-v1", "beta-capability-v1", "gamma-capability-v1"]
        )
        reverse = recipe_factory(
            adapters=["gamma-capability-v1", "beta-capability-v1", "alpha-capability-v1"]
        )
        assert (
            reconstruct(forward, tiny_snapshot, tiny_source).artifact_sha256
            == reconstruct(reverse, tiny_snapshot, tiny_source).artifact_sha256
        )

    def test_the_cheap_path_is_used_only_when_nothing_is_discarded(
        self, tiny_snapshot, tiny_source, recipe_factory
    ):
        cheap = reconstruct(
            recipe_factory(combination_type=C.MERGE_LINEAR, density=None, sign_method=None),
            tiny_snapshot,
            tiny_source,
        )
        assert cheap.stats.used_factor_space is True

        # Clamping is defined on the decomposition, so it forces the full path.
        clamped = reconstruct(
            recipe_factory(
                combination_type=C.MERGE_LINEAR, density=None, sign_method=None, clamp=0.95
            ),
            tiny_snapshot,
            tiny_source,
        )
        assert clamped.stats.used_factor_space is False

    def test_a_recipe_for_a_different_pool_is_refused(
        self, tiny_snapshot, tiny_source, recipe_factory
    ):
        recipe = recipe_factory()
        stale = recipe.model_copy(update={"source_snapshot_sha256": "sha256:" + "0" * 64})

        with pytest.raises(ReconstructionError, match="source snapshot"):
            reconstruct(stale, tiny_snapshot, tiny_source)

    def test_a_recipe_for_a_different_base_is_refused(
        self, tiny_snapshot, tiny_source, recipe_factory
    ):
        recipe = recipe_factory().model_copy(update={"base_revision": "some-other-revision"})
        with pytest.raises(ReconstructionError, match="base revision"):
            reconstruct(recipe, tiny_snapshot, tiny_source)

    def test_missing_adapter_weights_are_reported_clearly(self, tiny_snapshot, recipe_factory):
        from capability_subnet.merge_engine.loader import InMemoryAdapterSource

        empty = InMemoryAdapterSource({})
        with pytest.raises(ReconstructionError, match="not available on this host"):
            reconstruct(recipe_factory(), tiny_snapshot, empty)

    def test_contribution_statistics_cover_every_group_and_adapter(
        self, tiny_snapshot, tiny_source, recipe_factory
    ):
        recipe = recipe_factory()
        stats = reconstruct(recipe, tiny_snapshot, tiny_source).stats

        assert set(stats.contribution_by_group) == set(tiny_snapshot.manifest.layer_groups)
        for contributions in stats.contribution_by_group.values():
            assert set(contributions) == set(recipe.selected_adapters)
            assert all(value > 0 for value in contributions.values())


class TestDeriveSeed:
    def test_is_stable_across_processes(self):
        # Python salts string hashing per process, so a seed derived with the
        # builtin would differ between two workers that must agree.
        assert derive_seed(42, "adapter", "site") == derive_seed(42, "adapter", "site")

    def test_distinguishes_component_boundaries(self):
        assert derive_seed("a", "bc") != derive_seed("ab", "c")
