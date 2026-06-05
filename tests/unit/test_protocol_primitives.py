"""Hashing, commitments and the recipe contract.

These three carry the protocol's integrity guarantees. Hashing decides whether
two things are the same; commitments decide what a miner actually claimed; the
recipe schema decides what can enter the merge engine at all.
"""

from __future__ import annotations

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.commitments import (
    MAX_PAYLOAD_BYTES,
    CommitmentError,
    decode_commitment,
    encode_commitment,
    is_subnet_commitment,
    validate_recipe_uri,
)
from capability_subnet.common.hashing import (
    canonical_json_bytes,
    digests_equal,
    normalise_digest,
    sha256_bytes,
    sha256_json,
)
from capability_subnet.common.schemas import CompressionSpec, MergeSpec, Recipe


class TestCanonicalHashing:
    def test_key_order_does_not_change_the_digest(self):
        assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})

    def test_whitespace_does_not_change_the_digest(self):
        import json

        document = {"x": [1, 2, {"y": "z"}]}
        pretty = json.loads(json.dumps(document, indent=4))
        assert sha256_json(document) == sha256_json(pretty)

    def test_non_ascii_is_preserved_rather_than_escaped(self):
        raw = canonical_json_bytes({"text": "Öltemperatur"})
        assert "Öltemperatur".encode() in raw

    def test_nan_is_refused(self):
        # A NaN would serialise to a token no other JSON parser accepts, so two
        # implementations would disagree about the bytes being hashed.
        with pytest.raises(ValueError):
            canonical_json_bytes({"value": float("nan")})

    def test_digest_comparison_ignores_the_prefix(self):
        digest = sha256_bytes(b"payload")
        assert digests_equal(digest, digest.removeprefix("sha256:"))

    def test_malformed_digests_compare_false_rather_than_raising(self):
        assert digests_equal("not-a-digest", "also-not") is False

    def test_normalise_rejects_a_wrong_length(self):
        with pytest.raises(ValueError):
            normalise_digest("sha256:abcd")


class TestCommitments:
    URI = "hf:alice/capsub-recipes/final.json"

    def test_round_trip(self):
        digest = sha256_bytes(b"recipe")
        payload = encode_commitment(C.DEFAULT_WORKFLOW_ID, digest, self.URI)
        decoded = decode_commitment(payload)

        assert decoded.workflow_id == C.DEFAULT_WORKFLOW_ID
        assert digests_equal(decoded.recipe_sha256, digest)
        assert decoded.recipe_uri == self.URI

    def test_payload_fits_the_chain_field(self):
        payload = encode_commitment(C.DEFAULT_WORKFLOW_ID, sha256_bytes(b"x"), self.URI)
        assert len(payload.encode("utf-8")) <= MAX_PAYLOAD_BYTES

    def test_an_oversized_pointer_is_refused_with_a_usable_message(self):
        long_uri = "https://example.com/" + "a" * 200
        with pytest.raises(CommitmentError, match="over the"):
            encode_commitment(C.DEFAULT_WORKFLOW_ID, sha256_bytes(b"x"), long_uri)

    def test_foreign_payloads_are_not_mistaken_for_submissions(self):
        assert is_subnet_commitment("some other subnet's payload") is False
        with pytest.raises(CommitmentError, match="not a capsub1"):
            decode_commitment("other|v1|abc|def")

    def test_unknown_workflow_code_is_refused(self):
        with pytest.raises(CommitmentError, match="unknown workflow code"):
            decode_commitment("capsub1|zzzz|" + "A" * 43 + "|https://x.test/r")

    @pytest.mark.parametrize(
        "uri",
        [
            "hf:owner/repo/path/to/recipe.json",
            "ipfs:bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
            "https://example.com/recipes/final.json",
        ],
    )
    def test_accepted_pointer_forms(self, uri):
        assert validate_recipe_uri(uri) == uri

    @pytest.mark.parametrize(
        "uri",
        [
            "",
            "ftp://example.com/recipe.json",
            "file:///etc/passwd",
            "hf:owner/repo",  # missing the path
            "https://example.com/a|b",  # separator would corrupt the payload
        ],
    )
    def test_rejected_pointer_forms(self, uri):
        with pytest.raises(CommitmentError):
            validate_recipe_uri(uri)


class TestRecipeContract:
    def _minimal(self, **overrides):
        payload = {
            "base_revision": "rev-1",
            "source_snapshot_sha256": sha256_bytes(b"snapshot"),
            "selected_adapters": ["a-one", "b-two"],
            "merge": MergeSpec(combination_type=C.MERGE_LINEAR),
            "compression": CompressionSpec(output_rank=64),
        }
        payload.update(overrides)
        return Recipe(**payload)

    def test_a_valid_recipe_round_trips_through_its_canonical_form(self):
        recipe = self._minimal()
        rebuilt = Recipe.model_validate_json(recipe.canonical_bytes())
        assert rebuilt.digest() == recipe.digest()

    def test_adapter_order_does_not_change_the_load_order(self):
        forward = self._minimal(selected_adapters=["a-one", "b-two", "c-three"])
        reverse = self._minimal(selected_adapters=["c-three", "b-two", "a-one"])
        assert forward.sorted_adapters() == reverse.sorted_adapters()

    def test_unknown_fields_are_rejected(self):
        # An unknown field in a miner recipe usually means something is being
        # smuggled past the merge engine.
        with pytest.raises(Exception, match="[Ee]xtra"):
            Recipe.model_validate(
                {
                    "base_revision": "rev-1",
                    "source_snapshot_sha256": sha256_bytes(b"s"),
                    "selected_adapters": ["a", "b"],
                    "merge": {"combination_type": "linear"},
                    "compression": {"output_rank": 64},
                    "run_this": "rm -rf /",
                }
            )

    def test_a_single_adapter_is_not_a_candidate(self):
        with pytest.raises(Exception, match="at least"):
            self._minimal(selected_adapters=["only-one"])

    def test_duplicate_adapters_are_rejected(self):
        with pytest.raises(Exception, match="[Dd]uplicate"):
            self._minimal(selected_adapters=["a-one", "a-one", "b-two"])

    @pytest.mark.parametrize("weight", [-2.5, 2.5, 100.0])
    def test_out_of_range_coefficients_are_rejected(self, weight):
        with pytest.raises(Exception, match="outside"):
            self._minimal(global_weights={"a-one": weight})

    def test_coefficients_must_reference_selected_adapters(self):
        with pytest.raises(Exception, match="unselected"):
            self._minimal(global_weights={"not-selected": 1.0})

    def test_unknown_layer_groups_are_rejected(self):
        with pytest.raises(Exception, match="unknown layer group"):
            self._minimal(layer_group_overrides={"group_99": {"a-one": 1.0}})

    def test_density_is_required_where_it_applies_and_refused_where_it_does_not(self):
        with pytest.raises(Exception, match="requires 'density'"):
            self._minimal(merge=MergeSpec(combination_type=C.MERGE_TIES_SVD, majority_sign_method="total"))

        with pytest.raises(Exception, match="does not accept 'density'"):
            self._minimal(merge=MergeSpec(combination_type=C.MERGE_LINEAR, density=0.5))

    def test_sign_election_is_required_only_for_the_ties_family(self):
        with pytest.raises(Exception, match="requires 'majority_sign_method'"):
            self._minimal(merge=MergeSpec(combination_type=C.MERGE_TIES_SVD, density=0.5))

        with pytest.raises(Exception, match="does not accept 'majority_sign_method'"):
            self._minimal(
                merge=MergeSpec(
                    combination_type=C.MERGE_DARE_LINEAR_SVD,
                    density=0.5,
                    majority_sign_method="total",
                )
            )

    def test_only_allowed_output_ranks(self):
        with pytest.raises(Exception, match="not allowed"):
            self._minimal(compression=CompressionSpec(output_rank=77))

    def test_unknown_merge_method_is_rejected(self):
        with pytest.raises(Exception, match="unknown combination_type"):
            self._minimal(merge=MergeSpec(combination_type="my_custom_merge"))

    def test_effective_weight_resolution_order(self):
        recipe = self._minimal(
            selected_adapters=["a-one", "b-two"],
            global_weights={"a-one": 1.3},
            layer_group_overrides={"group_2": {"a-one": 0.7}},
        )
        assert recipe.effective_weight("a-one", "group_2") == pytest.approx(0.7)
        assert recipe.effective_weight("a-one", "group_0") == pytest.approx(1.3)
        assert recipe.effective_weight("b-two", "group_0") == pytest.approx(1.0)


class TestLayerGroups:
    def test_groups_partition_the_model_exactly(self):
        groups = C.build_layer_groups(C.NUM_HIDDEN_LAYERS)
        covered = [
            layer for _, (low, high) in groups.items() for layer in range(low, high + 1)
        ]
        assert sorted(covered) == list(range(C.NUM_HIDDEN_LAYERS))

    def test_group_sizes_differ_by_at_most_one(self):
        groups = C.build_layer_groups(C.NUM_HIDDEN_LAYERS)
        sizes = [high - low + 1 for low, high in groups.values()]
        assert max(sizes) - min(sizes) <= 1

    @pytest.mark.parametrize("depth", [4, 28, 32, 36, 40, 64])
    def test_partitioning_holds_at_every_plausible_depth(self, depth):
        groups = C.build_layer_groups(depth)
        covered = [
            layer for _, (low, high) in groups.items() for layer in range(low, high + 1)
        ]
        assert sorted(covered) == list(range(depth))
        assert len(groups) == C.NUM_LAYER_GROUPS

    def test_a_model_too_shallow_to_split_is_refused(self):
        with pytest.raises(ValueError, match="cannot be split"):
            C.build_layer_groups(2)

    def test_layer_lookup_rejects_an_out_of_range_index(self):
        with pytest.raises(ValueError, match="outside"):
            C.layer_group_of(C.NUM_HIDDEN_LAYERS + 5)
