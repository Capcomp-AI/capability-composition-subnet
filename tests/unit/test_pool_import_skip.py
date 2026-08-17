"""Re-importing a pool that is already correct.

Materialising the pool fetches two files per adapter and re-runs a decomposition
over 252 projections. Doing that for an adapter whose bytes are already on disk
and already match what the registry recorded buys nothing and costs minutes per
adapter, so a re-run is the difference between seconds and most of an hour.

What must not happen is skipping on the strength of the file merely existing: a
truncated or half-written artifact is exactly the case where skipping is wrong,
and it looks identical to a good one by presence alone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from capability_subnet.common.hashing import sha256_file

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "import_public_adapters.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("import_public_adapters", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def materialised(tmp_path):
    """One adapter on disk, and the digest it actually hashes to."""
    adapter = tmp_path / "an-adapter"
    adapter.mkdir()
    artifact = adapter / "adapter_model.safetensors"
    artifact.write_bytes(b"not really safetensors, but it hashes the same way")
    return tmp_path, "an-adapter", sha256_file(artifact)


class TestAVerifiedAdapterIsNotReImported:
    def test_a_matching_digest_is_skipped(self, script, materialised):
        out_dir, adapter_id, digest = materialised
        verified, detail = script._already_verified(out_dir, adapter_id, digest)
        assert verified is True
        assert "already materialised" in detail

    def test_a_mismatching_digest_is_re_imported_and_says_both(self, script, materialised):
        """The recorded digest changing is how a normalisation fix reaches the
        pool, so this path must re-import rather than trust what is there."""
        out_dir, adapter_id, digest = materialised
        verified, detail = script._already_verified(out_dir, adapter_id, "sha256:" + "0" * 64)
        assert verified is False
        # Both digests, because "mismatch" alone does not say which is stale.
        assert digest[:19] in detail
        assert "000000" in detail

    def test_presence_alone_is_not_verification(self, script, materialised):
        """A half-written artifact is present and wrong."""
        out_dir, adapter_id, digest = materialised
        (out_dir / adapter_id / "adapter_model.safetensors").write_bytes(b"truncated")
        verified, _ = script._already_verified(out_dir, adapter_id, digest)
        assert verified is False

    def test_an_adapter_with_no_recorded_digest_is_imported(self, script, materialised):
        """The state a first import starts from."""
        out_dir, adapter_id, _ = materialised
        assert script._already_verified(out_dir, adapter_id, "")[0] is False
        assert script._already_verified(out_dir, adapter_id, None)[0] is False

    def test_a_missing_adapter_is_imported_without_comment(self, script, materialised):
        out_dir, _, digest = materialised
        verified, detail = script._already_verified(out_dir, "absent", digest)
        assert verified is False
        assert detail == ""


class TestTheSelectionBoundsAreUsable:
    """The bounds have to leave a miner able to build something.

    They are set from reconstruction cost rather than from the pool's size, so
    the ceiling now sits well below the number of certified adapters. That is
    deliberate — but it means `capability-miner init`, which selects for a miner
    who has not named anything, has to choose a selection inside them. Taking the
    whole pool produced a recipe the schema rejected, so a miner's first command
    failed for what looked like the pool's fault.
    """

    def test_the_bounds_admit_a_selection_at_all(self):
        from capability_subnet.common import constants as C

        assert 0 < C.MIN_SELECTED_ADAPTERS <= C.MAX_SELECTED_ADAPTERS

    def test_the_pool_can_supply_a_full_selection(self):
        from capability_subnet.common import constants as C
        from capability_subnet.registry.snapshot import load_snapshot

        pool = len(load_snapshot().registry.capability_adapters())
        assert pool >= C.MAX_SELECTED_ADAPTERS, (
            f"the pool offers {pool} capability adapters but a recipe may select "
            f"{C.MAX_SELECTED_ADAPTERS}, so the ceiling can never be reached"
        )

    def test_the_starting_selection_is_admissible(self):
        """What `init` writes when the miner names nothing."""
        from capability_subnet.common import constants as C
        from capability_subnet.miner.cli import _starting_selection
        from capability_subnet.miner.recipe import check_recipe, new_recipe
        from capability_subnet.registry.snapshot import load_snapshot

        snapshot = load_snapshot()
        picks = _starting_selection(snapshot)
        assert C.MIN_SELECTED_ADAPTERS <= len(picks) <= C.MAX_SELECTED_ADAPTERS
        assert not check_recipe(new_recipe(picks, snapshot=snapshot))

    def test_the_reference_is_built_within_the_same_bounds(self):
        """The bar a miner clears is composed of no more than a miner may use."""
        from capability_subnet.common import constants as C
        from capability_subnet.registry.snapshot import load_snapshot

        adapters = sorted(load_snapshot().registry.capability_adapters())
        assert len(adapters[: C.MAX_SELECTED_ADAPTERS]) == C.MAX_SELECTED_ADAPTERS
