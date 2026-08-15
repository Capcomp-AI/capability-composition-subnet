"""A published recipe must hash to the digest committed for it.

A miner publishes a file and commits its digest on chain. A validator fetches
that file and hashes what arrived — raw, without parsing it first — then
compares. So the bytes on disk are the content address rather than a rendering
of it, and any file whose bytes hash to something else points at a recipe that
can never be resolved. The miner finds out by spending their one evaluation on
it.
"""

from __future__ import annotations

from capability_subnet.common.hashing import sha256_bytes
from capability_subnet.miner.recipe import load_recipe, new_recipe, write_recipe


class TestWhatIsWrittenIsWhatIsCommitted:
    def _recipe(self, snapshot):
        return new_recipe(sorted(snapshot.registry.capability_adapters())[:3], snapshot=snapshot)

    def test_the_file_hashes_to_the_digest_it_reports(self, tiny_snapshot, tmp_path):
        recipe = self._recipe(tiny_snapshot)
        path = tmp_path / "r.json"
        reported = write_recipe(recipe, path)

        assert sha256_bytes(path.read_bytes()) == reported, (
            "the bytes on disk do not hash to the digest that goes on chain, so a "
            "validator fetching this file would refuse it"
        )

    def test_it_matches_the_recipe_s_own_content_address(self, tiny_snapshot, tmp_path):
        recipe = self._recipe(tiny_snapshot)
        path = tmp_path / "r.json"
        write_recipe(recipe, path)
        assert path.read_bytes() == recipe.canonical_bytes()
        assert sha256_bytes(path.read_bytes()) == recipe.digest()

    def test_a_round_trip_through_disk_is_stable(self, tiny_snapshot, tmp_path):
        """Canonicalising an already-canonical file changes nothing — otherwise
        running the tool twice would invalidate a commitment made after the
        first run."""
        recipe = self._recipe(tiny_snapshot)
        first = tmp_path / "a.json"
        write_recipe(recipe, first)

        second = tmp_path / "b.json"
        write_recipe(load_recipe(str(first)), second)

        assert first.read_bytes() == second.read_bytes()

    def test_the_digest_survives_being_reloaded(self, tiny_snapshot, tmp_path):
        recipe = self._recipe(tiny_snapshot)
        path = tmp_path / "r.json"
        digest = write_recipe(recipe, path)
        assert load_recipe(str(path)).digest() == digest
