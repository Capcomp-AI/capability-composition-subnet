"""One recipe, one digest, whatever the file on disk looks like.

The bytes a miner seals are the bytes the engine identifies the recipe by, so
``capcomp digest`` and the commitment agree on one number. Any other compact
form would give a miner two numbers that differ for reasons nobody should have
to learn.
"""

from __future__ import annotations

import json

from capability_subnet.common.schemas import Recipe
from capability_subnet.miner import submit


def recipe_for(snapshot) -> Recipe:
    ids = list(snapshot.registry.selectable_ids)
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "base_revision": snapshot.manifest.revision,
            "source_snapshot_sha256": snapshot.sha256,
            "selected_adapters": ids[:2],
            "merge": {"combination_type": "linear"},
            "compression": {"output_rank": 64},
        }
    )


class TestOneDigestNotTwo:
    def test_the_body_is_the_protocol_s_canonical_form(self, tiny_snapshot):
        recipe = recipe_for(tiny_snapshot)
        body = submit.canonical_body(recipe)

        assert body == recipe.canonical_bytes()
        assert submit.digest_of(body) == recipe.digest()

    def test_formatting_on_disk_cannot_change_what_is_sealed(self, tiny_snapshot):
        """A stray byte in the miner's editor must not move the digest."""
        recipe = recipe_for(tiny_snapshot)
        spaced = Recipe.model_validate_json(json.dumps(recipe.model_dump(mode="json"), indent=4))

        assert submit.canonical_body(spaced) == submit.canonical_body(recipe)


class TestItStaysTwoPureFunctions:
    def test_nothing_here_reaches_for_the_network(self):
        """A transport here would be a second way into a run beside the chain,
        and the point of the chain path is that there is only one."""
        source = __import__("inspect").getsource(submit)

        assert "httpx" not in source
        assert "api_url" not in source
        assert "http://" not in source and "https://" not in source
        assert not hasattr(submit, "send")
        assert not hasattr(submit, "signing_message")
