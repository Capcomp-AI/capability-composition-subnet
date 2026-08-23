"""A merge already built is not built again.

Reconstruction is the one step that costs minutes before a card can serve
anything. A caller that builds ahead of the fleet, or one restarted part-way
through a run, would otherwise pay it twice — and on a four-card host that is
the difference between every card serving and every card waiting.

Reuse is keyed on the recipe, not on the directory being non-empty: serving one
miner's merge as another's would score the wrong package, which is the failure
this must not trade for speed.
"""

from __future__ import annotations

import json
from pathlib import Path

from capability_subnet.miner.local_eval import (
    ARTIFACT_RECORD,
    _existing_artifact,
    _record_artifact,
)

RECIPE = "sha256:" + "a" * 64
ARTIFACT = "sha256:" + "b" * 64


def _built(tmp_path: Path, recipe: str = RECIPE) -> Path:
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    _record_artifact(tmp_path, recipe, ARTIFACT, 7)
    return tmp_path


def test_a_merge_built_for_this_recipe_is_reused(tmp_path):
    assert _existing_artifact(_built(tmp_path), RECIPE) == (ARTIFACT, 7)


def test_a_merge_built_for_another_recipe_is_not(tmp_path):
    """The one that matters: this is a different package."""
    assert _existing_artifact(_built(tmp_path), "sha256:" + "c" * 64) is None


def test_a_directory_with_no_record_is_not_reused(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")

    assert _existing_artifact(tmp_path, RECIPE) is None


def test_a_record_with_no_weights_is_not_reused(tmp_path):
    """A half-written build must not be served as a finished one."""
    _record_artifact(tmp_path, RECIPE, ARTIFACT, 7)

    assert _existing_artifact(tmp_path, RECIPE) is None


def test_empty_weights_are_not_reused(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    _record_artifact(tmp_path, RECIPE, ARTIFACT, 7)

    assert _existing_artifact(tmp_path, RECIPE) is None


def test_an_unreadable_record_rebuilds_rather_than_raising(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / ARTIFACT_RECORD).write_text("{ not json")

    assert _existing_artifact(tmp_path, RECIPE) is None


def test_a_record_missing_its_digest_rebuilds(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / ARTIFACT_RECORD).write_text(json.dumps({"recipe_sha256": RECIPE}))

    assert _existing_artifact(tmp_path, RECIPE) is None


def test_an_empty_directory_is_not_reused(tmp_path):
    assert _existing_artifact(tmp_path, RECIPE) is None
