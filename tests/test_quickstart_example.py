"""The example miner is the first thing a miner runs. It has to work.

It is not covered by the package's own tests — it lives outside the package and
calls the miner-facing API from the outside, exactly as a miner would. That is
the point: if a rename breaks it, the failure belongs here and not in somebody
else's terminal on their first evening.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "quickstart_miner.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLE), *args],
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_it_produces_a_valid_recipe(tmp_path):
    out = tmp_path / "recipe.json"
    result = _run("--tries", "5", "--seed", "1", "--out", str(out))

    assert result.returncode == 0, result.stderr[-2000:]
    assert out.is_file(), "the example must write the recipe it selected"

    from capability_subnet.miner.recipe import check_recipe, load_recipe

    recipe = load_recipe(out)
    assert not check_recipe(recipe), "the example must only ever write an admissible recipe"
    assert recipe.digest() in result.stdout


def test_it_prints_a_commitment_that_fits(tmp_path):
    """The payload is capped at 128 bytes and the URI is most of the budget."""
    uri = "https://github.com/Capcomp-AI/lora-merger/raw/main/tn/m1.json"
    result = _run(
        "--tries",
        "3",
        "--seed",
        "2",
        "--out",
        str(tmp_path / "r.json"),
        "--recipe-uri",
        uri,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "capsub1|" in result.stdout, "a commitment payload must be printed"

    from capability_subnet.common.commitments import MAX_PAYLOAD_BYTES, decode_commitment

    line = next(x for x in result.stdout.splitlines() if "capsub1|" in x)
    payload = line.split("capsub1|", 1)[1]
    payload = "capsub1|" + payload.strip()
    assert len(payload.encode()) <= MAX_PAYLOAD_BYTES
    assert decode_commitment(payload).recipe_uri == uri


def test_it_says_the_placeholder_ranking_is_not_a_measurement(tmp_path):
    """A miner who mistakes the offline heuristic for a score wastes a window."""
    result = _run("--tries", "2", "--seed", "3", "--out", str(tmp_path / "r.json"))
    assert "not by measurement" in result.stdout
