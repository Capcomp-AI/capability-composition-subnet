"""Resolving a commitment as it actually arrives from the chain.

`read_commitments` yields a ChainCommitment, which carries the *decoded* payload
in `payload` and the 128-byte string in `raw`. Treating the decoded object as a
string raised AttributeError out of a function whose contract is
ResolutionError, and it did so for every real commitment — the validator's only
path to a miner's recipe.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capability_subnet.common.chain import ChainCommitment
from capability_subnet.common.commitments import CommitmentPayload, encode_commitment
from capability_subnet.validator.resolve import ResolutionError, resolve_recipe

DIGEST = "sha256:" + "ab" * 32
URI = "hf:alice/recipes/r.json"


def _unreachable(uri=URI):
    """A pointer that will fail at fetch, not at decode.

    Reaching a fetch error is the assertion: it means the digest and URI were
    extracted, which is the step that used to raise AttributeError.
    """
    return uri


class TestEveryCommitmentShapeIsUnderstood:
    def _assert_got_past_decoding(self, commitment):
        with pytest.raises(ResolutionError) as caught:
            resolve_recipe(commitment, timeout=1)
        message = str(caught.value)
        assert "unreadable commitment" not in message, "decoding failed rather than fetching"
        assert "carries no digest" not in message, "the digest and URI were not extracted"

    def test_a_chain_commitment_as_read_from_the_chain(self):
        payload = CommitmentPayload(
            workflow_id="lora_merger_logic_v1", recipe_sha256=DIGEST, recipe_uri=_unreachable()
        )
        self._assert_got_past_decoding(
            ChainCommitment(
                hotkey="5A",
                uid=1,
                block=1,
                raw=encode_commitment("lora_merger_logic_v1", DIGEST, _unreachable()),
                payload=payload,
            )
        )

    def test_the_raw_payload_string_on_its_own(self):
        raw = encode_commitment("lora_merger_logic_v1", DIGEST, _unreachable())
        self._assert_got_past_decoding(SimpleNamespace(payload=raw))

    def test_already_decoded_fields(self):
        self._assert_got_past_decoding(
            SimpleNamespace(recipe_sha256=DIGEST, recipe_uri=_unreachable())
        )

    def test_a_malformed_payload_is_still_reported_as_unreadable(self):
        with pytest.raises(ResolutionError, match="unreadable commitment"):
            resolve_recipe(SimpleNamespace(payload="not-a-capsub-payload"), timeout=1)

    def test_an_empty_commitment_says_what_is_missing(self):
        with pytest.raises(ResolutionError, match="carries no digest"):
            resolve_recipe(SimpleNamespace(), timeout=1)
