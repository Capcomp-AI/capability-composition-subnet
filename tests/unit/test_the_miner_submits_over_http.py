"""A miner's whole interaction with the network.

There is no chain write and nothing published anywhere: the recipe travels in
the request body, signed by the hotkey. What matters is that the bytes sent are
the bytes signed, that the signature binds both the run and the recipe, and that
a refusal tells the miner what to do about it.
"""

from __future__ import annotations

import json

import pytest

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


class TestWhatIsSentAndSigned:
    def test_the_body_is_the_protocol_s_canonical_form(self, tiny_snapshot):
        """One digest, not two.

        The bytes a miner signs are the bytes the engine identifies the recipe
        by, so `capability-miner digest` and the submission agree. Any other
        compact form would give the miner two numbers that differ for reasons
        nobody should have to learn.
        """
        recipe = recipe_for(tiny_snapshot)
        body = submit.canonical_body(recipe)

        assert body == recipe.canonical_bytes()
        assert submit.digest_of(body) == recipe.digest()

    def test_formatting_on_disk_cannot_change_what_is_signed(self, tiny_snapshot):
        recipe = recipe_for(tiny_snapshot)
        spaced = Recipe.model_validate_json(json.dumps(recipe.model_dump(mode="json"), indent=4))

        assert submit.canonical_body(spaced) == submit.canonical_body(recipe)

    def test_the_signed_string_binds_the_run_and_the_recipe(self):
        one = submit.signing_message(412, "sha256:aa")

        assert one != submit.signing_message(413, "sha256:aa")
        assert one != submit.signing_message(412, "sha256:bb")
        assert one.decode().startswith(submit.SIGNING_PREFIX)

    def test_the_prefix_is_the_one_the_service_checks(self):
        """Two spellings of this string would refuse every submission.

        Written out rather than imported: the service lives in the engine
        repository, which this package does not depend on. The engine's suite
        asserts the pair against each other, where both are importable.
        """
        assert submit.SIGNING_PREFIX == "capcomp-submit:v1"
        assert submit.signing_message(412, "sha256:aa") == b"capcomp-submit:v1:412:sha256:aa"


class Response:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class TestRefusalsAreExplained:
    """A miner reading the error should know what to change."""

    def test_a_bad_signature_says_what_to_sign(self):
        message = submit._explain(Response(401, {"detail": "signature does not match"}))

        assert "signature" in message
        assert "run" in message and "digest" in message

    def test_an_unregistered_hotkey_says_how_to_register(self):
        message = submit._explain(Response(403, {"detail": "not registered"}))

        assert "btcli subnet register" in message

    def test_a_spent_budget_says_so(self):
        message = submit._explain(Response(429, {"detail": "all 3 used for run 412"}))

        assert "no attempts left" in message
        assert "run 412" in message

    def test_an_unavailable_service_says_nothing_was_sent(self):
        """The one case where retrying is right, so it must not read as a
        rejection of the recipe."""
        message = submit._explain(Response(503, {"detail": "chain unreachable"}))

        assert "Nothing was submitted" in message

    def test_an_unexpected_status_still_carries_the_status(self):
        message = submit._explain(Response(500, None, text="upstream exploded"))

        assert "500" in message

    def test_a_response_that_is_not_json_does_not_crash(self):
        assert submit._explain(Response(502, None, text="<html>bad gateway</html>"))


class TestSending:
    def test_a_refusal_is_raised_not_returned(self, monkeypatch):
        import httpx

        monkeypatch.setattr(httpx, "post", lambda *a, **k: Response(429, {"detail": "spent"}))
        with pytest.raises(submit.SubmitError):
            submit.send("http://api.test", "5A", b"{}", "0xdeadbeef")

    def test_an_unreachable_service_raises(self, monkeypatch):
        import httpx

        def boom(*a, **k):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "post", boom)
        with pytest.raises(submit.SubmitError, match="could not reach"):
            submit.send("http://api.test", "5A", b"{}", "0x00")

    def test_an_accepted_submission_reports_the_budget(self, monkeypatch):
        import httpx

        monkeypatch.setattr(
            httpx,
            "post",
            lambda *a, **k: Response(
                200,
                {
                    "run_id": 412,
                    "uid": 72,
                    "recipe_sha256": "sha256:aa",
                    "submission_count": 2,
                    "remaining": 1,
                    "replaced": "sha256:bb",
                },
            ),
        )
        accepted = submit.send("http://api.test", "5A", b"{}", "0x00")

        assert accepted.run_id == 412 and accepted.uid == 72
        assert accepted.submission_count == 2 and accepted.remaining == 1
        assert accepted.replaced == "sha256:bb"
        assert accepted.unchanged is False

    def test_an_identical_resend_is_marked_unchanged(self, monkeypatch):
        """So the miner is told it cost nothing rather than left counting."""
        import httpx

        monkeypatch.setattr(
            httpx,
            "post",
            lambda *a, **k: Response(
                200,
                {
                    "run_id": 412,
                    "uid": 72,
                    "recipe_sha256": "sha256:aa",
                    "submission_count": 1,
                    "remaining": 2,
                    "unchanged": True,
                },
            ),
        )

        assert submit.send("http://api.test", "5A", b"{}", "0x00").unchanged is True
