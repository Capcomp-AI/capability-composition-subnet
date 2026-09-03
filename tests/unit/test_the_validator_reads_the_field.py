"""How a validator gets the field it measures.

Submissions live in the service, not on the chain, and a run's bodies are
readable only once that run is public. These cover what the client does with
that: bodies checked against their digests before anything is measured, a field
assembled correctly across the two source runs, and a refusal that says what
happened rather than looking like an empty subnet.

There is no signature and no credential here. The route is public or it
refuses, so nothing in this module can open a run early.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from capability_subnet.common import constants as C
from capability_subnet.common.chain import measured_in_run, run_opens_block
from capability_subnet.validator import field as F


class Response:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class Hotkey:
    ss58_address = "5Fabcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMN"

    def sign(self, message: bytes) -> bytes:
        # Bound to the message, so a test can assert what was signed without
        # standing up a keypair.
        return hashlib.sha256(message).digest()


class Wallet:
    hotkey = Hotkey()


def entry(hotkey: str, body: dict, block: int, uid: int = 1) -> dict:
    raw = json.dumps(body)
    return {
        "hotkey": hotkey,
        "uid": uid,
        "recipe_sha256": "sha256:" + hashlib.sha256(raw.encode()).hexdigest(),
        "recipe_raw": raw,
        "first_block": block,
        "submission_count": 1,
    }


def serving(by_run: dict[int, list[dict]]):
    """An httpx.get that answers /run/{id}/submissions from a dict."""
    captured: dict = {}

    def get(url, params=None, timeout=None):
        run_id = int(url.rstrip("/").split("/")[-2])
        captured.setdefault("params", []).append((run_id, dict(params or {})))
        return Response(200, {"run_id": run_id, "submissions": by_run.get(run_id, [])})

    get.captured = captured
    return get


class TestBodiesAreCheckedBeforeTheyAreMeasured:
    def test_a_body_that_does_not_hash_to_its_digest_is_refused(self, monkeypatch):
        """The service is not trusted for content, only for delivery.

        It could serve a recipe the miner never signed. It could not serve one
        that hashes to the digest stored beside it, and that is the whole of
        what this validator takes on faith.
        """
        import httpx

        tampered = entry("5Aaa", {"schema_version": 1}, 100)
        tampered["recipe_raw"] = json.dumps({"schema_version": 1, "extra": "added"})
        monkeypatch.setattr(httpx, "get", serving({411: [tampered]}))

        with pytest.raises(F.FieldError) as caught:
            F.fetch_run("http://api.invalid", 411)

        assert "hashing to" in str(caught.value)

    def test_a_good_body_arrives_as_the_exact_bytes_the_miner_sent(self, monkeypatch):
        import httpx

        body = {"schema_version": 1, "selected_adapters": ["a", "b"]}
        monkeypatch.setattr(httpx, "get", serving({411: [entry("5Aaa", body, 100)]}))

        got = F.fetch_run("http://api.invalid", 411)

        assert len(got) == 1
        assert json.loads(got[0].recipe_raw) == body
        digest = "sha256:" + hashlib.sha256(got[0].recipe_raw).hexdigest()
        assert digest == got[0].recipe_sha256

    def test_one_bad_body_fails_the_field_rather_than_shortening_it(self, monkeypatch):
        """A short field is worse than no field.

        The caller ranks and pays what comes back. A field quietly missing the
        entries that failed to verify would leave those miners looking like
        miners who never submitted, and nothing downstream could tell.
        """
        import httpx

        good = entry("5Aaa", {"schema_version": 1}, 100)
        bad = entry("5Bbb", {"schema_version": 1}, 101)
        bad["recipe_sha256"] = "sha256:" + "00" * 32
        monkeypatch.setattr(httpx, "get", serving({411: [good, bad]}))

        with pytest.raises(F.FieldError):
            F.fetch_run("http://api.invalid", 411)


class TestTheFieldSpansTwoSourceRuns:
    def test_a_late_submission_is_measured_a_run_later(self, monkeypatch):
        """The settling window, from the validator's side.

        A submission made in the last MIN_COMMITMENT_AGE_BLOCKS of a run has
        not stood long enough for the next one and is held over. A validator
        fetching only N-1 would never measure it, while its row sat in the
        service looking submitted.
        """
        import httpx

        # Runs at or after the epoch: before it they are a different length,
        # and the point here is the settling window, not the anchor.
        blocks = C.DEFAULT_RUN_BLOCKS
        opens = run_opens_block(413, blocks)
        early = opens + 10
        late = opens + blocks - 5

        assert measured_in_run(early, 414, blocks)
        assert measured_in_run(late, 415, blocks)

        by_run = {
            413: [
                entry("5Early", {"schema_version": 1}, early, uid=1),
                entry("5Late", {"schema_version": 1}, late, uid=2),
            ]
        }
        monkeypatch.setattr(httpx, "get", serving(by_run))

        measured_414 = F.field_for_run("http://api.invalid", 414)
        measured_415 = F.field_for_run("http://api.invalid", 415)

        assert [e.hotkey for e in measured_414] == ["5Early"]
        # Held over, not dropped: run 415 reaches back into run 413 for it.
        assert [e.hotkey for e in measured_415] == ["5Late"]

    def test_it_asks_for_both_source_runs(self, monkeypatch):
        import httpx

        get = serving({})
        monkeypatch.setattr(httpx, "get", get)
        F.field_for_run("http://api.invalid", 415)

        assert sorted(run for run, _ in get.captured["params"]) == [413, 414]


class TestRefusalsAreExplained:
    def test_an_unpublished_run_is_told_apart_from_an_empty_one(self, monkeypatch):
        """The refusal a validator meets every run it is asked to measure.

        Its field is published two runs after submission, so the run being
        scored is never readable. That has to arrive as a stated reason, not as
        a field of zero — the two produce the same weight vector and only one
        of them is a statement about the miners.
        """
        import httpx

        detail = "run 411 is not public yet; its bodies are published when run 413 opens"
        monkeypatch.setattr(httpx, "get", lambda *a, **k: Response(403, {"detail": detail}))

        with pytest.raises(F.FieldError) as caught:
            F.fetch_run("http://api.invalid", 411)

        assert "not published yet" in str(caught.value)
        assert "run 413 opens" in str(caught.value)

    def test_an_unreachable_service_is_not_an_empty_field(self, monkeypatch):
        """The two must not look alike.

        Both produce a burn, and only one of them is a statement about the
        miners. A validator that reported "nobody entered" when the service was
        down would be describing an outage as a subnet nobody wanted.
        """
        import httpx

        def boom(*a, **k):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", boom)

        with pytest.raises(F.FieldError) as caught:
            F.fetch_run("http://api.invalid", 411)

        assert "could not reach" in str(caught.value)
