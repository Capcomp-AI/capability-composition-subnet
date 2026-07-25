"""Adversarial behaviour.

Each test here plays a specific attacker: a miner smuggling executable content
past the merge engine, one copying the champion, one committing bytes that differ
from what it published, a candidate trying to reach the hidden material through a
tool, and a validator being handed a forged weight vector.

They are grouped by what the attacker is *trying to achieve*, because that is
what determines whether a defence is adequate — a filter that blocks today's
payload but not the goal behind it is not a defence.
"""

from __future__ import annotations

import json

import pytest

from capability_subnet.backend.monitor.admission import evaluate_commitment, parse_recipe
from capability_subnet.backend.monitor.anticopy import check_artifact_copy, check_for_copy
from capability_subnet.backend.monitor.fetch import FetchedRecipe, FetchError
from capability_subnet.common import constants as C
from capability_subnet.common.chain import ChainCommitment
from capability_subnet.common.commitments import CommitmentPayload
from capability_subnet.common.hashing import canonical_json_bytes, sha256_bytes
from capability_subnet.common.schemas import QueueEntry, WeightEntry, WeightVector
from capability_subnet.common.signing import SignatureError, require_trusted_signature


def commitment(hotkey: str, digest: str, block: int, uid: int = 1) -> ChainCommitment:
    return ChainCommitment(
        hotkey=hotkey,
        uid=uid,
        block=block,
        raw="capsub1|imde|x|https://example.test/r.json",
        payload=CommitmentPayload(
            workflow_id=C.DEFAULT_WORKFLOW_ID,
            recipe_sha256=digest,
            recipe_uri="https://example.test/r.json",
        ),
    )


def fetcher_for(raw: bytes):
    def _fetch(uri: str, expected: str, **_kwargs) -> FetchedRecipe:
        return FetchedRecipe(uri=uri, resolved_url=uri, raw=raw, sha256=sha256_bytes(raw))

    return _fetch


class TestSmugglingExecutableContent:
    """A recipe is the only miner-authored thing the engine reads, so it is the
    only place executable content could enter."""

    @pytest.mark.parametrize(
        "payload",
        [
            {"__class__": "os.system"},
            {"eval": "__import__('os').system('id')"},
            {"post_merge_hook": "http://attacker.test/payload.py"},
            {"custom_module": "attacker.merge"},
            {"auto_map": {"AutoModel": "attacker--model.Model"}},
        ],
    )
    def test_extra_fields_are_refused_outright(self, tiny_snapshot, payload):
        document = {
            "schema_version": 1,
            "workflow_id": C.DEFAULT_WORKFLOW_ID,
            "base_revision": tiny_snapshot.manifest.revision,
            "source_snapshot_sha256": tiny_snapshot.sha256,
            "selected_adapters": ["alpha-capability-v1", "beta-capability-v1"],
            "merge": {"combination_type": "linear"},
            "compression": {"output_rank": 64},
            **payload,
        }
        recipe, problems = parse_recipe(json.dumps(document).encode())

        assert recipe is None
        assert problems

    def test_an_adapter_path_cannot_be_supplied(self, tiny_snapshot):
        # There is no field to put one in: adapters are named, and the names are
        # resolved against the frozen pool.
        document = {
            "schema_version": 1,
            "base_revision": tiny_snapshot.manifest.revision,
            "source_snapshot_sha256": tiny_snapshot.sha256,
            "selected_adapters": ["../../etc/passwd", "beta-capability-v1"],
            "merge": {"combination_type": "linear"},
            "compression": {"output_rank": 64},
        }
        recipe, problems = parse_recipe(json.dumps(document).encode())

        assert recipe is not None  # the name is a legal string
        # …and the pool check is what refuses it.
        assert tiny_snapshot.registry.unknown_ids(recipe.selected_adapters)

    def test_binary_and_malformed_payloads_are_handled(self):
        assert parse_recipe(b"\x00\x01\x02\xff")[0] is None
        assert parse_recipe(b"not json at all")[0] is None
        assert parse_recipe(b"[1, 2, 3]")[0] is None  # a list, not an object
        assert parse_recipe(b"")[0] is None


class TestSubstitutingTheRecipe:
    """Commit one thing, publish another."""

    def test_bytes_that_do_not_match_the_commitment_are_rejected(self, tiny_snapshot, store):
        published = b'{"schema_version": 1}'

        result = evaluate_commitment(
            commitment("5Attacker", sha256_bytes(b"something else"), 100),
            snapshot=tiny_snapshot,
            store=store,
            registered_hotkeys={"5Attacker"},
            current_block=200,
            fetcher=fetcher_for(published),
        )

        assert not result.admitted
        assert "digest" in result.failed_gates()

    def test_a_recipe_whose_canonical_form_differs_is_rejected(self, tiny_snapshot, store):
        # Committing the digest of a specially-formatted file, then having the
        # engine score the canonical form of a different document, would let a
        # miner change their submission after committing.
        document = {
            "schema_version": 1,
            "workflow_id": C.DEFAULT_WORKFLOW_ID,
            "base_revision": tiny_snapshot.manifest.revision,
            "source_snapshot_sha256": tiny_snapshot.sha256,
            "selected_adapters": ["alpha-capability-v1", "beta-capability-v1"],
            "merge": {"combination_type": "linear", "random_seed": 0},
            "compression": {"output_rank": 8, "svd_clamp_quantile": 1.0},
            "global_weights": {},
            "layer_group_overrides": {},
            "output": {"dtype": "bfloat16", "adapter_name": "candidate"},
        }
        # Bytes with a stray field ordering AND an extra whitespace payload that
        # hashes differently from the canonical document.
        odd_bytes = json.dumps(document, indent=8).encode()

        result = evaluate_commitment(
            commitment("5Attacker", sha256_bytes(odd_bytes), 100),
            snapshot=tiny_snapshot,
            store=store,
            registered_hotkeys={"5Attacker"},
            current_block=200,
            fetcher=fetcher_for(odd_bytes),
        )

        # The canonical digest differs from the committed one, so it is refused
        # with an explanation rather than silently scored.
        assert not result.admitted
        assert "canonical" in result.reason

    def test_an_unreachable_pointer_is_a_rejection_not_a_crash(self, tiny_snapshot, store):
        def _fail(uri: str, expected: str, **_kwargs):
            raise FetchError(f"could not fetch {uri}")

        result = evaluate_commitment(
            commitment("5Attacker", sha256_bytes(b"x"), 100),
            snapshot=tiny_snapshot,
            store=store,
            registered_hotkeys={"5Attacker"},
            current_block=200,
            fetcher=_fail,
        )
        assert not result.admitted

    def test_an_unregistered_hotkey_never_reaches_the_fetch(self, tiny_snapshot, store):
        def _explode(uri: str, expected: str, **_kwargs):
            raise AssertionError("identity must be checked before anything is fetched")

        result = evaluate_commitment(
            commitment("5Nobody", sha256_bytes(b"x"), 100),
            snapshot=tiny_snapshot,
            store=store,
            registered_hotkeys=set(),
            current_block=200,
            fetcher=_explode,
        )
        assert not result.admitted
        assert "identity" in result.failed_gates()


class TestCopying:
    """Reading a published recipe and resubmitting it."""

    def _queue(
        self, store, hotkey: str, recipe_digest: str, block: int, artifact: str | None = None
    ):
        store.upsert_queue_entry(
            QueueEntry(
                hotkey=hotkey,
                uid=1,
                recipe_sha256=recipe_digest,
                recipe_uri="https://example.test/r.json",
                first_block=block,
                admitted_at_block=block,
            )
        )
        if artifact:
            store.set_artifact(hotkey, artifact)

    def test_the_earliest_commitment_wins(self, store):
        digest = sha256_bytes(b"recipe")
        self._queue(store, "5Original", digest, block=100)

        verdict = check_for_copy(store, hotkey="5Copycat", recipe_sha256=digest, first_block=200)

        assert verdict.is_copy
        assert verdict.original_hotkey == "5Original"

    def test_committing_earlier_than_the_stored_entry_is_not_a_copy(self, store):
        # The chain's ordering is the only one either miner could have relied on,
        # not the order the engine happened to read them in.
        digest = sha256_bytes(b"recipe")
        self._queue(store, "5Later", digest, block=500)

        verdict = check_for_copy(store, hotkey="5Earlier", recipe_sha256=digest, first_block=100)
        assert not verdict.is_copy

    def test_a_hotkey_does_not_copy_itself(self, store):
        digest = sha256_bytes(b"recipe")
        self._queue(store, "5Miner", digest, block=100)

        verdict = check_for_copy(store, hotkey="5Miner", recipe_sha256=digest, first_block=100)
        assert not verdict.is_copy

    def test_a_reworded_recipe_producing_the_same_weights_is_still_a_copy(self, store):
        # Reordering the adapter list or changing an ignored seed gives different
        # recipe bytes and identical weights. The artifact digest sees through it.
        artifact = sha256_bytes(b"identical weights")
        self._queue(store, "5Original", sha256_bytes(b"recipe-a"), block=100, artifact=artifact)

        verdict = check_artifact_copy(
            store, hotkey="5Copycat", artifact_sha256=artifact, first_block=200
        )
        assert verdict.is_copy
        assert verdict.original_hotkey == "5Original"

    def test_a_genuinely_different_package_is_not_flagged(self, store):
        self._queue(
            store,
            "5Original",
            sha256_bytes(b"recipe-a"),
            block=100,
            artifact=sha256_bytes(b"weights-a"),
        )
        verdict = check_artifact_copy(
            store, hotkey="5Other", artifact_sha256=sha256_bytes(b"weights-b"), first_block=200
        )
        assert not verdict.is_copy

    def test_a_terminated_hotkey_cannot_re_enter(self, store, tiny_snapshot):
        from capability_subnet.backend.monitor.admission import admit_new_commitments

        digest = sha256_bytes(b"first attempt")
        self._queue(store, "5Burned", digest, block=100)
        store.set_status("5Burned", "terminated", "lost decisively")

        # A fresh commitment from the same hotkey is skipped entirely: the one
        # shot has been spent.
        results = admit_new_commitments(
            [commitment("5Burned", sha256_bytes(b"second attempt"), 300)],
            snapshot=tiny_snapshot,
            store=store,
            registered_hotkeys={"5Burned"},
            current_block=400,
            fetcher=fetcher_for(b"{}"),
        )
        assert results == []
        assert store.get_queue_entry("5Burned").status == "terminated"


class TestReachingHiddenMaterial:
    """A candidate trying to read what it is being scored against."""

    def test_the_sql_tool_cannot_read_outside_the_snapshot(self, instance):
        from capability_subnet.sandbox.db_tool import SqliteMaintenanceDatabase, SqlRejected

        database = SqliteMaintenanceDatabase(instance.database)
        try:
            for statement in (
                "SELECT * FROM sqlite_master",
                "ATTACH DATABASE '/etc/passwd' AS leak",
                "SELECT load_extension('evil.so')",
            ):
                with pytest.raises(SqlRejected):
                    database.query(statement)
        finally:
            database.close()

    def test_generated_code_cannot_read_the_engine_process(self):
        from capability_subnet.sandbox.python_runner import PythonRunner

        # The runner is a separate interpreter, so the engine's objects — the
        # ground truth among them — are simply not in scope.
        source = (
            "def analyze(readings, threshold):\n"
            "    import sys\n"
            "    found = [m for m in sys.modules if 'capability_subnet' in m]\n"
            "    return {'peak': float(len(found)), 'exceedances': 0, 'longest_run': 0}\n"
        )
        outcome = PythonRunner().run(
            source, "analyze", [{"case_id": "c", "readings": [1.0], "threshold": 0.5}]
        )
        assert outcome.ok
        assert outcome.results[0]["output"]["peak"] == 0.0

    def test_generated_code_runs_with_an_isolated_import_path(self):
        from capability_subnet.sandbox.python_runner import PythonRunner

        source = (
            "def analyze(readings, threshold):\n"
            "    import sys\n"
            "    return {'peak': float('' in sys.path), 'exceedances': 0, 'longest_run': 0}\n"
        )
        outcome = PythonRunner().run(
            source, "analyze", [{"case_id": "c", "readings": [1.0], "threshold": 0.5}]
        )
        assert outcome.ok
        # An empty path entry would make the current directory importable, which
        # is where the engine's own modules live.
        assert outcome.results[0]["output"]["peak"] == 0.0

    def test_the_scorer_holds_truth_the_tools_never_see(self, instance):
        # The hidden diagnostic cases carry their expected outputs; what is sent
        # into the runner does not.
        from capability_subnet.sandbox.python_runner import hidden_cases

        sent = hidden_cases(instance.diagnostic.hidden_cases)
        assert all("expected" not in case for case in sent)
        assert all("expected" in case for case in instance.diagnostic.hidden_cases)


class TestForgedWeightVectors:
    """A validator being handed something the operator did not produce."""

    def _vector(self) -> WeightVector:
        return WeightVector(
            window_id=1,
            computed_at_block=100,
            spec_version=1000,
            entries=[WeightEntry(uid=5, hotkey="5Attacker", weight=1.0)],
        )

    def test_an_unsigned_vector_is_refused(self):
        with pytest.raises(SignatureError, match="unsigned"):
            require_trusted_signature(self._vector(), {"5Operator"})

    def test_a_signature_from_an_untrusted_signer_is_refused(self):
        vector = self._vector()
        vector.signature = "00" * 64
        vector.signer_hotkey = "5SomeoneElse"

        with pytest.raises(SignatureError, match="allow-list"):
            require_trusted_signature(vector, {"5Operator"})

    def test_a_forged_signature_from_a_trusted_hotkey_is_refused(self):
        vector = self._vector()
        vector.signature = "00" * 64
        vector.signer_hotkey = "5Operator"

        with pytest.raises(SignatureError, match="does not verify"):
            require_trusted_signature(vector, {"5Operator"})

    def test_the_signed_bytes_exclude_the_signature_itself(self):
        vector = self._vector()
        before = vector.signable_bytes()
        vector.signature = "deadbeef"
        vector.signer_hotkey = "5Operator"

        assert vector.signable_bytes() == before

    def test_tampering_with_the_payload_changes_the_signed_bytes(self):
        vector = self._vector()
        before = vector.signable_bytes()
        vector.entries[0].uid = 9

        assert vector.signable_bytes() != before

    def test_a_vector_that_does_not_sum_to_one_is_refused_by_the_schema(self):
        with pytest.raises(Exception, match="sum to 1.0"):
            WeightVector(
                window_id=1,
                computed_at_block=1,
                spec_version=1,
                entries=[
                    WeightEntry(uid=1, weight=0.9),
                    WeightEntry(uid=2, weight=0.9),
                ],
            )

    def test_a_vector_with_a_repeated_uid_is_refused_by_the_schema(self):
        with pytest.raises(Exception, match="duplicate uid"):
            WeightVector(
                window_id=1,
                computed_at_block=1,
                spec_version=1,
                entries=[
                    WeightEntry(uid=1, weight=0.5),
                    WeightEntry(uid=1, weight=0.5),
                ],
            )


class TestValidatorChecksAgainstTheChain:
    """A correctly-signed vector that is still not submittable."""

    def _vector(self, **overrides) -> WeightVector:
        payload = {
            "window_id": 10,
            "computed_at_block": 1000,
            "spec_version": 1000,
            "entries": [WeightEntry(uid=2, hotkey="5Champion", weight=1.0)],
        }
        payload.update(overrides)
        return WeightVector(**payload)

    def _validate(self, vector, **overrides):
        from capability_subnet.validator.client import validate_vector

        kwargs = {
            "metagraph_size": 8,
            "hotkeys": ["5A", "5B", "5Champion", "5D", "5E", "5F", "5G", "5H"],
            "current_window": 10,
            "max_stale_windows": 3,
            "burn_uid": 0,
        }
        kwargs.update(overrides)
        return validate_vector(vector, **kwargs)

    def test_a_healthy_vector_passes(self):
        assert (
            self._validate(
                self._vector(entries=[WeightEntry(uid=2, hotkey="5Champion", weight=1.0)])
            )
            == []
        )

    def test_a_uid_beyond_the_subnet_is_caught(self):
        problems = self._validate(
            self._vector(entries=[WeightEntry(uid=99, hotkey="5Champion", weight=1.0)])
        )
        assert any(problem.code == "uid_out_of_range" for problem in problems)

    def test_a_deregistered_champion_is_caught(self):
        # The UID is now someone else's; paying it would pay a stranger.
        problems = self._validate(
            self._vector(entries=[WeightEntry(uid=1, hotkey="5Champion", weight=1.0)])
        )
        assert any(problem.code == "hotkey_mismatch" for problem in problems)

    def test_a_stale_vector_is_caught(self):
        problems = self._validate(self._vector(window_id=1), current_window=20)
        assert any(problem.code == "stale" for problem in problems)

    def test_an_empty_vector_is_caught(self):
        vector = self._vector()
        vector.entries = []
        assert any(problem.code == "empty" for problem in self._validate(vector))

    def test_the_burn_entry_is_not_checked_against_a_hotkey(self):
        # The burn UID has no hotkey to match, and requiring one would make every
        # burning vector unsubmittable.
        vector = self._vector(entries=[WeightEntry(uid=0, hotkey="", weight=1.0, role="burn")])
        assert self._validate(vector) == []


class TestCanonicalFormCannotBeGamed:
    def test_two_documents_differing_only_in_formatting_share_a_digest(self):
        document = {"b": 2, "a": [1, {"z": 0, "y": 1}]}
        assert sha256_bytes(canonical_json_bytes(document)) == sha256_bytes(
            canonical_json_bytes(json.loads(json.dumps(document, indent=6)))
        )
