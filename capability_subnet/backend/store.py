"""Engine state.

A single SQLite database holds the queue, the champion record, the per-window
instance draws, every sample row, every published report and every weight vector.
SQLite rather than a server database because the engine is one process on one
host, the write rate is a handful of rows per evaluation, and an operator who
needs to inspect or repair state should be able to do it with a file and a shell.

Two things here are not conveniences:

* the champion record is written in the same transaction as the report that
  justified it, so state can never claim a champion no report supports;
* every sample row is retained. The comparator needs paired rows for the *same*
  instances, and reconstructing them later from aggregates is not possible.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from capability_subnet.common.schemas import (
    ChampionRecord,
    EvaluationReport,
    InstanceResult,
    QueueEntry,
    WeightVector,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS queue (
    hotkey            TEXT PRIMARY KEY,
    uid               INTEGER NOT NULL,
    recipe_sha256     TEXT NOT NULL,
    recipe_uri        TEXT NOT NULL,
    first_block       INTEGER NOT NULL,
    admitted_at_block INTEGER NOT NULL,
    status            TEXT NOT NULL,
    status_reason     TEXT NOT NULL DEFAULT '',
    artifact_sha256   TEXT
);
CREATE INDEX IF NOT EXISTS queue_status_block ON queue(status, first_block);
CREATE INDEX IF NOT EXISTS queue_recipe ON queue(recipe_sha256);
CREATE INDEX IF NOT EXISTS queue_artifact ON queue(artifact_sha256);

CREATE TABLE IF NOT EXISTS champion (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS windows (
    window_id    INTEGER PRIMARY KEY,
    opened_block INTEGER NOT NULL,
    hidden_seeds TEXT NOT NULL,
    ood_seeds    TEXT NOT NULL,
    finalized    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS samples (
    window_id    INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    instance_id  TEXT NOT NULL,
    split        TEXT NOT NULL,
    payload      TEXT NOT NULL,
    PRIMARY KEY (window_id, candidate_id, instance_id)
);
CREATE INDEX IF NOT EXISTS samples_lookup ON samples(window_id, candidate_id, split);

CREATE TABLE IF NOT EXISTS reports (
    report_sha256 TEXT PRIMARY KEY,
    window_id     INTEGER NOT NULL,
    candidate_id  TEXT NOT NULL,
    verdict       TEXT NOT NULL,
    created_block INTEGER NOT NULL,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS reports_window ON reports(window_id, candidate_id);

CREATE TABLE IF NOT EXISTS weights (
    window_id INTEGER PRIMARY KEY,
    payload   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
    window_id    INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    instance_id  TEXT NOT NULL,
    split        TEXT NOT NULL,
    instance_seed INTEGER NOT NULL,
    trace        TEXT NOT NULL,
    result       TEXT NOT NULL,
    PRIMARY KEY (window_id, candidate_id, instance_id)
);
CREATE INDEX IF NOT EXISTS traces_window ON traces(window_id);

CREATE TABLE IF NOT EXISTS compatibility (
    window_id    INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    payload      TEXT NOT NULL,
    PRIMARY KEY (window_id, candidate_id)
);
"""


class Store:
    """Durable engine state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        # WAL keeps the read-only API responsive while the control loop writes.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()
        self._set_meta("schema_version", str(SCHEMA_VERSION))

    # -- plumbing -----------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a unit of work atomically.

        Used wherever two writes must not be observable apart — crowning a
        champion and storing the report that justified it, most importantly.
        """
        with self._lock:
            try:
                yield self._connection
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _set_meta(self, key: str, value: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self._connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self._set_meta(key, value)

    # -- queue --------------------------------------------------------------

    def upsert_queue_entry(self, entry: QueueEntry) -> None:
        """Insert a newly admitted challenger.

        Existing rows are left alone. A hotkey gets one shot: re-committing a
        different recipe after being queued must not reset the entry, or the one
        shot rule would be worth nothing.
        """
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO queue(hotkey, uid, recipe_sha256, recipe_uri, first_block, "
                "admitted_at_block, status, status_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(hotkey) DO NOTHING",
                (
                    entry.hotkey,
                    entry.uid,
                    entry.recipe_sha256,
                    entry.recipe_uri,
                    entry.first_block,
                    entry.admitted_at_block,
                    entry.status,
                    entry.status_reason,
                ),
            )

    def get_queue_entry(self, hotkey: str) -> QueueEntry | None:
        row = self._connection.execute("SELECT * FROM queue WHERE hotkey = ?", (hotkey,)).fetchone()
        return _row_to_queue_entry(row) if row else None

    def next_challenger(self) -> QueueEntry | None:
        """The queue head: the earliest commitment still waiting.

        Ordering by commit block is what makes role assignment mechanical. Nobody
        chooses who challenges next; the chain does, by the order it accepted
        commitments in.
        """
        row = self._connection.execute(
            "SELECT * FROM queue WHERE status = 'queued' ORDER BY first_block, hotkey LIMIT 1"
        ).fetchone()
        return _row_to_queue_entry(row) if row else None

    def set_status(self, hotkey: str, status: str, reason: str = "") -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE queue SET status = ?, status_reason = ? WHERE hotkey = ?",
                (status, reason, hotkey),
            )

    def set_artifact(self, hotkey: str, artifact_sha256: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE queue SET artifact_sha256 = ? WHERE hotkey = ?",
                (artifact_sha256, hotkey),
            )

    def list_queue(self, status: str | None = None) -> list[QueueEntry]:
        if status:
            rows = self._connection.execute(
                "SELECT * FROM queue WHERE status = ? ORDER BY first_block, hotkey", (status,)
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM queue ORDER BY first_block, hotkey"
            ).fetchall()
        return [_row_to_queue_entry(row) for row in rows]

    def earliest_with_recipe(self, recipe_sha256: str) -> QueueEntry | None:
        """The first hotkey to commit a given recipe digest."""
        row = self._connection.execute(
            "SELECT * FROM queue WHERE recipe_sha256 = ? ORDER BY first_block, hotkey LIMIT 1",
            (recipe_sha256,),
        ).fetchone()
        return _row_to_queue_entry(row) if row else None

    def earliest_with_artifact(self, artifact_sha256: str) -> QueueEntry | None:
        """The first hotkey whose recipe reconstructed to a given artifact.

        Distinct from the recipe check: two different recipes can reconstruct to
        the same bytes, and a copy that paraphrases the champion's recipe is
        still a copy.
        """
        row = self._connection.execute(
            "SELECT * FROM queue WHERE artifact_sha256 = ? ORDER BY first_block, hotkey LIMIT 1",
            (artifact_sha256,),
        ).fetchone()
        return _row_to_queue_entry(row) if row else None

    # -- champion -----------------------------------------------------------

    def get_champion(self) -> ChampionRecord | None:
        row = self._connection.execute("SELECT payload FROM champion WHERE id = 1").fetchone()
        return ChampionRecord.model_validate_json(row["payload"]) if row else None

    def set_champion(
        self, champion: ChampionRecord, *, report: EvaluationReport | None = None
    ) -> None:
        """Crown a champion, optionally together with the report that justified it."""
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO champion(id, payload) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
                (champion.model_dump_json(),),
            )
            if report is not None:
                self._insert_report(connection, report)

    def clear_champion(self, *, reason: str) -> None:
        """Vacate the throne.

        Used when a champion can no longer be paid — its hotkey deregistered and
        its UID belongs to someone else. Leaving it seated would burn every
        window while still holding the dethrone bar at its score, so nothing
        could displace it and nothing could be paid: a deadlock with no exit.
        Vacating drops the bar back to the permanent references, which is where
        it sits before anyone has won.

        The champion-report pointer goes with it. A vector that named a champion
        no record supports is exactly what the transaction in ``set_champion``
        exists to prevent.
        """
        with self.transaction() as connection:
            connection.execute("DELETE FROM champion WHERE id = 1")
            connection.execute("DELETE FROM meta WHERE key = 'champion_report_sha256'")
        log.info("champion cleared: %s", reason)

    # -- windows ------------------------------------------------------------

    def record_window(
        self, window_id: int, opened_block: int, hidden_seeds: list[int], ood_seeds: list[int]
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO windows(window_id, opened_block, hidden_seeds, ood_seeds) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(window_id) DO NOTHING",
                (window_id, opened_block, json.dumps(hidden_seeds), json.dumps(ood_seeds)),
            )

    def get_window(self, window_id: int) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM windows WHERE window_id = ?", (window_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "window_id": row["window_id"],
            "opened_block": row["opened_block"],
            "hidden_seeds": json.loads(row["hidden_seeds"]),
            "ood_seeds": json.loads(row["ood_seeds"]),
            "finalized": bool(row["finalized"]),
        }

    def latest_window_id(self) -> int | None:
        row = self._connection.execute("SELECT MAX(window_id) AS m FROM windows").fetchone()
        return row["m"] if row and row["m"] is not None else None

    def finalize_window(self, window_id: int) -> None:
        with self.transaction() as connection:
            connection.execute("UPDATE windows SET finalized = 1 WHERE window_id = ?", (window_id,))

    # -- samples ------------------------------------------------------------

    def store_samples(
        self, window_id: int, candidate_id: str, results: list[InstanceResult]
    ) -> None:
        with self.transaction() as connection:
            connection.executemany(
                "INSERT INTO samples(window_id, candidate_id, instance_id, split, payload) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(window_id, candidate_id, instance_id) DO UPDATE SET "
                "payload = excluded.payload",
                [
                    (
                        window_id,
                        candidate_id,
                        result.instance_id,
                        result.split,
                        result.model_dump_json(),
                    )
                    for result in results
                ],
            )

    def load_samples(
        self, window_id: int, candidate_id: str, *, split: str | None = None
    ) -> list[InstanceResult]:
        if split:
            rows = self._connection.execute(
                "SELECT payload FROM samples WHERE window_id = ? AND candidate_id = ? "
                "AND split = ? ORDER BY instance_id",
                (window_id, candidate_id, split),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT payload FROM samples WHERE window_id = ? AND candidate_id = ? "
                "ORDER BY instance_id",
                (window_id, candidate_id),
            ).fetchall()
        return [InstanceResult.model_validate_json(row["payload"]) for row in rows]

    def candidates_in_window(self, window_id: int) -> list[str]:
        rows = self._connection.execute(
            "SELECT DISTINCT candidate_id FROM samples WHERE window_id = ? ORDER BY candidate_id",
            (window_id,),
        ).fetchall()
        return [row["candidate_id"] for row in rows]

    def has_samples(self, window_id: int, candidate_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM samples WHERE window_id = ? AND candidate_id = ? LIMIT 1",
            (window_id, candidate_id),
        ).fetchone()
        return row is not None

    # -- traces --------------------------------------------------------------

    def store_traces(self, window_id: int, candidate_id: str, entries: list[dict]) -> None:
        """Retain traces so a closed window can be independently re-scored."""
        with self.transaction() as connection:
            connection.executemany(
                "INSERT INTO traces(window_id, candidate_id, instance_id, split, "
                "instance_seed, trace, result) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(window_id, candidate_id, instance_id) DO UPDATE SET "
                "trace = excluded.trace, result = excluded.result",
                [
                    (
                        window_id,
                        candidate_id,
                        entry["instance_id"],
                        entry["split"],
                        entry["instance_seed"],
                        json.dumps(entry["trace"], sort_keys=True),
                        entry["result"],
                    )
                    for entry in entries
                ],
            )

    def load_traces(self, window_id: int) -> list[dict]:
        rows = self._connection.execute(
            "SELECT candidate_id, instance_id, split, instance_seed, trace, result "
            "FROM traces WHERE window_id = ? ORDER BY candidate_id, instance_id",
            (window_id,),
        ).fetchall()
        return [
            {
                "candidate_id": row["candidate_id"],
                "instance_id": row["instance_id"],
                "split": row["split"],
                "instance_seed": row["instance_seed"],
                "trace": json.loads(row["trace"]),
                "result": row["result"],
            }
            for row in rows
        ]

    # -- reports ------------------------------------------------------------

    def _insert_report(self, connection: sqlite3.Connection, report: EvaluationReport) -> str:
        from capability_subnet.common.hashing import sha256_bytes

        digest = sha256_bytes(report.signable_bytes())
        connection.execute(
            "INSERT INTO reports(report_sha256, window_id, candidate_id, verdict, "
            "created_block, payload) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(report_sha256) DO UPDATE SET payload = excluded.payload",
            (
                digest,
                report.window_id,
                report.candidate_id,
                report.verdict,
                report.evaluated_at_block,
                report.model_dump_json(),
            ),
        )
        return digest

    def store_report(self, report: EvaluationReport) -> str:
        with self.transaction() as connection:
            return self._insert_report(connection, report)

    def get_report(self, report_sha256: str) -> EvaluationReport | None:
        row = self._connection.execute(
            "SELECT payload FROM reports WHERE report_sha256 = ?", (report_sha256,)
        ).fetchone()
        return EvaluationReport.model_validate_json(row["payload"]) if row else None

    def list_reports(
        self, *, window_id: int | None = None, limit: int = 50
    ) -> list[tuple[str, EvaluationReport]]:
        if window_id is None:
            rows = self._connection.execute(
                "SELECT report_sha256, payload FROM reports ORDER BY created_block DESC, "
                "report_sha256 LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT report_sha256, payload FROM reports WHERE window_id = ? "
                "ORDER BY created_block DESC, report_sha256 LIMIT ?",
                (window_id, limit),
            ).fetchall()
        return [
            (row["report_sha256"], EvaluationReport.model_validate_json(row["payload"]))
            for row in rows
        ]

    # -- weights ------------------------------------------------------------

    def store_weights(self, vector: WeightVector) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO weights(window_id, payload) VALUES (?, ?) "
                "ON CONFLICT(window_id) DO UPDATE SET payload = excluded.payload",
                (vector.window_id, vector.model_dump_json()),
            )

    def latest_weights(self) -> WeightVector | None:
        row = self._connection.execute(
            "SELECT payload FROM weights ORDER BY window_id DESC LIMIT 1"
        ).fetchone()
        return WeightVector.model_validate_json(row["payload"]) if row else None

    def get_weights(self, window_id: int) -> WeightVector | None:
        row = self._connection.execute(
            "SELECT payload FROM weights WHERE window_id = ?", (window_id,)
        ).fetchone()
        return WeightVector.model_validate_json(row["payload"]) if row else None

    # -- compatibility history ----------------------------------------------

    def store_compatibility(self, window_id: int, candidate_id: str, payload: dict) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO compatibility(window_id, candidate_id, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(window_id, candidate_id) DO UPDATE SET payload = excluded.payload",
                (window_id, candidate_id, json.dumps(payload, sort_keys=True)),
            )

    def load_compatibility(self, *, limit: int = 1000) -> list[dict]:
        rows = self._connection.execute(
            "SELECT window_id, candidate_id, payload FROM compatibility "
            "ORDER BY window_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "window_id": row["window_id"],
                "candidate_id": row["candidate_id"],
                **json.loads(row["payload"]),
            }
            for row in rows
        ]


def _row_to_queue_entry(row: sqlite3.Row) -> QueueEntry:
    return QueueEntry(
        hotkey=row["hotkey"],
        uid=row["uid"],
        recipe_sha256=row["recipe_sha256"],
        recipe_uri=row["recipe_uri"],
        first_block=row["first_block"],
        admitted_at_block=row["admitted_at_block"],
        status=row["status"],
        status_reason=row["status_reason"] or "",
    )
