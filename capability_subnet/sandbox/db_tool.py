"""The maintenance-database tool.

The candidate gets a read-only connection and one verb: run a SELECT. Two
backends implement it — SQLite for the public pack and for local development,
PostgreSQL for the hidden snapshot — behind an interface that returns the same
shape, so a query developed against one runs unchanged against the other.

Read-only is enforced twice: by statement inspection before execution, and by the
connection itself (a SQLite connection opened in immutable mode, or a PostgreSQL
role with no write grants). Inspection alone would be a filter to be evaded;
the connection-level restriction is what actually holds.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from capability_subnet.sandbox.limits import ExecutionLimits
from capability_subnet.workflows.industrial_maintenance_de_v1.instance import DatabaseSnapshot

log = logging.getLogger(__name__)

#: Statements that may begin a query. Anything else is refused before it reaches
#: a connection.
_ALLOWED_PREFIXES = ("select", "with")

#: Keywords that must not appear as standalone words anywhere in the statement.
_FORBIDDEN_KEYWORDS = frozenset(
    {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "grant",
        "revoke",
        "attach",
        "detach",
        "pragma",
        "vacuum",
        "copy",
        "call",
        "do",
        "commit",
        "rollback",
    }
)

#: Identifier prefixes that address the database's own catalogs. Reading them is
#: not merely noise: the hidden snapshots for many instances live as separate
#: schemas in one PostgreSQL database, so catalog enumeration is a route from one
#: instance to another instance's data.
_FORBIDDEN_IDENTIFIER_PREFIXES = ("sqlite_", "pg_", "information_schema")

#: Functions that read or write outside the snapshot entirely.
_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "dblink",
        "lo_import",
        "lo_export",
        "current_setting",
        "set_config",
    }
)

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_COMMENT_RE = re.compile(r"(--[^\n]*)|(/\*.*?\*/)", re.DOTALL)


class SqlRejected(Exception):
    """Raised when a statement is refused before execution."""


@dataclass(frozen=True, slots=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "row_count": len(self.rows),
            "truncated": self.truncated,
        }


def validate_statement(sql: str) -> str:
    """Check that ``sql`` is a single read-only statement.

    Raises:
        SqlRejected: with a message the candidate sees, so a rejected query is a
            recoverable mistake it can correct on the next turn rather than a
            silent zero.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise SqlRejected("Leere Abfrage.")

    stripped = _COMMENT_RE.sub(" ", sql).strip().rstrip(";").strip()
    if not stripped:
        raise SqlRejected("Abfrage besteht nur aus Kommentaren.")

    if ";" in stripped:
        raise SqlRejected("Nur eine einzelne Anweisung ist zulässig.")

    lowered = stripped.lower()
    if not lowered.startswith(_ALLOWED_PREFIXES):
        raise SqlRejected("Nur SELECT-Abfragen sind zulässig.")

    words = {match.group(0).lower() for match in _WORD_RE.finditer(lowered)}

    forbidden = sorted(words & _FORBIDDEN_KEYWORDS)
    if forbidden:
        raise SqlRejected(f"Unzulässige Schlüsselwörter: {', '.join(forbidden)}.")

    catalogs = sorted(
        word for word in words if word.startswith(_FORBIDDEN_IDENTIFIER_PREFIXES)
    )
    if catalogs:
        raise SqlRejected(
            f"Zugriff auf Systemkataloge ist nicht zulässig: {', '.join(catalogs)}. "
            "Die Abfrage darf nur die dokumentierten Tabellen verwenden."
        )

    functions = sorted(words & _FORBIDDEN_FUNCTIONS)
    if functions:
        raise SqlRejected(f"Unzulässige Funktionen: {', '.join(functions)}.")

    return stripped


class MaintenanceDatabase(Protocol):
    """A read-only maintenance snapshot."""

    def query(self, sql: str) -> QueryResult: ...

    def close(self) -> None: ...


class SqliteMaintenanceDatabase:
    """SQLite-backed snapshot.

    Built in memory from the generated rows. Nothing is written to disk, so
    running many instances concurrently cannot collide on a file, and there is no
    snapshot left behind for a later run to read.
    """

    def __init__(self, snapshot: DatabaseSnapshot, limits: ExecutionLimits | None = None) -> None:
        self._limits = limits or ExecutionLimits()
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._build(snapshot)
        # Authorizer runs on every parsed statement and is the enforcement that
        # statement inspection only approximates.
        self._connection.set_authorizer(self._authorize)

    def _build(self, snapshot: DatabaseSnapshot) -> None:
        cursor = self._connection.cursor()
        for statement in snapshot.ddl:
            cursor.execute(statement)
        for table, rows in snapshot.rows.items():
            if not rows:
                continue
            placeholders = ", ".join("?" for _ in snapshot.columns[table])
            cursor.executemany(
                f"INSERT INTO {table} VALUES ({placeholders})", [tuple(row) for row in rows]
            )
        self._connection.commit()

    @staticmethod
    def _authorize(action: int, *_args: Any) -> int:
        allowed = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
        }
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    def query(self, sql: str) -> QueryResult:
        statement = validate_statement(sql)
        cursor = self._connection.cursor()
        try:
            cursor.execute(statement)
        except sqlite3.DatabaseError as exc:
            raise SqlRejected(f"SQL-Fehler: {exc}") from exc

        columns = tuple(description[0] for description in (cursor.description or ()))
        fetched = cursor.fetchmany(self._limits.max_sql_rows + 1)
        truncated = len(fetched) > self._limits.max_sql_rows
        rows = tuple(tuple(row) for row in fetched[: self._limits.max_sql_rows])
        return QueryResult(columns=columns, rows=rows, truncated=truncated)

    def close(self) -> None:
        self._connection.set_authorizer(None)
        self._connection.close()


class PostgresMaintenanceDatabase:
    """PostgreSQL-backed snapshot for hidden evaluation.

    Each instance gets its own schema inside a throwaway database, populated by
    the same generator output, and the candidate's connection uses a role with
    ``SELECT`` and nothing else. The statement timeout is set on the session so a
    pathological query is cancelled by the server rather than by the harness.
    """

    def __init__(
        self,
        dsn: str,
        snapshot: DatabaseSnapshot,
        *,
        schema: str,
        limits: ExecutionLimits | None = None,
        readonly_role: str | None = None,
    ) -> None:
        import psycopg  # imported lazily: only the engine host needs it

        self._limits = limits or ExecutionLimits()
        self._schema = schema
        self._admin = psycopg.connect(dsn, autocommit=True)
        self._build(snapshot, readonly_role)
        self._connection = psycopg.connect(dsn, autocommit=True)
        with self._connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {schema}")
            cursor.execute(
                f"SET statement_timeout TO {int(self._limits.sql_timeout_seconds * 1000)}"
            )
            cursor.execute("SET default_transaction_read_only TO on")

    def _build(self, snapshot: DatabaseSnapshot, readonly_role: str | None) -> None:
        with self._admin.cursor() as cursor:
            cursor.execute(f"DROP SCHEMA IF EXISTS {self._schema} CASCADE")
            cursor.execute(f"CREATE SCHEMA {self._schema}")
            cursor.execute(f"SET search_path TO {self._schema}")
            for statement in snapshot.ddl:
                cursor.execute(statement)
            for table, rows in snapshot.rows.items():
                if not rows:
                    continue
                placeholders = ", ".join("%s" for _ in snapshot.columns[table])
                cursor.executemany(
                    f"INSERT INTO {table} VALUES ({placeholders})",
                    [tuple(row) for row in rows],
                )
            if readonly_role:
                cursor.execute(f"GRANT USAGE ON SCHEMA {self._schema} TO {readonly_role}")
                cursor.execute(
                    f"GRANT SELECT ON ALL TABLES IN SCHEMA {self._schema} TO {readonly_role}"
                )

    def query(self, sql: str) -> QueryResult:
        statement = validate_statement(sql)
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(statement)
                columns = tuple(desc.name for desc in (cursor.description or ()))
                fetched = cursor.fetchmany(self._limits.max_sql_rows + 1)
        except Exception as exc:  # noqa: BLE001 - surfaced to the candidate verbatim
            raise SqlRejected(f"SQL-Fehler: {exc}") from exc

        truncated = len(fetched) > self._limits.max_sql_rows
        rows = tuple(tuple(row) for row in fetched[: self._limits.max_sql_rows])
        return QueryResult(columns=columns, rows=rows, truncated=truncated)

    def close(self) -> None:
        try:
            self._connection.close()
            with self._admin.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA IF EXISTS {self._schema} CASCADE")
        finally:
            self._admin.close()


def open_database(
    snapshot: DatabaseSnapshot,
    *,
    dsn: str | None = None,
    schema: str = "instance",
    limits: ExecutionLimits | None = None,
) -> MaintenanceDatabase:
    """Open the appropriate backend.

    A DSN selects PostgreSQL; without one the snapshot is served from SQLite.
    Both produce identical result sets for the queries this workflow asks for,
    which is what makes the public pack a faithful rehearsal.
    """
    if dsn:
        return PostgresMaintenanceDatabase(dsn, snapshot, schema=schema, limits=limits)
    return SqliteMaintenanceDatabase(snapshot, limits=limits)


def open_sqlite_file(path: str | Path, limits: ExecutionLimits | None = None) -> MaintenanceDatabase:
    """Open a pack-shipped SQLite file read-only.

    Used by miner tooling against the public pack, where the database is a file
    on disk rather than generated in memory.
    """
    uri = f"file:{Path(path).as_posix()}?mode=ro"

    class _FileBacked(SqliteMaintenanceDatabase):
        def __init__(self) -> None:
            self._limits = limits or ExecutionLimits()
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._connection.set_authorizer(self._authorize)

    return _FileBacked()
