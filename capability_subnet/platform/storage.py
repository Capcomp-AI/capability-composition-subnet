"""Artifact and report storage.

Two backends behind one interface: the local filesystem, which is what a
single-host deployment uses, and S3-compatible object storage, which is what a
deployment that wants published artifacts to survive the host uses.

Everything stored here is content-addressed, which makes the interface small —
there is no update, only put and get — and makes replication trivially safe: two
copies of an object with the same key are the same bytes or one of them is
corrupt, and the digest says which.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from capability_subnet.common.hashing import digests_equal, sha256_file

log = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when an object cannot be stored or retrieved intact."""


@dataclass(frozen=True, slots=True)
class StoredObject:
    key: str
    digest: str
    size_bytes: int
    location: str


class ObjectStore(Protocol):
    """Content-addressed storage."""

    def put(self, key: str, path: str | Path) -> StoredObject: ...

    def get(self, key: str, destination: str | Path) -> StoredObject: ...

    def exists(self, key: str) -> bool: ...

    def url_for(self, key: str) -> str: ...


class LocalObjectStore:
    """Filesystem-backed storage."""

    def __init__(self, root: str | Path, *, base_url: str = "") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")

    def _path(self, key: str) -> Path:
        # Two-character shard: a flat directory with tens of thousands of
        # artifacts is slow to list on most filesystems.
        safe = key.replace(":", "_").replace("/", "_")
        return self.root / safe[:2] / safe

    def put(self, key: str, path: str | Path) -> StoredObject:
        source = Path(path)
        if not source.is_file():
            raise StorageError(f"nothing to store at {source}")

        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Write beside the target and rename: a reader that arrives during a copy
        # must never see a partial object under a content-addressed key.
        staging = target.with_suffix(target.suffix + ".partial")
        shutil.copyfile(source, staging)
        staging.replace(target)

        return StoredObject(
            key=key,
            digest=sha256_file(target),
            size_bytes=target.stat().st_size,
            location=str(target),
        )

    def get(self, key: str, destination: str | Path) -> StoredObject:
        source = self._path(key)
        if not source.is_file():
            raise StorageError(f"no object stored under {key}")

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

        return StoredObject(
            key=key,
            digest=sha256_file(target),
            size_bytes=target.stat().st_size,
            location=str(target),
        )

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def url_for(self, key: str) -> str:
        if not self.base_url:
            return f"file://{self._path(key)}"
        safe = key.replace(":", "_").replace("/", "_")
        return f"{self.base_url}/{safe[:2]}/{safe}"


class S3ObjectStore:
    """S3-compatible storage.

    Works against AWS S3 and against the S3-compatible services most operators
    actually use. Credentials come from the environment; nothing here reads or
    writes a credential file, so the engine container can be given a role rather
    than a secret.
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        prefix: str = "",
        public_base_url: str = "",
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.public_base_url = public_base_url.rstrip("/")
        self._endpoint_url = endpoint_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise StorageError(
                    "S3 storage requires boto3. Install it, or use local storage."
                ) from exc
            self._client = boto3.client("s3", endpoint_url=self._endpoint_url)
        return self._client

    def _key(self, key: str) -> str:
        safe = key.replace(":", "_")
        return f"{self.prefix}/{safe}" if self.prefix else safe

    def put(self, key: str, path: str | Path) -> StoredObject:
        source = Path(path)
        if not source.is_file():
            raise StorageError(f"nothing to store at {source}")

        digest = sha256_file(source)
        self.client.upload_file(str(source), self.bucket, self._key(key))

        return StoredObject(
            key=key,
            digest=digest,
            size_bytes=source.stat().st_size,
            location=f"s3://{self.bucket}/{self._key(key)}",
        )

    def get(self, key: str, destination: str | Path) -> StoredObject:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, self._key(key), str(target))
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"could not fetch {key}: {exc}") from exc

        return StoredObject(
            key=key,
            digest=sha256_file(target),
            size_bytes=target.stat().st_size,
            location=f"s3://{self.bucket}/{self._key(key)}",
        )

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception:  # noqa: BLE001
            return False

    def url_for(self, key: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url}/{self._key(key)}"
        return f"s3://{self.bucket}/{self._key(key)}"


def fetch_verified(store: ObjectStore, key: str, destination: str | Path, digest: str) -> Path:
    """Fetch an object and refuse it if the bytes do not match ``digest``.

    Storage is content-addressed, so a mismatch means the object was corrupted or
    replaced. Deleting the local copy rather than leaving it on disk keeps a
    later run from picking up bytes this one already rejected.
    """
    target = Path(destination)
    stored = store.get(key, target)

    if not digests_equal(stored.digest, digest):
        target.unlink(missing_ok=True)
        raise StorageError(
            f"{key} does not match its expected digest: wanted {digest[:19]}…, "
            f"got {stored.digest[:19]}…"
        )
    return target
