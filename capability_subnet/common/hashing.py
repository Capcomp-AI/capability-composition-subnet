"""Canonical hashing.

Three things in this subnet are content-addressed and must hash identically on
every machine that touches them:

* the recipe bytes a miner uploads and commits a digest of,
* the frozen certified adapter snapshot,
* the reconstructed adapter artifact.

Canonicalisation is therefore part of the protocol, not an implementation
detail. All JSON is serialised with sorted keys, no insignificant whitespace,
and UTF-8 without a BOM.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

#: Prefix used everywhere a digest is written into a payload or report.
SHA256_PREFIX = "sha256:"

_CHUNK = 1024 * 1024


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialise ``obj`` to the one byte string the protocol considers canonical.

    ``sort_keys`` makes key order irrelevant, the compact separators strip
    insignificant whitespace, and ``ensure_ascii=False`` keeps the German text in
    workflow payloads readable rather than escaping it.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_str(obj: Any) -> str:
    """Human-facing canonical form: sorted keys, two-space indent, trailing newline.

    This is what a miner should write to disk. Hashing always runs over
    :func:`canonical_json_bytes` of the *parsed* document, so pretty-printing on
    disk never changes the digest.
    """
    return (
        json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def sha256_bytes(data: bytes) -> str:
    """Digest of a byte string, prefixed."""
    return SHA256_PREFIX + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Streaming digest of a file, prefixed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return SHA256_PREFIX + digest.hexdigest()


def sha256_json(obj: Any) -> str:
    """Digest of the canonical serialisation of ``obj``, prefixed."""
    return sha256_bytes(canonical_json_bytes(obj))


def sha256_tree(root: str | os.PathLike[str], *, suffixes: tuple[str, ...] | None = None) -> str:
    """Digest of a directory tree.

    The tree hash folds each file's relative POSIX path and its content digest
    into a single stream in sorted path order, so it is stable across
    filesystems and directory-listing order. Optionally restrict to files with
    one of ``suffixes``.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"{root_path} is not a directory")

    entries: list[tuple[str, str]] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        rel = path.relative_to(root_path).as_posix()
        entries.append((rel, sha256_file(path)))

    stream = hashlib.sha256()
    for rel, digest in sorted(entries):
        stream.update(rel.encode("utf-8"))
        stream.update(b"\0")
        stream.update(digest.encode("ascii"))
        stream.update(b"\n")
    return SHA256_PREFIX + stream.hexdigest()


def normalise_digest(value: str) -> str:
    """Accept a digest with or without the ``sha256:`` prefix; return it prefixed.

    Raises:
        ValueError: if the value is not a 64-character lowercase hex digest.
    """
    text = value.strip()
    if text.startswith(SHA256_PREFIX):
        text = text[len(SHA256_PREFIX) :]
    text = text.lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"not a sha256 digest: {value!r}")
    return SHA256_PREFIX + text


def digests_equal(left: str, right: str) -> bool:
    """Prefix-insensitive, constant-time digest comparison."""
    try:
        return hmac.compare_digest(normalise_digest(left), normalise_digest(right))
    except ValueError:
        return False


def short_digest(value: str, length: int = 12) -> str:
    """Shortened digest for log lines. Never used for comparison."""
    try:
        return normalise_digest(value)[len(SHA256_PREFIX) : len(SHA256_PREFIX) + length]
    except ValueError:
        return str(value)[:length]
