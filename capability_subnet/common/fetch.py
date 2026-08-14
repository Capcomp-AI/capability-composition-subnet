"""Fetching recipe bytes.

In the public package because every validator needs it. When one operator did
the evaluating, only that operator resolved pointers; a network where each
validator measures for itself has each of them fetching the same commitment
independently, and they must all apply the same rules to the same bytes.

A commitment carries a digest and a pointer. The pointer can be anything — a
Hugging Face file, an IPFS object, a plain URL — because the pointer is not
trusted. What makes the fetch safe is that the bytes are verified against the
on-chain digest before they are parsed, so a mutable pointer, a hijacked host or
a substituted file all fail the same way: the digest does not match, and the
submission is rejected.

The only real risks left are resource ones, and both are bounded here: the
response is size-capped so a hostile pointer cannot exhaust memory, and the
request is time-capped so a slow one cannot stall the queue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from capability_subnet.common.hashing import digests_equal, sha256_bytes

log = logging.getLogger(__name__)

#: A recipe is a small JSON document. Anything approaching this size is not one.
MAX_RECIPE_BYTES = 256 * 1024

#: Public gateway used to resolve content-addressed pointers. The digest check
#: makes an untrusted gateway harmless.
IPFS_GATEWAY = "https://ipfs.io/ipfs/"

HF_RESOLVE = "https://huggingface.co/{owner}/{repo}/resolve/main/{path}"


class FetchError(Exception):
    """Raised when recipe bytes cannot be retrieved or do not match the digest."""


@dataclass(frozen=True, slots=True)
class FetchedRecipe:
    uri: str
    resolved_url: str
    raw: bytes
    sha256: str


def resolve_uri(uri: str) -> str:
    """Turn a commitment pointer into an HTTPS URL."""
    text = uri.strip()

    if text.startswith("https://"):
        return text

    if text.startswith("ipfs:"):
        return IPFS_GATEWAY + text[len("ipfs:") :]

    if text.startswith("hf:"):
        parts = text[len("hf:") :].split("/", 2)
        if len(parts) != 3:
            raise FetchError(
                f"malformed Hugging Face pointer {uri!r}; expected hf:<owner>/<repo>/<path>"
            )
        owner, repo, path = parts
        return HF_RESOLVE.format(owner=owner, repo=repo, path=path)

    raise FetchError(f"unsupported pointer scheme in {uri!r}")


def fetch_recipe(uri: str, expected_sha256: str, *, timeout: float = 30.0) -> FetchedRecipe:
    """Retrieve and verify the recipe bytes a commitment points at.

    Raises:
        FetchError: on any transport failure, an oversized response, or a digest
            mismatch. Every one of these is the miner's problem to fix, and the
            message says which.
    """
    import httpx

    url = resolve_uri(uri)

    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
            response.raise_for_status()

            declared = response.headers.get("content-length")
            if declared is not None and int(declared) > MAX_RECIPE_BYTES:
                raise FetchError(
                    f"{uri} declares {declared} bytes, above the {MAX_RECIPE_BYTES}-byte limit"
                )

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > MAX_RECIPE_BYTES:
                    # Stop reading rather than finishing the download: a hostile
                    # pointer serving an endless stream must cost us one buffer,
                    # not the whole response.
                    raise FetchError(
                        f"{uri} exceeded the {MAX_RECIPE_BYTES}-byte limit while downloading"
                    )
                chunks.append(chunk)

    except FetchError:
        raise
    except Exception as exc:
        raise FetchError(f"could not fetch {uri} ({url}): {exc}") from exc

    raw = b"".join(chunks)
    actual = sha256_bytes(raw)

    if not digests_equal(actual, expected_sha256):
        raise FetchError(
            f"{uri} does not match the on-chain commitment: committed "
            f"{expected_sha256[:19]}…, fetched bytes hash to {actual[:19]}…"
        )

    return FetchedRecipe(uri=uri, resolved_url=url, raw=raw, sha256=actual)
