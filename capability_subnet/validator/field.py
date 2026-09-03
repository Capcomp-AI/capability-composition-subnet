"""Fetching a run's field of submissions, as a validator.

The chain used to carry the field: a miner committed a digest and a pointer,
and any validator could read both without asking anyone. It does not any more —
miners submit to the submission service — so a validator that still reads
commitments measures an empty subnet and burns every run.

**A run's bodies are readable only once it is public**, two runs after
submission. There is no early-access route and no credential that opens one:
the same rule applies to a validator, a miner and an auditor alike.

What that costs is worth naming, because it decides what this module can be
used for. A validator cannot read the field of the run it is measuring — those
bodies arrive at N+2, after that run has been paid — so it cannot derive that
run's weight vector from the submissions themselves in time. What it can do is
measure a published run and compare its own numbers against the archive, which
is verification after the fact rather than in the paying window.

Every body is checked against the digest the row carries before it becomes a
candidate. The service could serve a recipe that is not the one the miner
signed; it could not serve one that hashes to the digest stored beside it.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from capability_subnet.common import constants as C
from capability_subnet.common.chain import measured_in_run

log = logging.getLogger(__name__)


class FieldError(RuntimeError):
    """The field could not be fetched, or could not be trusted once it was."""


@dataclass(slots=True)
class FetchedSubmission:
    """One submission, verified against the digest it was stored under."""

    hotkey: str
    uid: int
    recipe_sha256: str
    recipe_raw: bytes
    first_block: int
    submission_count: int


def field_for_run(
    api_url: str,
    measuring_run: int,
    *,
    run_blocks: int = C.DEFAULT_RUN_BLOCKS,
    timeout: float = 60.0,
) -> list[FetchedSubmission]:
    """Everything ``measuring_run`` is supposed to measure.

    Two source runs, not one. A submission is measured in the run after the one
    it was made in — unless it was made inside the settling window, in which
    case it is held over one further run. So this run's field is the settled
    part of run N-1 plus the late part of N-2, and a validator that fetched only
    N-1 would leave the held-over miners unmeasured and unpaid while their rows
    sat in the service looking submitted.

    :func:`measured_in_run` decides which is which, from the submission block
    alone. Every validator reads the same blocks and selects the same field, so
    there is nothing here to disagree about.

    Both source runs must be public for this to return a complete field. When
    ``measuring_run`` is the run currently being measured, N-1 is not — it
    becomes public at N+1 — and this raises rather than returning the N-2 half
    alone. A field short of the run it is supposed to measure is one a caller
    would rank and pay from, with the missing miners looking like miners who
    never submitted.
    """
    seen: dict[str, FetchedSubmission] = {}
    for source in (measuring_run - 2, measuring_run - 1):
        if source < 0:
            continue
        for entry in fetch_run(api_url, source, timeout=timeout):
            if not measured_in_run(entry.first_block, measuring_run, run_blocks):
                continue
            # A hotkey appearing in both source runs submitted twice; the row
            # this run measures is the one whose block says so, and both have
            # already been filtered to exactly that.
            seen[entry.hotkey] = entry
    out = sorted(seen.values(), key=lambda e: e.first_block)
    log.info(
        "run %d: field of %d, drawn from runs %d and %d",
        measuring_run,
        len(out),
        measuring_run - 2,
        measuring_run - 1,
    )
    return out


def fetch_run(
    api_url: str,
    run_id: int,
    *,
    timeout: float = 60.0,
) -> list[FetchedSubmission]:
    """Every submission committed in ``run_id``, verified.

    Raises :class:`FieldError` rather than returning a short field. A field that
    is quietly missing the entries whose bodies failed to verify is a field the
    caller would rank and pay from, and the miners left out would look like
    miners who never submitted.
    """
    import httpx

    try:
        response = httpx.get(f"{api_url.rstrip('/')}/run/{run_id}/submissions", timeout=timeout)
    except Exception as exc:
        raise FieldError(f"could not reach the submission service: {exc}") from exc

    if response.status_code in (401, 403):
        # The run is not public yet. Not a permissions problem to work around:
        # nothing opens it early, so the reason is the whole answer.
        raise FieldError(f"run {run_id} is not published yet: {_detail(response)}")
    if response.status_code == 404:
        raise FieldError(f"run {run_id} has not closed: {_detail(response)}")
    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise FieldError(f"the submission service answered badly: {exc}") from exc

    out: list[FetchedSubmission] = []
    for entry in payload.get("submissions") or []:
        raw = entry.get("recipe_raw")
        digest = entry.get("recipe_sha256")
        who = str(entry.get("hotkey") or "?")
        if not raw or not digest:
            raise FieldError(f"{who[:12]}… came back without a body or a digest")
        body = raw.encode() if isinstance(raw, str) else bytes(raw)
        actual = "sha256:" + hashlib.sha256(body).hexdigest()
        if actual != digest:
            # The bytes are not the bytes that digest names. Whatever produced
            # that, it is not something to measure and pay.
            raise FieldError(f"{who[:12]}… served a body hashing to {actual}, stored as {digest}")
        out.append(
            FetchedSubmission(
                hotkey=who,
                uid=int(entry.get("uid") or 0),
                recipe_sha256=digest,
                recipe_raw=body,
                first_block=int(entry.get("first_block") or 0),
                submission_count=int(entry.get("submission_count") or 1),
            )
        )
    log.info("run %d: fetched %d submissions, every body matching its digest", run_id, len(out))
    return out


def _detail(response) -> str:
    try:
        return str(response.json().get("detail") or response.text)
    except Exception:
        return response.text[:200]
