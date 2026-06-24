"""The engine client.

A validator's entire trust decision happens here. It fetches a weight vector,
then answers three questions before anything touches the chain:

* is it signed by an operator this validator was configured to trust?
* is it internally consistent — do the weights sum to one, are the UIDs distinct,
  is the champion actually on the metagraph?
* is it recent enough to still mean something?

A vector that fails any of these is not submitted. That is the whole reason a
centralised evaluation engine is acceptable: the engine computes, but validators
decide, and each one answers to its own stake.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from capability_subnet.common.schemas import EvaluationReport, WeightVector
from capability_subnet.common.signing import SignatureError, require_trusted_signature

log = logging.getLogger(__name__)


class BackendUnavailable(Exception):
    """Raised when the engine cannot be reached or returns nothing usable."""


@dataclass(frozen=True, slots=True)
class ValidationProblem:
    """One reason a vector was refused."""

    code: str
    detail: str


class BackendClient:
    """Reads the engine's published state."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        trusted_signers: set[str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.trusted_signers = trusted_signers

        if trusted_signers is None:
            log.warning(
                "no operator allow-list configured. This validator will accept an "
                "unsigned weight vector from anything answering at %s, which is only "
                "safe for local development.",
                self.base_url,
            )

    def _get(self, path: str) -> dict:
        import httpx

        url = f"{self.base_url}{path}"
        try:
            response = httpx.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"could not read {url}: {exc}") from exc

    def health(self) -> dict:
        return self._get("/health")

    def window_blocks(self) -> int | None:
        """The engine's configured window length.

        Staleness is measured in windows, so computing it from a compiled-in
        default would be wrong for any deployment that tuned the window — either
        accepting vectors far past their useful life, or rejecting fresh ones.
        Returns ``None`` if the engine does not report it, and the caller then
        declines to judge staleness rather than guessing.
        """
        try:
            value = self.health().get("window_blocks")
        except BackendUnavailable:
            return None
        try:
            blocks = int(value)
        except (TypeError, ValueError):
            return None
        return blocks if blocks > 0 else None

    def fetch_weights(self) -> WeightVector:
        """Fetch and verify the latest published weight vector.

        Raises:
            BackendUnavailable: if the engine is unreachable or the payload is
                not a valid vector.
            SignatureError: if the vector is unsigned or signed by an operator
                this validator does not trust.
        """
        payload = self._get("/weights")

        try:
            vector = WeightVector.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"malformed weight vector: {exc}") from exc

        require_trusted_signature(vector, self.trusted_signers)
        return vector

    def fetch_report(self, report_sha256: str) -> EvaluationReport:
        payload = self._get(f"/reports/{report_sha256}")
        try:
            report = EvaluationReport.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"malformed report: {exc}") from exc

        require_trusted_signature(report, self.trusted_signers)
        return report

    def fetch_champion(self) -> dict:
        return self._get("/champion")


def validate_vector(
    vector: WeightVector,
    *,
    metagraph_size: int,
    hotkeys: list[str],
    current_window: int | None,
    max_stale_windows: int,
    burn_uid: int,
) -> list[ValidationProblem]:
    """Check a fetched vector against the chain the validator can see.

    A signature proves the engine produced the vector. It does not prove the
    vector is submittable: a champion may have deregistered since it was
    computed, the engine may have stalled, or the vector may reference a UID this
    subnet does not have. Those are the validator's problem to catch, because the
    chain will reject the extrinsic and the validator will have burned a
    transaction finding out.
    """
    problems: list[ValidationProblem] = []

    if not vector.entries:
        problems.append(ValidationProblem("empty", "the vector contains no entries"))
        return problems

    total = sum(entry.weight for entry in vector.entries)
    if abs(total - 1.0) > 1e-6:
        problems.append(
            ValidationProblem("not_normalised", f"weights sum to {total:.6f}, expected 1.0")
        )

    uids = [entry.uid for entry in vector.entries]
    if len(set(uids)) != len(uids):
        problems.append(ValidationProblem("duplicate_uid", f"repeated UIDs in {uids}"))

    out_of_range = [uid for uid in uids if uid < 0 or uid >= metagraph_size]
    if out_of_range:
        problems.append(
            ValidationProblem(
                "uid_out_of_range",
                f"UIDs {out_of_range} are outside this subnet's {metagraph_size} slots",
            )
        )

    if current_window is None:
        # The engine did not report its window length, so staleness cannot be
        # computed. Say so rather than assuming freshness — a validator that
        # silently skipped this check would keep paying a champion through an
        # engine outage, which is the exact failure the check exists for.
        problems.append(
            ValidationProblem(
                "staleness_unknown",
                "the engine did not report its window length, so the vector's "
                "freshness cannot be established",
            )
        )
    else:
        staleness = current_window - vector.window_id
        if staleness > max_stale_windows:
            problems.append(
                ValidationProblem(
                    "stale",
                    f"the vector is {staleness} windows behind the chain head, above the "
                    f"{max_stale_windows}-window tolerance. The engine may have stalled.",
                )
            )

    # A champion that deregistered leaves its UID occupied by someone else. Paying
    # that UID would pay a stranger for the champion's work.
    for entry in vector.entries:
        if entry.role == "burn" or entry.uid == burn_uid or not entry.hotkey:
            continue
        if entry.uid >= len(hotkeys):
            continue
        if hotkeys[entry.uid] != entry.hotkey:
            problems.append(
                ValidationProblem(
                    "hotkey_mismatch",
                    f"UID {entry.uid} is now {hotkeys[entry.uid][:12]}… but the vector "
                    f"expects {entry.hotkey[:12]}…; the champion appears to have "
                    "deregistered",
                )
            )

    return problems


def safe_fallback(burn_uid: int, vector: WeightVector) -> WeightVector:
    """The vector to submit when a fetched one cannot be trusted.

    Burning is the correct fallback rather than repeating the last known-good
    vector: a stale champion collecting emission indefinitely because the engine
    died is worse for the network than nobody collecting it.
    """
    from capability_subnet.common.schemas import WeightEntry

    return vector.model_copy(
        update={
            "entries": [WeightEntry(uid=burn_uid, hotkey="", weight=1.0, role="burn")],
            "champion_hotkey": None,
            "signature": None,
            "signer_hotkey": None,
        }
    )


__all__ = [
    "BackendClient",
    "BackendUnavailable",
    "SignatureError",
    "ValidationProblem",
    "safe_fallback",
    "validate_vector",
]
