"""License review for registry admission.

An adapter enters the certified pool only if its license permits redistribution
of derivative weights. The merged artifact the network produces *is* a
derivative of every adapter that went into it, so a non-derivative license makes
the adapter unusable regardless of how good it is.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Licenses that clearly permit redistributing derived weights.
PERMISSIVE = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0-1.0",
        "cc-by-4.0",
        "openrail",
        "bigscience-openrail-m",
    }
)

#: Copyleft-style licenses that permit derivatives but impose obligations the
#: operator must satisfy when publishing artifacts. Admissible, but flagged.
CONDITIONAL = frozenset(
    {
        "cc-by-sa-4.0",
        "gpl-3.0",
        "agpl-3.0",
        "lgpl-3.0",
        "llama3",
        "llama3.1",
        "gemma",
        "qwen",
    }
)

#: Licenses that forbid commercial use or derivative redistribution. Rejected.
BLOCKED = frozenset(
    {
        "cc-by-nc-4.0",
        "cc-by-nc-sa-4.0",
        "cc-by-nc-nd-4.0",
        "research-only",
        "non-commercial",
        "proprietary",
        "unknown",
    }
)


@dataclass(frozen=True, slots=True)
class LicenseVerdict:
    license_id: str
    admissible: bool
    requires_attention: bool
    reason: str


def review(license_id: str, *, declared_allows_derivatives: bool) -> LicenseVerdict:
    """Classify a license for pool admission.

    ``declared_allows_derivatives`` is the submitter's own assertion. It can only
    *narrow* the verdict — a submitter cannot promote a blocked license by
    ticking a box, but they can flag a permissive-looking license as unusable if
    they know of an extra restriction.
    """
    normalised = (license_id or "").strip().lower()

    if not declared_allows_derivatives:
        return LicenseVerdict(
            normalised,
            admissible=False,
            requires_attention=True,
            reason="submitter declared that the license does not permit derivative weights",
        )

    if normalised in BLOCKED:
        return LicenseVerdict(
            normalised,
            admissible=False,
            requires_attention=True,
            reason=f"{license_id!r} forbids commercial use or derivative redistribution",
        )
    if normalised in CONDITIONAL:
        return LicenseVerdict(
            normalised,
            admissible=True,
            requires_attention=True,
            reason=(
                f"{license_id!r} permits derivatives but carries obligations "
                "(attribution, share-alike or upstream terms) that the operator "
                "must satisfy when publishing merged artifacts"
            ),
        )
    if normalised in PERMISSIVE:
        return LicenseVerdict(
            normalised, admissible=True, requires_attention=False, reason="permissive"
        )

    return LicenseVerdict(
        normalised,
        admissible=False,
        requires_attention=True,
        reason=(
            f"{license_id!r} is not on the reviewed list. Add it to the permissive "
            "or conditional set after a legal review rather than assuming it is safe."
        ),
    )
