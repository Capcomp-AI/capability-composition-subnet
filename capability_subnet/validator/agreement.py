"""Whether two validators measuring the same instances agree.

Six of this protocol's seven merge methods run an SVD, and an SVD is not
bitwise reproducible across devices. Measured on the reference deployment:

    linear     cpu vs cuda : identical digest
    ties_svd   cpu vs cuda : different digest

So two honest validators on different hardware reconstruct *different artifact
bytes from the same recipe*, and score them differently. Any design that demands
byte equality either bans six of the seven methods — including the only family
that has ever beaten the base model here — or brands honest validators as
faulty for owning a different card.

The way out is to stop comparing artifacts and start comparing outcomes, with a
band wide enough for device divergence and narrow enough to catch a validator
that did not do the work. The shared core makes that a *paired* question: both
validators ran the same instances, so sampling noise cancels and what remains is
the divergence itself.

The band is a network parameter, not a fact of nature. The default below is
deliberately conservative and should be replaced with a measured value: run
``observed_discordance`` across validators on real hardware for a few windows
and set the band from the distribution. A band guessed too tight punishes honest
diversity, which is worse than one guessed too loose — a loose band still costs
a cheating validator everything the moment it disagrees on a *result*, because
the core is the same instances and the truth is checkable by replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Fraction of paired core instances two honest validators may resolve
#: differently. Provisional: device divergence has been demonstrated at the
#: artifact level here but not yet quantified at the outcome level, and this
#: number should be set from ``observed_discordance`` once more than one card is
#: in the network. Chosen loose on purpose — see the module docstring.
DEFAULT_DISCORDANCE_BAND: float = 0.10

#: Instances a comparison needs before its verdict is trusted. Below this a
#: single flipped instance moves the rate more than the band is wide.
MIN_PAIRED_INSTANCES: int = 30


@dataclass(frozen=True, slots=True)
class AgreementVerdict:
    """The result of comparing two validators on their shared core."""

    paired: int
    discordant: int
    rate: float
    band: float
    consistent: bool
    detail: str


def observed_discordance(
    a: dict[int, bool],
    b: dict[int, bool],
) -> tuple[int, int]:
    """How often two validators resolved the same instance differently.

    Args:
        a, b: instance seed -> whether that instance was scored a success.

    Returns:
        ``(paired, discordant)`` over the instances both measured. Instances only
        one of them holds are not evidence about either and are excluded.
    """
    shared = a.keys() & b.keys()
    return len(shared), sum(1 for seed in shared if a[seed] != b[seed])


def compare(
    a: dict[int, bool],
    b: dict[int, bool],
    *,
    band: float = DEFAULT_DISCORDANCE_BAND,
    min_paired: int = MIN_PAIRED_INSTANCES,
) -> AgreementVerdict:
    """Decide whether two validators' core results are consistent.

    Consistency is judged against the *upper* confidence bound on the observed
    rate rather than the rate itself, so a validator is not condemned by a small
    sample that happened to land badly. The test is one-sided: being unusually
    *close* to another validator is not a fault.
    """
    paired, discordant = observed_discordance(a, b)
    if paired == 0:
        return AgreementVerdict(0, 0, 0.0, band, True, "no shared instances to compare")

    rate = discordant / paired
    if paired < min_paired:
        return AgreementVerdict(
            paired,
            discordant,
            rate,
            band,
            True,
            f"{paired} shared instances is below {min_paired}; not enough to judge",
        )

    # Wilson-style one-sided upper bound at ~95%. Cheaper than a bootstrap and
    # well behaved at the small rates and sample sizes this sees.
    z = 1.645
    centre = (rate + z * z / (2 * paired)) / (1 + z * z / paired)
    half = (
        z
        * math.sqrt(rate * (1 - rate) / paired + z * z / (4 * paired * paired))
        / (1 + z * z / paired)
    )
    upper = centre + half

    consistent = upper <= band
    detail = (
        f"{discordant} of {paired} shared instances differ "
        f"({rate:.1%}, upper bound {upper:.1%}) against a {band:.1%} band"
    )
    if not consistent:
        detail += " — too far apart to be device divergence"
    return AgreementVerdict(paired, discordant, rate, band, consistent, detail)


def outlier_validators(
    results: dict[str, dict[int, bool]],
    *,
    band: float = DEFAULT_DISCORDANCE_BAND,
    min_paired: int = MIN_PAIRED_INSTANCES,
) -> dict[str, str]:
    """Validators inconsistent with the majority of their peers.

    A validator is only called out when it disagrees with *most* of the others,
    so a network split two ways does not produce two sets of accusations, and a
    single odd participant cannot brand everybody else.

    Returns:
        hotkey -> why, for each validator judged an outlier. Empty when the
        network agrees or when there are too few validators to have a majority.
    """
    hotkeys = sorted(results)
    if len(hotkeys) < 3:
        return {}

    flagged: dict[str, str] = {}
    for who in hotkeys:
        peers = [p for p in hotkeys if p != who]
        verdicts = [
            compare(results[who], results[p], band=band, min_paired=min_paired) for p in peers
        ]
        judged = [v for v in verdicts if v.paired >= min_paired]
        if not judged:
            continue
        disagreed = [v for v in judged if not v.consistent]
        if len(disagreed) * 2 > len(judged):
            worst = max(disagreed, key=lambda v: v.rate)
            flagged[who] = (
                f"inconsistent with {len(disagreed)} of {len(judged)} peers: {worst.detail}"
            )
    return flagged
