"""The general-capability retention probe.

A merged package can score well on the workflow and still be a worse model:
aggressive merging is exactly the operation that trades general instruction
following for task-specific behaviour. The retention gate exists to catch that,
and it can only catch it by asking the candidate something the workflow does not.

The previous implementation compared the candidate's *workflow* completion with
the base model's. That could not work as a safeguard. A challenger has to clear
the strongest reference by an absolute margin before it is ever compared, and the
base model is one of those references — so a candidate that reached the gate had
already beaten the base, the ratio was always above one, and the clamp reported
exactly ``1.0`` every time. A gate whose answer does not depend on its input is
not a gate.

The probe here is a held-out set of short, deterministically-scored instructions
covering the general behaviours a merge is most likely to destroy: following an
explicit format, refusing to pad an answer, arithmetic, ordering, and staying in
the language it was addressed in. Every item is checked by exact string or
numeric comparison, never by a model, for the same reason the workflow scorer is:
a judged probe would make the gate's verdict depend on a second model's mood.

Items are drawn from the window seed, so a candidate cannot tune to a fixed probe
set, and the base model is measured on the same draw in the same window — the
comparison is paired, like every other comparison in this engine.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from capability_subnet.common import constants as C

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProbeItem:
    """One general-capability question and its exact expected answer."""

    probe_id: str
    prompt: str
    expected: str
    #: What this item is protecting. Reported per-item so an operator can see
    #: *which* general behaviour a candidate destroyed, not merely that it did.
    behaviour: str

    def matches(self, reply: str | None) -> bool:
        """Whether ``reply`` is the expected answer.

        Compared on the stripped, case-folded text. Deliberately forgiving about
        surrounding whitespace and case, and deliberately unforgiving about
        everything else: a package that answers "42" when asked for "42" has
        followed the instruction, and one that answers "The result is 42!" has
        not followed the "answer with the number alone" it was given.
        """
        if reply is None:
            return False
        return reply.strip().casefold() == self.expected.strip().casefold()


def _arithmetic(rng: random.Random, index: int) -> ProbeItem:
    left = rng.randint(120, 990)
    right = rng.randint(12, 99)
    return ProbeItem(
        probe_id=f"probe-arith-{index:03d}",
        prompt=(
            f"Compute {left} + {right}. Answer with the resulting number alone, "
            "with no words, no punctuation and no explanation."
        ),
        expected=str(left + right),
        behaviour="arithmetic under a strict output instruction",
    )


def _ordering(rng: random.Random, index: int) -> ProbeItem:
    values = rng.sample(range(10, 99), 5)
    return ProbeItem(
        probe_id=f"probe-order-{index:03d}",
        prompt=(
            f"Sort these numbers ascending: {', '.join(str(v) for v in values)}. "
            "Answer with the sorted numbers separated by single commas and nothing else."
        ),
        expected=",".join(str(v) for v in sorted(values)),
        behaviour="ordering under an exact format",
    )


def _extraction(rng: random.Random, index: int) -> ProbeItem:
    words = rng.sample(
        ["turbine", "bearing", "gasket", "flange", "impeller", "coupling", "seal", "rotor"], 4
    )
    target = max(words, key=len)
    return ProbeItem(
        probe_id=f"probe-extract-{index:03d}",
        prompt=(
            f"Which of these words is the longest: {', '.join(words)}? Answer with that word alone."
        ),
        expected=target,
        behaviour="selection under a one-word instruction",
    )


def _format_only(rng: random.Random, index: int) -> ProbeItem:
    token = "".join(rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ") for _ in range(6))
    return ProbeItem(
        probe_id=f"probe-echo-{index:03d}",
        prompt=(
            f"Repeat the following token exactly once, with nothing before or after it: {token}"
        ),
        expected=token,
        behaviour="verbatim reproduction without embellishment",
    )


def _language(rng: random.Random, index: int) -> ProbeItem:
    pairs = [
        ("Wie lautet die deutsche Übersetzung von 'pump'? Antworte nur mit dem Wort.", "pumpe"),
        ("Wie lautet die deutsche Übersetzung von 'valve'? Antworte nur mit dem Wort.", "ventil"),
        ("Wie lautet die deutsche Übersetzung von 'pressure'? Antworte nur mit dem Wort.", "druck"),
        ("Wie lautet die deutsche Übersetzung von 'bearing'? Antworte nur mit dem Wort.", "lager"),
    ]
    prompt, expected = rng.choice(pairs)
    return ProbeItem(
        probe_id=f"probe-lang-{index:03d}",
        prompt=prompt,
        expected=expected,
        behaviour="answering in the language it was addressed in",
    )


#: The generators, in a fixed order. Adding one changes what the gate measures
#: and is therefore a consensus change.
GENERATORS = (_arithmetic, _ordering, _extraction, _format_only, _language)


def build_probe(seed: int, count: int = C.RETENTION_PROBE_ITEMS) -> list[ProbeItem]:
    """Draw a probe set from ``seed``.

    Deterministic in the seed, so an auditor replaying a closed window
    regenerates exactly the items the candidate faced — the same property the
    workflow instances have, for the same reason.
    """
    rng = random.Random(seed)
    items: list[ProbeItem] = []
    for index in range(count):
        generator = GENERATORS[index % len(GENERATORS)]
        items.append(generator(rng, index))
    return items


@dataclass(slots=True)
class ProbeOutcome:
    """How a package did on the probe."""

    correct: int = 0
    total: int = 0
    #: Behaviours the package got wrong, for the report.
    failed_behaviours: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failed_behaviours is None:
            self.failed_behaviours = []

    @property
    def score(self) -> float:
        return self.correct / self.total if self.total else 0.0


def run_probe(client, items: list[ProbeItem], *, max_tokens: int = 32) -> ProbeOutcome:
    """Ask one served package every probe item.

    Args:
        client: a :class:`~capability_subnet.sandbox.model_client.ModelClient`.
        max_tokens: small on purpose. Every expected answer is a handful of
            characters, and a package that needs more than this to produce one
            has already failed the instruction the item is testing.

    Returns:
        The outcome. An item whose request failed is dropped from the total
        rather than counted wrong — a serving hiccup is not a capability loss.
    """
    from capability_subnet.sandbox.model_client import ModelMessage

    outcome = ProbeOutcome()
    for item in items:
        reply = client.complete(
            [ModelMessage(role="user", content=item.prompt)],
            [],
            seed=abs(hash(item.probe_id)) % (2**31),
            max_tokens=max_tokens,
        )
        if not reply.ok:
            log.debug("probe %s could not be asked: %s", item.probe_id, reply.error)
            continue

        outcome.total += 1
        if item.matches(reply.content):
            outcome.correct += 1
        else:
            outcome.failed_behaviours.append(item.behaviour)

    return outcome


def relative_retention(candidate: ProbeOutcome, base: ProbeOutcome) -> float:
    """The candidate's probe score as a fraction of the base model's.

    Capped at 1: a package that answers *more* probe items than the base has
    lost nothing, and the gate is about loss.

    A base model that scores zero gives no signal, and the conservative reading
    is full retention — the gate exists to catch destroyed ability, and there is
    nothing to destroy in a baseline that answered nothing correctly.
    """
    if base.total == 0 or base.score <= 0.0:
        log.warning("the base model scored nothing on the probe; retention is unmeasurable")
        return 1.0
    if candidate.total == 0:
        return 0.0
    return min(1.0, candidate.score / base.score)
