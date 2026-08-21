"""Scoring, in two regimes, with no model judging either one.

**Exact match**, for the logic corpus. Every item states the wrapper it wants its
answer in — a fenced block, ``\\boxed{}``, or double square brackets — and carries
the exact answer, so scoring is a string comparison.

**Execution**, for the code corpus. The candidate's program is run against
stdin/stdout cases, which asks whether the code works rather than whether it
looks right. The stronger claim of the two, and the reason a quarter of every
run is drawn from it.

Both are checkable by someone who is not the operator — the property that makes a
published score worth anything.

Extraction tries each wrapper and then the whole reply. Falling back to the whole
reply is deliberate in both regimes: a package that produced the right answer, or
working code, without the requested fence has demonstrated the capability. Format
compliance is measured as its own axis rather than folded into correctness — and
on the same terms in both regimes, so the axis means one thing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE = re.compile(r"```(?:python|json)?\s*(.*?)```", re.S)
_BOXED = re.compile(r"\\boxed\{(.*?)\}", re.S)
_BRACKETS = re.compile(r"\[\[(.*?)\]\]", re.S)


def normalise(text: str) -> str:
    """Whitespace- and quote-insensitive comparison form.

    Deliberately forgiving about layout and quote style, because the corpus is
    inconsistent about both between families, and deliberately unforgiving about
    everything else.
    """
    text = text.strip().strip("`").strip()
    return re.sub(r"\s+", "", text).replace("'", '"').casefold()


def candidate_spans(reply: str) -> list[str]:
    """Every plausible answer span in a reply, best guess first."""
    if not reply:
        return []
    spans: list[str] = []
    for pattern in (_FENCE, _BOXED, _BRACKETS):
        found = pattern.findall(reply)
        if found:
            # The answer is stated last; earlier matches are usually the
            # worked example the prompt itself supplied.
            spans.append(found[-1])
    spans.append(reply)
    return spans


def answered_correctly(reply: str | None, answer: str) -> bool:
    target = normalise(answer)
    if not target or reply is None:
        return False
    for span in candidate_spans(reply):
        candidate = normalise(span)
        if candidate == target or candidate.strip('"[]') == target.strip('"[]'):
            return True
    return False


def used_requested_format(reply: str | None) -> bool:
    """Whether the reply wrapped its answer the way the prompt asked.

    Scored separately from correctness. A package that is right but cannot
    follow an output instruction is a different failure from one that is wrong,
    and a merged package losing format compliance is the single most common way
    merging goes bad — worth its own axis rather than hidden inside a pass/fail.
    """
    if not reply:
        return False
    return bool(_FENCE.search(reply) or _BOXED.search(reply) or _BRACKETS.search(reply))


@dataclass(frozen=True, slots=True)
class StageOutcome:
    score: float
    passed: bool
    detail: str = ""


#: Fenced Python, for pulling a submitted program out of a reply.
_PYTHON_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_program(reply: str | None) -> str:
    """The program a reply submitted, or empty.

    The last fenced block, because a reply commonly restates the problem's
    example before giving its own answer. Falls back to the whole reply, so a
    package that emitted bare code is judged on whether the code runs rather
    than on whether it remembered the fence — compliance is its own axis.
    """
    if not reply:
        return ""
    blocks = _PYTHON_FENCE.findall(reply)
    return (blocks[-1] if blocks else reply).strip()


def _normalise_output(text: str) -> str:
    """Compare stdout the way a judge would: per line, trailing space ignored."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def run_test_cases(program: str, cases, runner) -> tuple[int, int, str]:
    """Run ``program`` against every case. Returns ``(passed, total, detail)``.

    Every case is run even after one *fails*. A partial pass is real information —
    a program correct on the sample and wrong on an edge case is a different
    failure from one that does not run — and a crash is cheap to repeat, so
    collecting it costs nothing over discarding evidence already paid for.

    One exception, and it is a cost bound rather than a scoring choice: a program
    that exhausts its time or memory ceiling gets no further cases. It would
    exhaust the ceiling again on each of them, so continuing cannot change the
    verdict — passing requires every case — while it can multiply the ceiling by
    the case count. Stopping on exhaustion is deterministic in the program and the
    cases, which matters because two validators scoring the same trace must agree.
    """
    from capability_subnet.sandbox.python_runner import EXHAUSTED

    total = len(cases)
    if not program:
        return 0, total, "no program submitted"

    passed = 0
    first_failure = ""
    for index, case in enumerate(cases):
        stdout, detail = runner.run_program(program, case.stdin)
        if stdout is None:
            first_failure = first_failure or f"case {index + 1}: {detail}"
            if detail.startswith(EXHAUSTED):
                return passed, total, f"{first_failure} (stopped after {index + 1} case(s))"
            continue
        if _normalise_output(stdout) == _normalise_output(case.expected_stdout):
            passed += 1
        elif not first_failure:
            first_failure = f"case {index + 1}: wrong output"

    if passed == total:
        return passed, total, f"all {passed} case(s) passed"
    return passed, total, first_failure or "failed"


def score_instance(instance, trace, *, runner=None) -> dict[str, StageOutcome]:
    """Score one answered instance into its per-axis outcomes.

    Two scoring regimes, because the corpora differ in what they can prove. A
    logic item has one exact answer, so it is a string comparison. A code item
    has test cases, so the candidate's program is *executed* — which asks whether
    the code works rather than whether it looks right, and is the stronger claim
    of the two.
    """
    reply = trace.reply
    formatted = used_requested_format(reply)

    if instance.cases:
        if runner is None:
            from capability_subnet.sandbox.python_runner import PythonRunner

            runner = PythonRunner()
        program = extract_program(reply)
        passed, total, detail = run_test_cases(program, instance.cases, runner)
        solved = total > 0 and passed == total
        # Compliance is "used the fence the prompt asked for", the same question
        # the exact-match branch asks. Not `bool(program)`: extraction falls back
        # to the whole reply, so that would score every non-empty reply compliant
        # and the axis would mean something different on each corpus.
        fenced = bool(reply and _PYTHON_FENCE.search(reply))
        return {
            instance.task: StageOutcome(float(solved), solved, detail),
            "format_compliance": StageOutcome(
                float(fenced), fenced, "fenced its program" if fenced else "did not"
            ),
        }

    correct = answered_correctly(reply, instance.answer)
    return {
        instance.task: StageOutcome(
            float(correct), correct, "exact match" if correct else "did not match"
        ),
        "format_compliance": StageOutcome(
            float(formatted), formatted, "used the requested wrapper" if formatted else "did not"
        ),
    }
