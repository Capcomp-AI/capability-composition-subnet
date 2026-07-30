# Arena reference — LoRA Merger Logic

The default workflow. A single-turn arena over two pinned corpora: reasoning
puzzles scored by exact match, and programming problems scored by **running the
submitted program**.

```bash
python -m capability_subnet.workflows.cli show --workflow lora_merger_logic_v1 --seed 42
```

---

## What it measures

Whether a merged package beats its constituents, with as little between the
answer and the measurement as possible:

| Property | How |
|---|---|
| One turn | No state, no tool loop, no partial-credit chain to argue about |
| Pre-measured difficulty | Items carry a pass rate, so selection targets the band where packages actually differ |
| No model judges anything | Exact match, or the program runs and its stdout is compared |
| Paired | Every package sees the identical item set for a given window |
| Twelve axes | Ten logic families, code execution, format compliance |

Workflows register through the `capability_subnet.workflows` entry-point group.
None is hardcoded into the evaluator; `workflow_id` in `backend.yaml` selects
which runs.

---

## The corpora

Both are pinned to a revision. A score that cannot say what it was measured on
is not reproducible.

| Source | Items used | Scoring |
|---|---|---|
| `AffineFoundation/affine-lgc` @ `19765edac477` | ~3,193 across 10 families | exact match |
| `AffineFoundation/rl-python` @ `0cc711b1f059` | ~3,920 problems | execution |

A quarter of every window is drawn from the code corpus (`CODE_FRACTION`).
Execution is the stronger signal — it asks whether the code works rather than
whether the answer looks right — but every case is a subprocess, so it is
deliberately neither a tenth of the board nor all of it.

### Item selection

Logic items are kept only when the corpus's own `avg@16_qwen3_4b_instruct_2507`
column falls in **0.20–0.80**. Outside that band an item carries no information:
too easy and every package ties, too hard and every package scores zero.
Code problems are admitted at the `introductory` and `interview` tiers on the
same reasoning.

Selection is stratified by family and deterministic in the seed, so a window
cannot be dominated by whichever family the corpus is densest in — or a miner
happened to study.

### Limits

Two limits apply, and a miner needs both to decide whether to compete.

**The items are public.** What the hidden seed protects is *which* items a window
draws, not the items themselves — a weaker anti-overfitting property than a
generated workflow, and the price of corpora whose difficulty is already
measured. Corpus size is not the mitigation and is not offered as one. What
applies instead: stratified selection, the executed quarter of each window, and a
general-capability probe scored outside these corpora entirely.

**The difficulty labels overstate this harness.** They were measured at pass@16
with sampling. This engine scores pass@1, greedy, with the reasoning channel
disabled. Expect absolute scores well below the band.

---

## Scoring

### Logic items

Exact match after whitespace and quote normalisation. Extraction tries the
wrapper the prompt requested — a fenced block, `\boxed{}`, or `[[...]]` — then
falls back to the whole reply, so a package that produced the right answer
without the fence is not marked wrong for a formatting slip.

### Code items

The last fenced block is taken as the submitted program and run against
stdin/stdout cases in the same isolated interpreter the V1 workflow uses:
`-I -S`, no inherited environment, memory and CPU ceilings, a wall-clock timeout.
An instance passes only if **every** case passes.

Every case runs even after one fails — a program correct on the sample and wrong
on an edge case is different information from one that does not run. One
exception, which is a cost bound rather than a scoring choice: a program that
exhausts its time or memory ceiling gets no further cases. It would exhaust the
ceiling on each of them, passing already requires all of them, and continuing
would multiply the ceiling by the case count.

Cases are capped at **32 per problem**, applied at load time so the retained
prefix travels with the instance and a replay scores the same cases. The cap
bounds cost: the median problem carries 9 cases and the worst carries 565, and
each case is a subprocess with a 20-second ceiling.

### Format compliance

Its own axis rather than part of correctness, on both corpora and on the same
terms: did the reply use the wrapper the prompt asked for. A package that is
right but can no longer follow an output instruction has failed differently from
one that is wrong, and losing compliance is the most common way an
over-aggressive merge goes bad.

One caveat on that axis, since it is measurable: families whose answer is itself
a bracketed list — `arrow_maze`, the word-sorting pair — are compliant by
construction, because the answer's own shape is the requested wrapper. It carries
information mainly on the scalar families and on code.

---

## Reproducing a window

Everything an auditor needs is in the published record. A seed regenerates the
exact instance, including its test cases; every revision is pinned, so the items
cannot change underneath.

Every validator does this automatically: each round, `spot_check_window` pulls
the previous window's disclosure and re-scores it from the published traces,
reporting any disagreement with the operator's numbers. Because this workflow is
`publicly_verifiable`, that check is meaningful here — an operator cannot publish
a score the traces do not support.

To do it by hand:

```python
from capability_subnet.validator.client import BackendClient, spot_check_window

passed, detail = spot_check_window(BackendClient(...), window_id)
```

See [docs/validator.md](validator.md).
