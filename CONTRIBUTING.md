# Contributing

---

## Setup

```bash
git clone <repository-url> lora-merger && cd lora-merger
pip install -e ".[dev]"
make test-fast
```

## Before you open a change

```bash
make lint          # ruff
make test          # the full suite
python -m capability_subnet.workflows.cli selftest --count 20
```

That last command is not optional if you touched anything under `workflows/`. The scripted reference solver knows every answer by construction, so anything less than 20/20 means the environment itself is broken and no candidate score from those instances would mean anything.

---

## Things that need more care than usual

### Consensus-critical code

`merge_engine/` and `common/` decide what a recipe reconstructs to. A change that alters an artifact digest changes every champion's identity, invalidates the artifact cache, and breaks the anti-copy check for anything already submitted.

If you touch either:

1. Run `make test-repro` and understand why each test passes.
2. Bump `__version__` — the spec version derives from it and travels with every weight submission.
3. Say plainly in the change description that reconstruction output changed.

The determinism suite covers the failure modes that actually occur in practice: thread-count dependence, global RNG leakage, iteration-order dependence, tensor layout, and singular-vector sign conventions. If you add a numerical operation, add the case that would catch it going wrong.

### The workflow

Changing generation, tools or scoring changes what every historical score meant. Sample rows from before and after a scoring change are not comparable, and the comparator pairs across them.

Adding a stage or changing a threshold is a protocol change, not a fix.

### The failure classification

The engine distinguishes **miner failures** (fail closed, score zero) from **infrastructure failures** (fail open, hold the queue). One shot per hotkey is only defensible because the engine never spends that shot on its own bad night.

If you add a failure path, classify it deliberately and add a case to `tests/integration/test_failure_classification.py`. Getting this wrong is invisible in a normal run and terminates real miners.

---

## Style

The codebase follows a few conventions consistently. Matching them matters more than any individual preference.

**Explain why, not what.** The code says what it does. A comment earns its place by explaining a decision that is not obvious from reading it — why a threshold rather than a top-k selection, why the CPU generator rather than CUDA, why an axis with too few samples counts as worse.

**Docstrings state the contract.** What it does, what the arguments mean where it is not obvious, what it raises and when. Module docstrings explain what the module is *for* and what property it guarantees.

**Name the failure mode in the error.** `f"adapter {adapter_id!r} is not in the certified pool. Available: {...}"` beats `"unknown adapter"`. Somebody is going to read that message at three in the morning.

**Collect every problem, not just the first.** Validation returns a list. A miner fixing a recipe should not have to resubmit once per mistake.

**Prefer explicit construction over filtering.** The visible instance payload is built field by field rather than filtered from the full object, so a new ground-truth field is invisible until someone deliberately exposes it. Filtering fails open; construction fails closed.

Formatting is `ruff` with a 100-column line length. `make fmt` applies it.

---

## Tests

| Suite | Covers |
|---|---|
| `tests/unit/` | Protocol primitives, merge stages, scoring, comparator, weights |
| `tests/integration/` | Sandbox, workflow, tools, failure classification |
| `tests/adversarial/` | Smuggling, substitution, copying, extraction, forged vectors |
| `tests/reproducibility/` | Determinism of reconstruction and instance generation |
| `tests/e2e/` | The whole engine over a miniature pool |

Markers: `gpu`, `docker`, `chain` — skipped by `make test-fast`.

### What a good test looks like here

Write the property, not the implementation. `test_abandoning_one_capability_blocks_a_dethrone` survives a comparator rewrite; `test_compare_axis_returns_worse` does not.

Use the miniature fixtures. `tiny_snapshot`, `tiny_source` and `tiny_pool_dir` give a structurally identical pool at a fraction of the size, which is why the suite runs in under two minutes rather than needing an 8B model.

For scoring and comparator tests, build rows with known statistics using `make_results` rather than deriving them from a real run. Those tests are about arithmetic, and making them depend on model behaviour would make them fail for reasons that have nothing to do with what they check.

---

## Adding a workflow

The engine addresses workflows through a registry, so a second one is an addition rather than a rewrite. It needs:

- an instance generator that is a pure function of a seed,
- a deterministic scorer that reads only a stored trace and the ground truth,
- a tool schema set,
- a published contract,
- an entry in `capability_subnet/workflows/__init__.py`.

The bar for "objective" is specific: **no language model may decide any part of the result.** Every stage must be judged by execution, schema validation, exact comparison against generated truth, or a deterministic rule engine. If a stage needs a model to judge it, it does not belong in a workflow used for paired statistical comparison.

Write the scripted reference solver alongside it. Without one there is no way to know the environment is solvable, and an unsolvable instance produces scores that look real.

---

## Reporting security issues

Report privately to the subnet operator rather than opening a public issue. See [docs/architecture.md](docs/architecture.md#security-model).

Findings affecting scoring integrity — determinism, hidden-material exposure, or anything letting a candidate influence its own evaluation — are the highest priority, because they invalidate results rather than merely disrupting service.
