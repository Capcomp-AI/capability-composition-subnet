<div align="center">

<img src="assets/logo.png" alt="LoRA Merger" width="120" height="120">

# LoRA Merger

### The Capability Composition Subnet

**Which combination of existing specialist adapters actually finishes a real business workflow — and can it beat every off-the-shelf alternative?**

[Quick start](#quick-start) · [How it works](#how-it-works) · [Miner guide](docs/miner.md) · [Validator guide](docs/validator.md) · [Architecture](docs/architecture.md)

</div>

---

## The problem

The pieces of applied AI are all available. There are capable open base models, thousands of task-specific LoRA adapters, agent frameworks, tool APIs, evaluation libraries and serving systems.

What is still unsolved is the **composition decision**. For one real workflow: which adapters help, which are redundant, which actively conflict, what coefficients and per-layer weights to use, how much of each delta to prune, how aggressively to compress — and whether one static package can replace a runtime routing system while actually *improving* end-to-end completion.

A company does not buy "a translation adapter" or "a SQL adapter." It buys a system that finishes a business process. This subnet therefore optimises the finished process, not isolated benchmark scores.

## What miners actually submit

One JSON document. Not a model, not weights, not code.

```jsonc
{
  "workflow_id": "industrial_maintenance_de_v1",
  "base_revision": "<pinned commit>",
  "source_snapshot_sha256": "sha256:...",
  "selected_adapters": ["embedded-engineering-v1", "industrial-ifc-v1", "..."],
  "merge": { "combination_type": "dare_ties_svd", "density": 0.35,
             "majority_sign_method": "total", "random_seed": 937152 },
  "global_weights": { "industrial-ifc-v1": 1.20, "constrained-selection-v1": 1.15 },
  "layer_group_overrides": { "group_2": { "industrial-ifc-v1": 1.30 } },
  "compression": { "output_rank": 64, "svd_clamp_quantile": 0.99 }
}
```

Everything a recipe can express is a bounded number, a name drawn from a frozen registry, or an enum. That is what makes it safe to reconstruct inside the evaluation engine without ever executing anything a miner wrote.

A miner searches the composition space **privately** — any method, any hardware, no reporting. The network judges the artifact, not the research process.

## How it works

```
MINER          commits one recipe digest on-chain, publishes the bytes off-chain
  │
ENGINE         admit      → validate, verify the digest, check the frozen pool, anti-copy
  │            reconstruct→ deterministic merge, cached by artifact hash
  │            evaluate   → serve the package, run the fixed agent on hidden instances
  │            compare    → head-to-head with the champion and the reference baselines
  │            publish    → signed evaluation report + signed weight vector
  │
VALIDATOR      fetch weights → verify the signature → check against the chain → set_weights
  │
CHAIN          identity, commitments, consensus, emission
```

### Continuous champion-challenge

There is no round ceremony. A champion holds the throne continuously; challengers are drawn from a queue **in commit order** and evaluated one at a time. Nobody chooses who challenges next — the chain does, by the order it accepted commitments.

Every window (~24 h) the engine draws a **fresh set of hidden instances** and re-measures the champion and every baseline on them. Nobody defends on data they have already been scored on, and nothing needs to be revealed, because the instances did not exist in observable form before the window opened.

### Taking the throne is hard on purpose

A challenger must clear four independent bars:

| Bar | Requirement |
|---|---|
| **Per-axis dominance** | Clearly better on at least the required number of capability axes |
| **No abandoned capability** | Not worse on *every* remaining axis |
| **Absolute margin** | Beat the **strongest permanent reference** by 3 points of completion |
| **Defender's advantage** | Beat the incumbent by a further margin that decays to zero over ~30 days |
| **Statistical significance** | Paired bootstrap lower confidence bound above zero on shared instances |

The second bar stops a package trading away one capability for a better average.

The third bar is **off by default** (`require_beat_reference`): the highest score on the board is paid whether or not it cleared the strongest reference. References are still measured and published every window. Enable it for the stricter contract.

The incumbent is not one of the permanent references. It gets its own smaller margin, which decays to zero over roughly thirty days.

**One shot per hotkey.** A decisive loss terminates the challenger permanently. Failures that indict the engine — an unreadable memory counter, too few scored instances — hold the candidate for a later window instead of ending it.

### Losing well is worth something

Almost every submission will fail to take the top slot, and a recipe is one shot. So the top slot is winner-takes-most and everything below it is **graded**:

| Term | Weight | What it rewards |
|---|---|---|
| Quality | 50% | The qualified score — completion, stage balance, OOD, retention, latency, tokens, size |
| Improvement | 25% | How far past the strongest non-learned reference it got |
| Proximity | 15% | How close it came to the champion — a near miss is not a wasted registration |
| Cost | 10% | Token spend and latency, because two packages that finish equally are not equally valuable |

Only packages that cleared **every hard gate** are graded. If nobody qualifies the graded pool burns. Every grade is published broken into its four terms.

Miners still waiting in the queue earn a small tapered share, so an unevaluated challenger is not the first thing the chain prunes.

## Two arenas

A package is judged by a **workflow**, and the engine does not hardcode one. Workflows register through the `capability_subnet.workflows` entry-point group and `workflow_id` in `backend.yaml` selects which runs.

| Workflow | What it is | Role |
|---|---|---|
| `lora_merger_logic_v1` | Single-turn reasoning puzzles and execution-verified programming problems, twelve axes — [reference](docs/arena.md) | The default |
| `industrial_maintenance_de_v1` | A twelve-turn German maintenance chain, seven dependent axes — [reference](docs/arena.md#the-maintenance-workflow) | A multi-turn agent workflow |

### `lora_merger_logic_v1` — the arena

Two pinned corpora. ~3,193 logic puzzles across ten families, scored by exact match against the answer the prompt asked for. ~2,944 competitive-programming problems, scored by **running the submitted program** against stdin/stdout cases in an isolated interpreter — a quarter of every window, because execution asks whether the code works rather than whether the answer looks right.

Items are selected in the band where a model of this class discriminates — the corpus carries its own measured pass rate — stratified by family, and deterministic in a hidden seed.

Two limits apply. The corpora are public, so the seed protects *which* items a window draws rather than the items themselves. And the difficulty labels were measured at pass@16 with sampling while this engine scores pass@1 greedy, so absolute scores land below the band. See [docs/arena.md](docs/arena.md).

### `industrial_maintenance_de_v1` — the demonstration

A German industrial-maintenance agent works one fault from a controller log to a signed-off replacement decision:

```
manual interpretation → fault extraction → maintenance SQL → diagnostic Python
    → inventory action → safety validation → strict final JSON
```

Seven scored capability axes in one dependent chain — manual interpretation, fault extraction, maintenance SQL, diagnostic Python, inventory action, safety validation, strict final JSON. Later steps consume earlier outputs, so it cannot be decomposed into independent benchmark questions.

**No language model decides the result.** Manual facts come from generator metadata. Fault codes come from a deterministic machine schema. SQL is judged by executing it against a hidden PostgreSQL snapshot. Python is judged by hidden test cases the agent never sees. Inventory is judged by the simulator's final state, safety by a deterministic rule engine, and the final report by JSON Schema plus exact value comparison.

Determinism is what makes the paired statistics valid and lets a disputed evaluation be replayed later with the same answer.

## Two repositories

| Repository | What it holds | Who needs it |
|---|---|---|
| **this one** | The protocol: recipe contract, merge engine, workflows, and every rule that decides a score — aggregation, gates, retention, the comparator, ranking, contribution, the weight vector | Miners, validators, auditors |
| [`lora-merger-engine`](https://github.com/Capability-AI/lora-merger-engine) | The evaluation engine that *runs* those rules: window loop, candidate serving, store, read-only API, operator configuration | The subnet operator only |

The line is drawn where it is for one reason: a validator pays only for a score
it can recompute. So everything that turns evidence into a number is here and
public, and the engine depends on this package rather than reimplementing any of
it. Nothing here imports the engine, and
[a test](tests/unit/test_layering.py) enforces that direction.

What the operator keeps private is operational — the hidden seed root, wallet
material, filled-in configuration, host inventory — none of which changes what a
candidate scores.

## Quick start

```bash
git clone <repository-url> lora-merger && cd lora-merger

# Install what your role needs. A validator never touches the tensor stack.
pip install -e .              # validator, auditor  (~50 MB)
pip install -e ".[miner]"     # + reconstruction and local evaluation
pip install -e ".[dev]"       # everything, plus test and lint tooling

# What is the arena?
python -m capability_subnet.miner.cli pool          # the frozen certified adapter pool
python -m capability_subnet.miner.cli contract      # the full published contract

# What does one problem look like?
python -m capability_subnet.workflows.cli show --workflow lora_merger_logic_v1 --seed 42 --with-truth
python -m capability_subnet.workflows.cli show --seed 42 --with-truth   # the maintenance workflow

# Is the environment actually solvable?
python -m capability_subnet.workflows.cli selftest --count 10

# Build and check a recipe
python -m capability_subnet.miner.cli init --random --out recipe.json
python -m capability_subnet.miner.cli validate --recipe recipe.json
```

Then read the guide for your role:

| You are a… | Read | You need |
|---|---|---|
| **Miner** | [docs/miner.md](docs/miner.md) | Any hardware. A GPU only if you want to evaluate locally. |
| **Pool operator** | [`scripts/import_public_adapters.py`](scripts/import_public_adapters.py) | Materialises the certified pool from its pinned upstream sources. |
| **Validator** | [docs/validator.md](docs/validator.md) | A small VPS. **No GPU.** |
| **Subnet operator** | the engine repository (operator-only) | GPU hosts, Docker, PostgreSQL. |

## Why validators need no GPU

Validators do not reconstruct, serve or score anything. They fetch the signed weight vector the engine publishes, verify it, and set weights.

This concentrates evaluation in one operator. A validator is not a relay — before touching the chain it:

1. verifies the operator signature against an allow-list it controls,
2. checks the vector against the chain it can see — does the champion still hold that UID? is the engine stalled?
3. **re-scores a closed window from the engine's own published traces**, and
4. **burns rather than submitting anything it cannot verify.**

Step 3 is what turns a signature into evidence. Instance generation is a pure function of the seed and the scorer is deterministic, so a validator regenerates the problems the candidates faced and re-runs the scoring over the published traces — on a VPS, with no GPU and no model. **An engine whose scores do not follow from its own traces does not get paid.**

Every validator does this automatically. Beyond that, anyone can check the record by hand, also without a GPU:

```bash
capability-audit window --window <n>    # do the reports and the weight vector agree?
capability-audit replay --window <n>    # re-score a closed window from its own traces
```

Closed windows publish their instance seeds and the traces the scorer read.
Instance generation is a pure function of the seed, so an auditor regenerates the
exact problems a candidate faced and re-runs the deterministic scorer over them.
A published score that does not follow from its published trace is caught.

## Documentation

| Document | What it covers |
|---|---|
| [Architecture](docs/architecture.md) | How the pieces fit together, the security model, and how scores become emission |
| [Miner guide](docs/miner.md) | Building, evaluating and committing a recipe, and the full recipe reference |
| [Validator guide](docs/validator.md) | Running a validator, verification, failure modes, and deployment |
| [Arena reference](docs/arena.md) | Both workflows: corpora, scoring and limits |
| [Changelog](CHANGELOG.md) | Release history |

## Project layout

```
capability_subnet/
├── common/          protocol contracts: constants, schemas, hashing, commitments, signing
├── registry/        the pinned base model and the certified adapter pool
├── merge_engine/    deterministic reconstruction — the consensus-critical core
├── workflows/       workflow definitions, generators and deterministic scorers
├── sandbox/         isolated execution: agent loop, tool services, limits
├── scoring/         aggregation, gates, comparator, ranking, weight vector
├── miner/           recipe construction, local evaluation, commitment
├── validator/       the thin weight-setter
├── audit/           independent verification of published records
└── testing/         miniature-pool fixtures, published as a pytest plugin
```

## What a merged package is worth

A company does not buy "a translation adapter" or "a SQL adapter". It buys a system that finishes a business process. Whether that is cheaper as one merged model or as several specialists behind a router is an arithmetic question, and this is the arithmetic.

**The router stops existing.** A routed system pays something on every request deciding where to send it — tokens if the decision is made by a model, a hop and a service if by a classifier. A merged package answers directly. On a million requests a month with an LLM router at ~260 tokens per decision, that is ~260M tokens spent producing no answers. *Illustrative arithmetic on stated assumptions, not a measurement — substitute your own numbers.*

**The footprint stops depending on how many skills you offer.** One file under 524 MB on one 24 GB card. No adapter set to keep resident, no swap on a cold skill, no per-skill capacity planning.

**One artifact is one thing to certify.** Regulated buyers approve artifacts, not architectures. A reproducible, content-addressed file with a published evaluation record is a shorter conversation than seven adapters, a router and its training set.

### What we measured

On a 250-item paired benchmark, output tokens tracked merge health closely enough to be a diagnostic:

| Package | Score | Output tokens | |
|---|---|---|---|
| base model | 0.100 | 205,241 | the baseline |
| best single adapter | 0.132 | 182,070 | best score on this pool |
| equal-weight TIES merge | 0.056 | 64,999 | a third of the base model's tokens |
| equal-weight linear merge | 0.000 | 254,950 | collapsed — rambles, answers nothing |

Two things are true at once. A healthy merge was **three times cheaper to run** than the base model on identical work. And on this pool it also **scored lower** than the best single adapter — which is the trade the network exists to measure rather than assume. A collapsed merge announces itself in cost before it announces itself in score, which is why token efficiency is a scored term and not a footnote.

Cost is not a claim here: latency, token efficiency and artifact size are 15% of the quality score, so a package that wins is already one that is cheap to run.

### When merging does not win

- **Efficient multi-adapter serving is genuinely good.** S-LoRA and Punica amortise adapter swapping well. Where your router is a cheap classifier rather than a model, the token argument shrinks to a latency and operations argument.
- **A specialist can simply be better.** If one adapter dominates your traffic, merging trades capability for convenience you may not need. On the current pool, no merge has beaten the best single adapter.
- **Merging can destroy general ability** before it destroys task score, which is why a 0.98 retention floor on a held-out probe is a hard gate rather than a scored term.

## Scope

This is a V1 protocol, and it is deliberately narrow: one base model, one adapter
pool, one workflow running at a time, one declarative recipe format, no routing,
no distillation, no miner-hosted inference.

Narrowness makes the subnet measurable, secure and reproducible with technology
that already exists. The questions it is built to answer:

- Can composed adapters beat the best single adapter and the standard merges?
- Does that improvement survive out-of-distribution data?
- Is a static merged package cheaper than runtime adapter routing?

The adapter pool is assembled from public LoRAs trained on the pinned base model
and normalised to the canonical rank. Coverage is uneven: no public Qwen3-8B
adapter exists for German technical language, and none for text-to-SQL on the
pinned base, so those axes are carried by the base model alone.

Efficient multi-adapter serving is a real alternative to static merging. A
routed-adapter reference baseline is a planned addition.

## Common questions

### What does this subnet actually produce?

A verified, deployable package for one specific business workflow: a merged LoRA adapter that completes the workflow better than the base model, better than any single specialist adapter, better than standard merges, and better than whatever held the throne before it — under hard limits on size, memory, latency and general-capability retention.

Not a benchmark score. A thing you can deploy.

### Why composition instead of training?

Training a new adapter is a solved, well-served problem. What is not solved is deciding which of the adapters you *already have* should be combined, at what weights, at what depth, and how aggressively compressed — for one real workflow.

That decision is currently made by intuition and a handful of experiments. This subnet turns it into a measured competition.

### Is merging actually better than routing between adapters?

**Unknown, and the subnet is built to find out rather than assume it.**

Efficient multi-adapter serving is a genuine competitor. Static merging removes a runtime selector, a load/swap policy and several runtime states, giving one stable model identity and simpler batching — but it does not inherently reduce tokens, agent steps or base compute. Those savings only appear where merging eliminates a token-consuming mechanism.

A routed-adapter reference baseline is a planned addition. Until it exists, the honest position is that static merging is worth selling *where measured total cost is better*, not everywhere.

### What happens if composition turns out not to help?

Nobody gets paid. The permanent reference baselines — including plain equal-weight merges — sit on the board specifically so the network can discover the answer is "no" rather than paying miners to not discover it.

If a reference holds the throne, the workflow share burns.

---

## References

| | |
|---|---|
| LoRA | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| LoRAHub | [arXiv:2307.13269](https://arxiv.org/abs/2307.13269) |
| LoRA Soups | [arXiv:2410.13025](https://arxiv.org/abs/2410.13025) |
| AdapterFusion | [arXiv:2005.00247](https://arxiv.org/abs/2005.00247) |
| TIES-Merging | [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) |
| DARE | [arXiv:2311.03099](https://arxiv.org/abs/2311.03099) |
| S-LoRA | [arXiv:2311.03285](https://arxiv.org/abs/2311.03285) |
| Punica | [arXiv:2310.18547](https://arxiv.org/abs/2310.18547) |
| Qwen3 | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |
| PEFT · safetensors · vLLM · SGLang · Inspect AI · MergeKit · Bittensor SDK | see [docs/architecture.md](docs/architecture.md#prior-art) |

## License

MIT — see [LICENSE](LICENSE).
