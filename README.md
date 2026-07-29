<div align="center">

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

The second bar is the one that matters most. It stops a package from trading away, say, safety compliance for a better SQL score: the average would improve and the package would be worse at the job.

The third bar is what keeps the network honest at genesis. Standard, non-learned baselines sit on the board permanently — the base model, the best single adapter, three equal-weight merges, and the operator's own published recipe. **If a miner cannot beat all of them, composition has not added value and nobody gets paid.** A reference on the throne earns nothing; the share burns.

The incumbent is deliberately *not* one of the permanent references. Folding it in would mean every successive champion had to beat the previous one by a further three points — and since completion is bounded by one, that staircase stalls after a handful of dethrones, after which one package holds the throne forever and the network buys nothing more. The incumbent instead gets its own, smaller margin that decays: a defender's advantage for holding the throne well, not a freehold.

**One shot per hotkey.** A decisive loss terminates the challenger permanently. That rule is only defensible if the engine never spends a candidate's shot on its own bad night, so failures that indict the *engine* — an unreadable memory counter, too few scored instances — hold the candidate for a later window instead of ending it.

### Losing well is worth something

Winner-take-all throws away the network's most useful signal. Almost every submission that is ever evaluated will fail to take the throne, and a recipe is *one shot* — a miner cannot iterate on it the way a code-submitting miner can. Paying a miner who moved end-to-end completion from 0.41 to 0.58 exactly what it pays one that submitted a soup of distractors tells neither of them anything.

So the throne is winner-takes-most, and everything below it is **graded**:

| Term | Weight | What it rewards |
|---|---|---|
| Quality | 50% | The qualified score — completion, stage balance, OOD, retention, latency, tokens, size |
| Improvement | 25% | How far past the strongest non-learned reference it got |
| Proximity | 15% | How close it came to the champion — a near miss is not a wasted registration |
| Cost | 10% | Token spend and latency, because two packages that finish equally are not equally valuable |

Only packages that cleared **every hard gate** are graded. This is not a consolation prize for producing something undeployable, and if nobody qualifies the graded pool burns rather than becoming a bonus for an uncontested champion. Every grade is published broken into its four terms, so a miner can see what earned it.

Miners still waiting in the queue earn a small tapered share on top. Not payment for work: Bittensor prunes by lowest emission, so a strict winner-take-all split makes every unevaluated challenger the first thing the chain evicts.

## The V1 workflow: Industrial Maintenance DE

A German industrial-maintenance agent works one fault from a controller log to a signed-off replacement decision:

```
manual interpretation → fault extraction → maintenance SQL → diagnostic Python
    → inventory action → safety validation → strict final JSON
```

Seven scored capability axes in one dependent chain — manual interpretation, fault extraction, maintenance SQL, diagnostic Python, inventory action, safety validation, strict final JSON. Later steps consume earlier outputs, so it cannot be decomposed into independent benchmark questions.

**No language model decides the result.** Manual facts come from generator metadata. Fault codes come from a deterministic machine schema. SQL is judged by executing it against a hidden PostgreSQL snapshot. Python is judged by hidden test cases the agent never sees. Inventory is judged by the simulator's final state, safety by a deterministic rule engine, and the final report by JSON Schema plus exact value comparison.

That determinism is not fastidiousness — it is what makes the paired statistics valid and lets a disputed evaluation be replayed years later with the same answer.

## Quick start

```bash
git clone <repository-url> lora-merger && cd lora-merger

# Install what your role needs. The base install is deliberately small —
# a validator never touches the tensor stack, so it does not download one.
pip install -e .              # validator, auditor  (~50 MB)
pip install -e ".[miner]"     # + reconstruction and local evaluation
pip install -e ".[backend]"   # + serving, sandbox services, NVML
pip install -e ".[dev]"       # everything, plus test and lint tooling

# What is the arena?
python -m capability_subnet.miner.cli pool          # the frozen certified adapter pool
python -m capability_subnet.miner.cli contract      # the full published contract

# What does one problem look like?
python -m capability_subnet.workflows.cli show --seed 42 --with-truth

# Is the environment actually solvable?
python -m capability_subnet.workflows.cli selftest --count 10

# Build and check a recipe
python -m capability_subnet.miner.cli init --out recipe.json
python -m capability_subnet.miner.cli validate --recipe recipe.json
```

Then read the guide for your role:

| You are a… | Read | You need |
|---|---|---|
| **Miner** | [docs/miner.md](docs/miner.md) | Any hardware. A GPU only if you want to evaluate locally. |
| **Pool operator** | [`scripts/import_public_adapters.py`](scripts/import_public_adapters.py) | Materialises the certified pool from its pinned upstream sources. |
| **Validator** | [docs/validator.md](docs/validator.md) | A small VPS. **No GPU.** |
| **Subnet operator** | [docs/backend.md](docs/backend.md) | GPU hosts, Docker, PostgreSQL. |

## Why validators need no GPU

Validators do not reconstruct, serve or score anything. They fetch the signed weight vector the engine publishes, verify it, and set weights.

That is a real trade: it concentrates evaluation in one operator. What keeps it honest is that a validator is **not a relay**. Before touching the chain it:

1. verifies the operator signature against an allow-list it controls,
2. checks the vector against the chain it can see — does the champion still hold that UID? is the engine stalled?
3. **re-scores a closed window from the engine's own published traces**, and
4. **burns rather than submitting anything it cannot verify.**

Step 3 is the one that turns a signature into evidence. A signature proves the operator produced a vector; it says nothing about whether the evaluation behind it was honest. Because instance generation is a pure function of the seed and the scorer is deterministic, a validator regenerates exactly the problems the candidates faced and re-runs the scoring over the published traces — on a VPS, with no GPU and no model. **An engine whose scores do not follow from its own traces does not get paid.**

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
| [Architecture](docs/architecture.md) | How the pieces fit and why the design is shaped this way |
| [Miner guide](docs/miner.md) | Building, evaluating and committing a recipe |
| [Validator guide](docs/validator.md) | Running a validator, verification, failure modes |
| [Engine operations](docs/backend.md) | Operating the evaluation engine |
| [Workflow reference](docs/workflow.md) | The V1 workflow, its stages and how each is scored |
| [Recipe reference](docs/recipe.md) | Every field, bound and merge method |
| [Security model](docs/security.md) | Threats, defences and what is deliberately not defended |
| [Repositories](docs/repositories.md) | What is public, what an operator keeps private, and why |
| [Deployment](docs/deployment.md) | Local, testnet and mainnet |
| [FAQ](docs/faq.md) | Common questions |
| [Changelog](CHANGELOG.md) | Release history, including what an audit pass found |

## Project layout

```
capability_subnet/
├── common/          protocol contracts: constants, schemas, hashing, commitments, signing
├── registry/        the pinned base model and the certified adapter pool
├── merge_engine/    deterministic reconstruction — the consensus-critical core
├── workflows/       workflow definitions, generators and deterministic scorers
├── sandbox/         isolated execution: agent loop, tool services, limits
├── backend/         the evaluation engine (operator-only)
├── miner/           recipe construction, local evaluation, search, commitment
├── validator/       the thin weight-setter
├── audit/           independent verification of published records
└── platform/        storage, compatibility history, dashboard
```

## Status and honest limits

This is a V1 protocol, and it is deliberately narrow: one base model, one adapter pool, one workflow, one declarative recipe format, no routing, no distillation, no miner-hosted inference.

**Does composition beat the equal-weight merge? On this pool, measured: no.**

250 paired items from `AffineFoundation/affine-lgc`, ten task families, exact-match scored:

| package | score | 95% CI | output tokens | kind |
|---|---|---|---|---|
| `creative-writing-v1` | 0.132 | [0.096, 0.180] | 182,070 | single (**a declared distractor**) |
| `code-generation-v1` | 0.116 | [0.082, 0.162] | 202,666 | single |
| **base model** | **0.100** | **[0.069, 0.143]** | **205,241** | reference |
| `action-planner-v1` | 0.100 | [0.069, 0.143] | 205,211 | single |
| owner's tuned recipe | 0.060 | [0.037, 0.097] | 62,973 | **merge** |
| equal-weight TIES | 0.056 | [0.034, 0.092] | 64,999 | **merge** |
| equal-weight SVD | 0.004 | [0.001, 0.022] | 115,349 | **merge** |
| equal-weight linear | 0.000 | [0.000, 0.015] | 254,950 | **merge** |

Merge better on **0** of 10 tasks, single better on 7, tied on 3. Every merge landed at or below the base model, in the same order the retention probe found — TIES survives, linear collapses.

Two findings behind the headline. The best single adapter is a **declared distractor**, because the distractor labels were assigned from descriptions rather than measurement. And composition *did* lift one capability off the floor — `time_sequence`, where the base scores 0.00 and the TIES merge 0.08 — which is the shape a real positive result would take, on one family out of ten.

This measures *this* pool: twelve scavenged public adapters with no coherent capability coverage, six of which individually fall below the retention floor. It does not show composition cannot work; it shows composing these adapters does not — which is the question [the go/no-go](docs/architecture.md#gono-go) says to answer before launching.

**Earlier measurement, general-capability retention.** Forty exactly-scored general-capability probe items against the pinned base model, on a GPU:

| package | probe | retention | output tokens | 0.98 gate |
|---|---|---|---|---|
| best single adapter | 36/40 | 1.000 | 241 | pass |
| **base model** | **35/40** | **1.000** | **250** | pass |
| equal-weight TIES merge | 34/40 | 0.971 | 252 | **rejected** |
| operator's tuned recipe | 32/40 | 0.914 | 264 | **rejected** |
| equal-weight SVD merge | 10/40 | 0.286 | 861 | **rejected** |
| equal-weight linear merge | 0/40 | 0.000 | 1280 | **rejected** |

Two results matter. **Interference-aware merging is the difference between working and not** — the same ten adapters summed linearly retain *nothing*, while TIES retains 0.971. And **the operator's tuned recipe lost to the untuned equal-weight merge**, which is the question this subnet exists to ask.

The tuned recipe emphasises the structured-output and tool-calling adapters to buy workflow capability, and the probe does not measure workflow capability — so this shows the cost of that trade with the benefit invisible. It is the trade the retention gate exists to catch, and it caught it. A collapsed package also announces itself in cost: 5x the base model's output tokens, because it answers terse instructions with prose.

Read the limit with the result. This is the retention probe, not the workflow: it shows that a merge destroyed general ability, and cannot show that composition added workflow value.

**Every merge measured is rejected by the retention gate** — the best of them misses a 0.98 floor by nine thousandths. As configured against this pool the network would crown nobody and burn indefinitely. That is a calibration decision to make before genesis, and it is now an informed one: either the floor comes down to something a real merge can clear, or the pool gains an adapter actually trained to preserve capability under merging. Lowering the floor without fixing the pool would be choosing not to look.

**The pool is assembled from public adapters, and it does not cover every axis.** Every member is a real, permissively licensed LoRA trained on the pinned base model, verified and normalised to the canonical rank. But no public Qwen3-8B adapter exists for German technical language, and none for text-to-SQL that is both permissively licensed and trained on the pinned base rather than a quantised mirror. Those two axes are currently carried by the base model alone, which caps what composition can be shown to add on this workflow. Closing them means training the adapters rather than finding them.

Narrowness is the point. It makes the subnet measurable, secure and reproducible with technology that already exists. The open questions it is designed to answer — and might answer *no* to — are:

- Can composed adapters beat the best single adapter and standard merges on a real workflow?
- Does that improvement survive out-of-distribution data?
- Is a static merged package genuinely cheaper than runtime adapter routing?

Efficient multi-adapter serving is a strong competitor. Static merging is only worth selling where measured total cost is actually better, which is why the routed-adapter reference baseline is a planned addition rather than an assumption.

**If composition cannot beat strong baselines, the correct decision is not to launch.** The go/no-go criteria are in [docs/architecture.md](docs/architecture.md#go-no-go).

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
