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
  "selected_adapters": ["german-technical-v1", "text-to-sql-v1", "..."],
  "merge": { "combination_type": "dare_ties_svd", "density": 0.35,
             "majority_sign_method": "total", "random_seed": 937152 },
  "global_weights": { "text-to-sql-v1": 1.20, "safety-policy-v1": 1.15 },
  "layer_group_overrides": { "group_2": { "text-to-sql-v1": 1.30 } },
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
| **Absolute margin** | Beat the **strongest reference** — not just the incumbent — by 3 points of completion |
| **Statistical significance** | Paired bootstrap lower confidence bound above zero on shared instances |

The second bar is the one that matters most. It stops a package from trading away, say, safety compliance for a better SQL score: the average would improve and the package would be worse at the job.

The third bar is what keeps the network honest at genesis. Standard, non-learned baselines sit on the board permanently — the base model, the best single adapter, three equal-weight merges, and the operator's own published recipe. **If a miner cannot beat all of them, composition has not added value and nobody gets paid.** A reference on the throne earns nothing; the share burns.

**One shot per hotkey.** A decisive loss terminates the challenger permanently. Copying a published recipe costs a registration and buys nothing, because dethroning requires a genuine margin and a copy reproduces the champion's scores exactly.

## The V1 workflow: Industrial Maintenance DE

A German industrial-maintenance agent works one fault from a controller log to a signed-off replacement decision:

```
manual interpretation → fault extraction → maintenance SQL → diagnostic Python
    → inventory action → safety validation → strict final JSON
```

Eight capabilities in one dependent chain — German technical language, structured log extraction, fault reasoning, text-to-SQL, code generation, tool calling, safety-policy compliance, strict structured output. Later steps consume earlier outputs, so it cannot be decomposed into independent benchmark questions.

**No language model decides the result.** Manual facts come from generator metadata. Fault codes come from a deterministic machine schema. SQL is judged by executing it against a hidden PostgreSQL snapshot. Python is judged by hidden test cases the agent never sees. Inventory is judged by the simulator's final state, safety by a deterministic rule engine, and the final report by JSON Schema plus exact value comparison.

That determinism is not fastidiousness — it is what makes the paired statistics valid and lets a disputed evaluation be replayed years later with the same answer.

## Quick start

```bash
git clone <repository-url> lora-merger && cd lora-merger
pip install -e ".[dev]"

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
| **Validator** | [docs/validator.md](docs/validator.md) | A small VPS. **No GPU.** |
| **Subnet operator** | [docs/backend.md](docs/backend.md) | GPU hosts, Docker, PostgreSQL. |

## Why validators need no GPU

Validators do not reconstruct, serve or score anything. They fetch the signed weight vector the engine publishes, verify it, and set weights.

That is a real trade: it concentrates evaluation in one operator. What keeps it honest is that a validator is **not a relay**. Before touching the chain it verifies the operator signature against an allow-list it controls, checks the vector against the chain it can see (does the champion still hold that UID? is the engine stalled?), and **burns rather than submitting anything it cannot verify**. Every report the decision rests on is signed and published, so the weight vector can be re-derived independently by anyone.

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
| [Deployment](docs/deployment.md) | Local, testnet and mainnet |
| [FAQ](docs/faq.md) | Common questions |

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
└── platform/        storage, compatibility history, dashboard
```

## Status and honest limits

This is a V1 protocol, and it is deliberately narrow: one base model, one adapter pool, one workflow, one declarative recipe format, no routing, no distillation, no miner-hosted inference.

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
