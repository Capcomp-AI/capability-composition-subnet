# Architecture

How the pieces fit together, and — more usefully — why each one is shaped the way it is.

---

## The question the design answers

> Given one immutable base model, one certified pool of specialist adapters, one executable agentic workflow, and one deployment budget — which miner can produce the highest-performing static capability package that completes the entire hidden workflow at the lowest practical deployment cost?

Every design decision below follows from making that question **checkable**.

---

## The commodity

A validator-reproducible merged and compressed LoRA adapter that beats:

- the base model,
- every relevant individual source adapter,
- standard merge baselines,
- and the incumbent champion,

on hidden end-to-end workflow execution.

Note what is *not* in that list: a benchmark score, a loss value, or a human judgement.

---

## Why the protocol is narrow

V1 pins one of everything: one base model, one tokenizer, one adapter format, one certified pool, one agent harness, one workflow, one recipe schema, one reconstruction implementation. No routing, no distillation, no full checkpoints, no miner-hosted inference, no model-as-judge.

That narrowness is the design, not a limitation of it. Each pin removes a degree of freedom that would otherwise make results incomparable:

| Pinned | Because otherwise |
|---|---|
| One base model | Two packages measured on different bases are not comparable |
| One rank (64) | Merge behaviour, output size and resource cost all vary with rank |
| One agent harness | A difference in outcome could be a difference in scaffolding |
| One recipe format | Arbitrary miner code would have to be executed to be evaluated |
| Deterministic reconstruction | "The champion's package" would not be a well-defined object |
| No model-as-judge | The scorer's variance would contaminate every comparison |

Routing, hybrid packages and distillation are later phases with their own champion tracks.

---

## Layers

```
capability_subnet/
├── common/          protocol contracts — constants, schemas, hashing, commitments, signing, traces
├── registry/        the pinned base model and the frozen certified adapter pool
├── merge_engine/    deterministic reconstruction (consensus-critical)
├── workflows/       workflow definitions, instance generators, deterministic scorers
├── sandbox/         isolated execution — agent loop, tool services, limits
├── backend/         the evaluation engine (operator-only)
├── miner/           recipe construction, local evaluation, search, commitment
├── validator/       the thin weight-setter
└── platform/        object storage, compatibility history, dashboard
```

Dependencies run downward. `common` depends on nothing; `merge_engine` depends on `common` and `registry`; the engine depends on everything below it. Nothing depends on `backend` except the engine's own entry points, which is what lets a miner install the package without the evaluation stack.

---

## The four objects that matter

### 1. The recipe

One JSON document. Everything it can express is a bounded number, a name from a frozen registry, or an enum. Unknown fields are rejected outright — an unknown field in a miner-authored document usually means something is being smuggled past the merge engine.

Its digest is taken over the **canonical form of the parsed document**, not the file bytes. Pretty-printing is therefore free, but the engine re-derives the digest at admission, so a recipe whose canonical form differs from its commitment is refused. That closes the "commit one thing, publish another" gap.

### 2. The artifact

A merged adapter, serialised with sorted keys, no metadata, contiguous tensors, bfloat16. Its digest is the package's identity: the cache key, the anti-copy identity, and the thing independent workers must agree on.

### 3. The trace

The complete immutable record of one attempt: every tool call, every result, the final payload, and the state each tool service was left in. The scorer reads nothing else. Re-scoring a stored trace a year later gives the same answer.

### 4. The report

The signed, published account of one evaluation: gate verdicts, per-axis comparisons, paired statistics, and the decision with its reason. Anyone can re-derive the weight vector from a stream of these, which is what keeps a centralised engine auditable.

---

## Determinism

Reconstruction is the one place where two machines must agree **byte-for-byte**. Everything that could vary is pinned:

- **Load order.** Adapters load in sorted identifier order, never the order the miner listed them.
- **Precision.** Everything widens to float32 for arithmetic and narrows to bfloat16 only at write time.
- **Thread count.** Single-threaded, because multi-threaded accumulation reorders floating-point additions.
- **TF32.** Disabled — it silently truncates float32 mantissas on recent hardware.
- **Randomness.** Every stochastic step keys a CPU generator on the recipe seed *plus the identity of the tensor it is about to touch*, so the result cannot depend on iteration order, sharding or parallelism.
- **Tie-breaking.** Magnitude trimming cuts at a threshold rather than selecting `k` indices, because index selection breaks ties differently on different backends.
- **Singular vector signs.** A decomposition is only unique up to a joint sign flip per component, and different LAPACK builds choose differently. A canonical convention — the largest-magnitude entry of each left vector is forced positive — makes two correct implementations agree.

That last one is the subtle one. Without it, two perfectly correct engines produce different-but-equally-valid factorisations, and every artifact hash diverges.

**When workers disagree**, evaluation of that candidate *pauses*. It is neither scored nor terminated, because scoring one of two disagreeing artifacts would mean paying for a result nobody can reproduce.

---

## The merge pipeline

Seven method names, one explicit three-stage pipeline:

1. **Sparsify** — discard part of each update before it can interfere. Either the smallest-magnitude entries, or a random subset with survivors rescaled so the expected update is preserved.
2. **Elect signs** — decide per entry which direction the merged update moves in, so adapters that disagree cannot cancel into noise.
3. **Aggregate** — sum the survivors, or average only over the adapters that agree with the elected sign.

| Method | Sparsify | Sign election | Aggregate | Path |
|---|---|---|---|---|
| `linear` | none | none | weighted sum | factor space |
| `svd` | none | none | weighted sum | delta space |
| `cat_svd` | none | none | weighted sum | delta space |
| `ties_svd` | magnitude | majority | disjoint mean | delta space |
| `dare_ties_svd` | random rescale | majority | disjoint mean | delta space |
| `dare_linear_svd` | random rescale | none | weighted sum | delta space |
| `magnitude_prune_svd` | magnitude | none | weighted sum | delta space |

Naming the stages makes the redundancies visible rather than hiding them: with no sparsification and no sign election there is one sensible combination, so **`svd` and `cat_svd` produce the same update** and differ only in the factorisation path they document. `linear` is genuinely distinct — it sums the factors rather than their products, which carries cross terms between adapters and needs no decomposition at all when the output rank matches the pool's.

All arithmetic happens in **delta space** on the effective update `ΔW = (α/r)·B·A`. That matters because two adapters can hold numerically similar factors and apply updates that differ by an order of magnitude; without the normalisation, a coefficient of 1.2 would mean different things for different adapters.

---

## The continuous loop

```
open window   → draw fresh hidden instances, re-measure every reference and the incumbent
admit         → validate, verify digest, check the frozen pool, anti-copy      (no GPU)
take the head → the earliest commitment still waiting becomes the challenger
evaluate      → reconstruct, serve, run the fixed agent, score deterministically
compare       → per-axis verdicts, end-to-end margin, paired bootstrap
publish       → signed report, signed weight vector
```

Role assignment is **mechanical**: the queue is ordered by the block each commitment was made at, and the head is the challenger. Nobody chooses.

Re-measuring the references every window is the expensive part and is not optional. A challenger compared against last window's reference numbers would be compared on a different instance set, and the paired statistics would be meaningless.

### The two failure policies

This distinction matters more than any other rule in the engine:

- **Miner failures fail closed.** An invalid submission scores zero.
- **Infrastructure failures fail open.** The queue *holds* rather than terminating a candidate on flaky hardware.

A one-shot-per-hotkey rule is only defensible if the engine never spends that shot on its own bad night.

---

## The dethrone rule

Four independent bars, all of which must clear:

1. **Per-axis dominance** on at least the required number of capability axes.
2. **Not worse on every remaining axis.** This is the bar that stops a package trading a capability away for a better average.
3. **An absolute end-to-end margin** over the *strongest reference*, not merely the incumbent.
4. **Paired statistical significance** — a one-sided bootstrap lower confidence bound above zero on shared instances.

The test is **paired** because both packages ran on the same instances. Pairing removes instance difficulty from the comparison entirely: what gets resampled is the vector of per-instance differences, not two independent score distributions. That is a substantially tighter test, and it is only available because the engine controls which instances both sides saw.

An axis with too few paired samples counts as **worse**, not as a tie. Absence of evidence that a challenger kept a capability is not evidence that it did.

A loss is *decisive* — and terminates the hotkey — only when the challenger was genuinely measured. An axis with no paired samples at all means the engine failed to gather evidence, and terminating on that would punish a miner for an evaluation the engine did not complete.

---

## Permanent reference champions

A plain king-of-the-hill contest only requires beating whoever holds the throne. At genesis the throne is empty, and later a mediocre champion could hold it simply because nothing better challenged.

So the network keeps a set of references on the board permanently:

| Reference | What it represents |
|---|---|
| Base model | Doing nothing |
| Best single adapter | Picking the best specialist |
| Equal-weight linear merge | The obvious merge |
| Equal-weight TIES merge | The obvious interference-aware merge |
| Equal-weight DARE-TIES merge | The obvious stochastic variant |
| Owner reference recipe | The operator's own published attempt |

They are measured every window through **exactly the same code path** as candidates — if baselines were measured differently, "the challenger beat the strongest reference" would be a statement about two harnesses rather than two packages.

None of them can be terminated and **none of them earn emission**. If a reference holds the throne, the workflow share burns, because the network has not yet produced anything worth paying for.

---

## Anti-copy

Recipes are public — they have to be, or nobody could verify an evaluation. So the protocol makes copying *worthless* rather than impossible:

- **Earliest commit wins**, checked on the recipe digest at admission and again on the reconstructed artifact digest. Two differently-worded recipes that build the same bytes are the same package.
- **One shot per hotkey.** Copying costs a registration and buys nothing.
- **Defender advantage.** Dethroning requires a genuine margin, so a copy that exactly reproduces the champion's scores loses by construction.

The third point is what makes the first two sufficient. Even a copy nobody detected cannot win.

---

## Sandbox isolation

```
orchestrator (per instance)
  candidate_model_server   base + reconstructed adapter, OpenAI-compatible,
                           internal bridge only, no dynamic adapter loading
  agent_runner             fixed ReAct loop, fixed prompt and tool schemas
  postgres_tool            hidden snapshot, read-only role, per-instance schema
  python_runner            candidate code, no network, no writable filesystem
  inventory_simulator      deterministic stock state
  safety_engine            deterministic rule engine
  scorer                   holds the truth — NOT reachable by the agent
```

The scorer runs in the engine process, outside every network the agent container can see. A scorer the agent can reach is an answer key with a REST interface.

The Python runner is the only component that executes candidate-written code, so it gets the strongest boundary: no network at all, read-only root, all capabilities dropped, memory and PID limits, and a fresh interpreter in isolated mode that never sees the parent's memory.

Adapter hot-swapping is disabled. Restarting the runtime per candidate is slower, but hot-swapping leaves the previous candidate's state in the runtime's caches, and a measurement that depends on which package ran before it is not a measurement.

---

## Scoring

```
Q = 0.60·end_to_end + 0.15·stage_balance + 0.10·ood
  + 0.05·retention + 0.05·latency + 0.05·artifact_efficiency
```

Quality carries 85%, efficiency 15% — a cheap package that does not finish the job is worth less than an expensive one that does.

**Stage balance** is a geometric mean of the per-stage means, and the choice of mean is doing real work: a package scoring 1.0 on six stages and 0.1 on the seventh lands far below one scoring 0.8 everywhere, even though their arithmetic means are close. The workflow needs every stage, so a package that abandoned one has not solved it.

---

## Go/no-go

The network should not register until:

- optimised composition beats the best single adapter;
- optimised composition beats standard TIES/DARE baselines;
- the improvement survives hidden out-of-distribution data;
- reconstruction is deterministic across independent workers;
- one full candidate fits the hardware budget;
- the continuous engine processes the expected miner count within the window budget;
- security tests pass;
- the commercial package has a clear buyer or pilot.

**If composition cannot beat strong baselines, the correct decision is not to launch.**

That is not a rhetorical hedge. Efficient multi-adapter serving is a strong competitor, and static merging is only worth selling where measured total cost is genuinely better. The reference baselines exist precisely so the network can discover the answer is "no" rather than paying miners to not discover it.

---

## Prior art

The design draws on published research and existing production code. None of it is novel infrastructure; the contribution is the composition-as-commodity framing and the evaluation mechanism around it.

| Area | Reference |
|---|---|
| Low-rank adaptation | [LoRA](https://arxiv.org/abs/2106.09685) |
| Composition | [LoRAHub](https://arxiv.org/abs/2307.13269) · [LoRA Soups](https://arxiv.org/abs/2410.13025) · [AdapterFusion](https://arxiv.org/abs/2005.00247) |
| Interference | [TIES-Merging](https://arxiv.org/abs/2306.01708) · [DARE](https://arxiv.org/abs/2311.03099) |
| Multi-adapter serving | [S-LoRA](https://arxiv.org/abs/2311.03285) · [Punica](https://arxiv.org/abs/2310.18547) · [vLLM](https://docs.vllm.ai/en/latest/features/lora.html) |
| Base model | [Qwen3](https://arxiv.org/abs/2505.09388) |
| Merge implementations | [PEFT](https://github.com/huggingface/peft) · [MergeKit](https://github.com/arcee-ai/mergekit) |
| Agentic evaluation | [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) |
| Tensor format | [safetensors](https://github.com/huggingface/safetensors) |
| Chain and SDK | [Bittensor](https://github.com/opentensor/bittensor) · [subtensor](https://github.com/opentensor/subtensor) |

The continuous champion-challenge shape — a queue ordered by commit block, per-window task refresh, single-recipient weight vectors, per-instance sample rows feeding a comparator — is an established pattern in this ecosystem, adapted here to the merged-adapter commodity.

---

## Where this subnet sits

| Neighbour | What miners submit | Relationship |
|---|---|---|
| Distributed pre-training subnets | Pre-training compute for one shared base | **Upstream.** They produce base models; this subnet adapts a pinned one and never trains. |
| Fine-tuning-as-a-service subnets | A model or adapter trained on a supplied dataset | **Closest neighbour.** They train new adapters on data; this composes existing certified ones with no training, and scores end-to-end workflow completion rather than single-task loss. |
| From-scratch model subnets | Full model weights | **Different layer.** Architecture training versus declarative composition above a frozen base. |
| Environment-evaluated model subnets | A model plus revision, run through RL environments | **Adjacent mechanism**, different artifact and different target. |

What makes this one distinct: the artifact is a declarative recipe over a frozen pool, not a trained model; the objective is end-to-end completion of an executable business workflow judged by non-model truth; no gradient work is ever verified; and the output is a deployable, cost-bounded package with hard memory, size, latency and retention gates.
