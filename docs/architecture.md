# Architecture

How the pieces fit together.

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
├── scoring/         aggregation, gates, comparator, ranking, weight vector
├── miner/           recipe construction, local evaluation, commitment
├── validator/       the thin weight-setter
├── audit/           independent verification of published records
└── testing/         miniature-pool fixtures, published as a pytest plugin
```

Dependencies run downward. `common` depends on nothing; `merge_engine` depends on `common` and `registry`; `scoring` depends on those. The evaluation engine ships in a separate operator-only repository and depends on this package — nothing here depends on it, which is what makes the two separable and what a test in `tests/unit/test_layering.py` enforces.

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

Five independent bars, all of which must clear:

1. **Per-axis dominance** on at least the required number of capability axes.
2. **Not worse on every remaining axis.** This is the bar that stops a package trading a capability away for a better average.
3. **An absolute end-to-end margin** over the *strongest permanent reference*. This is the bar that says composition added value at all, and it never moves. **Off by default** — see below.
4. **A defender's margin** over the incumbent, which starts small and decays to zero over roughly thirty days.
5. **Paired statistical significance** — a one-sided bootstrap lower confidence bound above zero on shared instances.

### Bar 3 is optional, and ships off

`require_beat_reference` defaults to **False**: the highest score on the board is paid, whether or not it cleared the strongest reference. The product is the best composition anyone has found, not proof that composition was worth attempting — and the strict rule had a real failure attached, where a network producing perfectly good comparative information would burn its emission indefinitely because nothing cleared an absolute bar. References are still measured and published every window, so the question stays answerable from the record; it just stops gating payment. Set it True for the stricter contract.

**Base retention does not move with it.** A package that destroyed the base model's general ability is not deployable whatever it scored, so that gate stays hard in both modes.

Turning bar 3 off opens a hole that bar 4 was quietly closing, and it is closed separately. Under a margin rule a copy of the leader could not displace it, because identical scores are not a margin. Under highest-score-wins a copy *ties* — and since no two evaluations of two distinct artifacts land on exactly the same number, a copy with one coefficient nudged takes the top slot roughly half the time on sampling noise alone. Recipes are public, so this is read-and-resubmit rather than a hypothetical.

So submissions closer together than the window can **resolve** are ranked as tied, and ties resolve to the earliest commitment. A copier commits later by construction, so it has to be measurably better — the same bar the margin enforced, expressed in the units the evidence actually supports. Indistinguishability is not transitive, so ranking groups maximal runs into equivalence classes rather than swapping pairs.

Bars 3 and 4 are separate. If the incumbent counted among the references, every successive champion would have to beat the previous one by a further margin, and since completion is bounded by one that bar walks upward until nothing can move it.

The decay means an incumbent nothing has displaced for a month loses its advantage rather than holding the throne on the strength of being unopposed.

The test is **paired** because both packages ran on the same instances. Pairing removes instance difficulty from the comparison entirely: what gets resampled is the vector of per-instance differences, not two independent score distributions. That is a substantially tighter test, and it is only available because the engine controls which instances both sides saw.

An axis with too few paired samples counts as **worse**, not as a tie. Absence of evidence that a challenger kept a capability is not evidence that it did.

A loss is *decisive* — and terminates the hotkey — only when the challenger was genuinely measured. An axis with no paired samples at all means the engine failed to gather evidence, and terminating on that would punish a miner for an evaluation the engine did not complete.

The same principle governs the hard gates, which are split into two sets. A candidate that used 40 GB genuinely failed the memory limit; a candidate on a host whose memory counter was unreadable has not been shown to fail anything. The second kind holds the candidate for a later window and leaves its single shot intact. A one-shot-per-hotkey rule is only defensible if the engine never charges a candidate for its own bad night.

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

They are measured through **exactly the same code path** as candidates — if baselines were measured differently, "the challenger beat the strongest reference" would be a statement about two harnesses rather than two packages.

The base model, the three standard merges and the owner recipe are re-measured every window. The per-adapter single-adapter references are **rotated**, a few per window on a schedule derived from the window id. Measuring all of them every window is correct and costs most of a window's GPU budget before any challenger is looked at — and a window that cannot finish never evaluates anybody. Rotation keeps the "beat the best specialist" bar honest over time while leaving room to actually run the queue.

The **incumbent is not one of them**. It is re-measured every window and reported alongside them, but it does not set the absolute bar; see the dethrone rule above for why.

None of them can be terminated and **none of them earn emission**. Under the strict contract (`require_beat_reference=True`) a reference can hold the throne, and then the workflow share burns because the network has not yet produced anything worth paying for. Under the default it cannot: the best *submission* is paid regardless of where the references landed, and the references serve as published context for whether that submission was worth anything rather than as a gate on it.

---

## What a centralised engine is and is not trusted for

Evaluation runs in one place. That is a concentration of trust, and this is what it does and does not buy an operator.

The operator **is** trusted to choose the hidden instances (from a secret root), to run the sandbox, and to publish honestly and promptly.

The operator is **not** trusted to be believed. Every score is published with the trace it came from, every window's seeds are disclosed once it closes, and instance generation is a pure function of the seed — so the claim "this candidate scored 0.62" is checkable by anyone who can run a deterministic scorer, which is anyone with a laptop. Validators do this on every pass before paying, and burn when the record contradicts itself.

What that leaves genuinely unaddressed: an operator who *withholds* disclosures entirely, or who is slow. Validators tolerate a missing disclosure rather than treating it as fraud — the opposite policy would make an outage indistinguishable from dishonesty — so a determined operator can degrade verification by simply not publishing. The counterweight is that validators refuse a vector more than `max_stale_windows` behind the chain head, so withholding cannot be sustained without also stopping emission.

---

## Who chooses the problems

Hidden instances are drawn per window from a secret root the operator holds.
Refreshing per window stops a *miner* tuning to a fixed set. On its own it does
nothing about the operator, and the difference matters.

Seeds derive deterministically from the root and the window id. An operator free
to try roots until the draw suited a candidate they had already evaluated would
pass every check in this document: the seeds would be real, the instances would
match them, and the scores would follow from the traces. What was chosen was the
draw.

Two things close it, and both are needed.

**The root is committed.** `seed_root_commitment` is a hash of the root, published
in the contract and in every closed window's disclosure. It reveals nothing and
binds the operator to one root: the same value has to appear in every window, so a
commitment that moves is the draw being re-rolled where anyone can see it.
`commitments_agree()` checks a run of disclosures for exactly that.

**The draw is bound to a value the operator does not choose.** Each window mixes
in the hash of its own first block — `window_id x window_blocks`, not whatever
block the engine happened to open at. That is public, it is not the operator's to
pick, and it does not exist until the window begins, so a draw cannot be selected
after seeing a candidate.

Deriving it from the window rather than from the moment of opening is what makes
it checkable at all: an auditor knows which block a window started at and can
fetch the same hash, where a block only the operator knew would prove nothing.
It also makes re-opening a window idempotent — an engine restarted mid-window
re-derives the seeds it already published instead of quietly evaluating a
different test.

**What this does and does not buy.** The beacon is the part with teeth: a
validator compares it against the real block hash, and a fabricated one fails.
The commitment is weaker — nothing reveals the root, so a constant fabricated
value passes. What it forces is that the operator pick one root and keep it, which
means picking it before any candidate exists; `check_draw_was_not_re_rolled()`
catches a value that moves across recent windows.

So the honest summary is: **re-rolling the draw between windows is caught, and a
draw not bound to its block is caught. A single root chosen dishonestly at genesis
is not** — closing that needs an eventual reveal of the root, which this protocol
does not yet do.

Where the beacon is absent — a local run, or an endpoint that was down — the
disclosure says so and the auditor raises `unbound_draw` rather than implying a
guarantee that is not there.

---

## Anti-copy

Recipes are public — they have to be, or nobody could verify an evaluation. So the protocol makes copying *worthless* rather than impossible:

- **Earliest commit wins**, checked on the recipe digest at admission and again on the reconstructed artifact digest. Two differently-worded recipes that build the same bytes are the same package.
- **One shot per hotkey.** Copying costs a registration and buys nothing.
- **Defender advantage.** A copy has to be *measurably* better, not merely higher. Under the strict contract that is the dethrone margin; under the default it is the tie rule — scores closer than the window can resolve are ranked equal and ties resolve to the earliest commitment, so a later copy cannot displace what it copied on sampling noise.

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
Q = 0.55·end_to_end + 0.15·stage_balance + 0.10·ood + 0.05·retention
  + 0.05·latency + 0.05·token_efficiency + 0.05·artifact_efficiency
```

Quality carries 85%, efficiency 15% — a cheap package that does not finish the job is worth less than an expensive one that does.

**Token efficiency** is measured per *completed* instance, not per attempted one. Dividing by attempts would make a package that gives up after one turn the most efficient thing on the board; charging its tokens against the workflows it actually finished makes cheap failure expensive, which is the correct direction. An unfinished workflow has no value to divide a cost into.

---

## Who gets paid

Ranking decides who leads — the dethrone rule under the strict contract, the tie-aware leaderboard by default. Either way it is far too blunt to also decide who gets *paid*, because almost every submission that is ever evaluated will fail to lead — and a recipe is *one shot*, so a miner cannot iterate on it the way a code-submitting miner can. Paying a miner who moved completion from 0.41 to 0.58 exactly what it pays one that submitted a soup of distractors leaves the second attempt no better informed than the first, in a network whose entire purpose is to learn which adapters compose.

So the top slot is winner-takes-most and everything below it is graded:

| Term | Weight | What it rewards |
|---|---|---|
| Quality | 50% | The qualified score above |
| Improvement | 25% | Distance past the strongest permanent reference, scaled by the headroom that remained |
| Proximity | 15% | How close it came to the champion |
| Cost | 10% | Token spend and latency |

Only candidates that cleared **every hard gate** are graded — this is not a consolation prize for producing something undeployable. If nobody qualifies, the graded pool burns rather than folding into the champion's share, because holding an uncontested throne is not an achievement. Each grade is published broken into its four terms, so a miner can act on it.

Proximity is not a gate. Rewarding closeness alone would pay for copying the leader, so it is one term of four and the anti-copy check runs *before* evaluation.

**Stage balance** is a geometric mean of the per-stage means, and the choice of mean is doing real work: a package scoring 1.0 on all but one axis and 0.1 on the last lands far below one scoring 0.8 everywhere, even though their arithmetic means are close. A workflow needs every axis it declares — seven for the maintenance chain, twelve for the arena — so a package that abandoned one has not solved it.

**Retention** is measured on a held-out general-capability probe, *not* on the workflow. That distinction is the whole reason the term exists, and it is the one term that does not weaken when bar 3 is off: a comparison against the base model's *workflow* completion cannot detect collapse under the strict contract either, since a candidate only reaches the gate after beating the base by a margin, so the ratio is always above one and the clamp returns exactly `1.0` for every candidate that could possibly be crowned. The probe asks short, exactly-scored questions about the behaviours aggressive merging actually destroys — following a format, not padding an answer, arithmetic, ordering, answering in the language it was addressed in — drawn per window from their own secret seed and asked of the base model on the same draw.

---

## The commercial question

The subnet optimises a finished process rather than isolated benchmark scores,
because that is what is bought. Whether a *merged* package is the right shape for
that process is a separate question, and the honest answer is that it depends on
numbers the buyer has.

What merging removes is structural: the routing decision on every request, the
adapter set held resident, and the per-skill capacity planning. What it risks is
also structural: a merge can be worse than the specialist it replaced, and an
over-aggressive one loses general instruction-following before it loses task
score.

Both are measured rather than assumed. Latency, token efficiency and artifact
size carry 15% of the qualified score, so cost is priced into who wins. Base
retention is a hard gate at 0.98 on a held-out probe, so a package that traded
away general ability cannot be crowned whatever it scored. And the permanent
references — the base model, the best single adapter, the standard merges — exist
so the network can discover that composition did *not* help, which on the current
pool is still the standing result.

See the economics tab of the published dashboard for the worked arithmetic, and
[the README](../README.md#what-a-merged-package-is-worth) for the summary.

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

---

# Security model

## The shape of the problem

Three things make this system's threat model unusual:

1. **Recipes are public.** They must be, or nobody could verify an evaluation. So copying is trivially easy and has to be made *worthless* rather than impossible.
2. **Candidate-written code executes.** The diagnostic stage is not scoreable otherwise.
3. **Evaluation is centralised.** One operator holds the hidden instances and the signing key, which concentrates both capability and the incentive to misuse it.

Each is addressed by a different mechanism, and the mechanisms are described honestly below, including where they stop.

---

## Threats and defences

### A malicious recipe

**Goal:** get the engine to execute something, load something, or reach somewhere.

| Defence | How |
|---|---|
| Declarative-only schema | Every expressible value is a bounded number, a name from a frozen registry, or an enum |
| Unknown fields rejected | An unknown field in a miner document usually means something is being smuggled |
| Names, not paths | Adapters are named and resolved against the frozen pool; there is no field to put a path in |
| Strict bounds | Coefficients, densities, ranks and quantiles are all range-checked |
| Size cap | Recipe fetches are capped, and reading stops mid-download when exceeded |

There is no code path from a recipe to execution. The merge engine reads numbers and multiplies matrices.

### A malicious adapter file

**Goal:** get code to run when the pool is loaded.

Adapters are untrusted input until certified. Admission requires: safetensors only (no pickle-backed formats, which execute on load), no scripts or binaries in the directory, no remote-code hooks in the config, exact shape validation against the pinned base, and an exhaustive NaN/infinity scan.

A single non-finite entry would propagate through every merge that selected that adapter, which is why the scan is exhaustive rather than sampled.

### Substituting the recipe after committing

**Goal:** commit one thing, be scored on another.

Bytes are verified against the on-chain digest before parsing. Then the digest is **re-derived from the canonical form of the parsed document** and compared again. That second check closes the gap where a miner commits the digest of a specially-formatted file and the engine scores the canonical form of something else.

The pointer itself is not trusted, so a mutable host is fine for integrity. It is not fine for availability: if the engine cannot fetch the bytes, the submission is not admitted.

### Copying

**Goal:** read a published recipe and resubmit it.

Three mechanisms, and the third is what makes the first two sufficient:

1. **Earliest commit wins**, checked on the recipe digest at admission and again on the reconstructed **artifact** digest. Two differently-worded recipes that build the same bytes are the same package.
2. **One shot per hotkey.** Copying costs a registration.
3. **Defender advantage.** Dethroning requires a genuine margin, so a copy that exactly reproduces the champion's scores loses by construction.

Even a copy nobody detected cannot win. That is the point.

### Extracting hidden material

**Goal:** learn the hidden instances, the truth, or the seed root.

| Boundary | Enforcement |
|---|---|
| The scorer | Runs in the engine process, outside every network the agent container can see |
| Hidden diagnostic cases | Their **expected outputs never enter the runner** — only inputs are sent |
| The seed root | Never leaves the engine; the API has no route that exposes it |
| Instance visibility | The visible payload is built explicitly, field by field, rather than filtered from the full object |
| Per-window refresh | Instances rotate, so anything learned about one window is worth little in the next |

That fourth row is a deliberate design choice: a new field added to the ground truth is invisible to the agent until someone *deliberately* adds it to the visible payload. A filter-based approach fails open; this fails closed.

### SQL injection and exfiltration

**Goal:** read or write outside the instance's snapshot.

Statement inspection rejects anything that is not a single read-only statement, along with catalog access (`sqlite_*`, `pg_*`, `information_schema`) and file-access functions.

**Inspection alone would be a filter to be evaded.** What actually holds is the connection: a SQLite authorizer that denies everything but read operations, or a PostgreSQL role with no write grants, `default_transaction_read_only`, a statement timeout, and `SELECT` granted per instance schema.

Catalog blocking matters specifically because hidden snapshots for many instances live as separate schemas in one PostgreSQL database. Without it, catalog enumeration is a route from one instance to another instance's data.

### Sandbox escape

**Goal:** break out of the code runner.

Candidate code is the only component that executes arbitrary instructions, so it gets the strongest boundary — two layers:

**The container:** no network at all, read-only root filesystem, all capabilities dropped, `no-new-privileges`, non-root user, memory/CPU/PID limits, and an image with no compiler, no package manager and no network client.

**The process:** a fresh interpreter in isolated mode (`-I -S`) that ignores environment variables and keeps the current directory off the import path, with address-space, CPU-time, process-count, file-size and core-dump limits applied before `exec`.

Builtins are deliberately **not** restricted. Restricting them inside an already-isolated container is security theatre, and it would silently fail correct submissions that use `sum` or `max`.

### Prompt injection

**Goal:** use the workflow content to make the agent leak or misbehave.

Largely neutralised by architecture rather than filtering: the agent has no network, no tool that reveals correctness, and no route to the scorer. There is nothing to exfiltrate *to* and nothing to exfiltrate. A successful injection can make a candidate fail its own evaluation, which is a self-inflicted wound.

### Result fabrication by the operator

**Goal:** the operator pays whoever they like.

This is the honest weak point of a centralised engine, and it is mitigated rather than eliminated:

| Mitigation | What it gives |
|---|---|
| Signed reports | Nothing published can be repudiated, and nothing unsigned can be attributed |
| Published recipes and artifacts | Anyone can rebuild a champion's artifact and confirm its digest |
| Validators verify, not relay | Each validator checks the signature against **its own** allow-list and the vector against the chain it can see |
| Burn on doubt | A validator that cannot verify burns rather than submitting |
| Chain consensus | Weights still pass through the chain's own aggregation |
| Sample rows retained | Every per-instance outcome is stored, so a claimed aggregate can be checked against its parts |

| Closed-window disclosure | Once a window closes, its seeds and the traces the scorer read are published, so anyone can regenerate the instances and re-run the scorer over them without a GPU |

Since v1.0.0 the reported scores **can** be checked independently:
`capability-audit replay --window <n>` regenerates each disclosed instance from
its seed and re-scores the engine's own trace. A published score that does not
follow from its published trace is caught, specifically and by anyone.

What this still does **not** give: proof that the hidden draw was fair, or that a
trace faithfully records what the model did. A determined operator could publish
a fabricated trace that scores exactly as claimed — but it would have to stay
internally consistent, per turn, with a workflow the auditor regenerates
independently.

If that residual trust is unacceptable, it is better to know before registering than after.

### Forged weight vectors

**Goal:** get a validator to pay an attacker.

A validator refuses a vector that is unsigned, signed by a hotkey outside its allow-list, or whose signature does not verify. The signature covers canonical bytes that exclude the signature fields themselves, so signing is idempotent and tampering with any payload field invalidates it.

Beyond the signature, the validator checks the vector against the chain: weights summing to one, distinct UIDs, UIDs inside the subnet, the champion still holding its UID, and the vector not being many windows stale. **A correctly-signed vector can still be unsubmittable**, and a deregistered champion is the common case.

---

### Operator-side threats

| Threat | What stops it |
|---|---|
| Publishing scores the traces do not support | Validators re-score every closed window and burn on disagreement |
| Binding a draw to a block it did not open at | `verify_beacon_against_chain()` compares the beacon with the real block hash |
| Re-rolling the draw between windows | `check_draw_was_not_re_rolled()` compares the seed-root commitment across recent windows |
| Choosing one root dishonestly at genesis | **Not defended.** The root is never revealed, so a constant fake commitment passes |
| Quietly not re-measuring references | Every window's reference scores are published in its report |
| Withholding disclosures | Validators refuse a vector more than `max_stale_windows` behind the chain head |

## What is deliberately not defended

Being explicit about this is more useful than a longer list of defences.

**A miner's private search.** Miners may use any hardware and any method. The network judges the artifact, not the process. There is no attempt to detect or constrain how a recipe was found.

**Operator honesty about the hidden set.** Nothing proves the hidden instances were drawn fairly. The generator is published and the draw is deterministic given the root, but the root is secret, so the fairness of the draw rests on the operator.

**Availability of a miner's pointer.** If a recipe's host goes down before admission, the submission is not admitted. The engine keeps its own copy afterwards so a champion can keep defending, but it does not fetch pre-emptively.

**Model-level attacks on the base model.** The base is pinned upstream and taken as given.

**Collusion between the operator and a miner.** The mitigations above make it *detectable in principle* — published recipes, reproducible artifacts, retained sample rows — but nothing prevents it.

---

## Reporting a vulnerability

Report privately to the subnet operator rather than opening a public issue. Include what you found, how to reproduce it, and what it would let an attacker do.

Findings that affect scoring integrity — determinism, hidden-material exposure, or anything that lets a candidate influence its own evaluation — are the highest priority, because they invalidate results rather than merely disrupting service.

---

## Common questions

### Why is beating the champion not enough?

Because at genesis the throne is empty, and later a mediocre champion could hold it simply because nothing better challenged. The permanent references — base model, best single adapter, three equal-weight merges, the operator's own recipe — mean the network can never crown a package that an off-the-shelf merge already beats.

### Why must a challenger be "not worse" on every axis?

Otherwise a package could trade a capability away for a better average. Drop safety compliance, gain SQL accuracy: the mean improves and the package is worse at the job. The workflow needs every stage, so a package that abandoned one has not solved it.

### Why a paired bootstrap instead of just comparing scores?

With a hundred instances, a package that is genuinely no better than the incumbent scores higher about half the time. The question is not "did it score higher" but "is the difference larger than the noise."

Pairing works because both packages ran on **the same instances**, so instance difficulty drops out of the comparison entirely. That is a much tighter test, and it is only available because the engine controls the draw.

### Why does an axis with few samples count as *worse*?

Absence of evidence that a challenger kept a capability is not evidence that it did. Treating it as a tie would let a challenger win an axis by not being measured on it.

### Why does everything have to be byte-for-byte deterministic?

The artifact digest is the package's identity — the anti-copy check, the cache key, and the thing independent workers must agree on. Without determinism, "the champion's package" is not a well-defined object.

### What if two workers disagree on the artifact hash?

Evaluation of that candidate **pauses**. It is neither scored nor terminated, because scoring one of two disagreeing artifacts would mean paying for a result nobody can reproduce. The operator investigates the software mismatch.

### Why is no language model used to judge?

A model-based scorer carries its own variance, and two packages could differ on a scored axis without differing in behaviour at all. It would also make re-scoring a stored trace non-reproducible.

Every score here comes from comparing a trace with truth computed before the candidate saw the instance.

### How do the layer groups work if the base model has 36 layers?

There are always four groups splitting the decoder stack into quarters. The group *names* are protocol and never change; the layer ranges follow the pinned model's depth, so repinning to a model of different depth does not invalidate existing recipes. `miner.cli pool` prints the current ranges.

### Why are `svd` and `cat_svd` the same thing?

Because concatenating factorisations is algebraically the sum of the updates, so with no sparsification and no sign election there is one sensible combination. Both names are kept because both appear in the reference merge implementations this engine mirrors. It is documented rather than hidden.

`linear` is genuinely different — it sums the factors rather than their products.

---

### How long does a window take?

Roughly `(references + 1) × instances × per-instance-time`. With the default 100 hidden instances and around ten references, opening a window is the dominant cost. Size `window_blocks` accordingly.

### Can I change the comparator thresholds?

Yes, and they are published in the contract so miners can see them. Do not tune them in response to a specific candidate — that converts the engine from a measuring instrument into a decision-maker.

### What if a champion's recipe URL goes dead?

The engine keeps its own copy of every admitted recipe under `state/recipes/`, so a champion whose pointer went dead keeps defending.

### Can I run the engine without Docker?

Yes, but the container boundaries are the primary isolation for candidate-written code. The in-process resource limits are defence in depth, not a substitute.

### Do I earn anything while I wait in the queue?

A small, tapered share — most at the front of the queue, least at the back. It
is not payment for work: Bittensor prunes by lowest emission, so a miner holding
exactly zero is the first the chain evicts. The engine evaluates roughly one
challenger per window, so without it you could be deregistered before your
single evaluation ever ran.

### The champion looks unbeatable. Is the subnet finished?

No. The margin a challenger must clear over the *incumbent* decays to zero over
roughly thirty days, so an unopposed champion progressively loses its defender's
advantage. What does not decay is the margin over the permanent references —
beating an off-the-shelf equal-weight merge is the bar that says composition
added value at all, and that question does not get easier because someone
already answered it once.

### I did not take the throne. Did I earn anything?

Yes, if your package cleared every hard gate. The champion takes a fixed share
and everything below it is graded on quality, how far past the strongest
permanent reference you got, how close you came to the champion, and what your
package costs to run. That grade earns a proportional share for several windows,
and it is published broken into its four terms so you can see what earned it.

Clearing the gates is the threshold, and it is not negotiable. Grading applies
within the qualified set — it is not a consolation prize for producing something
undeployable.

### Does the subnet owner run code the rest of us cannot see?

Not for anything that decides a score. Everything that turns evidence into a
number — instance generation, the deterministic scorers, aggregation, the hard
gates, retention, the comparator, ranking, contribution and the weight vector —
is in this repository, and a test enforces that none of it depends on the
operator's engine.

The engine itself is operator-only: the window loop, candidate serving, the
store, the read-only API and the configuration surface. None of that changes what
a candidate scores, and a validator does not need it — it recomputes a published
score from published traces using the public rules, and burns rather than paying
for a number that does not follow.

What an operator keeps private beyond the engine is the hidden seed root, wallet
material, filled-in configuration, host inventory and runbooks.
