# Miner guide

You have one job: find the composition of certified adapters that completes the workflow better than anything else on the board — then commit it once.

Your entire on-chain footprint is a single commitment. You never serve inference, never answer a query, and never run a process the network talks to. How you search is your own business.

---

## Before anything else: a commitment is measured once

Your commitment is evaluated in the run **after** the one you made it in, and
it earns from that measurement alone. It is not re-measured and it does not keep
earning afterwards. To earn again, commit again.

That gives you one evaluated attempt per run — a floor of one run between
attempts, because a commitment made after a run opened is not measured in it.
Nothing is terminated: a package that loses costs you that run, not the
hotkey.

The floor is what keeps copying expensive. Reading a published recipe, tweaking
it and resubmitting costs a full run per attempt, against an anti-copy check
that compares you to every commitment already admitted and a champion whose
margin you still have to clear.

So: validate locally, evaluate locally, and only then commit — a wasted
commitment costs you a run.

**Start from the worked example.** [`examples/quickstart_miner.py`](../examples/quickstart_miner.py)
does the whole loop in one file — builds valid recipes, rejects the inadmissible
ones, scores the survivors, writes the winner and prints its commitment:

```bash
python examples/quickstart_miner.py --tries 20 --out recipe.json
```

It runs with no GPU and no chain. Its *search* is random sampling, which is the
weakest search there is and the part you are meant to replace; everything around
it — validation, digests, commitment encoding, local scoring — is the part you
can rely on. See [examples/README.md](../examples/README.md).

> An *infrastructure* failure costs you nothing beyond the run. If a validator cannot serve your package or the sandbox falls over, you are not scored down for it — you are simply not measured, exactly as if you had not committed.

---

## What a run pays

Four fifths of every run burns to the subnet owner's UID. The remaining fifth
is the miner pool, and the best measured package takes 90% of it, and the graded runners-up split the rest by rank.

| Recipient | Share of the miner pool | Share of the run |
|---|---|---|
| Burn (subnet owner's UID) | — | 80% |
| Best measured package | 90% | 18% |
| Graded runners-up, ranks 2–10 | 10%, by rank | 2% |

Ten miners are paid at most: the leader and nine graded runners-up. Ranking is by
graded contribution — quality 50%, improvement over the strongest reference 25%,
proximity to the champion 15%, running cost 10% — over the packages that cleared
every hard gate. Anything nobody earned burns rather than being promoted into the
leader's share.

---

## 1. Install

```bash
git clone <repository-url> lora-merger && cd lora-merger
pip install -e .

# Add the reconstruction + evaluation extras if you have a GPU
pip install -e ".[miner]"
```

## 2. Learn the arena

Everything you are judged on is published. None of it is secret except the specific hidden instances.

```bash
# The frozen certified adapter pool
python -m capability_subnet.miner.cli pool

# The complete contract: bounds, gates, scoring weights, dethrone rule
python -m capability_subnet.miner.cli contract

# One section at a time
python -m capability_subnet.miner.cli contract --section champion_challenge
python -m capability_subnet.miner.cli contract --section hard_gates
```

The pool contains capability adapters **and controlled distractors**. The distractors are selectable on purpose: recognising that a plausible-looking adapter actively hurts is part of the composition problem. `miner.cli pool` marks them, and `miner.cli validate` warns when you select one.

## 3. Understand the problem

```bash
# One instance, with the ground truth revealed
python -m capability_subnet.workflows.cli show --seed 42 --with-truth

# Confirm the environment is solvable at all
python -m capability_subnet.workflows.cli selftest --count 10
```

Then generate the public development pack — 120 complete instances with ground truth, plus SQLite copies of each maintenance database:

```bash
python -m capability_subnet.workflows.cli generate-public-pack --out data/public_pack
```

The pack is reproducible from a published seed. Compare the printed tree digest against the published one to confirm your copy matches.

**What the pack gives you:** the exact generator, the exact tools, the exact scorer. You can reproduce the whole evaluation offline.
**What it does not give you:** the hidden draw. Local scores predict; they do not decide.

## 4. Build a recipe

```bash
python -m capability_subnet.miner.cli init --out recipe.json
```

Then edit it. The interesting fields:

| Field | What it controls |
|---|---|
| `selected_adapters` | Which specialists take part at all |
| `merge.combination_type` | How their updates are combined |
| `merge.density` | How much of each update survives before combining |
| `global_weights` | Per-adapter coefficient across the whole model |
| `layer_group_overrides` | Per-adapter coefficient at a specific depth |
| `compression.output_rank` | Artifact size and memory, paid for with reconstruction error |
| `compression.svd_clamp_quantile` | How much one direction may dominate |

See [Recipe reference](#recipe-reference) below for every field, bound and merge method.

Validate before you go further:

```bash
python -m capability_subnet.miner.cli validate --recipe recipe.json
```

This runs **exactly the checks the engine runs at admission** and prints both hard problems and advisories. Advisories are not rejections — they flag choices that are legal but usually unintended, such as selecting a distractor or leaving a stochastic merge on the default seed.

Check the artifact size before you build anything:

```bash
python -m capability_subnet.miner.cli size --recipe recipe.json
```

> Rank 128 produces roughly 666 MB against the pinned base model, which exceeds the 500 MB artifact gate. It is a legal value in the schema and an automatic rejection at evaluation. The `size` command tells you in a second.

## 5. Evaluate locally

Reconstruct your package with the same engine the network uses, serve it, and score it against public instances:

```python
from capability_subnet.miner.local_eval import evaluate_locally
from capability_subnet.miner.recipe import load_recipe
from capability_subnet.sandbox.model_client import OpenAICompatibleClient

recipe = load_recipe("recipe.json")

# Serve the reconstructed adapter yourself (vLLM, SGLang, whatever you use),
# then point the client at it.
client = OpenAICompatibleClient("http://127.0.0.1:8000", "candidate")

result = evaluate_locally(
    recipe,
    client,
    pool_dir="pool",
    artifact_dir="build/candidate",
    instance_count=40,
    ood_count=15,
)
print(result.summary())
```

Two things worth knowing:

- **The artifact digest you compute here is the digest the engine will compute.** If they differ, your host disagrees with the engine about determinism — worth finding out before committing, not after.
- **Twenty instances is not enough to distinguish two similar recipes.** The variance of an end-to-end completion rate over a small sample is wide. If you are comparing candidates that differ by a few points, you need many more instances or a cheaper proxy metric.

## 6. Search

No search is shipped. The starting point is a random valid recipe:

```python
from capability_subnet.miner.baseline import random_recipe

recipe = random_recipe(seed=1, adapter_count=4)
```

Equivalently, `miner.cli init --random`. It picks adapters at random and assigns
arbitrary coefficients — enough to have something that validates, and nothing
more. Building a search is the work.

Things worth investigating:

- **Which pairs interfere.** Two adapters that are individually good can cancel each other's updates. Sign election exists because of this.
- **Depth.** Reading a manual and writing SQL are not the same kind of behaviour and do not live in the same layers.
- **Density against rank.** Aggressive trimming plus a high rank is a different package from light trimming plus a low rank, even at the same artifact size.
- **What to leave out.** The best package is usually not the one with the most adapters.

The engine publishes a compatibility history at `/compatibility` — co-selection frequencies, marginal contributions, method and rank effects across every evaluation the network has run. It is the accumulated answer to exactly these questions.

## 7. Publish and commit

Publish the **exact recipe bytes** at an immutable, content-addressed location. Accepted pointer forms:

```
hf:<owner>/<repo>/<path>
ipfs:<cid>
https://<host>/<path>
```

The pointer is not trusted — the bytes are verified against your on-chain digest before anything is parsed — so a mutable host is safe from an integrity standpoint. It is not safe from an *availability* standpoint: if the engine cannot fetch your bytes, your submission is not admitted.

Write the file in canonical form so its digest matches what you commit:

```bash
python -m capability_subnet.miner.cli canonicalise --recipe recipe.json
python -m capability_subnet.miner.cli digest --recipe recipe.json
```

Preview the exact on-chain payload:

```bash
python -m capability_subnet.miner.cli commitment \
    --recipe recipe.json \
    --recipe-uri hf:alice/capsub-recipes/final.json
```

Then a dry run, which performs every check and touches nothing:

```bash
python neurons/miner.py \
    --netuid <netuid> \
    --wallet.name <coldkey> --wallet.hotkey <hotkey> \
    --recipe recipe.json \
    --recipe_uri hf:alice/capsub-recipes/final.json
```

And finally, when you are sure:

```bash
python neurons/miner.py ... --confirm
```

## 8. Track it

```bash
curl https://<engine-host>/queue/<your-hotkey>
```

| Status | Meaning |
|---|---|
| `queued` | Admitted, waiting. Challengers are evaluated in commit-block order. |
| `evaluating` | Being measured right now. |
| `champion` | It took the throne. |
| `terminated` | It lost, or failed a hard gate. This hotkey is finished. |
| *404* | Not admitted. Either the engine has not read the chain yet, or admission rejected it. |

The full evaluation report is published at `/reports` — every gate verdict, every per-axis comparison, the paired statistics, and the reason for the decision.

---

## Hardware

| What you are doing | What you need |
|---|---|
| Building and validating recipes | Any machine |
| Reconstructing an artifact | ~32 GB RAM. A GPU is optional and ~30x faster on the trimming methods. |
| Evaluating locally | A GPU that fits the base model in bfloat16 (24 GB+) |
| Searching seriously | As much as you want to spend — this is where competition happens |

See [min_compute.yml](../min_compute.yml) for detail.

## Common mistakes

**Committing a digest that does not match the canonical form.** The engine re-derives the digest from the parsed document. Use `canonicalise` and the two always agree.

**Choosing rank 128.** It exceeds the artifact-size gate against the pinned base model. Run `size` first.

**Omitting the retention adapter.** The base-retention gate rejects packages that traded away general ability for workflow score. It is measured on a held-out probe of short, exactly-scored general instructions — arithmetic, ordering, exact formats, answering in the language you were addressed in — so a package can score well on the workflow and still fail it. The retention anchor exists for exactly that.

**Tuning to the public pack.** The hidden set is drawn fresh each run and includes out-of-distribution mutations — renamed components, converted units, aliased database columns, reformatted fault codes. A package that memorised surface patterns fails on those, and out-of-distribution robustness carries 10% of the qualified score directly.

**Assuming only the throne pays.** It does not. Every package that clears all
hard gates is graded on quality, improvement over the strongest reference,
proximity to the champion, and running cost — and earns a share of emission for
several runs whether or not it dethroned anything. Getting close is worth
something; producing something undeployable is not. See
[what a run pays](#what-a-run-pays) for the split.

**Ignoring token spend.** It is now a scored component, and it is measured per
*completed* instance rather than per attempted one — so giving up early makes it
worse, not better. Two packages that finish the same fraction of workflows are
not equally valuable if one costs twice as much to run.

**Committing before evaluating.** A commitment is measured once and the next attempt is a run away. There is no reason to spend one on a package you have not measured.

**Assuming a loss is the end.** It is, if you were genuinely measured and genuinely lost. It is not when the engine could not evaluate you — an unreadable memory counter or too few scored instances holds your submission for a later run rather than terminating it. While it waits it earns a small share of emission, which is what keeps it from being deregistered before its turn comes.

---

# Recipe reference

## The document

Every value below is a placeholder, chosen to be neutral rather than good: equal
weights, no layer emphasis, a middling density. It shows the shape of the
document and nothing about which composition wins — that is the search, and it is
yours. `examples/quickstart_miner.py` writes a valid recipe you can run.

```jsonc
{
  "schema_version": 1,
  "workflow_id": "lora_merger_logic_v1",
  "base_revision": "<from: capability-miner pool>",
  "source_snapshot_sha256": "<from: capability-miner pool>",

  "selected_adapters": ["<adapter-id>", "<adapter-id>"],

  "merge": {
    "combination_type": "ties_svd",
    "density": 0.5,
    "majority_sign_method": "total",
    "random_seed": 0
  },

  "global_weights": { "<adapter-id>": 1.0, "<adapter-id>": 1.0 },

  "layer_group_overrides": {},

  "compression": {
    "output_rank": 64,
    "svd_clamp_quantile": 0.99
  },

  "output": {
    "dtype": "bfloat16",
    "adapter_name": "candidate"
  }
```

**No executable content is accepted, and unknown fields are rejected outright.**

---

## Fields

### Scope

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | Must be `1`. |
| `workflow_id` | string | Must match the pool's workflow. |
| `base_revision` | string | The pinned base commit. Fill it from the pool, not by hand. |
| `source_snapshot_sha256` | string | Digest of the frozen adapter pool. Same. |

Get both from `python -m capability_subnet.miner.cli pool`, or let `miner.cli init` fill them in. A recipe declaring a different snapshot was built against a pool that no longer exists and is rejected at admission.

### `selected_adapters`

Between **2 and 12** adapter IDs from the certified pool. Duplicates are rejected.

Order does not matter — reconstruction always loads in sorted identifier order, so the same set produces the same artifact however you wrote it.

A single adapter is not a candidate; it is one of the reference baselines.

> The pool contains **controlled distractors**. They are selectable on purpose. Recognising that a plausibly-relevant adapter actively hurts is part of the composition problem, and `miner.cli validate` warns when you select one.

### `merge`

| Field | Type | Bounds | Required for |
|---|---|---|---|
| `combination_type` | enum | see below | always |
| `density` | float | `0.05 … 1.0` | TIES, DARE and magnitude-prune families |
| `majority_sign_method` | `"total"` \| `"frequency"` | — | TIES family only |
| `random_seed` | int | `0 … 4294967295` | always (only *affects* stochastic methods) |

Supplying a parameter a method does not use is an error, not a silent ignore. That is deliberate: silently ignoring `density` on a linear merge would let a miner believe they were tuning something.

### `global_weights`

Per-adapter coefficient across the whole model. Range **`-2.0 … 2.0`**. Unlisted adapters default to `1.0`. Only selected adapters may appear.

Negative coefficients are legal and occasionally useful — subtracting an adapter's update is a real operation — but they interact badly with the base-retention gate, which measures general instruction-following on a held-out probe rather than anything about this workflow.

### `layer_group_overrides`

Per-adapter coefficient at a specific depth, keyed by group then adapter. Same range. An override **replaces** the global weight for that group.

There are always four groups, splitting the decoder stack into quarters:

```bash
python -m capability_subnet.miner.cli pool   # prints the layer ranges
```

The group *names* are part of the protocol and never change. The layer ranges behind them follow the pinned model's depth, so repinning to a model of different depth does not invalidate existing recipes.

This is where the interesting structure lives. Reading a manual and writing SQL are not the same kind of behaviour and do not live in the same layers.

### `compression`

| Field | Allowed | Effect |
|---|---|---|
| `output_rank` | `8, 16, 32, 48, 64, 96, 128` | Artifact size and memory, paid for with reconstruction error |
| `svd_clamp_quantile` | `0.90 … 1.00` | Caps how much one direction may dominate; `1.0` disables it |

> **Rank 128 exceeds the 500 MB artifact gate against the pinned base model** (≈666 MB). It is a legal schema value and an automatic rejection at evaluation. Run `miner.cli size --recipe recipe.json` before committing.

Setting `svd_clamp_quantile` below `1.0` forces the full decomposition path even for `linear`, because clamping is defined on the decomposition and has no factor-space equivalent.

### `output`

| Field | Value |
|---|---|
| `dtype` | `"bfloat16"` (fixed) |
| `adapter_name` | letters, digits, `-`, `_`; max 64 chars |

Both are fixed in V1 but stated explicitly so the recipe fully determines the artifact bytes.

---

## Merge methods

Each name is a preset over one three-stage pipeline: **sparsify → elect signs → aggregate**.

| Method | Sparsify | Sign election | Aggregate |
|---|---|---|---|
| `linear` | none | none | weighted sum of the factors |
| `svd` | none | none | weighted sum of the updates |
| `cat_svd` | none | none | weighted sum of the updates |
| `ties_svd` | magnitude trim | majority | disjoint mean |
| `dare_ties_svd` | random drop + rescale | majority | disjoint mean |
| `dare_linear_svd` | random drop + rescale | none | weighted sum |
| `magnitude_prune_svd` | magnitude trim | none | weighted sum |

### What each stage does

**Magnitude trim** keeps the largest-magnitude entries of each update and zeroes the rest. The cut is made at a *threshold* rather than by selecting exactly `k` entries — index selection has to break ties somehow, and different backends break them differently.

**Random drop and rescale** keeps a fraction `d` at random and divides the survivors by `d`, leaving the expected update unchanged while removing most of the entries that could interfere with another adapter. This is the only stage that uses `random_seed`.

**Sign election** decides, per entry, which direction the merged update moves in. `total` weighs each adapter by how much it wants to move; `frequency` gives every adapter one vote regardless of magnitude. Exact ties resolve positive.

**Disjoint mean** averages only over the adapters that agree with the elected sign. Averaging rather than summing keeps the merged magnitude comparable to a single adapter's: two adapters that both want to move an entry the same way should reinforce each other's confidence, not double the step size.

### Two honest notes

**`svd` and `cat_svd` produce the same update.** Concatenating factorisations is algebraically the sum of the updates, so with no sparsification and no sign election there is one sensible combination. Both names are kept because both appear in the reference merge implementations this engine mirrors.

**`linear` is genuinely different.** It sums the *factors* rather than their products, which carries cross terms between adapters. It is also the only method that needs no decomposition at all when the output rank matches the pool's, which makes it by far the cheapest to reconstruct.

---

## Coefficient resolution

```
layer group override  →  global weight  →  1.0
```

```python
recipe.effective_weight("<adapter-id>", "group_2")
```

---

## Where the parameters are applied

Order matters, and this order is fixed:

1. Each adapter's update is materialised as `ΔW = (α/r)·B·A`.
2. **Sparsification runs first, before the coefficient.** Applying the coefficient first would let a large coefficient protect entries that magnitude trimming should have discarded, which would make `density` mean different things at different coefficients.
3. The coefficient for this layer group is applied.
4. Signs are elected and the updates aggregated.
5. The result is decomposed back to `output_rank`, with clamping if requested.

The emitted adapter is written with `lora_alpha == rank`, so its own scaling factor is exactly 1 and `B @ A` *is* the update, with no hidden multiplier.

---

## Hard gates

Any failure zeroes the candidate.

| Gate | Requirement |
|---|---|
| Identity | Registered hotkey with a valid commitment |
| Digest | Recipe bytes match the on-chain digest |
| Schema | Valid V1 recipe |
| Source pool | Only adapters from the frozen snapshot |
| Anti-copy | Not a duplicate of an earlier commitment |
| Reconstruction | Independent workers agree on the artifact hash |
| Compatibility | Exact base model and module shapes |
| Numerical | No NaN or infinity |
| Security | No executable miner content |
| Artifact size | ≤ 500 MB |
| Peak VRAM | ≤ 24 GB |
| Latency | p95 workflow ≤ 30 s |
| Agent limits | ≤ 12 turns, ≤ 8192 output tokens |
| Safety | Zero critical unsafe actions |
| Stage floors | Every critical stage above its floor |
| Base retention | ≥ 98% of the base model's score on the general-capability probe |
| Baseline | Exceeds the strongest **permanent reference** by the absolute margin |
| Defender's margin | Exceeds the incumbent by its remaining, decaying margin |
| Statistics | Paired lower confidence bound above zero |

Two of these say the validator could not measure you rather than that you fell
short: an unreadable GPU memory counter, and too few instances scored to compare
on. Neither is scored against you — the run simply does not pay you, and the
next commitment is measured on its own terms.

---

## Committing

The on-chain payload is compact — the digest travels as unpadded base64url rather than hex to leave room for the pointer:

```
capsub1|imde|<43-char digest>|<uri>
```

Maximum **128 bytes**. Accepted pointer forms:

| Form | Example |
|---|---|
| Hugging Face | `hf:owner/repo/path/recipe.json` |
| IPFS | `ipfs:<cid>` |
| HTTPS | `https://host/path/recipe.json` |

The pointer is not trusted — bytes are verified against the digest before parsing — so a mutable host is safe for integrity. It is not safe for availability: if the engine cannot fetch your bytes, the submission is not admitted.

```bash
python -m capability_subnet.miner.cli commitment \
    --recipe recipe.json --recipe-uri hf:alice/recipes/final.json
```

---

## Validation

```bash
python -m capability_subnet.miner.cli validate --recipe recipe.json   # every admission check
python -m capability_subnet.miner.cli size     --recipe recipe.json   # predicted artifact size
python -m capability_subnet.miner.cli digest   --recipe recipe.json   # canonical digest
python -m capability_subnet.miner.cli canonicalise --recipe recipe.json
```

`validate` reports two kinds of finding. **Problems** would cause rejection at admission. **Advisories** are legal choices that are usually unintended — selecting a distractor, omitting the retention anchor, coefficients above 1.5 in magnitude, or a stochastic merge left on the default seed.

---

## Common questions

### What hardware do I need?

Building and validating recipes: any machine. Reconstructing an artifact: ~32 GB RAM; a GPU is optional but roughly thirty times faster on the trimming methods, which have to decompose a full update per projection. Evaluating locally: a GPU that fits the base model in bfloat16. Searching seriously: as much as you want to spend — that is where the competition is.

### How often can I submit?

Once per run, in effect. A commitment is measured in the run after the one
it was made in, so committing again immediately does not buy you a second
measurement in the same run — it replaces what will be measured in the next
one. Nothing is terminated, and a loss costs you that run rather than the
hotkey.

### Doesn't that make copying cheap?

It makes it slow, which is the part that matters. Reading a published recipe and
tweaking it costs a run per attempt, and each attempt still has to clear the
anti-copy check against every commitment already admitted and beat the champion
by its margin. Iterating toward a win that way is strictly more expensive than
searching properly, and it is visible the whole time.

### How do I know my recipe is valid before committing?

```bash
python -m capability_subnet.miner.cli validate --recipe recipe.json
```

This runs exactly the checks the engine runs at admission, and separates hard problems from advisories.

### Why is my artifact too large?

You probably chose rank 128. Against the pinned base model that produces roughly 666 MB, over the 500 MB gate. `miner.cli size` tells you in a second.

### Should I select the distractor adapters?

Almost certainly not — but they are selectable on purpose. One is a German legal-contract adapter, superficially relevant because the workflow is German and actually harmful. Recognising that is part of the problem.

### Can I use negative coefficients?

Yes, within `-2.0 … 2.0`. Subtracting an adapter's update is a real operation. It interacts badly with the base-retention gate — which measures general instruction-following on a held-out probe, not anything about this workflow — so measure it.

### How much do local scores predict hidden scores?

Directionally, quite well — same generator, same tools, same scorer. But the hidden set is drawn fresh each run and includes out-of-distribution mutations. And a completion rate over twenty instances has wide enough variance that two recipes differing by a few points are indistinguishable at that sample size.

### Why is my submission not in the queue?

Either the engine has not read the chain yet, or admission rejected it. Check `/queue/<your-hotkey>`; a 404 means it was not admitted. The usual causes are a stale snapshot digest, an unfetchable recipe URI, or a digest computed over non-canonical bytes.

### What is the compatibility history for?

It is the accumulated answer to the questions a grid search cannot answer: which adapter pairs reinforce each other, which interfere, which capabilities need a specialist at which depth, when trimming beats dropping, and how much rank the workflow actually needs.

```bash
curl https://<engine-host>/compatibility
```

---
