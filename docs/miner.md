# Miner guide

You have one job: find the composition of certified adapters that completes the workflow better than anything else on the board — then send it.

Your whole interaction with the network is one HTTP request. **Nothing goes on chain and nothing is published anywhere.** You never serve inference, never answer a query, and never run a process the network talks to. How you search is your own business.

You need a registered hotkey and nothing else: no commitment, no transaction, no fee, no wallet unlocked for anything but signing a short string. Your recipe travels in the request body, signed by that hotkey, and is held privately until the run that pays it opens — so nobody can read it, let alone copy it, while it is being measured.

```bash
capcomp submit --recipe recipe.json \
    --wallet.name <coldkey> --wallet.hotkey <hotkey> --confirm
```

That is the entire protocol contract. The rest of this guide is about making the recipe good.

---

## Before anything else: a run is a day, and the pipeline is three runs deep

A run is **7200 blocks — 24 hours**, and boundaries are anchored: run 412 opens
at block 8,908,667, which the chain reaches at about **12:00 Eastern on Sunday
23 August 2026**, and every run after it opens 24 hours later. So a run opens at
about noon Eastern, every day.

Your recipe then walks three runs:

| | what happens |
|---|---|
| **run N** | you submit |
| **run N+1** | it is evaluated and its score recorded |
| **run N+2** | that score sets the weight, and your recipe and its score become public |

The gap between measuring and paying is not a delay for its own sake. A weight
vector is a statement about a **closed** run's leaderboard. Paying inside the
run doing the measuring means paying from a leaderboard still being written — a
candidate measured early faces an empty field, one measured late faces a full
one, and the vector moves under both as the queue is worked through. A closed
run has a final leaderboard, and every validator reading the same chain
computes the same vector from it.

Your submission earns from that one measurement alone. It is not re-measured
and it does not keep earning afterwards. To earn again, submit again — which
gives you one evaluated attempt per day, because a recipe sent after a run
opened is not measured in it. Nothing is terminated: a package that loses costs
you that run, not the hotkey.

**Three submissions per run.** Only your last one is measured, and only it is
stored — the ones it replaced survive as digests and a count, so you can always
check what you used. Re-sending an identical recipe does not cost an attempt, so
retrying a request that timed out is safe.

The limit is real because there is now a record to enforce it against. It is
high enough to correct a mistake and low enough that you cannot iterate against
the measurement: a budget you could search with would turn the run into a free
evaluation service.

**Do not replace your recipe once the run you submitted in has closed.** A
recipe sent in run N is measured in run N+1. Sending a new one *during* N+1
files it under N+1, so N+1 measures nothing from you and the new recipe waits
for N+2 — and its payment for N+3. You did not fail a gate or lose a comparison:
you withdrew the submission that was about to be judged, and you skip a day.

**Submissions close an hour before the run does.** A submission must have been
in for `MIN_COMMITMENT_AGE_BLOCKS` — 300 blocks, about an hour — when the run
that would measure it opens. Inside that last hour the API **refuses it**:
`409`, nothing stored, no attempt spent. Send it again once the next run opens.

You are told at once, while you can still act on it, rather than watching a
blank row sit through a run you thought you had entered.

The window also removes the advantage of submitting at the closing block after
watching the whole run — every result published and every recipe disclosed —
before choosing.

Ask the tooling rather than doing the arithmetic. Pass the current block and
`capcomp timing` says which run will measure a submission made now and how
long you have left:

```bash
capcomp timing --block $(btcli subnet show --netuid 103 | grep -i block)

# run 412: measured in run 413, paid in run 414. 6890 blocks (~23.0h) left to
# change your mind.
```

Inside the closing window it says so instead:

```
run 412 closes in 200 blocks, inside the 60-minute settling window.
A submission made now is refused: it has to have been in for 300 blocks when a
run opens to be measured by it, and it cannot be.
Nothing is stored and no attempt is spent. Wait for run 413 to open, then send
it — it is measured in run 414.
```

Add `--strict-timing` to make that a non-zero exit, so a script does not submit
into a run that will not measure it.

The rule in three lines: **replace it freely early in the run, stop an hour
before it closes — after that it is refused — then leave it alone until you
have been measured.**

The floor is what keeps copying expensive. Reading a published recipe, tweaking
it and resubmitting costs a full run per attempt, against an anti-copy check
that compares you to every submission already admitted and a champion whose
margin you still have to clear.

So: validate locally, evaluate locally, and only then submit — a wasted
attempt costs you a run.

**Start from the worked example.** [`examples/quickstart_miner.py`](../examples/quickstart_miner.py)
does the whole loop in one file — builds valid recipes, rejects the inadmissible
ones, scores the survivors, writes the winner and prints its digest:

```bash
python examples/quickstart_miner.py --tries 20 --out recipe.json
```

It needs no chain, and its scoring step needs a card like any other. Its
*search* is random sampling, which is the weakest search there is and the part
you are meant to replace; everything around it — validation, digests, signing,
local scoring — is the part you can rely on. See
[examples/README.md](../examples/README.md).

> An *infrastructure* failure costs you nothing beyond the run. If a validator cannot serve your package or the sandbox falls over, you are not scored down for it — you are simply not measured, exactly as if you had not submitted.

---

## What a run pays

**A run pays whatever cleared its hard gates.** Clear them and you are ranked
by grade and paid by rank. The bar that matters is the entry gate — **0.02** of
end-to-end completion over the strongest permanent reference — and it is
absolute, so whether you earn depends on your package rather than on how strong
some earlier run happened to be. A run where nothing clears the gates burns
entirely.

**Taking the throne is a separate thing**, and it does not change what you are
paid. The leader takes it by exceeding the reigning champion's grade by
**0.0005**; that decides who is recorded as champion for the network, and
nothing else.

No fixed share of a run burns. A run that produces a new champion and fills
every paid rank pays its **whole emission** to miners:

| Rank | Share of the run |
|---|---|
| 1st | 90% |
| 2nd | 5% |
| 3rd | 3% |
| 4th | 1% |
| 5th | 0.5% |
| 6th–10th | 0.5%, in proportion to grade |

Ten miners are paid at most. A run can still burn, in two ways, and both are
deliberate:

- **No throne, nothing paid.** If no candidate clears the reigning grade the
  entire run burns.
- **An unfilled rank in the first five burns** rather than being promoted into
  the leader's share, so a field of five pays 99.5% and burns 0.5%. The
  sixth-to-tenth share is different: it is split across whoever occupies those
  ranks, so it is paid in full as soon as any one of them is filled.

Ranking is by **grade**, one number per candidate over the packages that
cleared every hard gate:

| Term | Weight | What it rewards |
|---|---|---|
| Quality | 50% | The qualified score below |
| Improvement | 40% | How far past the base model the package got |
| Cost | 10% | Token spend |

The qualified score is itself weighted: end-to-end completion 55%, stage
balance 15%, out-of-distribution 10%, token efficiency 10%, base retention 5%,
artifact efficiency 5%. So end-to-end reaches the grade twice — through quality
and again through improvement — and is roughly two-thirds of what decides rank.

Stage balance is the geometric mean of your score across the twelve capability
axes: it punishes gaps rather than averaging them away, so being absent on a few
axes costs more than being merely mediocre everywhere.

The other terms are hard gates first and scored terms second: clearing their
floors is required, exceeding them is worth very little.

Every term is measured against the run's own instances and the base model,
never against the incumbent, so a grade means the same thing in every run.

The throne does not move on its own. A run nobody wins leaves it where it was,
so the next run faces the same grade rather than a rising one.

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
capcomp pool

# The complete contract: bounds, gates, scoring weights, dethrone rule
capcomp contract

# One section at a time
capcomp contract --section champion_challenge
capcomp contract --section hard_gates
```

The pool contains capability adapters **and controlled distractors**. The distractors are selectable on purpose: recognising that a plausible-looking adapter actively hurts is part of the composition problem. `capcomp pool` marks them, and `capcomp validate` warns when you select one.

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
capcomp init --out recipe.json
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
capcomp validate --recipe recipe.json
```

This runs **exactly the checks the engine runs at admission** and prints both hard problems and advisories. Advisories are not rejections — they flag choices that are legal but usually unintended, such as selecting a distractor or leaving a stochastic merge on the default seed.

Check the artifact size before you build anything:

```bash
capcomp size --recipe recipe.json
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

- **The artifact digest you compute here is the digest the engine will compute.** If they differ, your host disagrees with the engine about determinism — worth finding out before submitting, not after.
- **Twenty instances is not enough to distinguish two similar recipes.** The variance of an end-to-end completion rate over a small sample is wide. If you are comparing candidates that differ by a few points, you need many more instances or a cheaper proxy metric.

## 6. Search

No search is shipped. The starting point is a random valid recipe:

```python
from capability_subnet.miner.baseline import random_recipe

recipe = random_recipe(seed=1, adapter_count=4)
```

Equivalently, `capcomp init --random`. It picks adapters at random and assigns
arbitrary coefficients — enough to have something that validates, and nothing
more. Building a search is the work.

Things worth investigating:

- **Which pairs interfere.** Two adapters that are individually good can cancel each other's updates. Sign election exists because of this.
- **Depth.** Reading a manual and writing SQL are not the same kind of behaviour and do not live in the same layers.
- **Density against rank.** Aggressive trimming plus a high rank is a different package from light trimming plus a low rank, even at the same artifact size.
- **What to leave out.** The best package is usually not the one with the most adapters.

The engine publishes a compatibility history at `/compatibility` — co-selection frequencies, marginal contributions, method and rank effects across every evaluation the network has run. It is the accumulated answer to exactly these questions.

## 7. Submit

There is nothing to publish and nothing to commit.

**The API is the only way in.** A recipe written to the chain as a commitment,
or left at an `hf:` or `https:` URL for the engine to fetch, is not a
submission: it is not admitted, not stored and not scored. That was the route
before miners moved to signing a request body, and nothing reads it now — a
commitment made today produces no queue entry and no scoreboard row, and you
will see no error, because nothing is looking at it to raise one. If `capcomp
submit` did not return success, you are not in the run.

**First, ask whether it would be admitted.** This costs nothing — no signature,
no hotkey, no attempt — and it is the engine's own contract answering, not a
local approximation:

```bash
capcomp check --recipe recipe.json
```

```
ok — this recipe would be admitted
```

Iterate until it says that. Then send it:

```bash
capcomp submit --recipe recipe.json \
    --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

Without `--confirm` this is a dry run: it validates the recipe, reads the run
from the API, prints the digest, says what it would replace and how many of the
run's three attempts you would have left — and sends nothing.

```
run 412: measured in run 413, paid in run 414
digest      sha256:38f6428d92f241679146202cc68bba9372591e5916c0085fdf49f3ba3c805576
attempts    none used, 3 available this run

Nothing was sent. Re-run with --confirm to submit.
```

Add `--confirm` when you are sure.

The digest is the canonical digest of your recipe — the same number
`capcomp digest` prints, and the same one the engine identifies your
package by. You sign a string binding that digest to the run, so a signature
cannot be replayed into a later run or against a different recipe. The client
serialises the recipe canonically before sending, so formatting in your editor
is irrelevant and there is exactly one number to check.

`--api.url` (or `CAPSUB_API_URL`) points at the submission service if you need
to override it.

## 8. Track it

```bash
capcomp status --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

```
run 412: measured in run 413, paid in run 414
  holding   sha256:38f6428d…
  attempts  1 used, 2 left
```

Or over HTTP:

```bash
curl https://<api-host>/status/<your-hotkey>
```

```json
{"run_id": 412, "submitted": true, "submission_count": 1,
 "remaining": 2, "recipe_sha256": "sha256:38f6428d…", "superseded": []}
```

`superseded` lists the digests of recipes this one replaced, so the attempt
count is checkable rather than something you are told.

Your recipe, its score and the run's weight vector all become public when the
run that pays it opens — two runs after the one you submitted in:

| route | what it gives |
|---|---|
| `GET /run/{id}` | what was submitted, and by whom |
| `GET /run/{id}/results` | every score, gate verdict and rank |
| `GET /run/{id}/weights` | what the run pays |
| `GET /run/{id}/instances/{hotkey}` | what each package was asked and answered |

Ask before then and they answer `released: false` with the run they open in,
rather than an error.

The full evaluation report is published at `/reports` — every gate verdict, every per-axis comparison, the paired statistics, and the reason for the decision.

**Your score appears the day after you submit; the emission the day after
that.** A weight vector states a closed run's leaderboard, so the run that
measures you is not the run that pays you. Seeing a report with a good score
and no emission on the same day is the pipeline working, not a failure.

---

## Hardware

Submitting takes any machine: a recipe is a few hundred bytes of JSON, signed
by your hotkey and sent over HTTP.

Evaluating takes a 32 GB GPU, and evaluating is how you find a recipe worth
submitting. You have three submissions a run and each unmeasured guess spends
one. The entry gate alone is 0.02 of end-to-end completion over the strongest
reference, and adjacent paid ranks are routinely under 0.001 apart.

| What you are doing | What you need |
|---|---|
| Building and validating recipes | Any machine |
| Reconstructing an artifact | ~32 GB RAM. A GPU is optional and ~30x faster on the trimming methods. |
| Evaluating locally | A GPU that can hold the 24 GiB serving reservation (32 GB+) |
| Searching seriously | As much as you want to spend — this is where competition happens |

See [min_compute.yml](../min_compute.yml) for detail.

## Common mistakes

**Assuming your file's formatting matters.** It does not: the client serialises your recipe canonically before signing and sending, so the digest is the same whatever your editor did. `canonicalise` writes that form to disk if you want to see it.

**Choosing rank 128.** It exceeds the artifact-size gate against the pinned base model. Run `size` first.

**Omitting the retention adapter.** Retention carries 5% of the qualified score. It is measured on a held-out probe of short, exactly-scored general instructions — arithmetic, ordering, exact formats, answering in the language you were addressed in — so a package can score well on the workflow and still lose ground here. The retention anchor exists for exactly that.

**Tuning to the public pack.** The hidden set is drawn fresh each run and includes out-of-distribution mutations — renamed components, converted units, aliased database columns, reformatted fault codes. A package that memorised surface patterns fails on those, and out-of-distribution robustness carries 10% of the qualified score directly.

**Assuming only the throne pays.** It does not. Every package that clears all
hard gates is graded on quality, improvement over the strongest reference,
and running cost — and earns a share of emission for
several runs whether or not it dethroned anything. Getting close is worth
something; producing something undeployable is not. See
[what a run pays](#what-a-run-pays) for the split.

**Ignoring token spend.** It is now a scored component, and it is measured per
*completed* instance rather than per attempted one — so giving up early makes it
worse, not better. Two packages that finish the same fraction of workflows are
not equally valuable if one costs twice as much to run.

**Submitting before evaluating.** Only your last recipe is measured and you get three a run. There is no reason to spend one on a package you have not measured.

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
  "base_revision": "<from: capcomp pool>",
  "source_snapshot_sha256": "<from: capcomp pool>",

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

Get both from `capcomp pool`, or let `capcomp init` fill them in. A recipe declaring a different snapshot was built against a pool that no longer exists and is rejected at admission.

### `selected_adapters`

Between **2 and 10** adapter IDs from the frozen pool. Duplicates are rejected.

All 30 pool adapters are selectable; certification does not gate selection. The
bound is `MIN_SELECTED_ADAPTERS`/`MAX_SELECTED_ADAPTERS` and is exported in the
published JSON Schema as `minItems`/`maxItems`, so `capcomp validate`
catches a violation before you submit.

Order does not matter — reconstruction always loads in sorted identifier order, so the same set produces the same artifact however you wrote it.

A single adapter is not a candidate; it is one of the reference baselines.

> The pool contains **controlled distractors**. They are selectable on purpose. Recognising that a plausibly-relevant adapter actively hurts is part of the composition problem, and `capcomp validate` warns when you select one.

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

Negative coefficients are legal and occasionally useful — subtracting an adapter's update is a real operation — but they are hard on retention, which measures general instruction-following on a held-out probe rather than anything about this workflow. It is scored and published, so you can see what the subtraction cost.

### `layer_group_overrides`

Per-adapter coefficient at a specific depth, keyed by group then adapter. Same range. An override **replaces** the global weight for that group.

There are always four groups, splitting the decoder stack into quarters:

```bash
capcomp pool   # prints the layer ranges
```

The group *names* are part of the protocol and never change. The layer ranges behind them follow the pinned model's depth, so repinning to a model of different depth does not invalidate existing recipes.

This is where the interesting structure lives. Reading a manual and writing SQL are not the same kind of behaviour and do not live in the same layers.

### `compression`

| Field | Allowed | Effect |
|---|---|---|
| `output_rank` | `8, 16, 32, 48, 64, 96, 128` | Artifact size and memory, paid for with reconstruction error |
| `svd_clamp_quantile` | `0.90 … 1.00` | Caps how much one direction may dominate; `1.0` disables it |

> **Rank 128 exceeds the 500 MB artifact gate against the pinned base model** (≈666 MB). It is a legal schema value and an automatic rejection at evaluation. Run `capcomp size --recipe recipe.json` before submitting.

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
| Identity | Registered hotkey, and a signature that matches it |
| Digest | Recipe bytes match the digest you signed |
| Schema | Valid V1 recipe |
| Source pool | Only adapters from the frozen snapshot |
| Anti-copy | Not a duplicate of an earlier submission |
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
next submission is measured on its own terms.

---

## Submitting

One request. The recipe is the body, signed by your hotkey:

```
POST /submit
{"hotkey": "5…", "recipe": "<the recipe JSON>", "signature": "0x…"}
```

The signature is over this exact string, where the run is the one the API is
currently in and the digest is your recipe's canonical digest:

```
capcomp-submit:v1:<run_id>:sha256:<hex>
```

Both are inside it, so a signature cannot be replayed into a later run or
against a different recipe. `capcomp submit` does all of this; the shape is
here so you can build your own client if you would rather.

| | |
|---|---|
| Attempts per run | 3, and only the last is measured |
| Identical resend | Free — it costs no attempt |
| Max recipe size | 256 KB |
| Refusals | `401` signature, `403` not registered, `429` no attempts left |

`GET /contract` returns all of this as JSON.

---

## Validation

```bash
capcomp validate --recipe recipe.json   # every admission check
capcomp size     --recipe recipe.json   # predicted artifact size
capcomp digest   --recipe recipe.json   # canonical digest
capcomp canonicalise --recipe recipe.json
```

`validate` reports two kinds of finding. **Problems** would cause rejection at admission. **Advisories** are legal choices that are usually unintended — selecting a distractor, omitting the retention anchor, coefficients above 1.5 in magnitude, or a stochastic merge left on the default seed.

---

## Common questions

### What hardware do I need?

Building and validating recipes: any machine. Reconstructing an artifact: ~32 GB RAM; a GPU is optional but roughly thirty times faster on the trimming methods, which have to decompose a full update per projection. Evaluating locally: a GPU that fits the base model in bfloat16. Searching seriously: as much as you want to spend — that is where the competition is.

### How often can I submit?

Three times a run, and once a day in effect. A submission is measured in the run
after the one it was made in, so sending another immediately does not buy you a
second measurement in the same run — it replaces what will be measured in the
next one. Nothing is terminated, and a loss costs you that run rather than the
hotkey.

### Doesn't that make copying cheap?

There is nothing to copy while it matters. Recipes are held privately and
published two runs after they were submitted — by which point the run they
competed in is paid and closed. A recipe you can read is one that has already
earned what it was going to earn.

Copying an old one is still possible and still slow: it costs a run per attempt,
has to clear the anti-copy check against every submission already admitted, and
has to beat the champion by its margin. Iterating toward a win that way is
strictly more expensive than searching properly.

### How do I know my recipe is valid before submitting?

```bash
capcomp validate --recipe recipe.json
```

This runs exactly the checks the engine runs at admission, and separates hard problems from advisories.

### Why is my artifact too large?

You probably chose rank 128. Against the pinned base model that produces roughly 666 MB, over the 500 MB gate. `capcomp size` tells you in a second.

### Should I select the distractor adapters?

Almost certainly not — but they are selectable on purpose. One is a German legal-contract adapter, superficially relevant because the workflow is German and actually harmful. Recognising that is part of the problem.

### Can I use negative coefficients?

Yes, within `-2.0 … 2.0`. Subtracting an adapter's update is a real operation. It is hard on retention — which measures general instruction-following on a held-out probe, not anything about this workflow — so measure it. Retention is scored, not gated, so it costs you rather than refusing you.

### How much do local scores predict hidden scores?

Directionally, quite well — same generator, same tools, same scorer. But the hidden set is drawn fresh each run and includes out-of-distribution mutations. And a completion rate over twenty instances has wide enough variance that two recipes differing by a few points are indistinguishable at that sample size.

### Why is my submission not in the queue?

Admission rejected it, or it never arrived. Check `/queue/<your-hotkey>`; a 404 means it was not admitted. The usual causes are a stale snapshot digest, a recipe that fails the schema, or a digest computed over bytes other than the ones you sent.

If you wrote a commitment to the chain rather than calling `capcomp submit`, that is the reason: the engine does not read commitments. See [Submit](#7-submit).

### What is the compatibility history for?

It is the accumulated answer to the questions a grid search cannot answer: which adapter pairs reinforce each other, which interfere, which capabilities need a specialist at which depth, when trimming beats dropping, and how much rank the workflow actually needs.

```bash
curl https://<engine-host>/compatibility
```

---
