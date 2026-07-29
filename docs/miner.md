# Miner guide

You have one job: find the composition of certified adapters that completes the workflow better than anything else on the board — then commit it once.

Your entire on-chain footprint is a single commitment. You never serve inference, never answer a query, and never run a process the network talks to. How you search is entirely your business.

---

## Before anything else: one recipe per hotkey is final

A hotkey gets **one evaluation**. When your submission reaches the head of the queue it is evaluated against the champion and the reference baselines, and:

- if it wins, it takes the throne,
- if it loses decisively, that hotkey is **terminated permanently**.

There is no resubmission, no second attempt, no "I'll fix it next window." A new package needs a new hotkey, which costs a registration. That is deliberate — it is what makes copying a published recipe worthless.

So: validate locally, evaluate locally, and only then commit.

> An *infrastructure* failure never costs you your shot. If the engine cannot serve your package or the sandbox falls over, your submission returns to the queue untouched. Only a measured loss terminates you.

---

## 1. Install

```bash
git clone <repository-url> lora-merger && cd lora-merger
pip install -e .

# Add the local search + evaluation extras if you have a GPU
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

The pool contains capability adapters **and two controlled distractors**. The distractors are selectable on purpose: recognising that a plausible-looking adapter actively hurts is part of the composition problem. One is a legal-citation adapter — superficially relevant because it is formal and heavily structured, actually harmful here. The other is creative writing, which is directly antagonistic to strict structured output.

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

See [the recipe reference](recipe.md) for every field, bound and merge method.

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

There is a reference search to get you started, and it will not win:

```python
from capability_subnet.miner.search import coarse_search, refine_layer_groups

report = coarse_search(my_scoring_function, budget=24, shuffle_seed=1)
refined = refine_layer_groups(
    report.best.recipe, my_scoring_function,
    adapters=["industrial-ifc-v1", "embedded-engineering-v1"],
)
```

A grid does not see the interactions between adapters, and the interactions are where the value is. Things worth investigating that a grid will not find for you:

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

**Tuning to the public pack.** The hidden set is drawn fresh each window and includes out-of-distribution mutations — renamed components, converted units, aliased database columns, reformatted fault codes. A package that memorised surface patterns fails on those, and out-of-distribution robustness carries 10% of the qualified score directly.

**Assuming only the throne pays.** It does not. Every package that clears all
hard gates is graded on quality, improvement over the strongest reference,
proximity to the champion, and running cost — and earns a share of emission for
several windows whether or not it dethroned anything. Getting close is worth
something; producing something undeployable is not.

**Ignoring token spend.** It is now a scored component, and it is measured per
*completed* instance rather than per attempted one — so giving up early makes it
worse, not better. Two packages that finish the same fraction of workflows are
not equally valuable if one costs twice as much to run.

**Committing before evaluating.** One shot per hotkey. There is no reason to spend it on a package you have not measured.

**Assuming a loss is the end.** It is, if you were genuinely measured and genuinely lost. It is not when the engine could not evaluate you — an unreadable memory counter or too few scored instances holds your submission for a later window rather than terminating it. While it waits it earns a small share of emission, which is what keeps it from being deregistered before its turn comes.
