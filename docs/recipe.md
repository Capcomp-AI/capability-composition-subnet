# Recipe reference

Every field, every bound, every merge method. This is the complete specification of what a miner may submit.

The authoritative machine-readable version is always:

```bash
python -m capability_subnet.miner.cli contract --section recipe
```

---

## The document

```jsonc
{
  "schema_version": 1,
  "workflow_id": "industrial_maintenance_de_v1",
  "base_revision": "<pinned commit sha>",
  "source_snapshot_sha256": "sha256:...",

  "selected_adapters": [
    "embedded-engineering-v1",
    "chained-reasoning-v1",
    "industrial-ifc-v1",
    "code-generation-v1",
    "action-planner-v1",
    "constrained-selection-v1",
    "structured-explanation-v1"
  ],

  "merge": {
    "combination_type": "dare_ties_svd",
    "density": 0.35,
    "majority_sign_method": "total",
    "random_seed": 937152
  },

  "global_weights": {
    "industrial-ifc-v1": 1.20,
    "constrained-selection-v1": 1.15
  },

  "layer_group_overrides": {
    "group_2": { "industrial-ifc-v1": 1.30 }
  },

  "compression": {
    "output_rank": 64,
    "svd_clamp_quantile": 0.99
  },

  "output": {
    "dtype": "bfloat16",
    "adapter_name": "candidate"
  }
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

> The pool contains two **controlled distractors**. They are selectable on purpose. Recognising that a plausibly-relevant adapter actively hurts is part of the composition problem, and `miner.cli validate` warns when you select one.

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
recipe.effective_weight("industrial-ifc-v1", "group_2")
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

Two of these fail *without* ending your run, because they say the engine could
not measure you rather than that you fell short: an unreadable GPU memory
counter, and too few instances scored to compare on. Those hold your submission
for a later window with its single shot intact.

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
