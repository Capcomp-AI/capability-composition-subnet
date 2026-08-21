# Subnet owner guide

What the owner runs, what everyone else runs, and the configuration the live subnet uses.

---

## What the owner has to run

Validators measure candidates on their own hardware, so the owner is not in the scoring path. Three things still belong to the owner, and only the first is mandatory.

### 1. The pool and the base model — mandatory, one-off

The certified adapter pool and the pinned base model are the shared ground every recipe is written against. The owner materialises them once, publishes the registry, and every miner and validator reproduces the same pool from it.

```bash
python scripts/import_public_adapters.py --out pool --write-registry
capability-registry snapshot
```

The snapshot digest that prints is the pool's identity. Every recipe declares it, and a recipe declaring a different one is rejected. Publish it alongside the registry.

This is a one-off per arena version. Re-running it is cheap: an adapter already on disk whose bytes still hash to the digest the registry records is verified and skipped.

### 2. The evaluation engine — optional

The engine runs runs, serves candidates, and publishes signed reports and a signed weight vector.

Nothing on the network depends on it. Validators measure for themselves, so no weight anyone sets comes from here — which is the point, and also why this is optional. Run it if you want a reference set of published numbers to compare validators against, a console for the subnet, or a disclosure feed that third parties can replay without a GPU. It needs its own 32 GB card and the same serving toolchain a validator needs.

```bash
capability-backend --config backend.yaml
capability-backend-api --config backend.yaml
```

If you run it, it needs a signing hotkey. Reports and weight vectors published unsigned are refused by any validator enforcing an allow-list, which means the engine runs, publishes, and moves no emission at all.

### 3. The console — optional

A read-only web front end that proxies to the engine. It carries no engine, no protocol package, no torch, and no key: a compromise of it yields a proxy and an HTML file. Deployed from `lora-merger-engine`, which is the only part of this system that belongs on a public platform.

### What the owner does *not* run

No inference endpoint for miners to query. No recipe hosting — miners publish their own. No per-miner infrastructure of any kind.

---

## Mainnet configuration

Netuid **103** on **finney**.

`backend.yaml`, for an owner running the optional engine:

```yaml
netuid: 103
network: finney
chain_endpoint: ""            # empty uses finney's public endpoint

wallet_name: capability
wallet_hotkey: default

# Commitments at or before this block are ignored. Set it to the block the arena
# opens at, so leftovers from a previous arena version cannot enter the queue.
min_commit_block: <current finney block at launch>

state_dir: /var/lib/capsub/state
adapter_pool_dir: /var/lib/capsub/pool
report_dir: /var/lib/capsub/state/reports

workflow_id: lora_merger_logic_v1
window_blocks: 21600          # ~72h at 12s blocks
hidden_instances: 1350
ood_instances: 100
end_to_end_margin: 0.03       # 1350 instances resolve ~0.024

base_model_path: /var/lib/capsub/base-model/Qwen3-8B
serving_gpu_uuid: "GPU-<uuid>"
serving_max_model_len: 8192
serving_gpu_memory_utilization: 0.5     # 48 GB card; see the formula below
serving_python: /opt/vllm/bin/python
serving_extra_args: "--kv-cache-dtype fp8 --max-num-seqs 32"

evaluator_image_digest: sha256:<digest of the built engine image>
reconstruction_workers: 2
merge_device: cuda
disclosure_traces: 10
```

`.env` carries the per-host details and the one secret. **Environment variables override the YAML**, so anything set here wins silently over the file above — check it first when a setting appears not to apply.

```bash
CAPSUB_NETUID=103
CAPSUB_NETWORK=finney
CAPSUB_HIDDEN_SEED_ROOT=<python3 -c "import secrets; print(secrets.randbits(63))">
CAPSUB_BASE_MODEL_PATH=/var/lib/capsub/base-model/Qwen3-8B
CAPSUB_ADAPTER_POOL_DIR=/var/lib/capsub/pool
CAPSUB_EVALUATOR_IMAGE_DIGEST=sha256:<digest>
```

`CAPSUB_HIDDEN_SEED_ROOT` is the root every hidden instance draw derives from. Anyone who learns it can predict every future hidden instance. Generate it once, keep it out of version control and out of logs, and never put it on the API surface.

### Sizing the serving fraction

`gpu_memory_utilization` is a fraction of the **whole card** and is a reservation rather than a requirement. Peak memory tracks it closely:

    peak ≈ gpu_memory_utilization × total_VRAM + 0.9 GiB

A candidate's peak is gated at 24 GiB, so the fraction follows from the card:

| Card | Fraction | Resulting peak |
|---|---|---|
| 80 GB | 0.25 | ~20.9 GiB |
| 48 GB | 0.42 | ~20.9 GiB |
| 32 GB | 0.66 | ~20.9 GiB |
| 24 GB | 0.91 | ~20.9 GiB |

The model needs about 15.3 GiB of weights and about 0.6 GiB of KV cache for the 8192 context at one sequence. A larger card needs a *smaller* fraction, not a larger one — the reservation is absolute, so peak lands near 20.9 GiB on every card that can hold it.

---

## Where miners submit recipes

A miner's entire on-chain footprint is one commitment. There is no upload endpoint and nothing the owner hosts.

**1. Publish the recipe JSON** anywhere fetchable, under 256 KB. Three URI schemes are accepted:

| Scheme | Form | Resolves to |
|---|---|---|
| Hugging Face | `hf:<owner>/<repo>/<path>` | `https://huggingface.co/<owner>/<repo>/resolve/main/<path>` |
| IPFS | `ipfs:<cid>` | `https://ipfs.io/ipfs/<cid>` |
| HTTPS | `https://...` | itself |

**2. Commit the digest and the URI on chain**, under the subnet's commitment key:

```bash
capability-miner commitment \
    --recipe recipe.json \
    --recipe-uri hf:<owner>/<repo>/final.json
```

which prints the payload to set:

```
capsub1|lmlg|<base64url digest>|hf:<owner>/<repo>/final.json
```

Format is `capsub1|<workflow code>|<recipe digest>|<uri>`. Validators read the commitment from chain, fetch the URI, and check the bytes against the committed digest — so the recipe cannot be swapped after commitment, and the URI only has to stay reachable.

A commitment gets one evaluation, and a hotkey may resubmit once per run for another. Publishing the file is not the commitment; the on-chain payload is.

---

## The adapter pool

Thirty adapters, of which twenty-six are selectable: twenty-three capability adapters and three distractors that are selectable on purpose, because recognising that they hurt is part of the composition problem.

Capability measurement does **not** gate selection. An adapter that is structurally sound but that nobody has characterised sits in the pool and can be selected; its measurements, where they exist, are published as information a miner can use rather than a door they must pass.

Structural admission does not relax. Safetensors only, the canonical config, the pinned base revision, correct shapes, finite values, and a licence that permits redistributing derivative weights. These tensors are loaded into a process that also holds hidden evaluation material, and no incentive argument touches that.

Four adapters are held out of selection on the licence gate: their upstream licence is unstated, so it is not on the reviewed list. Admitting them is a legal decision, not a technical one. If the owner completes that review, record the reviewed licence for those entries and they become selectable like any other.

---

## Launch checklist

- [ ] Pool imported, registry published, snapshot digest announced
- [ ] Base model materialised at the pinned revision on every measuring host
- [ ] `min_commit_block` set to the block the arena opens at
- [ ] `CAPSUB_HIDDEN_SEED_ROOT` generated, kept out of git and logs
- [ ] Serving fraction set from the card so peak lands under 24 GiB
- [ ] Signing hotkey configured, if the engine is being run
- [ ] Validators told to install `capability-subnet[merge]` and run `own` mode
- [ ] Spec version bumped if any consensus constant changed
