# Subnet owner guide

What the owner runs, what everyone else runs, and the configuration the live subnet uses.

---

## What the owner has to run

Validators measure candidates on their own hardware, so the owner is not in the scoring path. Three things still belong to the owner, and only the first is mandatory.

### 1. The pool and the base model - mandatory, one-off

The certified adapter pool and the pinned base model are the shared ground every recipe is written against. The owner materialises them once, publishes the registry, and every miner and validator reproduces the same pool from it.

```bash
python scripts/import_public_adapters.py --out pool --write-registry
capability-registry snapshot
```

The snapshot digest that prints is the pool's identity. Every recipe declares it, and a recipe declaring a different one is rejected. Publish it alongside the registry.

This is a one-off per arena version. Re-running it is cheap: an adapter already on disk whose bytes still hash to the digest the registry records is verified and skipped.

### 2. The evaluation engine - optional

The engine runs runs, serves candidates, and publishes signed reports and a signed weight vector.

Nothing on the network depends on it. Validators measure for themselves, so no weight anyone sets comes from here - which is the point, and also why this is optional. Run it if you want a reference set of published numbers to compare validators against, a console for the subnet, or a disclosure feed that third parties can replay without a GPU. It needs its own 32 GB card and the same serving toolchain a validator needs.

```bash
python -m capability_engine.service --config backend.yaml   # control loop
python -m capability_engine.api     --config backend.yaml   # read-only API
```

The engine ships in `lora-merger-engine`, not in this package; see
[engine operations](https://github.com/Capcomp-AI/lora-merger-engine/blob/main/docs/operations.md).

If you run it, it needs a signing hotkey. Reports and weight vectors published unsigned are refused by any validator enforcing an allow-list, which means the engine runs, publishes, and moves no emission at all.

### 3. The console - optional

A read-only web front end that proxies to the engine. It carries no engine, no protocol package, no torch, and no key: a compromise of it yields a proxy and an HTML file. Deployed from `lora-merger-engine`, which is the only part of this system that belongs on a public platform.

### What the owner does *not* run

No inference endpoint for miners to query. No recipe hosting - miners publish their own. No per-miner infrastructure of any kind.

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
run_blocks: 7200           # 24h at 12s blocks
run_epoch_block: 8908667   # run 412 opens here, ~12:00 ET on 23 Aug 2026
run_epoch_id: 412          # everything before it is frozen history
min_commitment_age_blocks: 300
hidden_instances: 1350
ood_instances: 100
end_to_end_margin: 0.02       # 1350 resolve ~0.024; the bar is arithmetic, not a test

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

`.env` carries the per-host details and the one secret. **Environment variables override the YAML**, so anything set here wins silently over the file above - check it first when a setting appears not to apply.

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

`gpu_memory_utilization` is a fraction of the **whole card** and is a reservation rather than a requirement. What vLLM sets aside tracks it closely:

    reserved ≈ gpu_memory_utilization × total_VRAM

A candidate reserves a fixed **24 GiB** to serve - peak memory is deliberately neither gated nor scored, and it lands near 21 GiB whatever the package merged - so the fraction follows from the card:

| Card | Fraction | Reserved |
|---|---|---|
| 80 GB | 0.30 | ~24 GiB |
| 48 GB | 0.50 | ~24 GiB |
| 32 GB | 0.78 | ~25 GiB |

A 24 GB card cannot serve: it does not clear the reservation plus the driver context and the merge sharing the card, so the smallest card a validator may use is 32 GB.

The model needs about 15.3 GiB of weights and about 0.6 GiB of KV cache for the 8192 context at one sequence. A larger card needs a *smaller* fraction, not a larger one - the reservation is absolute at 24 GiB, so the same package is served identically on every card that can hold it, and measured peak lands near 21 GiB throughout.

---

## Where miners submit recipes

To the submission API the owner runs. There is no on-chain step, no upload to a
third party, and nothing for a miner to host.

`capcomp commit` writes a timelocked recipe into the commitments pallet, and is
the preview of the chain-native path. Nothing reads it: the engine builds its
field from the API, so a commitment made with it is inert until engine ingest
ships. The command says so on every run.

`capcomp commit` writes a timelocked recipe into the commitments pallet and is
the chain-native path being built beside this one. Nothing reads it yet, so a
commitment made today produces no queue entry and no scoreboard row. It is a
rehearsal, not a second way in.

A miner POSTs the recipe signed by their hotkey. The service checks the
signature, that the hotkey is registered on the subnet, and that they have
attempts left in the run, then stores it - replacing whatever it held for them.

```
POST /submit
{"hotkey": "5…", "recipe": "<the recipe JSON>", "signature": "0x…"}
```

Signed over `capcomp-submit:v1:<run_id>:sha256:<hex>`, which binds the signature
to both the run and the recipe so it cannot be replayed into either.

| | |
|---|---|
| Attempts per run | `RESUBMISSION_LIMIT`, currently 3; only the last is measured |
| Identical resend | Costs no attempt |
| Max recipe size | 256 KB |
| Stored | The final recipe only - the rest survive as digests and a count |

Bodies are held privately until the run that pays them opens, two runs after
the one submitted in. That is what makes copying pointless: by the time a
recipe is readable, the run it competed in is closed and paid.

The anti-copy check covers the same span from the other side - it compares a
submission against everything admitted in the last `COPY_LOOKBACK_RUNS` (2)
runs, so a duplicate arriving while the original is still unpaid is refused,
and one arriving after the original has been paid and published is not.

Miners do not write to the chain at all. That is a deliberate trade: the
submission set is this operator's record rather than something a third party can
rebuild from a public ledger. What it buys is a recipe nobody can copy before it
has been scored, a resubmission limit that can be enforced, and a recipe that
stays retrievable rather than depending on a repository its author may delete.

Validators read the record from the API - `/run/{id}`, `/run/{id}/results`,
`/run/{id}/weights` and `/run/{id}/instances/{hotkey}` - all of which open when
the run that pays them opens.

The settling rule still applies: a submission must have been in for
`MIN_COMMITMENT_AGE_BLOCKS` when the run that would measure it opens, or it is
held over to the run after. That is what keeps a miner from submitting at the
closing block after watching the whole run.

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
- [ ] Serving fraction derived from the card so the 24 GiB reservation fits
- [ ] Signing hotkey configured, if the engine is being run
- [ ] Validators told to install `capability-subnet[merge]` and run `--neuron.mode local` (or `endpoint`, if they verify signed reports instead of measuring)
- [ ] Spec version bumped if any consensus constant changed
