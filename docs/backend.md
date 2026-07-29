# Engine operations

The evaluation engine is operated by the subnet owner. Miners and validators do not run any of this.

It has one job and one obligation: measure every candidate identically, and publish enough signed evidence that anyone can check it did.

---

## What it runs

```
engine   ─ the control loop. Reads the chain, admits, evaluates, decides, publishes.
           Holds the hidden instances and the signing key.
api      ─ read-only. Serves what validators and miners need. No route changes state.
sandbox  ─ per-instance tool services: PostgreSQL, the code runner, inventory, safety.
serving  ─ one GPU per candidate, OpenAI-compatible, bfloat16, no adapter hot-swapping.
```

The engine and the API are **separate processes on purpose**. The API is exposed to the internet so validators can poll it; it must not share a process with the loop that holds the hidden instances and the operator key.

---

## First run

```bash
git clone <repository-url> lora-merger && cd lora-merger
pip install -e ".[backend,dev]"

cp backend.example.yaml backend.yaml
cp .env.example .env
```

### The things you must change

The engine **refuses to start** until these are set, because each one, left at its default, would make its own results indefensible.

**1. Materialise the pool.** The base model is already pinned to an immutable commit; the certified adapters have to be fetched and normalised before anything can be reconstructed:

```bash
python scripts/import_public_adapters.py --out pool --write-registry
```

This fetches exactly two files per adapter at the pinned upstream revision — the config and the weights, nothing else — refuses any upstream whose config has drifted from what the registry recorded, re-factorises the update to the canonical rank, and writes each artifact digest back into the registry. Then measure each adapter and record it with `capability-registry certify`; the engine refuses to start while any adapter is uncertified.

**2. Point `base_model_path` at a local copy of the base model.** Managed serving starts a runtime per candidate from it, and evaluation runs with `HF_HUB_OFFLINE=1` — a candidate evaluation must never reach the network.

**3. Generate a hidden seed root.**

```bash
python3 -c "import secrets; print(secrets.randbits(63))"
```

Put it in `.env` as `CAPSUB_HIDDEN_SEED_ROOT`. **Anyone who learns this can predict every future hidden instance.** Keep it out of version control, out of logs, and off the API.

**4. Pin the evaluator image digest.**

```bash
docker inspect --format='{{index .RepoDigests 0}}' capability-subnet/engine:latest
```

It is written into every published report, so a report identifies the exact software that produced it.

**5. Set `min_commit_block`** to the block the arena opened at, so commitments from a previous arena version cannot enter the queue.

Then verify:

```bash
python -m capability_subnet.backend.service --config backend.yaml --dry-run --once
```

Preflight lists every remaining problem at once rather than one per restart.

---

## Certifying the adapter pool

The pool is the foundation. An adapter enters it only after passing every admission gate, because the merge engine loads these tensors into a process that also holds hidden evaluation material — an adapter file is untrusted input until proven to be nothing but finite numbers of the expected shape.

The gates run against the file `import_public_adapters.py` wrote, not against anything upstream published. That is deliberate: the importer fetches only the config and the weights, so a repository that also ships scripts, pickles or stray tensors contributes none of them, and what gets certified is what will actually be loaded.

```bash
python -m capability_subnet.registry.cli certify /path/to/adapter \
    --adapter-id embedded-engineering-v1 \
    --license Apache-2.0 --allows-derivatives \
    --capability-score 0.82 \
    --base-retention 0.991
```

| Gate | Checks |
|---|---|
| Security | safetensors and metadata only — no pickle formats, scripts, binaries, or remote-code hooks |
| Config | Matches the canonical spec exactly: rank, alpha, dropout, bias, target modules |
| Base revision | Declares the exact pinned revision |
| Shapes | Every expected tensor present, right shape, nothing extra |
| Numerical | Exhaustive NaN and infinity scan |
| License | Permits redistributing derivative weights |
| Capability | Actually good at its declared capability, and did not destroy general ability getting there |
| Conversion | An adapter normalised to the canonical rank was **recertified afterwards** |

That last one matters: rank conversion is lossy, and certifying the original then admitting the converted file would put unmeasured weights in the pool.

Record the printed `artifact_sha256` in `registry/data/adapter_registry.json`, set `certified: true`, then confirm the snapshot digest:

```bash
python -m capability_subnet.registry.cli snapshot
```

That digest is what every recipe must declare. **Changing it invalidates every outstanding recipe**, so freeze the pool before announcing the arena.

---

## Running

### Docker

```bash
docker compose -f docker/docker-compose.sandbox.yml up -d   # tool services
docker compose -f docker/docker-compose.engine.yml up -d    # engine + API
```

### Directly

```bash
python -m capability_subnet.backend.service --config backend.yaml   # control loop
python -m capability_subnet.backend.api     --config backend.yaml   # read-only API
```

Under a process manager:

```bash
pm2 start "python -m capability_subnet.backend.service --config backend.yaml" --name capsub-engine
pm2 start "python -m capability_subnet.backend.api --config backend.yaml"     --name capsub-api
```

---

## One pass of the loop

```
block = chain head
admit new commitments             ← cheap checks only, nothing touches a GPU
ensure the window is open         ← if it changed: redraw, re-measure every reference
evaluate the queue head           ← reconstruct, serve, run, gate, compare, decide
publish weights                   ← signed vector for validators
sleep, repeat
```

Opening a window is the expensive step. It re-measures the base model, every capability adapter alone, three equal-weight merges, the operator recipe and the incumbent — all on the freshly drawn instances. It is not optional: a challenger compared against last window's numbers would be compared on a different instance set, and the paired statistics would be meaningless.

Budget roughly `(references + 1) × instances × per-instance-time` per window and size `window_blocks` accordingly.

---

## Failure handling

The distinction the engine draws everywhere:

| Failure | Response |
|---|---|
| Invalid miner submission | **Fails closed** — score zero |
| Backend or infrastructure failure | **Fails open** — the queue holds; no negative verdict until the failure is reproduced |
| Workers disagree on an artifact hash | Evaluation of that candidate **pauses**. Investigate the software-version mismatch. |
| Hidden generator failure | Window not finalised; existing weights stay active. Emission never stops. |
| No qualified candidate | The incumbent remains if valid; otherwise 100% burn. |
| Tool platform failure | Retry. Never counted as a model failure. |

A one-shot-per-hotkey rule is only defensible if the engine never spends that shot on its own bad night. If you change one thing in this document, do not change that.

---

## Configuration

Every field can be overridden with `CAPSUB_<FIELD_NAME>`. Full annotated reference: [backend.example.yaml](../backend.example.yaml).

### Windows

| Field | Default | Notes |
|---|---|---|
| `window_blocks` | `7200` | ≈24 h. Shorter re-measures references more often; longer gives a champion more time on one draw. |
| `hidden_instances` | `100` | Canonical instances per window. |
| `ood_instances` | `30` | Out-of-distribution instances per window. |
| `hidden_seed_root` | — | **Secret.** Set it. |
| `single_adapter_rotation` | `3` | Single-adapter references measured per window, rotated by window id. `0` measures all of them. |

`single_adapter_rotation` is a throughput knob with a real trade behind it.
Measuring every single-adapter reference each window is the most defensible
thing to do and, at one adapter per full instance draw, consumes most of a
window's GPU budget before a single challenger is looked at — and a window that
cannot finish never evaluates anybody. Rotating keeps the "beat the best
specialist" bar honest over time while leaving room to run the queue. Set it to
`0` only if your hardware can genuinely afford the full sweep every window.

### Serving

| Field | Default | Notes |
|---|---|---|
| `serving_mode` | `managed` | `managed` starts a vLLM process per candidate with that candidate's adapter applied |
| `base_model_path` | — | Local path to the pinned base model. Required in managed mode. |
| `serving_gpu_index` | `0` | GPU assigned to the candidate endpoint |
| `serving_max_model_len` | `16384` | Context window |
| `serving_gpu_memory_utilization` | `0.90` | Fraction of the card vLLM may claim |
| `serving_python` | `""` | Interpreter vLLM runs under; empty uses the engine's own |
| `tool_call_parser` | `hermes` | Qwen3 emits Hermes-style tool calls |
| `reasoning_parser` | `""` | Empty: this subnet disables the model's thinking channel |

Set `serving_python` when vLLM lives in its own virtualenv, which is common:
vLLM pins torch tightly enough that operators routinely keep it separate from
the engine. Empty means "the interpreter the engine runs under".

The engine probes vLLM's `--help` for optional flags rather than assuming them,
because vLLM removes options between releases without a deprecation window and
an unknown option is an immediate argparse exit — presenting as every candidate
being unservable. Flags the protocol depends on are *not* probed: a build
without `--enable-auto-tool-choice` cannot run this workflow, and failing loudly
at start-up is the right outcome.

**Use `managed`.** The `external` mode points at a runtime it does not own and
therefore cannot apply a candidate's adapter — every candidate and every
reference would be measured against whatever process is already running, all of
them would post identical scores, no challenger could ever be distinguished, and
the network would burn its emission with no report explaining why. The engine
refuses to start in `external` mode for exactly that reason; it exists for
development against the base model alone.

### How much GPU memory

Measured against the pinned base model, not estimated:

| | |
|---|---|
| base weights (bf16) | 15.26 GB |
| KV cache, one 16384-token sequence | 2.25 GB (144 KB/token) |
| merged adapter at rank 64 | 0.33 GB |
| CUDA graphs, activations, workspace | ~1.50 GB |
| **minimum to serve one candidate** | **19.33 GB** |

Reconstruction is a separate and much smaller cost: **~1.5 GB peak, and flat in
the number of selected adapters** — the merge streams one update at a time
rather than stacking them, so a twelve-adapter recipe costs the same as a
two-adapter one. It can share the serving card or use any other.

**24 GB is a real floor rather than a comfortable one.** It leaves about 4.5 GB
of KV cache — 32k tokens, which covers the single sequence the engine runs at a
time with nothing to spare. 40 GB or more is what to buy if the workflow's
context grows or evaluation is ever made concurrent.

One consequence worth knowing: because the base model dominates, `peak_vram`
barely distinguishes candidates. A rank-128 artifact is 0.33 GB heavier than a
rank-64 one, against a 24 GB limit — so the memory gate answers "does this fit
on the card" and not "is this candidate leaner than that one", and the
`artifact_efficiency` score component is correspondingly compressed. Artifact
*size* is the term that actually separates packages there.

**A faulty GPU anywhere in the machine can make higher-indexed ones unreachable.**
Device enumeration walks the physical indices in order, so a card NVML cannot
describe stops the walk — every GPU above it then reports as absent, and vLLM
fails with `local rank 0 is out of bounds for 0 devices` rather than anything
naming the real cause. If `nvidia-smi -i <n>` reports "No devices were found"
for any card, treat every index above it as unusable until that card is repaired
or removed, and keep `serving_gpu_index` and `merge_gpu_index` below it. A host
fault rather than a subnet one, but it presents as the engine being unable to
serve anything.

`tool_call_parser` is not optional either. Without it vLLM renders the tool
schemas into the prompt but never parses the reply, so `message.tool_calls`
comes back empty and `tool_choice: "auto"` is rejected outright — every instance
fails for a reason that has nothing to do with the candidate.

### Reconstruction

| Field | Default | Notes |
|---|---|---|
| `merge_device` | `cuda` | Where the merge arithmetic runs |
| `merge_gpu_index` | `0` | GPU used for reconstruction |
| `reconstruction_workers` | `2` | Independent rebuilds whose digests must agree |

`merge_device` is consensus-relevant. The trimming methods materialise a full
update per projection and must decompose it densely; on the pinned base model
that is roughly six minutes per build on a GPU against nearly three hours on a
CPU. But cuSOLVER and LAPACK do not agree bit-for-bit, so an artifact digest
reproduces only on the same device class. Every published report records the
device it was built on, and **every worker in one deployment must use the same
one** — which the cross-worker digest check enforces automatically, by failing.

### Comparator

| Field | Default | Notes |
|---|---|---|
| `axis_margin` | `0.02` | Absolute margin to count as dominant on an axis |
| `axis_tolerance` | `0.01` | Relative band that still counts as not-worse |
| `min_dominant_axes` | `1` | Axes a challenger must dominate |
| `min_axis_samples` | `20` | Fewer paired samples than this counts as worse |
| `end_to_end_margin` | `0.03` | Absolute completion margin over the strongest **permanent reference** |
| `champion_margin` | `0.01` | Defender's advantage over the incumbent, at the moment it is crowned |
| `champion_margin_decay_blocks` | `216000` | Blocks over which that advantage falls to zero (~30 days) |
| `strict_pareto` | `false` | `true` requires dominance on *every* axis |

Raising `end_to_end_margin` makes the throne harder to take and improvements more meaningful; lowering it risks crowning noise. Do not tune it in response to a specific candidate.

`champion_margin` is deliberately separate from `end_to_end_margin`, and setting them equal recreates a bug rather than simplifying the configuration: the incumbent would then effectively count as a reference, every successive champion would have to beat the previous one by a further fixed margin, and the bar would walk upward until nothing could move it. Setting `champion_margin_decay_blocks` to `0` disables the decay and reintroduces a permanent defender's advantage — an incumbent that nothing displaces then holds the throne indefinitely.

### Incentive

| Field | Default | Notes |
|---|---|---|
| `incentive_mode` | `graded_contribution` | or `winner_take_all`, `graded_top3` |
| `champion_base_share` | `0.55` | The champion's fixed cut under the graded mode |
| `contribution_memory_windows` | `7` | How long a graded result keeps earning |
| `tail_share` | `0.20` | Spread across queued miners as a linear taper |
| `burn_percentage` | `0.0` | The operational safety valve |
| `burn_uid` | `0` | Fallback only — see below |

`graded_contribution` is the default because winner-take-all discards the
network's most useful signal. Almost every submission that is ever evaluated
will fail to dethrone, and a recipe is one shot — a miner cannot iterate on it
the way a code-submitting miner can. Paying a miner who moved completion from
0.41 to 0.58 exactly what it pays one that submitted a distractor soup tells
neither of them anything, and the second attempt is no better informed than the
first.

The throne is still the prize: the champion takes `champion_base_share` of the
payable emission outright. Everything below it is split by a grade blending the
qualified score (50%), improvement over the strongest permanent reference (25%),
proximity to the champion (15%) and running cost — tokens and latency (10%).
Only candidates that cleared **every hard gate** are graded; this is not a
consolation prize for producing something undeployable. If nobody qualifies, the
graded pool burns rather than folding into the champion's share, because holding
an uncontested throne is not an achievement.

Each candidate's grade and its four terms are published on its report, so a
miner that earned a partial share can see which part of its package earned it.

`burn_percentage` is what you reach for during an incident: it routes part of the share to burn without changing who the champion is, which is better than stopping the engine and freezing emission entirely.

`tail_share` is not payment for work. Bittensor prunes by lowest emission, so a miner holding exactly zero is the first the chain evicts — and under a strict winner-take-all split that is every challenger still waiting to be evaluated. The engine evaluates roughly one challenger per window, so that wait is long enough to matter, and a queue that empties itself is a subnet with one participant. Setting it to `0` restores pure winner-take-all and reintroduces that failure.

`burn_uid` is a **fallback for offline tooling only**. Live components resolve the subnet owner's UID from the metagraph instead, because UID 0 is not an incinerator — it belongs to whichever neuron registered into the first slot, and weighting it pays that miner for nothing. A validator that cannot resolve the owner submits nothing at all rather than paying a stranger.

---

## Monitoring

```bash
curl localhost:8080/health       # window, champion, queue depth
curl localhost:8080/champion
curl localhost:8080/queue
curl "localhost:8080/reports?limit=20"
curl localhost:8080/compatibility
```

Render the dashboard:

```bash
python -m capability_subnet.platform.dashboard --config backend.yaml --out dashboard.html
```

### Worth alerting on

- **Worker disagreement on an artifact hash.** A determinism regression. Evaluation of that candidate is stuck until you resolve it.
- **A reference that could not be measured.** The window's comparison bar is incomplete.
- **Queue depth growing steadily.** Evaluation is slower than submissions arrive; shorten the window or add capacity.
- **Repeated infrastructure failures on the same candidate.** Something about that package breaks the serving stack.

---

## Operational cautions

**Never edit a score by hand.** The engine's authority rests entirely on running the published protocol. One manual adjustment and every report becomes an assertion rather than a record.

**Never publish the hidden seed root.** It is the whole arena.

**Never disable the cross-worker check** on a live network. `reconstruction_workers: 1` turns a determinism regression into silent mis-scoring.

**Keep the recipes you admit.** Champions are re-measured every window, so their recipe must be retrievable long after the miner published it. The engine stores its own copy under `state/recipes/` for exactly this reason — a champion whose pointer went dead must not silently stop defending.

**Back up `state/`.** It holds the queue, the champion record, every sample row and every report. Sample rows in particular cannot be reconstructed from aggregates, and the comparator needs them.

---

## Security posture

| Boundary | Enforcement |
|---|---|
| Candidate code | Own container: no network, read-only root, all capabilities dropped, memory/CPU/PID limits, fresh isolated interpreter |
| SQL | Statement inspection **and** a read-only role with per-instance schema grants. Catalog access blocked — instances share a database. |
| Hidden truth | Lives in the engine process, outside every network the agent can reach |
| Serving | Internal bridge only; adapter hot-swapping disabled |
| Reports | Signed with the operator hotkey |
| API | Read-only allow-list of routes; no route changes engine state |

See [security.md](security.md) for the full threat model.

---

## Requirements

| | Per evaluation worker |
|---|---|
| CPU | 32 cores |
| RAM | 128 GB |
| GPU | 1 × 80 GB, compute capability 8.0+ |
| Disk | 2 TB NVMe |
| Docker | 24.0+ |

The control plane (loop, API, monitor) can share a smaller host. See [min_compute.yml](../min_compute.yml).
