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

### The four things you must change

The engine **refuses to start** until these are set, because each one, left at its default, would make its own results indefensible.

**1. Pin the base model.** Edit `capability_subnet/registry/data/base_manifest.json` and replace `PIN_BEFORE_GENESIS` with an immutable upstream commit. Every certified adapter must declare the same revision.

**2. Generate a hidden seed root.**

```bash
python3 -c "import secrets; print(secrets.randbits(63))"
```

Put it in `.env` as `CAPSUB_HIDDEN_SEED_ROOT`. **Anyone who learns this can predict every future hidden instance.** Keep it out of version control, out of logs, and off the API.

**3. Pin the evaluator image digest.**

```bash
docker inspect --format='{{index .RepoDigests 0}}' capability-subnet/engine:latest
```

It is written into every published report, so a report identifies the exact software that produced it.

**4. Set `min_commit_block`** to the block the arena opened at, so commitments from a previous arena version cannot enter the queue.

Then verify:

```bash
python -m capability_subnet.backend.service --config backend.yaml --dry-run --once
```

Preflight lists every remaining problem at once rather than one per restart.

---

## Certifying the adapter pool

The pool is the foundation. An adapter enters it only after passing every admission gate, because the merge engine loads these tensors into a process that also holds hidden evaluation material — an adapter file is untrusted input until proven to be nothing but finite numbers of the expected shape.

```bash
python -m capability_subnet.registry.cli certify /path/to/adapter \
    --adapter-id german-technical-v1 \
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

### Comparator

| Field | Default | Notes |
|---|---|---|
| `axis_margin` | `0.02` | Absolute margin to count as dominant on an axis |
| `axis_tolerance` | `0.01` | Relative band that still counts as not-worse |
| `min_dominant_axes` | `1` | Axes a challenger must dominate |
| `min_axis_samples` | `20` | Fewer paired samples than this counts as worse |
| `end_to_end_margin` | `0.03` | Absolute completion margin over the strongest reference |
| `strict_pareto` | `false` | `true` requires dominance on *every* axis |

Raising `end_to_end_margin` makes the throne harder to take and improvements more meaningful; lowering it risks crowning noise. Do not tune it in response to a specific candidate.

### Incentive

| Field | Default |
|---|---|
| `incentive_mode` | `winner_take_all` (or `graded_top3`) |
| `burn_percentage` | `0.0` — the operational safety valve |
| `burn_uid` | `0` |

`burn_percentage` is what you reach for during an incident: it routes part of the share to burn without changing who the champion is, which is better than stopping the engine and freezing emission entirely.

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
