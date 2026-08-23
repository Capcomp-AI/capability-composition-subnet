# Validator guide

A validator decides where emission goes, and it earns that by measuring. There is one mode: every number you submit comes from work you did, on your hardware, against instances you regenerated yourself.

| What you need | |
|---|---|
| GPU | **4 × RTX 5090 (32 GB) minimum**, 8 recommended — one candidate per card |
| Adapter pool on disk | ~9 GB |
| Install | `capability-subnet[merge]` |
| An operator to trust | None |

A validator that cannot measure a candidate refuses to start rather than score every miner zero for a dependency it is missing. There is no lighter configuration to fall back to: taking numbers from somebody else is a weaker claim than measuring, and a network that offers both ends up described by the stronger one while running on the weaker.

Checking this network needs no GPU and no validator registration. `capability-audit` replays any published run from its seeds and traces, and is worth running whether or not you set weights.

---

## What you are doing

Each run, your validator reads the commitments on chain, fetches each miner's recipe from the URI it committed, checks the recipe against its digest, reconstructs the merged adapter locally, serves it through your own endpoint, and scores it against hidden instances it regenerates from seeds derived from a block hash. It then ranks what it measured, writes the result to a run report, and submits the weights from the run **before** this one.

A run is 24 hours (7200 blocks) and boundaries are anchored: run 412 opens at block 8,908,667, about 12:00 Eastern on 23 August 2026, and each run after it opens 7200 blocks later. So the pipeline is three runs deep:

| | |
|---|---|
| **run N** | a miner commits |
| **run N+1** | you measure it and write `runs/run-N+1.json` |
| **run N+2** | you submit the weights that report holds |

The gap is not latency for its own sake. A weight vector states a *closed* run's leaderboard. Submitting the vector you have just computed means submitting a leaderboard still being written — a candidate measured early in the run faces an empty field, one measured late a full one — so two validators that reached the queue in a different order would submit different vectors from the same evidence. Reading it back from the report also means a validator restarted between runs pays what it measured rather than starting again with nothing.

If the report for the run being paid is missing, unreadable, or measured nobody, the validator burns. It never invents a vector.

Validators are not required to agree on artifact bytes. Six of the seven merge methods run an SVD, and an SVD is not bitwise reproducible across devices, so agreement is on **outcomes** rather than on hashes. The artifact digest is still recorded — a validator whose digest matches another's is stronger evidence — but it is evidence, not a gate.

## What you are not doing

You are not a relay. Before anything touches the chain your validator:

1. **Verifies signatures** against an allow-list *you* configure, when it takes numbers from an engine. A vector it cannot attribute to a trusted operator is refused.
2. **Checks the vector against the chain it can see.** Does the champion still hold that UID, or did it deregister and leave the slot to a stranger? Is every UID inside this subnet? Do the weights sum to one? Are there duplicate UIDs the chain would reject?
3. **Checks freshness**, against the run length the engine reports rather than a compiled-in default.
4. **Burns rather than submitting anything it cannot verify.**

---

## Hardware

`own` mode reconstructs and serves an 8B model with a rank-64 adapter applied.

| | |
|---|---|
| GPU | **4 × RTX 5090 (32 GB) minimum**, 8 recommended — one candidate per card |
| CPU | 16 cores minimum, 32 recommended |
| RAM | 64 GB |
| Disk | 120 GB — base model ~16 GB, adapter pool ~9 GB, artifacts and state |
| Network | 100 Mbps |

The GPU floor is enforced, not advisory: `MIN_VALIDATOR_CARDS` and
`MIN_VALIDATOR_CARD_GIB` are protocol constants and the engine refuses to start
below them, with the arithmetic in the message.

**Why 32 GB a card.** A candidate reserves a fixed **24 GiB** while it is
served, the driver context holds about 1 GiB before anything loads, and a merge
sharing the card peaks near 2.5 GiB — about 27.5 GiB before a card can serve and
reconstruct at once. A card that cannot clear the reservation is refused at
start-up rather than measuring a package it cannot fit. The reservation is
absolute and the fraction is derived from whatever card you have: 0.78 on a
32 GB card, 0.50 on a 48 GB one. That is what keeps what a candidate answers
with a property of the package rather than of your hardware — the same KV cache
everywhere. A larger card therefore does not run a bigger candidate; it runs the
same one with a smaller fraction.

**Why four cards.** Not the reference schedule — batched serving finishes the
eight reference packages in about 3.7 hours on a single card, which fits a
24-hour run with room to spare. It is challenger throughput. At the current
rate one card covers about 35 challengers in a day-long run and four cover
about 149, against the 29 commitments the network already carries. One card
would be at its limit today and past it with any growth; four is what keeps a
daily run viable.

**Why the CPU and RAM are higher than they look.** Reconstruction is pinned to a
single torch thread — byte-identical merges across machines require it — so each
merge saturates exactly one core and no more. The engine runs merges for
upcoming candidates while the fleet serves the current batch, so a 4-card
validator has up to four merges and four servers in flight at once, each merge
holding 1–4 GB resident. Cores, not clock speed, are what shorten a run.

Cards are the unit of parallelism. Each one measures a whole candidate at a
time, so eight cards measure eight candidates at once and the run's throughput
scales with the count. Two candidates sharing a card would contend for memory
and each would measure the other's footprint as its own.

| Cards | Candidates in flight | Roughly per 24 h run |
|---|---|---|
| 1 | 1 | ~35 |
| 4 | 4 | ~149 |
| 8 | 8 | ~299 |

Not linear in the card count: the reference schedule is measured once per run
whatever the fleet size, so the first card pays for it and the rest are pure
challenger throughput. These are the batched-serving rates — an earlier version
of this table carried ~23 per **72 h** run on one card, which was the
sequential rate and is about a fifth of what the same hardware does now.

One card still works and is still honest; it measures fewer candidates per
run. A validator that cannot finish its queue should bound it with
`--neuron.max_candidates_per_run` rather than fall behind silently.

Point the validator at the cards with `--neuron.devices`:

```bash
--neuron.devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7
```

Each device gets a port of its own, counting up from the one in
`--neuron.serve_url`, because the runtimes are alive at the same time.

### What one candidate costs

For the default run, per candidate:

| Stage | Cost |
|---|---|
| Reconstruction, trimming merge (`ties_svd`, rank 64) | **~15 min** |
| Reconstruction, `linear` | seconds |
| Runtime start-up | ~1 min |
| Evaluation, 540 instances at ~19 s each | **~2.8 h** |
| **Total** | **~3 h** |

Evaluation dominates and is bounded by the latency gate rather than by the
hardware: a package whose p95 exceeds 25 s per instance fails that gate, so
anything still in the running costs at most 540 × 25 s. Reconstruction is a
twelfth of the total, which is why there is no artifact cache — it would buy
about 8%.

The 540 is this validator's own slice: the shared core plus the tail drawn for
its hotkey, out of the run's 2000 instances.

> It is the package's latency gate that fails when a host is slow, so a card
> materially slower than an RTX 5090 can fail candidates for a reason that has
> nothing to do with them.

---

## Setup for `own` mode

The reconstruction stack is not in the base install, and the serving runtime is not installed by this package at all.

```bash
git clone <repository-url> lora-merger && cd lora-merger
pip install -e ".[merge]"
```

vLLM pins torch tightly, so keep it in its own virtualenv and point the validator at that interpreter:

```bash
python -m venv .venv-vllm
.venv-vllm/bin/pip install vllm ninja
```

**Match the wheel to your driver.** vLLM's compiled extensions link a specific
CUDA runtime, and a wheel built against a newer one than your driver supports
fails at import with `libcudart.so.<N>: cannot open shared object file` no
matter which torch is installed. Check what your driver offers with
`nvidia-smi`, then pick a vLLM release built against it and install the matching
torch from the index for that CUDA version:

```bash
# example: a driver offering CUDA 12.8
.venv-vllm/bin/pip install "vllm==0.18.0" ninja
.venv-vllm/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall --no-deps torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128
```

vLLM patches `torch._inductor.standalone_compile.FakeTensorMode` when it builds
its compile cache. Some torch versions rebind that name from the submodule to
the function they export, and the patch then fails with *"does not have the
attribute 'FakeTensorMode'"* and the engine core dies before serving anything.
If you hit it, give the function the attribute — it stays callable, which is
what the rest of torch uses it as:

```bash
cat > .venv-vllm/lib/python3.*/site-packages/_capsub_vllm_shim.py <<'PY'
try:
    import torch._inductor
    from torch._subclasses.fake_tensor import FakeTensorMode
    _sc = getattr(torch._inductor, "standalone_compile", None)
    if callable(_sc) and not hasattr(_sc, "FakeTensorMode"):
        _sc.FakeTensorMode = FakeTensorMode
except Exception:
    pass
PY
echo 'import _capsub_vllm_shim' > .venv-vllm/lib/python3.*/site-packages/zz_capsub_vllm_shim.pth
```

A `.pth` rather than `sitecustomize.py`: only one `sitecustomize` is imported
per interpreter and the system one usually wins, while every `.pth` runs.

Do not reach for `--enforce-eager` to get past this. It starts, but it runs slow
enough to put packages over the p95 latency gate — turning your environment into
their failed gate.

vLLM JIT-compiles attention kernels on first start and needs a toolchain to do it. On a clean Ubuntu host:

```bash
apt-get install -y build-essential python3-dev
```

The CUDA compiler and the CUDA runtime headers must be the same version, or the kernel build fails with either `Unsupported .version` from ptxas or `CUDA compiler and CUDA toolkit headers are incompatible`. Check them against each other:

```bash
nvcc --version                                    # compiler
grep 'define CUDART_VERSION' $CUDA_HOME/include/cuda_runtime_api.h
```

The linker also needs the development symlinks that the pip CUDA packages omit — `libcudart.so` pointing at `libcudart.so.13`, and `lib64` beside `lib` — or the build ends in `cannot find -lcudart`.

Materialise the pool and the base model before starting; evaluation runs with `HF_HUB_OFFLINE=1` and must never reach the network:

```bash
python scripts/import_public_adapters.py --out pool --write-registry
huggingface-cli download Qwen/Qwen3-8B --revision <pinned> --local-dir base-model/Qwen3-8B
```

Register the hotkey and run:

```bash
btcli subnet register --netuid 103 --wallet.name <coldkey> --wallet.hotkey <hotkey>

python neurons/validator.py \
    --netuid 103 \
    --wallet.name <coldkey> \
    --wallet.hotkey <hotkey> \
    --neuron.device cuda \
    --neuron.serve_url http://127.0.0.1:8000 \
    --neuron.pool_dir pool \
    --neuron.base_model_path base-model/Qwen3-8B \
    --neuron.serving_python .venv-vllm/bin/python
```

`--neuron.serve_url` is the address the validator **binds its own runtime to**, not an endpoint you stand up beforehand. Each candidate gets a runtime started with that candidate's adapter applied, and stopped afterwards, so nothing carries from one submission to the next.

You do not set a memory fraction. A candidate's peak memory is gated, and a fraction of the card would make that gate a statement about your hardware rather than about the package: the same candidate measures 25.4 GiB at 0.78 of a 32 GB card and 22.9 GiB at 0.70. The reservation is fixed by the protocol and the fraction is derived from whatever card you have.

The validator checks at start-up that it has a serving endpoint, a CUDA device, an importable reconstruction stack and a pool on disk, reports every problem at once, and refuses to run otherwise. A validator that cannot measure must not vote on who deserves emission.

Expect roughly 15 minutes of reconstruction per candidate that uses a trimming merge, and about twice that when the cross-worker digest check is on. Linear merges take seconds.

## Configuration

| Flag | Default | What it does |
|---|---|---|
| `--neuron.weight_interval` | `300` | Minimum blocks between submissions |
| `--neuron.poll_interval` | `60` | Seconds between polls |
| `--neuron.burn_percentage` | `0.0` | Additional fraction *you* route to burn |
| `--neuron.disable_set_weights` | off | Compute and log without submitting |
| `--neuron.device` | `cuda` | Device the merge runs on. A non-CUDA value is refused at start-up. |
| `--neuron.serve_url` | *empty* | Address the validator binds each candidate's runtime to. **Required.** |
| `--neuron.pool_dir` | `pool` | The certified adapter pool on disk |
| `--neuron.base_model_path` | *empty* | Local copy of the pinned base model. **Required.** |
| `--neuron.serving_python` | *empty* | Interpreter that starts each candidate's runtime |
| `--neuron.devices` | *empty* | Cards to measure on, comma-separated. Empty uses `--neuron.device` alone |
| `--neuron.max_candidates_per_run` | `0` | Stop after this many candidates, in commit order. `0` measures everything eligible |
| `--burn_percentage` | `0` | Burn *more* than the protocol asks, never less |

### How a run's emission splits

Four fifths of every run burns to the subnet owner's UID. The remaining fifth
is the miner pool, and it is paid only if the run's best candidate took the
throne — exceeding the reigning champion's grade by `0.002`. If it did not, the
whole miner share burns; second place does not inherit a run its leader did not
win.

| Rank | Share of the run |
|---|---|
| Burn | 80% |
| 1st | 18% |
| 2nd | 1% |
| 3rd | 0.6% |
| 4th | 0.2% |
| 5th | 0.1% |
| 6th–10th | 0.1%, in proportion to grade |

Ten miners are paid at most, and a rank nobody filled burns rather than being
promoted into the leader's share.

Ranking is by grade: quality 60%, improvement over the base model 30%, cost
10%. A candidate that failed a hard gate is not graded at all. Every term is
measured against the run's own instances and the base model, never against the
incumbent, so a grade means the same thing in every run — which is what lets
the dethrone margin be a fixed number.

The throne is carried between runs in the run reports. A run that crowns nobody
records the grade it inherited, so the bar stays where it was rather than
rising or resetting.

### Burning more than the protocol asks

`--neuron.burn_percentage` compounds with the split above rather than replacing
it: everything already in the vector is scaled by what remains and the extra is
added on top, so `0.5` leaves miners a tenth of the run rather than a fifth. A
validator may burn more than the protocol asks and never less — burning less
would let one validator quietly override the rule the rest are applying.

### Your own burn

`--neuron.burn_percentage` lets you burn **more** than the engine asked for, never less.

Burned emission goes to the **subnet owner's UID**, resolved from the metagraph
on every pass. It is not UID 0: that slot belongs to whichever neuron registered
into it first, so weighting it would pay that miner rather than burning
anything. If the owner holds no UID at all, this validator submits nothing for
that pass — there is no address that "burn" could honestly mean, and paying an
arbitrary neuron is worse than skipping a run.

Allowing less would let a validator quietly override an operator's incident response. Allowing more is you declining to pay a champion you do not trust, with your own stake — which is a decision you are entitled to make, and one of the few levers you have if you disagree with an evaluation.

---

## Verifying an evaluation yourself

Nothing stops you checking the engine's work. Everything needed is published.

```bash
# What is the current weight vector, and which report justifies it?
curl https://<engine-host>/weights
curl https://<engine-host>/champion

# The full report: gate verdicts, per-axis comparison, paired statistics
curl https://<engine-host>/reports/<report-sha256>

# Every evaluation in a run
curl "https://<engine-host>/reports?run_id=<n>"
```

A report states the recipe digest, the artifact digest, the evaluator image digest, every hard-gate verdict, every per-axis comparator verdict, the paired bootstrap bound, and the reason for the decision. If you have the published recipe and the certified pool, you can rebuild the artifact and confirm its digest matches:

```bash
python -m capability_subnet.miner.cli digest --recipe <the-published-recipe>
```

### The audit tool does this for you

```bash
# Every report in a run, plus the weight vector derived from them
capability-audit --trusted-signers <operator-hotkey> run --run <n>

# One report
capability-audit --trusted-signers <operator-hotkey> report --digest <sha256>
```

It checks that the qualified score follows from its own published components,
that the claimed strongest reference really is the strongest one published, that
a dethrone is supported by the gates and the comparator, and that the weight
vector pays only someone a report crowned. A fabricated number has to be
fabricated *consistently* across a signed record that was published the moment it
was produced.

### Re-scoring a closed run

Stronger still, and it needs no GPU:

```bash
capability-audit --trusted-signers <operator-hotkey> replay --run <n>
```

Hidden instances are drawn fresh every run and never reused, so once a run
closes its seeds have no value as a secret. The engine publishes them together
with the traces it scored. The tool regenerates each instance from its seed —
generation is a pure function of the seed, so this reproduces the exact problem
the candidate faced — and re-runs the deterministic scorer over the published
trace.

If the engine's arithmetic was honest, every stage score matches. If it was not,
you get the specific disagreement: this instance, this stage, this number against
that one.

What this cannot check is whether a trace faithfully records what the model
actually did. A determined operator could publish a fabricated trace that scores
as claimed. But the fabrication has to be carried down to per-turn tool calls
that stay consistent with a workflow you can regenerate, which is considerably
harder than adjusting an aggregate.

The current run is never disclosed — its challenger is still sitting that test.

---

## Failure modes and what happens

| Situation | What your validator does |
|---|---|
| Engine unreachable | Logs it, waits, retries. No submission — the previous weights stay in force. |
| Vector unsigned or from an untrusted signer | **Burns.** |
| Signature does not verify | **Burns.** |
| Champion deregistered since the vector was computed | **Burns** — paying that UID now would pay a stranger. |
| Vector many runs stale | **Burns** — the engine has stalled. |
| Engine does not report its run length | **Burns** — freshness cannot be established, so it is not assumed. |
| Weights malformed (bad sum, duplicate UID, out-of-range) | **Burns.** |
| Chain rate limit | Treated as a no-op, retried next interval. |
| Healthy vector | Re-scores the last closed run, then applies your burn setting and submits. |
| Graded payments | Every non-champion recipient must have a published report showing it cleared all hard gates and carries a contribution grade; `capability-audit` checks this. |
| Run does not re-score | **Burns.** The engine's published scores contradict its own published traces. |
| Run not yet disclosed | Submits. Absence of a disclosure is absence of evidence, not proof of dishonesty — treating it otherwise would turn an outage into a punishment and give validators a reason to race the disclosure. |

Burning is the deliberate fallback rather than resubmitting the last known-good vector, because a dead engine must not pin emission to a stale champion forever.

---

## Monitoring

The validator logs every decision with its reason. Worth alerting on:

- repeated `burning this run's share` — the engine is unhealthy or the allow-list is wrong
- repeated `engine unavailable` — network or operator problem
- `weight submission failed` other than rate limiting — a chain problem

Check the engine's own health directly:

```bash
curl https://<engine-host>/health
```

---

## Requirements

See [Hardware](#hardware) above: a validator needs 32 GB cards, and measures one candidate per card across every card it has.

See [min_compute.yml](../min_compute.yml).

---

## The honest trade

Concentrating evaluation in one operator is a real cost. It buys reproducible scoring, cheap validation and a straightforward comparison mechanism; it costs decentralisation at the evaluation layer.

The mitigations are signed reports, published artifacts and recipes, validators that verify rather than relay, and the chain's own consensus at the weight layer. Optional validator audit re-runs — where a validator re-executes a sampled instance and checks the engine's answer — are a planned hardening step, not something that exists today.

If that trade is not acceptable to you, it is better to know before you register than after.

---

# Deployment

## Local

Everything runs without a chain, without a GPU and without the real 8B pool. Nothing here measures model quality; it exercises every other moving part.

```bash
git clone <repository-url> lora-merger && cd lora-merger
pip install -e ".[dev]"

make test-fast          # the suite, skipping GPU/docker/chain-marked tests
```

### Is the environment sound?

```bash
python -m capability_subnet.workflows.cli selftest --count 20
python -m capability_subnet.workflows.cli selftest --count 20 --split ood
```

The scripted reference solver knows every answer, so anything less than 20/20 means the workflow itself is broken.

### A full engine pass

```bash
./scripts/run_dev_engine.sh
```

This builds a *synthetic* adapter pool — structurally identical to the real one, random weights — generates a small public pack, runs one engine pass in dry-run mode, and renders the dashboard. It exercises reconstruction, hashing and the loop without downloading four gigabytes of real weights, and it cannot reach a live network: synthetic adapters carry no certification record, and preflight refuses to start while any pool member is uncertified.

For the real pool, `make import-pool` fetches and normalises the certified adapters from their pinned upstream sources. Then:

```bash
CAPSUB_STATE_DIR=.dev-state python -m capability_subnet.backend.api
curl localhost:8080/health
```

### With a real model

If you have a GPU and the pinned base model, serve a reconstructed adapter and point local evaluation at it:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --served-model-name candidate \
    --enable-lora --lora-modules candidate=build/candidate \
    --dtype bfloat16
```

Then use `capability_subnet.miner.local_eval.evaluate_locally` — see the [miner guide](miner.md#5-evaluate-locally).

---

## Local chain

For anything involving commitments or weights, run a chain rather than guessing.

```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor && ./scripts/localnet.sh
```

Create wallets, create a subnet, register:

```bash
btcli wallet new_coldkey --wallet.name owner
btcli wallet new_hotkey  --wallet.name owner --wallet.hotkey default

btcli subnet create --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9944
btcli subnet register --netuid 1 --wallet.name owner --wallet.hotkey default \
    --subtensor.chain_endpoint ws://127.0.0.1:9944
```

Run the pieces against it:

```bash
# Engine
CAPSUB_NETUID=1 CAPSUB_CHAIN_ENDPOINT=ws://127.0.0.1:9944 \
python -m capability_subnet.backend.service --config backend.yaml

# Validator
python neurons/validator.py --netuid 1 \
    --wallet.name owner --wallet.hotkey default \
    --subtensor.chain_endpoint ws://127.0.0.1:9944 \
    --neuron.serve_url http://127.0.0.1:8000 \
    --neuron.pool_dir pool

# Miner
python neurons/miner.py --netuid 1 \
    --wallet.name miner --wallet.hotkey default \
    --subtensor.chain_endpoint ws://127.0.0.1:9944 \
    --recipe recipe.json --recipe_uri https://example.test/recipe.json --confirm
```

### What to verify before moving on

- A commitment is admitted and appears in `/queue`.
- An invalid submission is rejected with a clear reason and does **not** enter the queue.
- A run opens, references are measured, and `/weights` appears.
- A validator fetches, verifies and submits.
- The engine survives a restart with its queue and champion intact.

---

## Testnet

Testnet is where you find out whether the thing runs continuously, not whether it computes correctly.

```bash
btcli subnet register --netuid <testnet-netuid> --subtensor.network test \
    --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

Run everything with `--subtensor.network test`.

### Pre-mainnet checklist

| | |
|---|---|
| ☐ | Base model pinned to an immutable commit |
| ☐ | Every adapter certified, digests recorded, pool frozen |
| ☐ | Snapshot digest published |
| ☐ | Hidden seed root generated, stored securely, never committed |
| ☐ | Evaluator image digest pinned |
| ☐ | `min_commit_block` set to the arena's opening block |
| ☐ | PostgreSQL provisioned with a read-only role |
| ☐ | At least two reconstruction workers agreeing |
| ☐ | Signing key configured — unsigned reports are refused by validators |
| ☐ | API behind TLS |
| ☐ | `state/` backed up |
| ☐ | Three or more independent validators setting weights from published reports |
| ☐ | A full run completing unattended |
| ☐ | Champion ranking consistent across validators |
| ☐ | Commitments surviving an engine restart |
| ☐ | Invalid submissions failing closed |
| ☐ | Adversarial suite passing |
| ☐ | Validator running cost measured |

The last one is not bureaucracy. If validation costs more than it earns, validators leave and the subnet stops setting weights.

---

## Mainnet

```bash
btcli subnet register --netuid <netuid> --subtensor.network finney \
    --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

### Engine

```bash
docker compose -f docker/docker-compose.sandbox.yml up -d
docker compose -f docker/docker-compose.engine.yml up -d
```

Or directly, under a process manager:

```bash
pm2 start "python -m capability_subnet.backend.service --config backend.yaml" --name capsub-engine
pm2 start "python -m capability_subnet.backend.api --config backend.yaml"     --name capsub-api
pm2 save && pm2 startup
```

Put the API behind TLS. It is read-only and cannot change engine state, but validators are trusting what it returns, and a plaintext channel invites a party that is not you to answer.

### Validator

```bash
pm2 start "python neurons/validator.py \
    --netuid <netuid> \
    --wallet.name <coldkey> --wallet.hotkey <hotkey> \
    --neuron.serve_url http://127.0.0.1:8000 \
    --neuron.pool_dir pool \
    --neuron.base_model_path base-model/Qwen3-8B" \
  --name capsub-validator
```

**Give it a GPU.** The validator refuses to start without a CUDA device, a serving endpoint, an importable reconstruction stack and a pool on disk — a host that cannot measure must not vote on who deserves emission.

### Miner

Miners run nothing continuously. Commit once, then watch:

```bash
curl https://<engine-host>/queue/<your-hotkey>
```

---

## Upgrades

Changing anything that affects how a recipe is reconstructed or scored is a **coordinated network upgrade**, not a rolling deploy.

1. Bump the version — the spec version is derived from it and travels with every weight submission.
2. Announce it with a target block.
3. Upgrade the engine at that block.
4. Validators upgrade; the spec version keeps mismatched submissions distinguishable.

### What forces a spec bump

- The merge engine, the canonical writer, or anything else that changes an artifact digest
- The recipe schema
- Stage scoring or the qualified-score weights
- The comparator or the dethrone rule
- The base model or the adapter pool

### What does not

- The API, dashboard, logging or documentation
- Operational settings such as run length or poll interval

### Repinning the base model

This creates a **new arena**, not a new run. Existing recipes target the old base and are not valid against the new one. Champions do not carry over. Plan it as a relaunch.

---

## Backup and recovery

Back up `state/` — it holds the queue, the champion record, every sample row, every report and every weight vector.

```bash
sqlite3 state/engine.sqlite ".backup 'backup/engine-$(date +%F).sqlite'"
tar czf backup/reports-$(date +%F).tar.gz state/reports state/recipes
```

Sample rows in particular cannot be reconstructed from aggregates, and the comparator needs them for paired statistics.

To recover: restore `state/`, restart the engine. It re-reads the chain, rebuilds any missing artifacts from cached recipes, and continues from the current run. Losing the artifact cache costs rebuild time and nothing else — artifacts are content-addressed and reproducible.

---

## Common questions

### Why don't validators need a GPU?

Evaluation is centralised. Validators fetch a signed weight vector, verify it, and set weights.

### Then what stops a dishonest operator?

A validator is not a relay. Before touching the chain it verifies the operator signature against an allow-list **it** configures, checks the vector against the chain it can see, and burns rather than submitting anything it cannot verify. Every report a decision rests on is signed and published, so the weight vector can be re-derived independently.

What that does **not** give is independent verification that the hidden instances were fair. Validator audit re-runs are planned; they do not exist today. If that residual trust is unacceptable, better to know before registering.

### What if the engine goes down?

The validator keeps the last submitted weights in force while retrying. If the published vector goes many runs stale, the validator **burns** rather than continuing to pay a champion the engine can no longer defend.

### Can I burn more than the engine asked?

Yes — more, never less. Allowing less would let a validator override an operator's incident response. Allowing more is you declining to pay a champion you do not trust, with your own stake.

### How do I verify an evaluation myself?

Fetch the report, read the gate verdicts and the comparator's per-axis verdicts, then rebuild the champion's artifact from its published recipe and confirm the digest matches. See the [validator guide](validator.md#verifying-an-evaluation-yourself).

---
