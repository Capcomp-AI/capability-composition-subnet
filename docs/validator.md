# Validator guide

Running a validator on this subnet costs a small VPS. No GPU, no model, no adapter pool, no reconstruction.

That does not make it a relay. Before it pays anyone, a validator **re-scores a closed window from the engine's own published traces** — regenerating the instances from their seeds and re-running the deterministic scorer over what the engine said happened. A score that does not follow from its own trace is caught by the party about to pay for it. Install size is under 50 MB: the base package excludes the tensor stack, which a validator never touches.

That is unusual, and it is worth understanding *why* before you run one — because the thing you are actually providing is not compute.

---

## What you are doing

The evaluation engine reconstructs candidates, serves them, runs them against hidden workflow instances, compares them with the champion and the reference baselines, and publishes a **signed weight vector** together with the signed reports that justify it.

Your validator fetches that vector and sets weights on-chain.

## What you are not doing

You are not a relay. Before anything touches the chain, your validator:

1. **Verifies the operator signature** against an allow-list *you* configure. A vector you cannot attribute to a trusted operator is refused.
2. **Checks the vector against the chain you can see.** Does the champion still hold that UID, or did it deregister and leave the slot to a stranger? Is every UID inside this subnet? Do the weights sum to one? Are there duplicate UIDs the chain would reject?
3. **Checks freshness**, against the window length the engine reports rather than a compiled-in default — a deployment that tuned its window is judged by the window it actually runs. A vector many windows behind the head means the engine has stalled, and a stale champion collecting emission indefinitely is worse for the network than nobody collecting it.
4. **Burns rather than submitting anything it cannot verify.**

The engine computes; validators decide. Each of you answers to your own stake, and every report a decision rests on is published and signed, so the weight vector can be re-derived independently by anyone who cares to.

---

## Setup

```bash
git clone <repository-url> lora-merger && cd lora-merger
pip install -e .
```

Register your hotkey:

```bash
btcli subnet register --netuid <netuid> --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

Run:

```bash
python neurons/validator.py \
    --netuid <netuid> \
    --wallet.name <coldkey> \
    --wallet.hotkey <hotkey> \
    --backend.url https://<engine-host> \
    --backend.trusted_signers <operator-hotkey-ss58>
```

### The one setting that matters

`--backend.trusted_signers` is the allow-list of operator hotkeys whose signatures you accept.

**Leaving it empty is a startup error.** The validator refuses to run rather than submit whatever the configured URL returns — that is not a behaviour anyone should inherit by omission. If you genuinely want it on a local network, pass `--backend.allow_unsigned` to say so deliberately.

Get the operator hotkey from the subnet's published channels, not from the engine you are about to trust.

### Under a process manager

```bash
pm2 start "python neurons/validator.py \
    --netuid <netuid> \
    --wallet.name <coldkey> --wallet.hotkey <hotkey> \
    --backend.url https://<engine-host> \
    --backend.trusted_signers <operator-hotkey>" \
  --name capsub-validator
```

Or with environment variables (`.env.example` has the full list):

```bash
export CAPSUB_NETUID=<netuid>
export CAPSUB_BACKEND_URL=https://<engine-host>
export CAPSUB_TRUSTED_SIGNERS=<operator-hotkey>
python neurons/validator.py --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

---

## Configuration

| Flag | Default | What it does |
|---|---|---|
| `--backend.url` | `http://127.0.0.1:8080` | The engine's read-only API |
| `--backend.trusted_signers` | *empty* | Operator hotkeys you accept. **Required.** |
| `--backend.allow_unsigned` | off | Explicitly accept unsigned vectors. Local development only. |
| `--backend.timeout` | `30` | HTTP timeout in seconds |
| `--neuron.weight_interval` | `300` | Minimum blocks between submissions |
| `--neuron.poll_interval` | `60` | Seconds between polls |
| `--neuron.burn_percentage` | `0.0` | Additional fraction *you* route to burn |
| `--neuron.no_spot_check` | off | Stop re-scoring the last closed window before paying |
| `--neuron.max_stale_windows` | `3` | Refuse a vector this far behind the head |
| `--neuron.disable_set_weights` | off | Compute and log without submitting |

### Your own burn

`--neuron.burn_percentage` lets you burn **more** than the engine asked for, never less.

Burned emission goes to the **subnet owner's UID**, resolved from the metagraph
on every pass. It is not UID 0: that slot belongs to whichever neuron registered
into it first, so weighting it would pay that miner rather than burning
anything. If the owner holds no UID at all, this validator submits nothing for
that pass — there is no address that "burn" could honestly mean, and paying an
arbitrary neuron is worse than skipping a window.

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

# Every evaluation in a window
curl "https://<engine-host>/reports?window_id=<n>"
```

A report states the recipe digest, the artifact digest, the evaluator image digest, every hard-gate verdict, every per-axis comparator verdict, the paired bootstrap bound, and the reason for the decision. If you have the published recipe and the certified pool, you can rebuild the artifact and confirm its digest matches:

```bash
python -m capability_subnet.miner.cli digest --recipe <the-published-recipe>
```

### The audit tool does this for you

```bash
# Every report in a window, plus the weight vector derived from them
capability-audit --trusted-signers <operator-hotkey> window --window <n>

# One report
capability-audit --trusted-signers <operator-hotkey> report --digest <sha256>
```

It checks that the qualified score follows from its own published components,
that the claimed strongest reference really is the strongest one published, that
a dethrone is supported by the gates and the comparator, and that the weight
vector pays only someone a report crowned. A fabricated number has to be
fabricated *consistently* across a signed record that was published the moment it
was produced.

### Re-scoring a closed window

Stronger still, and it needs no GPU:

```bash
capability-audit --trusted-signers <operator-hotkey> replay --window <n>
```

Hidden instances are drawn fresh every window and never reused, so once a window
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

The current window is never disclosed — its challenger is still sitting that test.

---

## Failure modes and what happens

| Situation | What your validator does |
|---|---|
| Engine unreachable | Logs it, waits, retries. No submission — the previous weights stay in force. |
| Vector unsigned or from an untrusted signer | **Burns.** |
| Signature does not verify | **Burns.** |
| Champion deregistered since the vector was computed | **Burns** — paying that UID now would pay a stranger. |
| Vector many windows stale | **Burns** — the engine has stalled. |
| Engine does not report its window length | **Burns** — freshness cannot be established, so it is not assumed. |
| Weights malformed (bad sum, duplicate UID, out-of-range) | **Burns.** |
| Chain rate limit | Treated as a no-op, retried next interval. |
| Healthy vector | Re-scores the last closed window, then applies your burn setting and submits. |
| Graded payments | Every non-champion recipient must have a published report showing it cleared all hard gates and carries a contribution grade; `capability-audit` checks this. |
| Window does not re-score | **Burns.** The engine's published scores contradict its own published traces. |
| Window not yet disclosed | Submits. Absence of a disclosure is absence of evidence, not proof of dishonesty — treating it otherwise would turn an outage into a punishment and give validators a reason to race the disclosure. |

Burning is the deliberate fallback rather than resubmitting the last known-good vector, because a dead engine must not pin emission to a stale champion forever.

---

## Monitoring

The validator logs every decision with its reason. Worth alerting on:

- repeated `burning this window's share` — the engine is unhealthy or the allow-list is wrong
- repeated `engine unavailable` — network or operator problem
- `weight submission failed` other than rate limiting — a chain problem

Check the engine's own health directly:

```bash
curl https://<engine-host>/health
```

---

## Requirements

| | |
|---|---|
| CPU | 2 cores |
| RAM | 4 GB |
| Disk | 20 GB |
| GPU | **None** |
| Network | 100 Mbps |

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

# Validator — no allow-list is acceptable here and nowhere else
python neurons/validator.py --netuid 1 \
    --wallet.name owner --wallet.hotkey default \
    --subtensor.chain_endpoint ws://127.0.0.1:9944 \
    --backend.url http://127.0.0.1:8080

# Miner
python neurons/miner.py --netuid 1 \
    --wallet.name miner --wallet.hotkey default \
    --subtensor.chain_endpoint ws://127.0.0.1:9944 \
    --recipe recipe.json --recipe_uri https://example.test/recipe.json --confirm
```

### What to verify before moving on

- A commitment is admitted and appears in `/queue`.
- An invalid submission is rejected with a clear reason and does **not** enter the queue.
- A window opens, references are measured, and `/weights` appears.
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
| ☐ | A full window completing unattended |
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
    --backend.url https://<engine-host> \
    --backend.trusted_signers <operator-hotkey>" \
  --name capsub-validator
```

**Set `--backend.trusted_signers`.** Without it the validator submits whatever the URL returns.

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
- Operational settings such as window length or poll interval

### Repinning the base model

This creates a **new arena**, not a new window. Existing recipes target the old base and are not valid against the new one. Champions do not carry over. Plan it as a relaunch.

---

## Backup and recovery

Back up `state/` — it holds the queue, the champion record, every sample row, every report and every weight vector.

```bash
sqlite3 state/engine.sqlite ".backup 'backup/engine-$(date +%F).sqlite'"
tar czf backup/reports-$(date +%F).tar.gz state/reports state/recipes
```

Sample rows in particular cannot be reconstructed from aggregates, and the comparator needs them for paired statistics.

To recover: restore `state/`, restart the engine. It re-reads the chain, rebuilds any missing artifacts from cached recipes, and continues from the current window. Losing the artifact cache costs rebuild time and nothing else — artifacts are content-addressed and reproducible.

---

## Common questions

### Why don't validators need a GPU?

Evaluation is centralised. Validators fetch a signed weight vector, verify it, and set weights.

### Then what stops a dishonest operator?

A validator is not a relay. Before touching the chain it verifies the operator signature against an allow-list **it** configures, checks the vector against the chain it can see, and burns rather than submitting anything it cannot verify. Every report a decision rests on is signed and published, so the weight vector can be re-derived independently.

What that does **not** give is independent verification that the hidden instances were fair. Validator audit re-runs are planned; they do not exist today. If that residual trust is unacceptable, better to know before registering.

### What if the engine goes down?

The validator keeps the last submitted weights in force while retrying. If the published vector goes many windows stale, the validator **burns** rather than continuing to pay a champion the engine can no longer defend.

### Can I burn more than the engine asked?

Yes — more, never less. Allowing less would let a validator override an operator's incident response. Allowing more is you declining to pay a champion you do not trust, with your own stake.

### How do I verify an evaluation myself?

Fetch the report, read the gate verdicts and the comparator's per-axis verdicts, then rebuild the champion's artifact from its published recipe and confirm the digest matches. See the [validator guide](validator.md#verifying-an-evaluation-yourself).

---
