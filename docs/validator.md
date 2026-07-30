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
