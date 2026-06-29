# Deployment

Local development, testnet, then mainnet. In that order — each stage exists to catch a class of problem the next one is too expensive to catch.

---

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

This builds a synthetic adapter pool (structurally identical to the real one, random weights), generates a small public pack, runs one engine pass in dry-run mode, and renders the dashboard. Then:

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
