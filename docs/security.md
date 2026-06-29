# Security model

What is defended, how, and — just as usefully — what is deliberately **not** defended.

---

## The shape of the problem

Three things make this system's threat model unusual:

1. **Recipes are public.** They must be, or nobody could verify an evaluation. So copying is trivially easy and has to be made *worthless* rather than impossible.
2. **Candidate-written code executes.** The diagnostic stage is not scoreable otherwise.
3. **Evaluation is centralised.** One operator holds the hidden instances and the signing key, which concentrates both capability and the incentive to misuse it.

Each is addressed by a different mechanism, and the mechanisms are described honestly below, including where they stop.

---

## Threats and defences

### A malicious recipe

**Goal:** get the engine to execute something, load something, or reach somewhere.

| Defence | How |
|---|---|
| Declarative-only schema | Every expressible value is a bounded number, a name from a frozen registry, or an enum |
| Unknown fields rejected | An unknown field in a miner document usually means something is being smuggled |
| Names, not paths | Adapters are named and resolved against the frozen pool; there is no field to put a path in |
| Strict bounds | Coefficients, densities, ranks and quantiles are all range-checked |
| Size cap | Recipe fetches are capped, and reading stops mid-download when exceeded |

There is no code path from a recipe to execution. The merge engine reads numbers and multiplies matrices.

### A malicious adapter file

**Goal:** get code to run when the pool is loaded.

Adapters are untrusted input until certified. Admission requires: safetensors only (no pickle-backed formats, which execute on load), no scripts or binaries in the directory, no remote-code hooks in the config, exact shape validation against the pinned base, and an exhaustive NaN/infinity scan.

A single non-finite entry would propagate through every merge that selected that adapter, which is why the scan is exhaustive rather than sampled.

### Substituting the recipe after committing

**Goal:** commit one thing, be scored on another.

Bytes are verified against the on-chain digest before parsing. Then the digest is **re-derived from the canonical form of the parsed document** and compared again. That second check closes the gap where a miner commits the digest of a specially-formatted file and the engine scores the canonical form of something else.

The pointer itself is not trusted, so a mutable host is fine for integrity. It is not fine for availability: if the engine cannot fetch the bytes, the submission is not admitted.

### Copying

**Goal:** read a published recipe and resubmit it.

Three mechanisms, and the third is what makes the first two sufficient:

1. **Earliest commit wins**, checked on the recipe digest at admission and again on the reconstructed **artifact** digest. Two differently-worded recipes that build the same bytes are the same package.
2. **One shot per hotkey.** Copying costs a registration.
3. **Defender advantage.** Dethroning requires a genuine margin, so a copy that exactly reproduces the champion's scores loses by construction.

Even a copy nobody detected cannot win. That is the point.

### Extracting hidden material

**Goal:** learn the hidden instances, the truth, or the seed root.

| Boundary | Enforcement |
|---|---|
| The scorer | Runs in the engine process, outside every network the agent container can see |
| Hidden diagnostic cases | Their **expected outputs never enter the runner** — only inputs are sent |
| The seed root | Never leaves the engine; the API has no route that exposes it |
| Instance visibility | The visible payload is built explicitly, field by field, rather than filtered from the full object |
| Per-window refresh | Instances rotate, so anything learned about one window is worth little in the next |

That fourth row is a deliberate design choice: a new field added to the ground truth is invisible to the agent until someone *deliberately* adds it to the visible payload. A filter-based approach fails open; this fails closed.

### SQL injection and exfiltration

**Goal:** read or write outside the instance's snapshot.

Statement inspection rejects anything that is not a single read-only statement, along with catalog access (`sqlite_*`, `pg_*`, `information_schema`) and file-access functions.

**Inspection alone would be a filter to be evaded.** What actually holds is the connection: a SQLite authorizer that denies everything but read operations, or a PostgreSQL role with no write grants, `default_transaction_read_only`, a statement timeout, and `SELECT` granted per instance schema.

Catalog blocking matters specifically because hidden snapshots for many instances live as separate schemas in one PostgreSQL database. Without it, catalog enumeration is a route from one instance to another instance's data.

### Sandbox escape

**Goal:** break out of the code runner.

Candidate code is the only component that executes arbitrary instructions, so it gets the strongest boundary — two layers:

**The container:** no network at all, read-only root filesystem, all capabilities dropped, `no-new-privileges`, non-root user, memory/CPU/PID limits, and an image with no compiler, no package manager and no network client.

**The process:** a fresh interpreter in isolated mode (`-I -S`) that ignores environment variables and keeps the current directory off the import path, with address-space, CPU-time, process-count, file-size and core-dump limits applied before `exec`.

Builtins are deliberately **not** restricted. Restricting them inside an already-isolated container is security theatre, and it would silently fail correct submissions that use `sum` or `max`.

### Prompt injection

**Goal:** use the workflow content to make the agent leak or misbehave.

Largely neutralised by architecture rather than filtering: the agent has no network, no tool that reveals correctness, and no route to the scorer. There is nothing to exfiltrate *to* and nothing to exfiltrate. A successful injection can make a candidate fail its own evaluation, which is a self-inflicted wound.

### Result fabrication by the operator

**Goal:** the operator pays whoever they like.

This is the honest weak point of a centralised engine, and it is mitigated rather than eliminated:

| Mitigation | What it gives |
|---|---|
| Signed reports | Nothing published can be repudiated, and nothing unsigned can be attributed |
| Published recipes and artifacts | Anyone can rebuild a champion's artifact and confirm its digest |
| Validators verify, not relay | Each validator checks the signature against **its own** allow-list and the vector against the chain it can see |
| Burn on doubt | A validator that cannot verify burns rather than submitting |
| Chain consensus | Weights still pass through the chain's own aggregation |
| Sample rows retained | Every per-instance outcome is stored, so a claimed aggregate can be checked against its parts |

What this does **not** give: independent verification that the hidden instances were fair or that the reported scores match what actually ran. Validator audit re-runs — where a validator re-executes a sampled instance and checks the engine's answer — are a planned hardening step and do not exist today.

If that residual trust is unacceptable, it is better to know before registering than after.

### Forged weight vectors

**Goal:** get a validator to pay an attacker.

A validator refuses a vector that is unsigned, signed by a hotkey outside its allow-list, or whose signature does not verify. The signature covers canonical bytes that exclude the signature fields themselves, so signing is idempotent and tampering with any payload field invalidates it.

Beyond the signature, the validator checks the vector against the chain: weights summing to one, distinct UIDs, UIDs inside the subnet, the champion still holding its UID, and the vector not being many windows stale. **A correctly-signed vector can still be unsubmittable**, and a deregistered champion is the common case.

---

## What is deliberately not defended

Being explicit about this is more useful than a longer list of defences.

**A miner's private search.** Miners may use any hardware and any method. The network judges the artifact, not the process. There is no attempt to detect or constrain how a recipe was found.

**Operator honesty about the hidden set.** Nothing proves the hidden instances were drawn fairly. The generator is published and the draw is deterministic given the root, but the root is secret, so the fairness of the draw rests on the operator.

**Availability of a miner's pointer.** If a recipe's host goes down before admission, the submission is not admitted. The engine keeps its own copy afterwards so a champion can keep defending, but it does not fetch pre-emptively.

**Model-level attacks on the base model.** The base is pinned upstream and taken as given.

**Collusion between the operator and a miner.** The mitigations above make it *detectable in principle* — published recipes, reproducible artifacts, retained sample rows — but nothing prevents it.

---

## Reporting a vulnerability

Report privately to the subnet operator rather than opening a public issue. Include what you found, how to reproduce it, and what it would let an attacker do.

Findings that affect scoring integrity — determinism, hidden-material exposure, or anything that lets a candidate influence its own evaluation — are the highest priority, because they invalidate results rather than merely disrupting service.
