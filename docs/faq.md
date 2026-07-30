# FAQ

---

## General

### What does this subnet actually produce?

A verified, deployable package for one specific business workflow: a merged LoRA adapter that completes the workflow better than the base model, better than any single specialist adapter, better than standard merges, and better than whatever held the throne before it — under hard limits on size, memory, latency and general-capability retention.

Not a benchmark score. A thing you can deploy.

### Why composition instead of training?

Training a new adapter is a solved, well-served problem. What is not solved is deciding which of the adapters you *already have* should be combined, at what weights, at what depth, and how aggressively compressed — for one real workflow.

That decision is currently made by intuition and a handful of experiments. This subnet turns it into a measured competition.

### Is merging actually better than routing between adapters?

**Unknown, and the subnet is built to find out rather than assume it.**

Efficient multi-adapter serving is a genuine competitor. Static merging removes a runtime selector, a load/swap policy and several runtime states, giving one stable model identity and simpler batching — but it does not inherently reduce tokens, agent steps or base compute. Those savings only appear where merging eliminates a token-consuming mechanism.

A routed-adapter reference baseline is a planned addition. Until it exists, the honest position is that static merging is worth selling *where measured total cost is better*, not everywhere.

### What happens if composition turns out not to help?

Nobody gets paid. The permanent reference baselines — including plain equal-weight merges — sit on the board specifically so the network can discover the answer is "no" rather than paying miners to not discover it.

If a reference holds the throne, the workflow share burns.

---

## Mining

### What hardware do I need?

Building and validating recipes: any machine. Reconstructing an artifact: ~32 GB RAM; a GPU is optional but roughly thirty times faster on the trimming methods, which have to decompose a full update per projection. Evaluating locally: a GPU that fits the base model in bfloat16. Searching seriously: as much as you want to spend — that is where the competition is.

### Can I really only submit once?

Once per hotkey. A decisive loss terminates that hotkey permanently; a new package needs a new registration.

That is what makes copying worthless. It is also why the tooling makes it easy to check everything locally first, and why an *infrastructure* failure never costs you your shot.

### Why can't I resubmit after fixing a mistake?

Because then copying would cost nothing: read the champion's published recipe, tweak it, resubmit until something sticks. One shot plus the margin requirement makes that strategy strictly worse than doing the work.

### How do I know my recipe is valid before committing?

```bash
python -m capability_subnet.miner.cli validate --recipe recipe.json
```

This runs exactly the checks the engine runs at admission, and separates hard problems from advisories.

### Why is my artifact too large?

You probably chose rank 128. Against the pinned base model that produces roughly 666 MB, over the 500 MB gate. `miner.cli size` tells you in a second.

### Should I select the distractor adapters?

Almost certainly not — but they are selectable on purpose. One is a German legal-contract adapter, superficially relevant because the workflow is German and actually harmful. Recognising that is part of the problem.

### Can I use negative coefficients?

Yes, within `-2.0 … 2.0`. Subtracting an adapter's update is a real operation. It interacts badly with the base-retention gate — which measures general instruction-following on a held-out probe, not anything about this workflow — so measure it.

### How much do local scores predict hidden scores?

Directionally, quite well — same generator, same tools, same scorer. But the hidden set is drawn fresh each window and includes out-of-distribution mutations. And a completion rate over twenty instances has wide enough variance that two recipes differing by a few points are indistinguishable at that sample size.

### Why is my submission not in the queue?

Either the engine has not read the chain yet, or admission rejected it. Check `/queue/<your-hotkey>`; a 404 means it was not admitted. The usual causes are a stale snapshot digest, an unfetchable recipe URI, or a digest computed over non-canonical bytes.

### What is the compatibility history for?

It is the accumulated answer to the questions a grid search cannot answer: which adapter pairs reinforce each other, which interfere, which capabilities need a specialist at which depth, when trimming beats dropping, and how much rank the workflow actually needs.

```bash
curl https://<engine-host>/compatibility
```

---

## Validating

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

## The mechanism

### Why is beating the champion not enough?

Because at genesis the throne is empty, and later a mediocre champion could hold it simply because nothing better challenged. The permanent references — base model, best single adapter, three equal-weight merges, the operator's own recipe — mean the network can never crown a package that an off-the-shelf merge already beats.

### Why must a challenger be "not worse" on every axis?

Otherwise a package could trade a capability away for a better average. Drop safety compliance, gain SQL accuracy: the mean improves and the package is worse at the job. The workflow needs every stage, so a package that abandoned one has not solved it.

### Why a paired bootstrap instead of just comparing scores?

With a hundred instances, a package that is genuinely no better than the incumbent scores higher about half the time. The question is not "did it score higher" but "is the difference larger than the noise."

Pairing works because both packages ran on **the same instances**, so instance difficulty drops out of the comparison entirely. That is a much tighter test, and it is only available because the engine controls the draw.

### Why does an axis with few samples count as *worse*?

Absence of evidence that a challenger kept a capability is not evidence that it did. Treating it as a tie would let a challenger win an axis by not being measured on it.

### Why does everything have to be byte-for-byte deterministic?

The artifact digest is the package's identity — the anti-copy check, the cache key, and the thing independent workers must agree on. Without determinism, "the champion's package" is not a well-defined object.

### What if two workers disagree on the artifact hash?

Evaluation of that candidate **pauses**. It is neither scored nor terminated, because scoring one of two disagreeing artifacts would mean paying for a result nobody can reproduce. The operator investigates the software mismatch.

### Why is no language model used to judge?

A model-based scorer carries its own variance, and two packages could differ on a scored axis without differing in behaviour at all. It would also make re-scoring a stored trace non-reproducible.

Every score here comes from comparing a trace with truth computed before the candidate saw the instance.

### How do the layer groups work if the base model has 36 layers?

There are always four groups splitting the decoder stack into quarters. The group *names* are protocol and never change; the layer ranges follow the pinned model's depth, so repinning to a model of different depth does not invalidate existing recipes. `miner.cli pool` prints the current ranges.

### Why are `svd` and `cat_svd` the same thing?

Because concatenating factorisations is algebraically the sum of the updates, so with no sparsification and no sign election there is one sensible combination. Both names are kept because both appear in the reference merge implementations this engine mirrors. It is documented rather than hidden.

`linear` is genuinely different — it sums the factors rather than their products.

---

## Operating

### How long does a window take?

Roughly `(references + 1) × instances × per-instance-time`. With the default 100 hidden instances and around ten references, opening a window is the dominant cost. Size `window_blocks` accordingly.

### Can I change the comparator thresholds?

Yes, and they are published in the contract so miners can see them. Do not tune them in response to a specific candidate — that converts the engine from a measuring instrument into a decision-maker.

### What if a champion's recipe URL goes dead?

The engine keeps its own copy of every admitted recipe under `state/recipes/`, so a champion whose pointer went dead keeps defending.

### Can I run the engine without Docker?

Yes, but the container boundaries are the primary isolation for candidate-written code. The in-process resource limits are defence in depth, not a substitute.

### Do I earn anything while I wait in the queue?

A small, tapered share — most at the front of the queue, least at the back. It
is not payment for work: Bittensor prunes by lowest emission, so a miner holding
exactly zero is the first the chain evicts. The engine evaluates roughly one
challenger per window, so without it you could be deregistered before your
single evaluation ever ran.

### The champion looks unbeatable. Is the subnet finished?

No. The margin a challenger must clear over the *incumbent* decays to zero over
roughly thirty days, so an unopposed champion progressively loses its defender's
advantage. What does not decay is the margin over the permanent references —
beating an off-the-shelf equal-weight merge is the bar that says composition
added value at all, and that question does not get easier because someone
already answered it once.

### I did not take the throne. Did I earn anything?

Yes, if your package cleared every hard gate. The champion takes a fixed share
and everything below it is graded on quality, how far past the strongest
permanent reference you got, how close you came to the champion, and what your
package costs to run. That grade earns a proportional share for several windows,
and it is published broken into its four terms so you can see what earned it.

Clearing the gates is the threshold, and it is not negotiable. Grading applies
within the qualified set — it is not a consolation prize for producing something
undeployable.

### Does the subnet owner run code the rest of us cannot see?

Not for anything that decides a score. The engine, the workflow generators and
the scorers are all in this repository, and they have to be: the reason to
believe a single operator's numbers is that anyone can regenerate a closed
window's instances from their published seeds and re-run the scoring over the
published traces. Move the scorer somewhere private and that check stops
existing.

What an operator does keep private is the hidden seed root, wallet material,
filled-in configuration, host inventory and runbooks — operations, not protocol.
