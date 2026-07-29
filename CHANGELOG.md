# Changelog

Notable changes to the Capability Composition Subnet.

Because reconstruction and scoring are consensus-relevant, entries are marked
**consensus** where they change what a recipe produces or what a package is
worth. Those require a coordinated upgrade rather than a rolling deploy.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html), with
the spec version derived from the release version and submitted alongside every
weight vector.

## [Unreleased]

### Changed — consensus

**The arena is the default workflow.** `DEFAULT_WORKFLOW_ID` and
`backend.example.yaml` now select `lora_merger_logic_v1` rather than the German
maintenance chain. Both remain registered and selectable; this changes which one
an operator gets without configuring anything. Consensus-relevant because a
recipe's commitment names its workflow — a submission for one is not a submission
for the other.

**Highest score wins.** `require_beat_reference` defaults to False: a
submission no longer has to clear the strongest permanent reference to be paid.
The product is the best composition anyone has found, not proof that
composition was worth attempting — and the old rule had a real failure attached,
where a network producing perfectly good comparative information burned its
emission indefinitely because nothing cleared an absolute bar. References are
still measured and published every window, so the question stays answerable from
the record; it just stops gating payment. Set True for the stricter contract.

Base retention does **not** move with it. A package that destroyed the base
model's general ability is not deployable whatever it scored.

**A copy can no longer take a slot on noise.** This is the hole the change above
opens, and it needed closing in the same release. Under a margin rule a copy of
the leader could not displace it — identical scores are not a margin. Under
highest-score-wins a copy *ties*, and since no two evaluations of two distinct
artifacts land on the same number, a copy with one coefficient nudged takes the
top slot roughly half the time on sampling noise alone. Recipes are public, so
this is read-and-resubmit rather than a hypothetical.

Submissions closer together than the window can resolve are now ranked as tied,
and ties resolve to the earliest commitment. A copier is later by construction,
so it has to be *measurably* better — the same bar the margin enforced, in the
units the evidence supports. Indistinguishability is not transitive, so ranking
groups maximal runs into equivalence classes rather than swapping pairs; a
single pass leaves a later entry ahead of an earlier one it cannot be shown to
beat, which is precisely the hole being closed.

**Capability certification is advisory, not a gate.** The two contracts need
different things from it: under an absolute bar an unmeasured adapter was a hole
in the argument, while under highest-score-wins a miner who picks a poor adapter
is answered by its score, immediately, over a pool far larger than an operator
can characterise by hand. Structural admission does not relax — those tensors
load into a process that also holds hidden evaluation material.

### Added

**A second workflow, `lora_merger_logic_v1`, and it is the one a launch runs.**
Single-turn, twelve axes, two pinned corpora: ~3,193 reasoning puzzles across ten
families scored by exact match, and ~3,920 competitive-programming problems
scored by **executing** the submitted program against stdin/stdout cases. A
quarter of each window is drawn from the executed corpus.

Built because the maintenance workflow cannot answer the question this subnet
exists to ask. Its oracle needs ten of twelve turns and the pool contains nothing
German and nothing SQL, so a null result there measures calibration rather than
composition. This one puts as little as possible between the answer and the
measurement: one turn, no tools, no state, no model judging anything, and items
selected inside the band where the corpus's own measured pass rate says a model
of this class discriminates. Full reference in [docs/arena.md](docs/arena.md).

Its limitations are published rather than argued away. The corpora are public, so
the hidden seed protects *which* items a window draws and not the items
themselves — a weaker anti-overfitting property than a generated workflow, and
the price of corpora whose difficulty is already measured. The difficulty labels
were measured at pass@16 with sampling while this engine scores pass@1 greedy, so
absolute scores land far below the band; the first live run of a merged package
scored 4/30.

`affine-lgc-xlarge` was measured and **rejected**. It advertises 1,081,566 rows
against the chosen corpus's 81,566, and a larger pool would have been a real
mitigation for corpus visibility. But its difficulty column is populated on 8% of
rows, all inside its first shard, and banding that shard yields exactly the same
3,193 usable items — its labelled subset *is* the smaller corpus. Preferring it
cost four shards of download and bought no additional problems. The mitigations
that do apply are stratified selection, the executed quarter, and a
general-capability probe scored outside these corpora entirely.

**`PythonRunner.run_program`** runs a whole program against stdin and returns its
stdout, with the same isolation as the existing function-calling path: `-I -S`,
no inherited environment, memory and CPU ceilings, a wall-clock timeout.
Competitive-programming problems are stated as "read stdin, write stdout" and
cannot be scored by calling a named function.

Execution is bounded two ways, both found by measuring the corpus rather than by
reasoning about it. Cases are **capped at 32 per problem**: the median problem
carries 9 but the worst carries 565, and at a 20-second ceiling one instance would
have held a validator for three hours against a 15-second budget. And a program
that exhausts its time or memory ceiling gets **no further cases** — it would
exhaust the ceiling on each of them, passing already requires all of them, so
continuing cannot change the verdict but can multiply the ceiling by the case
count. A *crash* still runs every case, because it is cheap and the next case may
behave differently. Both rules are deterministic in the program and the cases,
which matters because two validators scoring the same trace must agree; the cap
is applied at load time so the retained prefix travels with the instance and a
replay scores the same cases.

**The pool is 30 adapters, 26 admitted.** A strict sweep found 209 eligible
LoRAs declaring `Qwen/Qwen3-8B` with full canonical target coverage and no
DoRA, rslora or `modules_to_save`. Eighteen were added, deduplicated by training
run so one author's checkpoint series cannot fill the pool, capped at two per
owner, rank 16 or above.

Four were **rejected by the licence gate** for an unstated licence. That gate
governs weights redistributed inside a merged artifact, which is a different
exposure from evaluation data, so it stays strict by default; overriding it is
an operator decision.

Six conversions were lossy — rank 128 or 256 down to the canonical 64 — with
retained energy between 0.870 and 0.970, now recorded per adapter. Those numbers
are only meaningful because of the `retained_energy` fix earlier in this
release; before it, every conversion reported perfect retention.

That leaves **23 selectable capability adapters**, or about 5.5 million adapter
subsets before a miner chooses a method, a rank or a single coefficient.

### Fixed — consensus

**Nothing is keyed by whichever workflow is default any more.** Pointing
`DEFAULT_WORKFLOW_ID` at the arena broke fifty-six tests at once, and every
failure was the same mistake made in a different file: an identity *read from the
default* instead of stated. The maintenance workflow was registered under
`C.DEFAULT_WORKFLOW_ID`, so moving the default **unregistered it** — the workflow
stopped existing rather than stopping being the default. Its own `WORKFLOW_ID`
came from the same constant, so it then called itself by the new default's name,
which recipes and reports are keyed by. Its on-chain commitment code was
registered the same way, so no commitment for it could be encoded or decoded. And
`WindowDisclosure.workflow_id` defaulted to it, so a disclosure that omitted the
field claimed to be whatever workflow was configured *at replay time* — meaning a
disclosure written before an operator switched workflows would regenerate the
wrong instances and score them with the wrong scorer, silently, because both
would succeed.

`workflow_id` on a disclosure is now **required**. An audit record has to name its
own subject. A regression test asserts the whole class: every workflow registers
under its own id, survives a change of default, has a commitment code that
round-trips, and publishes a complete contract.

**Both workflows publish the same protocol facts.** The base model, the frozen
pool, the recipe schema and its bounds, the hard gates, the qualified-score
weights, the ranking rule, the window sizes and the incentive split are facts
about the protocol, not about whichever workflow is judging — but they were
written out inside one workflow's contract, so the second shipped a contract with
**no base model and no pool in it**. That is not a contract a miner can build
against. They now come from `workflows/shared_contract.py`, which both call and a
third would too.

Unifying them surfaced two stale statements in the published contract, both now
correct: `incentive.default_mode` still said `winner_take_all` when settings
default to `graded_contribution`, and the ranking rule described the
champion-challenge margin as unconditional when `require_beat_reference` ships
off. The contract now states both modes and marks which is the default.

**`docs/architecture.md` described an incentive the engine no longer runs.** It
documented champion-challenge throughout — including "if a reference holds the
throne, the workflow share burns", which is the strict contract's behaviour, not
the shipped one. Corrected to state both modes with the default marked, and the
anti-copy section now credits the tie rule rather than a margin that is off.

**`workflows.cli show` crashed on any workflow but one.** It accepted
`--workflow` and then read the maintenance workflow's instance fields directly, so
selecting the arena raised `AttributeError`. It now falls back to a field-driven
rendering that a third workflow gets for free.

**Execution details were reported in German.** `run_program` reused the
resource-limit messages written for the German maintenance workflow, so an
English-language arena published `Zeitlimit oder Speichergrenze überschritten` in
its traces. The two maps are now separate rather than shared, because translating
in place would have put the wrong language in the other workflow.


**The comparator demanded a margin its sample size could not resolve.** A
paired comparison over *n* instances resolves roughly `2.8 · sqrt(0.15 / n)`.
The shipped configuration asked for a 0.03 margin over 100 hidden instances,
which resolves 0.108 — so a challenger with a genuine three-point edge could
not have demonstrated it, no matter how good it was.

That is not a strict network. It is one where the throne cannot be taken, and
nothing anywhere says so: the bootstrap declines correctly, every individual
verdict reads as an ordinary loss, and the shortfall is invisible in exactly the
records built to make things visible.

Three changes, all mechanism rather than tuning:

* `minimum_detectable_effect()` computes what a sample size can resolve, and
  **preflight refuses a deployment where the margin sits below it**, naming both
  ways out with numbers.
* Every published report carries the window's `minimum_detectable_effect`, so a
  reader can tell a package that genuinely lost from one the engine never had
  the evidence to judge — previously identical in every field of a report.
* A challenger ahead by less than the sample can resolve is now reported as
  *not enough evidence either way*, distinct from a loss.

Defaults are now chosen as a pair rather than independently: **400 hidden
instances** (resolving 0.054) against a **0.06 margin**. That is four times the
per-package GPU cost, which is the real price of a decision rule that means
anything.

**Preflight also refuses a window that cannot finish its own schedule.** The
symptom otherwise is silence — the engine re-measures references forever, the
queue never moves, and nothing records that the budget was impossible from the
start. `single_adapter_rotation` drops to 2 so the shipped defaults pass their
own check: 4,000 runs in 16.7h of a 24h window.

### Measured — the hypothesis, tested directly

Composition should make one package better than *each of its constituents on
the tasks that constituent is not the specialist for*. The earlier run answered
a different question — it compared merges against the best single adapter *per
task*, which is an oracle router, the strongest possible single-adapter
strategy and not what a deployer chooses between.

Retested with a merge of only the five adapters that individually certify:

| package | score | 95% CI |
|---|---|---|
| best single (`creative-writing-v1`) | 0.132 | [0.096, 0.180] |
| **`selective_ties`** | **0.112** | [0.079, 0.157] |
| base model | 0.100 | [0.069, 0.143] |
| `selective_linear` | 0.088 | [0.059, 0.130] |
| `owner_tuned` (all ten adapters) | 0.060 | [0.037, 0.097] |

**Selection nearly doubled the best merge** — 0.060 to 0.112 — purely by
excluding adapters that individually fail the retention floor. That is a
statement about miner strategy, and the recipe format already supports it.

Against the hypothesis itself: broader than **2 of 5** constituents on away
tasks, short of a majority. But on `word_sorting` the merge scored **0.44
against a best member of 0.36 and a base of 0.20** — composition beating every
one of its parts, which is precisely the predicted shape, on one task in ten and
on a margin of two items.

Nothing here is established at 250 items. The honest summary is that the effect,
if it exists, is smaller than this experiment could resolve — which is the same
finding as the comparator fix above, arrived at from the other direction.

### Measured — does composition beat the equal-weight merge?

**On this pool, no.** 250 paired items from `AffineFoundation/affine-lgc`
(revision `19765edac477`), ten task families, 25 items each, difficulty-banded
on the corpus's own measured pass rate, scored by exact match against ground
truth. Seventeen packages, identical items, greedy, thinking disabled.

| package | score | 95% CI | output tokens | kind |
|---|---|---|---|---|
| `creative-writing-v1` | 0.132 | [0.096, 0.180] | 182,070 | single (**a declared distractor**) |
| `code-generation-v1` | 0.116 | [0.082, 0.162] | 202,666 | single |
| **base model** | **0.100** | **[0.069, 0.143]** | **205,241** | reference |
| `action-planner-v1` | 0.100 | [0.069, 0.143] | 205,211 | single |
| owner's tuned recipe | 0.060 | [0.037, 0.097] | 62,973 | **merge** |
| equal-weight TIES | 0.056 | [0.034, 0.092] | 64,999 | **merge** |
| equal-weight SVD | 0.004 | [0.001, 0.022] | 115,349 | **merge** |
| equal-weight linear | 0.000 | [0.000, 0.015] | 254,950 | **merge** |

Merge better on **0** of 10 tasks. Single better on 7. Tied on 3.

What the confidence intervals actually support, at 250 items:

* `equal_linear` and `equal_svd` are worse than the base model — decisively,
  intervals do not overlap.
* `equal_ties` and the tuned recipe are worse than the base model
  *directionally*; their intervals overlap it, so this sample does not settle it.
* Best single beating best merge is directional, not established — the
  intervals touch. The consistent finding is the direction, reproduced across
  four merges and ten tasks.

Three things worth acting on:

**Every merge scored at or below the base model.** The ordering matches the
retention probe exactly — TIES survives, linear collapses — which is two
independent measurements agreeing on the same mechanism.

**The best single adapter is a declared distractor.** `creative-writing-v1`
outscored every capability adapter and the base model. The distractor labels in
this pool were assigned from descriptions, not measurement, and this is what
that costs. Nothing in the registry currently earns the label `is_distractor`
by evidence.

**Composition did lift one capability off the floor.** On `time_sequence` the
base model scores 0.00 and the TIES merge scores 0.08 — a capability the base
does not have. It is one family out of ten and a single adapter still beat it,
but it is the only positive signal in the run and it is the shape a real result
would take.

Read the scope: this measures *this* pool — twelve scavenged public adapters
with no coherent capability coverage, six of which individually fall below the
retention floor. It does not show that composition cannot work. It shows that
composing these adapters does not, which is exactly the question
`docs/architecture.md` says to answer before launching.

### Measured

First run of the real pool against the real pinned base model on a GPU. Forty
exactly-scored general-capability probe items, greedy decoding, thinking
disabled — the same items and the same settings the engine uses.

| package | probe | retention | output tokens | 0.98 gate |
|---|---|---|---|---|
| best single adapter | 36/40 | 1.000 | 241 | pass |
| **base model** | **35/40** | **1.000** | **250** | pass |
| equal-weight TIES merge | 34/40 | 0.971 | 252 | **rejected** |
| operator's tuned recipe | 32/40 | 0.914 | 264 | **rejected** |
| equal-weight SVD merge | 10/40 | 0.286 | 861 | **rejected** |
| equal-weight linear merge | 0/40 | 0.000 | 1280 | **rejected** |

Four findings, all actionable:

**Tuning lost to the equal-weight merge.** The operator's own reference recipe —
hand-chosen coefficients, layer-group emphasis, density 0.45, clamp 0.995 —
retains *less* than the same method run with every coefficient at 1.0. That is
the question this subnet exists to ask, and on this measurement the answer is
no.

Read it precisely, because it is not "tuning does not work". The tuned recipe
emphasises the structured-output and tool-calling adapters, which is meant to
buy workflow capability, and the probe does not measure workflow capability. So
what the number shows is the *cost* side of that trade with the benefit side
invisible — a package paying general ability for task ability, which is exactly
the trade the retention gate exists to detect. It caught it.

**Interference-aware merging is not a refinement, it is the difference between
working and not.** TIES with sign election retains 0.971; the same ten adapters
summed linearly retain **0.000** — the package cannot follow a one-line
instruction at all. The gap is the whole result.

**A collapsed package announces itself in token spend.** The linear merge burns
5.1x the base model's output tokens and the SVD merge 3.4x, because they answer
terse instructions with prose instead of answers. The token-efficiency component
added in this release scores exactly that, and it would have separated these
packages even without the retention probe.

**The pool's designated retention anchor was the most destructive adapter in
it.** `general-instruction-v1` scored 0/40 on its own, with the identical
failure fingerprint as the linear merge — which the linear merge inherited from
it. It has been reclassified as a controlled distractor on the evidence, and
this pool now has no retention anchor, which is the honest state: no public
Qwen3-8B adapter was trained to preserve capability under merging.

Measured `base_retention` is now recorded for all twelve adapters.
`capability_score` deliberately is not — the probe measures retention, not
whether an adapter is good at its declared capability, and inventing that number
is precisely what the certification gate exists to prevent.

Read the limit as well as the result: the probe is a *necessary* condition, not
the workflow. It can show that a merge destroyed general ability; it cannot show
that composition added workflow value.

**Every merge measured is rejected by the retention gate** — the best, TIES at
0.971, misses a 0.98 floor. As configured against this pool the network would
crown nobody and burn indefinitely. That is a calibration decision the operator
has to make before genesis, and it is now an informed one: either the floor
comes down to something a real merge can clear, or the pool gets an adapter that
was actually trained to preserve capability under merging. Lowering the floor
without fixing the pool would be choosing not to look.

### Verified on hardware

A merged rank-64 adapter served on a **single RTX 4090** through the engine's
own `ManagedVllmServer`, with both request shapes the protocol uses succeeding —
a bare completion and a tool call:

| | |
|---|---|
| startup | 67 s |
| peak GPU memory, NVML under load | 22.89 GB |
| GPU KV cache | 73,968 tokens |
| concurrency at full 16384 context | 4.51x |

So 24 GB is sufficient rather than merely adequate on paper, using
`--kv-cache-dtype fp8` (which quantises the cache, not the weights, leaving
canonical scores in bfloat16) and `--max-num-seqs 1` (which matches what the
engine does anyway). Quantising the *weights* would free ~8 GB and is
forbidden by the contract.

The 16384 context is justified, not padding: measured with the real tokenizer
the opening prompt is 4.3–4.6k tokens and a complete oracle run reaches 8.0–8.3k.

### Fixed — consensus

Both of these were found by building the real pool on a real GPU. Neither was
visible to a CPU-only test suite, and both would have surfaced on the first
production window rather than in review.

- **The stochastic merges could not run on a GPU at all.** The DARE family draws
  its drop mask on the CPU by design — CUDA generators differ across drivers and
  architectures, so a GPU-drawn mask would make an artifact depend on which card
  a worker was assigned — and the mask was then applied to a CUDA delta without
  being moved to it. Every `dare_*` reconstruction failed on the first
  projection. The mask is still drawn on the CPU; it is now transferred.
- **A legal recipe could exhaust the engine's memory.** The merge stacked every
  selected adapter's full update at once: twelve adapters at the base model's
  widest projection is 2.3 GB in float32 before the aggregation allocates
  anything, and ten took a 24 GB card out of memory. Since
  `MAX_SELECTED_ADAPTERS` is 12, a recipe that passed every validation check
  could not be built — and `ReconstructionError` is classed as a *miner* failure,
  so it would have terminated that miner's single evaluation for a limit it was
  never told about.

  The merge now streams one adapter at a time. Sum aggregation accumulates;
  sign-elected aggregation makes two passes, recomputing each contribution
  rather than holding it, which is exact because every step is deterministic
  given the seed. Peak memory no longer depends on how many adapters a miner
  selected. **This changes artifact bytes** — accumulation order differs from a
  single stacked reduction — hence the consensus marking.
- **The probe could never have reached a real endpoint.** Every request sent
  `tool_choice: "auto"`, including the general-capability probe, which
  deliberately offers no tools — and an OpenAI-compatible server answers that
  combination with a 400. The retention gate would have read the failures as a
  candidate answering nothing. `tool_choice` and `tools` are now sent only
  together.
- **The serving subprocess could not find tools shipped beside its
  interpreter.** vLLM JIT-compiles kernels with `ninja`, which lives in its
  virtualenv's `bin/`; pointing `serving_python` at another environment without
  putting that directory on PATH produced a bare `FileNotFoundError: 'ninja'`
  from deep inside engine start-up. Note the first attempt at this fix resolved
  the interpreter path, which follows a virtualenv's `python` symlink to
  `/usr/bin` — the one directory already on PATH — so the test pins the
  symlinked case specifically.
- **A serving start-up failure now explains itself.** The message attached the
  last 2000 characters of the runtime's log, which reliably hid the cause: vLLM
  prints thousands of lines of banner and shutdown noise around the exception
  that killed it, so every distinct failure — out of memory, an unrecognised
  option, an unreachable GPU — arrived as the same fragment of a file path. The
  cause is now extracted by matching exception markers, deduplicated across
  worker and parent, and an unmatched log says it is falling back rather than
  presenting its tail as the reason.
- **vLLM options are probed rather than assumed.** `--disable-log-requests` was
  removed rather than deprecated in vLLM 0.26, and an unknown option is an
  immediate argparse exit — so the engine could not serve *anything* against a
  current vLLM. Optional flags are now checked against the runtime's own
  `--help`. Flags the protocol depends on are deliberately not probed: a build
  without `--enable-auto-tool-choice` cannot run this workflow, and failing
  loudly at start-up is correct.
- `serving_python` selects the interpreter vLLM runs under, for the common
  deployment where it lives in its own virtualenv.
- Out-of-memory during reconstruction is now an *engine* failure, so a candidate
  is held rather than terminated when the host runs out of room.

### Added

- **Validators re-score a closed window before paying.** The disclosure and
  replay machinery already existed and was already exposed over the API;
  nothing consumed it. A published record that nobody reads before paying is
  documentation, not a control. Validators now regenerate a closed window's
  instances from their seeds, re-run the deterministic scorer over the
  published traces, and **burn if the engine's scores do not follow from its
  own traces**. No GPU and no model — the same VPS as before. A window that has
  not been disclosed yet is not treated as dishonesty; that would make an
  outage indistinguishable from fraud. `--neuron.no_spot_check` disables it.
- **Losing well is worth something.** `graded_contribution` is the new default
  incentive mode. The champion still takes a fixed share outright, but every
  candidate that cleared *every hard gate* is graded on quality (50%),
  improvement over the strongest permanent reference (25%), proximity to the
  champion (15%) and running cost (10%), and earns a proportional share for a
  bounded number of windows.

  Winner-take-all discarded the network's most useful signal: almost every
  submission that is ever evaluated fails to dethrone, a recipe is one shot, and
  paying a miner who moved completion from 0.41 to 0.58 exactly what it pays one
  that submitted a distractor soup leaves the second attempt no better informed
  than the first. If nobody qualifies the graded pool burns rather than becoming
  a bonus for an uncontested champion. Each grade is published broken into its
  four terms. `verify_weight_vector` gained a matching rule, because a mode that
  pays more people needs a rule about who may be paid.
- **Token spend is scored.** It was measured and reported but never scored, so a
  package that reached the same answer twice as expensively ranked identically.
  Counted per *completed* instance rather than per attempted one — dividing by
  attempts rewards giving up early.
- **Workflows are pluggable.** Discovery now goes through the
  `capability_subnet.workflows` entry point group, so a workflow can ship as its
  own distribution without forking this repository. Intended for the one case
  that genuinely needs it: a workflow built on a customer's real business
  process, whose generator cannot be published. Such a workflow declares
  `publicly_verifiable=False` and the engine says so loudly — a workflow nobody
  can install is a workflow nobody can replay. See `docs/repositories.md`.
- `tests/unit/test_layering.py` enforces the dependency boundary that the
  packaging change below depends on, so it cannot regress silently.

### Changed

- **The base install no longer contains the tensor stack.** `torch`, `numpy`
  and `safetensors` moved to a `merge` extra, pulled in by `[miner]`,
  `[backend]` and `[registry]`. A validator was downloading 1.7 GB of tensor
  library it never imports, against a documented 20 GB VPS; the base install is
  now roughly 50 MB. The layering this reflects was already true in the import
  graph — `common`, `workflows`, `miner`, `validator`, `audit` and `platform`
  have no module-level path to a heavy dependency — and only `pyproject.toml`
  contradicted it.
- `[sandbox]` extra added for the tool services, which run in their own
  container and need neither the tensor stack nor the chain SDK.

## [2.0.0] — 2026-07-29

A launch-blocking pass. The previous release could not have run: its adapter
pool did not exist, its engine never served the artifact it built, and its
declared SDK dependency resolved to an API that no longer had the methods the
code called. Everything below either removes one of those blockers or fixes a
rule that would have emptied the subnet once it did run.

### Changed — consensus

**The pool is real.** Twelve public Qwen3-8B LoRA adapters, each verified
against the Hugging Face API to declare `Qwen/Qwen3-8B` as its base, full
canonical target coverage, `bias: none`, and no DoRA, rslora or
`modules_to_save`. The base model is pinned to `b968826d`. Adapters that failed
inspection were rejected rather than adapted: three otherwise-attractive
candidates are built on a different base model, a 4-bit quantised mirror, and an
unverifiable local path respectively. `scripts/import_public_adapters.py`
fetches only the config and the weights, refuses any upstream whose config has
drifted from what the registry recorded, and normalises the update to the
canonical rank.

**Retention measures something else.** The gate compared the candidate's
*workflow* completion with the base model's, but a candidate only reaches the
gate after beating the base by an absolute margin — so the ratio was always
above one and the clamp returned exactly `1.0` for every candidate that could
possibly be crowned. Retention is now a held-out, deterministically-scored
general-capability probe drawn per window from its own seed.

**The bar no longer ratchets.** The incumbent counted among the references a
challenger had to clear by three points, so every successive champion had to
beat the previous one by a further three. Completion is bounded by one, so that
staircase stalls after a handful of dethrones and then one package holds the
throne permanently. The absolute margin now applies to the permanent references
only; the incumbent gets a separate, smaller margin that decays to zero over
roughly thirty days.

**Thinking mode is off.** Qwen3's chat template enables it by default, and one
`<think>` block consumes the whole output budget before the agent calls a tool.
Now explicit, and published in the contract.

**`svd` and `cat_svd` are one package.** They always resolved to the same
pipeline and built identical bytes, so two miners who independently chose
different names collided and the later was terminated for copying. Folded to a
canonical spelling before a recipe is hashed.

**Reconstruction is exact and fast.** A merged update with no sparsification is
a sum of low-rank products, so it is now decomposed from the factors rather than
from a materialised matrix — exact to 6e-7 and ~2800x faster on the real
projections. The trimming methods still need a dense decomposition and now run
on the GPU: 6 minutes per build against 2.8 hours. cuSOLVER and LAPACK do not
agree bit-for-bit, so `merge_device` is recorded in every published report and
every worker in one deployment must use the same one.

### Fixed

- **The engine now serves the artifact it builds.** `service.py` hardwired
  `ExternalServer`, which discards the adapter path, so every candidate and
  every reference would have been scored against one static endpoint —
  identical scores, no challenger ever distinguishable, emission burned forever.
  `ManagedVllmServer` is selectable via `serving_mode` and is the default;
  `ExternalServer` now refuses an adapter instead of silently ignoring it.
- **vLLM tool-calling is configured.** Without `--enable-auto-tool-choice` and a
  `--tool-call-parser`, `message.tool_calls` is never populated and
  `tool_choice: "auto"` is rejected outright.
- **Bittensor 11.** The 10.x `Subtensor` methods the code called do not exist in
  11.x. Commitments, weights, the metagraph, wallets and the config layer are
  rewritten against the intent/read API, and the dependency is pinned below 12.
- **Infrastructure failures no longer spend a miner's one shot.** An unreadable
  memory counter, too few scored instances or a latency figure computed from no
  completed runs are the engine's failures; they now hold the candidate instead
  of terminating it.
- **Peak VRAM is measured where the model actually runs.** It was read with
  `torch.cuda.max_memory_allocated` in the engine process while the model ran in
  a separate vLLM subprocess, so it reported essentially zero — and with the
  measurement required, every candidate failed the gate. Now sampled across the
  run through NVML.
- **Queued miners are no longer pruned before evaluation.** Bittensor evicts by
  lowest emission, and winner-take-all gave every waiting challenger exactly
  zero. A tapered tail share keeps the queue alive.
- **Burn goes to the subnet owner.** UID 0 is whichever neuron registered first,
  not an incinerator. Resolved from the metagraph; a validator that cannot
  resolve it submits nothing rather than paying a stranger.
- **A deregistered champion no longer deadlocks the subnet.** Its UID belongs to
  someone else, so every window burned while the dethrone bar stayed pinned to a
  package nobody could be paid for. The throne is now vacated.
- **`recipe_dir` derives from `state_dir`.** Admission wrote to one path and the
  champion loader read from another; they agreed only on the default.
- **`retained_energy` measured nothing.** Computed after truncation, so it
  reported perfect retention at every output rank.
- The vLLM subprocess inherits its environment and uses `sys.executable`, so a
  venv or conda deployment can find vLLM at all.
- The anti-copy recipe check runs before evaluation rather than after it.
- Reference measurement rotates the single-adapter baselines, so opening a
  window no longer consumes most of it before any challenger is evaluated.

## [1.0.0] — 2026-07-25

First complete implementation of the protocol: one pinned base model, one
certified adapter pool, one executable workflow, one declarative recipe format,
and a continuous champion-challenge engine that decides who holds the throne.

### Added

**Protocol.** The declarative recipe contract — bounded numbers, names from a
frozen registry, enums, and nothing else. Canonical hashing so key order and
whitespace cannot change a digest. Compact on-chain commitments carrying an
unpadded base64url digest and an immutable pointer.

**Registry.** The pinned base manifest and the certified adapter pool, with
admission gates covering executable content, tensor shapes, non-finite values,
licences, capability certification, and recertification after rank conversion.

**Merge engine.** Deterministic reconstruction across seven merge presets over
one explicit sparsify → elect signs → aggregate pipeline, with a canonical
singular-vector sign convention, threshold-based magnitude trimming and
per-tensor derived seeds. Two workers running the same image produce the same
bytes.

**Workflow.** Industrial Maintenance DE: seven dependent stages judged entirely
by execution, schema validation, exact comparison and a deterministic rule
engine. No language model decides any part of a result. Out-of-distribution
mutations restate the same problem in five named ways.

**Sandbox.** One isolated environment per instance: the fixed agent loop, seven
tool services, a code runner with no network and no writable filesystem, and a
scorer the agent cannot reach.

**Evaluation engine.** Continuous champion-challenge with per-window instance
refresh, permanent reference baselines, a partial-Pareto dethrone rule, and
paired bootstrap significance testing. Signed reports and signed weight vectors.

**Neurons.** Miner recipe tooling with local evaluation against the public pack,
and a thin validator that needs no GPU but verifies rather than relays.

**Platform.** Object storage, the compatibility history, a self-contained
dashboard, container images and operator configuration.

**Verification.** `capability-audit` independently checks published records
without a GPU and without the operator's cooperation, and closed windows are
disclosed so anyone can regenerate their instances and re-score the engine's own
traces.

### Fixed

Defects found by an audit pass rather than by a failing test, each now covered by
a regression test:

- **Admitted recipes were never persisted.** The engine could not re-measure a
  champion or evaluate any challenger, and the only symptom was one error line
  per pass. It now stores its own verified copy at admission.
- **An unreadable commitment block defaulted to zero**, which sorts to the
  *front* of an ascending queue and handed priority to exactly the commitments
  the engine understood least. Now skipped and retried, loudly.
- **An unreadable GPU counter reported 0 GB**, which passes a 24 GB limit, so a
  host with a broken counter cleared every candidate unchecked. Now reported as
  unmeasured, with an operator policy for whether that fails.
- **`all([])` is True**, so an evaluation that returned before the gates ran read
  as having passed all of them, and an ungated report counted as qualified for
  graded emission.
- **A tool crash marked a run unscorable**, letting a candidate exclude exactly
  the instances it was failing by crafting an argument that raised. Tool
  exceptions are now failed tool calls.
- **Tool calls were unbounded per turn**, so the twelve-turn budget could be
  evaded by batching fifty calls into one reply.
- **An empty validator signer allow-list silently accepted anything** answering
  at the configured URL. Now a startup error unless explicitly waived.
- **Validator staleness used a compiled-in window length** rather than the one
  the engine reports.
- **An incomplete reference set could still crown a champion**, against a bar
  quietly lower than the protocol promises.
- **A determinism failure was suppressed**, so a build unable to enable
  deterministic kernels would produce mismatched artifact digests with no signal.
- **Execution traces were lossy**, omitting the SQL submissions and diagnostic
  results the scorer reads. Found by the disclosure replay, which could not
  reproduce those stages — the mechanism catching a real defect on its first
  serious use.

### Security

- Database catalog access is blocked. Hidden snapshots share one PostgreSQL
  instance as separate schemas, so catalog enumeration was a route from one
  instance to another instance's data.
- The opening prompt no longer names the fault code, directly or through the
  safety policy identifier. Both leaks made the fault-extraction stage trivially
  solvable and broke the dependency between stages.

### Known limitations

Stated because they bear on whether to participate:

- **A centralised engine requires residual trust.** Signed reports, published
  recipes, disclosed windows and independent re-scoring narrow it considerably;
  they do not remove it. Nothing proves the hidden draw was fair.
- **Two merge presets coincide.** `svd` and `cat_svd` produce the same update.
  Both names are kept because both appear in the reference implementations this
  engine mirrors.
- **Static merging is not yet shown to beat routing.** The routed-adapter
  reference baseline is planned, not implemented. Until it exists the network
  cannot say when static merging is actually cheaper.
- **The base model and adapter pool ship unpinned and uncertified.** The engine
  refuses to start against them, by design.

[Unreleased]: https://github.com/favoroot/lora-merger/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/favoroot/lora-merger/releases/tag/v1.0.0
