# Changelog

Entries marked **consensus** change what a recipe produces or what a package is
worth, and require a coordinated upgrade rather than a rolling deploy.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html), with
the spec version derived from the release version and submitted alongside every
weight vector.

## [Unreleased]

### Added

- `lora_merger_logic_v1`, a single-turn arena over two pinned corpora: reasoning
  puzzles scored by exact match, and competitive-programming problems scored by
  executing the submitted program against stdin/stdout cases. Twelve axes. A
  quarter of each window is drawn from the executed corpus. See
  [docs/arena.md](docs/arena.md).
- Workflows are pluggable through the `capability_subnet.workflows` entry-point
  group. `workflow_id` in `backend.yaml` selects which one runs.
- `PythonRunner.run_program`, which runs a whole program against stdin under the
  same isolation as the function-calling path.
- Graded contribution scoring: quality 50%, improvement 25%, proximity 15%,
  cost 10%.
- A held-out general-capability retention probe, scored outside the arena.
- Tie-aware ranking: submissions closer than the window can resolve are ranked
  equal, and ties resolve to the earliest commitment.
- Streaming merge, so peak memory no longer scales with adapter count.
- `factorize_product`, an exact low-rank factorisation via thin QR and a small
  core SVD.
- Adapter importer that normalises public LoRAs to the canonical rank and
  rejects wrong-base, DoRA, rslora and `modules_to_save` sources.
- Pool expanded to 30 adapters, 26 structurally admitted, 23 selectable.
- Managed vLLM serving: the engine serves the artifact it builds, probes optional
  flags against `--help`, and samples peak VRAM through NVML.
- `--random` on `miner init`, drawing a random valid recipe from
  `miner/baseline.py`.

### Changed

- **The evaluation engine ships separately**, in an operator-only repository. It
  depends on this package; nothing here depends on it, and a test enforces that
  direction. Everything that decides a published number moved from
  `capability_subnet.backend.{scorer,comparator,baselines,weights}` into
  `capability_subnet.scoring`, which is public and stays public.
- `capability_subnet.testing` publishes the miniature-pool fixtures as a pytest
  plugin so both repositories test against the identical structure.
- Documentation consolidated from ten files to four: `architecture.md` (with the
  security model), `miner.md` (with the recipe reference), `validator.md` (with
  deployment) and `arena.md` (with the maintenance workflow).

### Changed — consensus

- Default workflow is `lora_merger_logic_v1`. Both shipped workflows remain
  registered and selectable.
- `require_beat_reference` defaults to False: the highest score on the board is
  paid whether or not it cleared the strongest permanent reference. References
  are still measured and published every window. The base-retention floor
  applies in both modes.
- `incentive_mode` defaults to `graded_contribution`.
- Capability certification is advisory rather than a gate. Structural admission
  is unchanged.
- `svd` and `cat_svd` resolve to one canonical method name.
- Thinking mode is disabled for every candidate.
- Retention is measured on a held-out probe rather than against workflow
  completion.
- The incumbent is excluded from the permanent reference set, so the bar does
  not ratchet with each dethrone.
- `workflow_id` is required on a window disclosure.
- Registry version 4: the pool declares `lora_merger_logic_v1`.

### Removed

- `miner/search.py`. The shipped starting point is a naive random recipe;
  composition search is left to miners.

### Fixed — consensus

- **The operator could choose which problems a candidate faced.** Instance seeds
  derive from a root only the operator held, and nothing bound them to it: trying
  roots until the draw suited an already-evaluated candidate would have passed
  every replay, because the seeds were real and the instances matched them. The
  draw now mixes in the hash of the block the window opened at — public, and not
  the operator's to pick — and every window publishes `root_commitment`, a hash of
  the seed root that must be identical across a deployment. `commitments_agree()`
  checks a run of disclosures for a root that moved, `verify_beacon_against_chain()`
  compares the published beacon with the real block hash, and the validator runs
  the first of those every round. What this does not close, and the architecture
  guide says so: a single root chosen dishonestly before any candidate exists.
  Nothing reveals the root, so a constant fabricated commitment passes.
- **A quarter of the code corpus could be passed without reading the input.**
  Competitive-programming statements print a worked example and the corpus keeps
  it as a test case — harmless alongside other cases, but 971 problems had only
  that one, putting the expected output in the question. A program that ignored
  its input and printed that constant passed. Problems are now admitted only if
  a constant program cannot pass — that is, the cases do not all expect the same
  output that the statement prints. 976 problems were exploitable; the pool goes
  from 3,920 to 2,944.

### Fixed

- `retained_energy` was computed after truncation and always reported 1.0.
- DARE merges failed on CUDA because the sparsity mask was drawn on CPU. The
  mask stays CPU-drawn for hardware independence and is moved to the delta's
  device.
- Reconstruction of a legal twelve-adapter recipe exhausted memory. Resource
  exhaustion is now classified as an engine failure, not a miner failure.
- vLLM start-up failures were truncated past the point of diagnosis.
- `tool_choice` was sent without `tools`, which the endpoint rejects.
- Serving `PATH` resolved through the virtualenv symlink and lost the
  interpreter's own `bin`.
- Nothing is keyed by whichever workflow is currently the default. Each workflow
  registers under its own id, states its own `WORKFLOW_ID`, and has its own
  on-chain commitment code.
- Both workflows publish the same protocol facts — base model, pool, recipe
  schema and bounds, gates, weights, ranking rule, windows, incentive — from
  `workflows/shared_contract.py`.
- `workflows.cli show` accepted `--workflow` and then read one workflow's
  instance fields. It now falls back to a generic rendering.
- Execution details were reported in German by the English-language arena.
- Code problems are scored on at most 32 test cases, applied at load time so the
  retained prefix travels with the instance. A program that exhausts its time or
  memory ceiling gets no further cases.

## [2.0.0] — 2026-07-29

### Changed — consensus

- The adapter pool is twelve public Qwen3-8B LoRAs, each verified against the
  pinned base and normalised to the canonical rank.
- Retention is measured on a held-out general-capability probe.
- The incumbent no longer counts among the permanent references.
- Thinking mode is disabled.
- `svd` and `cat_svd` are one method.
- Exact reconstruction for merges with no sparsification.

### Added

- Bittensor 11 SDK: intent-based writes and the current read API.
- vLLM tool-calling configuration.
- Determinism tests across CPU and CUDA.

## [1.0.0] — 2026-07-25

### Added

- **Protocol.** The declarative recipe contract: bounded numbers, names from a
  frozen registry, enums. Canonical JSON, content digests, on-chain commitments.
- **Registry.** The pinned base manifest and the certified adapter pool.
- **Merge engine.** Deterministic reconstruction across seven merge presets.
- **Workflow.** Industrial Maintenance DE: seven dependent stages, none judged
  by a model.
- **Sandbox.** One isolated environment per instance — the fixed agent loop,
  tool services, and execution limits.
- **Evaluation engine.** Continuous champion-challenge with per-window instance
  draws, paired comparison and signed reports.
- **Neurons.** Miner recipe tooling with local evaluation; a validator that
  verifies before setting weights.
- **Platform.** Object storage, compatibility history, dashboard.
- **Verification.** `capability-audit` checks published records independently.

### Security

- Recipes are data, never code. Nothing a miner submits is executed.
- Candidate code runs in an isolated interpreter with memory, CPU and wall-clock
  ceilings and no inherited environment.
- SQL tools accept read-only statements only.
- Validators verify operator signatures against an allow-list and burn rather
  than submit an unverifiable vector.
