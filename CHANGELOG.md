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

Nothing yet.

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
