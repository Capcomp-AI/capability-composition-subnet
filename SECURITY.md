# Security policy

## Reporting

Report privately through [GitHub security advisories](https://github.com/favoroot/lora-merger/security/advisories/new)
rather than opening a public issue.

Include what you found, how to reproduce it, and what it would let an attacker
do. A seed reproduces an instance exactly, so `--seed <n>` is usually the
shortest path to a demonstration.

## What matters most

Findings that affect **scoring integrity** are the highest priority, because
they invalidate results rather than merely disrupting service:

- **Determinism.** Anything that makes two workers reconstruct the same recipe
  differently. Every artifact digest the network has recorded depends on this.
- **Hidden material.** Any route from a candidate, a tool, or the public API to
  the hidden instances, their seeds, the seed root, or the ground truth.
- **Self-influence.** Anything letting a candidate affect its own evaluation —
  including forcing an instance to be excluded from scoring, which is worth real
  score to a candidate that is failing.
- **Gate bypass.** Any way to clear a hard gate without meeting it, including a
  measurement that degrades to a value which happens to pass.
- **Attribution.** Anything letting a party other than the operator produce a
  weight vector or report that validators would accept.

Please report these privately even if they look minor. A public issue explains
the technique to everyone before it can be fixed.

## What is deliberately not defended

Stated so you can tell a finding from a known limitation. The reasoning is in
[docs/architecture.md](docs/architecture.md#security-model).

- **A miner's private search.** Any hardware, any method. The network judges the
  artifact, not the process.
- **Operator honesty about the hidden draw.** The generator is published and the
  draw is deterministic given the root, but the root is secret, so the fairness
  of the draw rests on the operator. Signed reports and published recipes make
  fabrication detectable in principle; they do not prevent it.
- **Availability of a miner's pointer.** If a recipe's host is unreachable at
  admission, the submission is not admitted.
- **The base model itself.** Pinned upstream and taken as given.

## Supported versions

Only the latest release. Changing reconstruction or scoring is a coordinated
network upgrade rather than a patch, so backporting a fix to an older spec
version would produce a client that disagrees with the network about what a
recipe is worth.
