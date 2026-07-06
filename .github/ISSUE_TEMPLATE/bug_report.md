---
name: Bug report
about: Something behaves differently from what the contract or docs say
labels: bug
---

## What happened

## What the contract or docs say should happen

<!-- Quote the relevant part. If they disagree with each other, say so — that is
     itself the bug. -->

## Reproducing it

<!-- A seed is worth more than a description. Instances are a pure function of
     their seed, so `--seed <n>` reproduces one exactly:

     python -m capability_subnet.workflows.cli show --seed <n> --with-truth
-->

## Environment

- Version (`python -c "import capability_subnet; print(capability_subnet.__version__)"`):
- Role: miner / validator / operator
- Python and OS:

## Scoring integrity

- [ ] This could affect a score, an artifact digest, or the hidden material.

<!-- If you ticked that box, please report it privately instead. See
     docs/security.md. A scoring-integrity bug invalidates results rather than
     merely disrupting service, and a public issue tells everyone how to use it
     before it is fixed. -->
