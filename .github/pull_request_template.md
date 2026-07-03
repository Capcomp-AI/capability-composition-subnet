## What this changes

<!-- What behaviour is different afterwards, and why it should be. -->

## Worth a closer look

<!-- The decisions a reviewer would otherwise have to reconstruct: a trade-off
     you made, an approach you rejected, a place the obvious reading is wrong. -->

## Checks

- [ ] `make lint` and `make test` pass
- [ ] `python -m capability_subnet.workflows.cli selftest --count 15` passes if the workflow changed

## Consensus impact

<!-- Delete whichever does not apply. -->

- [ ] **No consensus impact.** Cannot change an artifact digest or a score.
- [ ] **Consensus-relevant.** Changes reconstruction, scoring, the comparator or
      the workflow. `__version__` is bumped, and the upgrade needs coordinating
      with validators before it is deployed.

## Failure classification

<!-- Only if this adds a failure path. Getting it wrong is invisible in a normal
     run and terminates real miners. -->

- [ ] New failure paths are classified as miner failures (fail closed, score
      zero) or infrastructure failures (fail open, hold the queue), and covered
      in `tests/integration/test_failure_classification.py`.
