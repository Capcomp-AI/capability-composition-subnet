---
name: Workflow proposal
about: Propose a new workflow for the network to optimise
labels: workflow
---

## The business process

<!-- What repeatable process does this automate, and for whom? A workflow is not
     a benchmark list — "German + SQL + Python" is not a workflow, it is three
     capabilities looking for a task. -->

## Stage chain

<!-- The ordered stages, and what each consumes from the ones before it. If the
     stages are independent, this is a benchmark suite rather than a workflow. -->

## How each stage is judged

<!-- The bar: no language model may decide any part of the result. Every stage
     must be judged by execution, schema validation, exact comparison against
     generated truth, or a deterministic rule engine.

     If a stage needs a model to judge it, it cannot take part in a paired
     statistical comparison — the judge's own variance would be
     indistinguishable from a difference between packages. -->

| Stage | Judged by |
|---|---|
|  |  |

## Which capabilities it requires

## Commercial case

<!-- Who would deploy the resulting package, and what does it replace? -->

## Out-of-distribution mutations

<!-- How would you restate the same problem so a package that memorised the
     surface fails while one that understood it does not? -->
