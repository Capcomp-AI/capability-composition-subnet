# Experiments

The measurements the go/no-go decision rests on, and the harnesses that produced
them. Kept in the repository because a result nobody can reproduce is an
assertion, and this subnet's whole argument is that assertions are not enough.

| file | what it does |
|---|---|
| `affine_bench.py` | Single adapters versus merges on `AffineFoundation/affine-lgc`, pinned revision, difficulty-banded, paired, exact-match scored |
| `analyse_affine.py` | Per-task best-single versus best-merge, with Wilson intervals |
| `affine_results.json` | 17 packages × 250 paired items |
| `probe_results.json` | General-capability retention across the same packages |

## Reproducing

```bash
python experiments/affine_bench.py \
    <base-model-dir> <pool-dir> <packages-dir> results.json <vllm-python> 25 <gpu>
python experiments/analyse_affine.py results.json
```

Both are seeded and pin the dataset revision, so a rerun draws the same items.

## What they found

Composition lost. Every merge scored at or below the base model, on 0 of 10
tasks did a merge beat the best single adapter, and the ordering matched the
retention probe exactly — which is two independent measurements agreeing.

The scope matters and is stated in the changelog: this measures a pool of twelve
scavenged public adapters with no coherent capability coverage, six of which
individually fall below the retention floor. It does not show that composition
cannot work.
