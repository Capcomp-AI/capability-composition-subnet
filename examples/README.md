# Examples

## `quickstart_miner.py`

A miner that works, start to finish, in one file: read the pool, build valid
recipes, keep the best one, write it out, print the commitment.

```bash
pip install -e .
python examples/quickstart_miner.py --tries 20 --out recipe.json
```

No GPU, no chain, no engine. It draws recipes, rejects the invalid ones, and
ranks the survivors.

**It ranks them badly on purpose.** Without `--serve` the ranking is a
placeholder that rewards nothing the network rewards — it exists so the file
runs on a laptop. Choosing between two recipes on that number is choosing at
random with extra steps.

### Ranking on what is actually measured

Serve the merged adapter and point the script at it:

```bash
python examples/quickstart_miner.py --tries 20 \
  --serve http://127.0.0.1:8000/v1 --pool ./pool --instances 30
```

Now each candidate is reconstructed, served, and scored by the same scorer the
engine runs, on public instances. That is the number worth searching on.

### Submitting

Publish the recipe file somewhere that will still resolve tomorrow, then commit
the digest and the pointer:

```bash
capability-miner commitment --recipe recipe.json --recipe-uri https://…/recipe.json
```

Order matters. The commitment names bytes by digest, and the engine fetches the
pointer and checks it against that digest — so publish first, commit second. A
pointer that resolves to different bytes is rejected, and a rejected commitment
costs you the run.

The whole payload is capped at 128 bytes, of which 57 are the magic, the
workflow code and the digest. That leaves **71 characters for the URI**, which
is tighter than it sounds:
`https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>` usually will
not fit, and `https://github.com/<owner>/<repo>/raw/<branch>/<path>` is eleven
characters shorter.

Long names are what break it. This repository's own raw URL is 79 characters
and does not fit at all, so a recipe cannot be served from it. Host the file
somewhere with a short name — a bare domain costs about 30 characters and
leaves room to spare.

## What to replace

Everything except `draw_recipe`. The scaffolding — validation, digests,
commitment encoding, local scoring — is the part you can rely on. The search is
the part you compete on, and random sampling is the weakest search there is.

Things the contract lets you decide, that random sampling decides by coin flip:

- **which adapters**, and how many (2 to 10). They are not interchangeable; the
  pool publishes what each one claims to do.
- **the merge method**. `linear` is cheap and reproduces byte-for-byte anywhere;
  the SVD methods trim and re-factorise, which is usually better and costs a far
  slower merge.
- **density and sign election**, for the trimming methods.
- **per-adapter weights**, and **per-layer-group overrides** — the same adapter
  can be worth more in early layers than late ones.

Read `capability-miner contract` before optimising anything. Every threshold that
decides your score is in it, including the ones that terminate a submission
outright.
