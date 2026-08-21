#!/usr/bin/env python3
"""A working miner, start to finish, in one file.

It does the whole loop — read the pool, build recipes, keep the best one, write
it out, print the commitment — and it does the interesting part badly on
purpose. The search here is *random*: draw N recipes, score them, keep the
winner. That is the part you replace. Everything around it is the part you can
rely on.

Run it:

    python examples/quickstart_miner.py --tries 20 --out recipe.json

That needs nothing but the package: no GPU, no chain, no engine. It builds
valid recipes and ranks them with a cheap local heuristic.

To rank them by what the network actually measures, serve the merged adapter
and pass its endpoint:

    python examples/quickstart_miner.py --tries 20 --serve http://127.0.0.1:8000/v1

Then commit the winner (this writes to the chain):

    capability-miner commitment --recipe recipe.json --recipe-uri https://…/recipe.json

The commitment carries a digest and a pointer. The pointer has to resolve to
the exact bytes you committed — publish the file first, commit second.
"""

from __future__ import annotations

import argparse
import random
import sys

from capability_subnet.common import constants as C
from capability_subnet.common.commitments import MAX_PAYLOAD_BYTES, encode_commitment
from capability_subnet.common.schemas import Recipe
from capability_subnet.miner.recipe import advise, check_recipe, describe, new_recipe, write_recipe
from capability_subnet.registry.snapshot import PoolSnapshot, load_snapshot

# Methods worth drawing from. `linear` is the cheapest and reproduces
# byte-for-byte on any device; the SVD methods trim and re-factorise, which is
# usually better and costs a much slower merge.
METHODS = (C.MERGE_LINEAR, C.MERGE_TIES_SVD, C.MERGE_DARE_TIES_SVD)


def draw_recipe(rng: random.Random, snapshot: PoolSnapshot) -> Recipe:
    """One valid recipe, drawn at random.

    Valid, not good. Every field here is a decision the contract lets you make
    and this function makes all of them by coin flip — which adapters, how many,
    how to combine them, at what weights. Choosing them for a *reason* is the
    whole competition.
    """
    selectable = sorted(snapshot.registry.selectable_ids)
    if len(selectable) < C.MIN_SELECTED_ADAPTERS:
        raise SystemExit(f"the pool has only {len(selectable)} selectable adapters")

    count = rng.randint(C.MIN_SELECTED_ADAPTERS, min(C.MAX_SELECTED_ADAPTERS, len(selectable)))
    chosen = rng.sample(selectable, count)
    method = rng.choice(METHODS)

    return new_recipe(
        chosen,
        combination_type=method,
        # Density and sign election only apply to the trimming methods; passing
        # them for `linear` is what the schema rejects.
        density=None if method == C.MERGE_LINEAR else round(rng.uniform(0.3, 0.7), 2),
        majority_sign_method=None if method == C.MERGE_LINEAR else "total",
        random_seed=rng.randrange(1 << 30),
        global_weights={a: round(rng.uniform(0.8, 1.4), 2) for a in chosen},
        snapshot=snapshot,
    )


def cheap_score(recipe: Recipe, snapshot: PoolSnapshot) -> float:
    """A stand-in for measurement, so the loop runs with no GPU.

    It rewards nothing the network rewards. It exists so this file is runnable
    end to end on a laptop; if you are choosing between two recipes on this
    number you are choosing at random with extra steps. Use --serve.
    """
    penalty = len(advise(recipe, snapshot))
    return 1.0 / (1.0 + penalty) + len(recipe.selected_adapters) * 1e-3


def measured_score(recipe: Recipe, args, snapshot: PoolSnapshot) -> float:
    """What the network measures: end-to-end completion on public instances.

    Reconstructs the recipe, serves it through the endpoint you point at, and
    scores it with the same scorer the engine runs. This is the number worth
    searching on.
    """
    from capability_subnet.miner.local_eval import evaluate_locally
    from capability_subnet.sandbox.model_client import OpenAICompatibleClient

    client = OpenAICompatibleClient(args.serve, args.model)
    result = evaluate_locally(
        recipe,
        client,
        pool_dir=args.pool,
        artifact_dir=args.artifacts,
        instance_count=args.instances,
        snapshot=snapshot,
    )
    return result.scores.end_to_end


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tries", type=int, default=20, help="how many recipes to draw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="recipe.json")
    parser.add_argument("--recipe-uri", default="", help="where you will publish the file")
    parser.add_argument("--serve", default="", help="an endpoint serving the merged adapter")
    parser.add_argument("--model", default="candidate")
    parser.add_argument("--pool", default="pool", help="certified adapters on disk")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--instances", type=int, default=30)
    args = parser.parse_args(argv)

    snapshot = load_snapshot()
    rng = random.Random(args.seed)
    print(f"pool {snapshot.sha256[:19]}… · {len(snapshot.registry.selectable_ids)} selectable\n")

    best: tuple[float, Recipe] | None = None
    for attempt in range(1, args.tries + 1):
        recipe = draw_recipe(rng, snapshot)

        # Never submit a recipe that fails this. Admission runs the same checks
        # and a rejected commitment costs a run.
        problems = check_recipe(recipe, snapshot)
        if problems:
            print(f"  {attempt:3d}  invalid: {problems[0]}")
            continue

        score = (
            measured_score(recipe, args, snapshot) if args.serve else cheap_score(recipe, snapshot)
        )
        marker = " "
        if best is None or score > best[0]:
            best, marker = (score, recipe), "*"
        print(
            f"  {attempt:3d}{marker} {score:.4f}  {recipe.merge.combination_type:14s}"
            f" {len(recipe.selected_adapters)} adapters"
        )

    if best is None:
        print("\nnothing valid was drawn — widen the search")
        return 1

    score, recipe = best
    write_recipe(recipe, args.out)
    print(f"\nbest {score:.4f}\n")
    print(describe(recipe, snapshot))
    print(f"\nwritten to {args.out}")
    print(f"digest      {recipe.digest()}")

    if args.recipe_uri:
        payload = encode_commitment(recipe.workflow_id, recipe.digest(), args.recipe_uri)
        print(f"commitment  {payload}")
        print(f"            {len(payload.encode())} of {MAX_PAYLOAD_BYTES} bytes")
    else:
        print("\nPublish the file, then commit it:")
        print(f"  capability-miner commitment --recipe {args.out} --recipe-uri <url>")

    if not args.serve:
        print("\nThese were ranked by a placeholder, not by measurement. Serve the")
        print("merged adapter and re-run with --serve to rank on what is scored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
