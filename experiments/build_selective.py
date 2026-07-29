"""A merge of only the adapters that individually beat or matched the base model.

The reference merges combine every capability adapter, which is the right
definition for a *baseline* — it is what you get without doing any composition
research. It is not what a miner would submit. Six of those ten adapters
individually fall below the retention floor, and a miner can see that from the
published pool. The recipe format exists precisely so they can select a subset.
"""
import json, sys, time, pathlib
from capability_subnet.registry.snapshot import load_snapshot
from capability_subnet.merge_engine.engine import reconstruct
from capability_subnet.merge_engine.loader import SafetensorsAdapterSource
from capability_subnet.common import constants as C
from capability_subnet.common.schemas import CompressionSpec, MergeSpec, OutputSpec, Recipe

POOL, OUT, DEV = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
snap = load_snapshot(); source = SafetensorsAdapterSource(POOL)

# Measured >= the base model on the benchmark, and at 1.000 retention on the probe.
STRONG = ["action-planner-v1", "code-generation-v1", "constrained-selection-v1",
          "creative-writing-v1", "legal-citation-v1"]

VARIANTS = {
  "selective_ties": dict(m=C.MERGE_TIES_SVD, d=0.5, s="total"),
  "selective_linear": dict(m=C.MERGE_LINEAR, d=None, s=None),
}
manifest_path = OUT / "manifest.json"
man = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
for name, cfg in VARIANTS.items():
    r = Recipe(workflow_id=snap.registry.workflow_id, base_revision=snap.manifest.revision,
        source_snapshot_sha256=snap.sha256, selected_adapters=sorted(STRONG),
        merge=MergeSpec(combination_type=cfg["m"], density=cfg["d"],
                        majority_sign_method=cfg["s"], random_seed=1000003),
        compression=CompressionSpec(output_rank=64, svd_clamp_quantile=1.0),
        output=OutputSpec(adapter_name="sel"))
    t=time.time(); res = reconstruct(r, snap, source, output_dir=OUT/name, device=DEV)
    man[name] = {"path": str(OUT/name), "method": cfg["m"], "adapters": len(STRONG),
                 "path_used": res.stats.merge_path, "artifact_sha256": res.artifact_sha256}
    print(f"  {name:18s} {time.time()-t:6.1f}s  {res.stats.merge_path:12s} {len(STRONG)} adapters", flush=True)
json.dump(man, open(manifest_path,"w"), indent=1)
