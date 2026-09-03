#!/usr/bin/env bash
# Build everything this repository can build locally, without a chain or a GPU.
#
# A synthetic adapter pool and a public workflow pack, which together exercise
# reconstruction, hashing and the workflow contract. It cannot tell you anything
# about model quality: the adapters are synthetic and no model is served behind
# them.
#
# It does not run the evaluation engine. The engine is a separate component the
# subnet operator runs - it is not in this package and not something a miner or
# a validator installs - so the steps that used to start it here called modules
# that no longer exist and failed halfway through, after writing state.
set -euo pipefail

STATE_DIR="${STATE_DIR:-.dev-state}"
POOL_DIR="${POOL_DIR:-.dev-pool}"

echo "==> synthetic adapter pool"
if [ ! -d "$POOL_DIR" ]; then
    python3 scripts/setup_dev_pool.py --out "$POOL_DIR"
fi

echo "==> public workflow pack"
python3 -m capability_subnet.workflows.cli generate-public-pack \
    --out "$STATE_DIR/public_pack" --count 12 --ood-count 4

echo
echo "Pool in $POOL_DIR, pack in $STATE_DIR/public_pack."
echo "Check a recipe against them with:"
echo "  capcomp validate --recipe <recipe.json>"
