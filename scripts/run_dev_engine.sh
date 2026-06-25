#!/usr/bin/env bash
# Run the whole engine locally, end to end, without a chain or a GPU.
#
# Useful for exercising the control loop, the store, the report publisher and the
# API. It cannot tell you anything about model quality: the adapter pool is
# synthetic and there is no served model behind it.
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

echo "==> one engine pass (dry run)"
CAPSUB_STATE_DIR="$STATE_DIR" \
CAPSUB_ADAPTER_POOL_DIR="$POOL_DIR" \
CAPSUB_HIDDEN_INSTANCES=4 \
CAPSUB_OOD_INSTANCES=2 \
CAPSUB_RECONSTRUCTION_WORKERS=2 \
python3 -m capability_subnet.backend.service --dry-run --once --state-dir "$STATE_DIR"

echo "==> dashboard"
CAPSUB_STATE_DIR="$STATE_DIR" python3 -m capability_subnet.platform.dashboard \
    --out "$STATE_DIR/dashboard.html"

echo
echo "State in $STATE_DIR. Serve the API with:"
echo "  CAPSUB_STATE_DIR=$STATE_DIR python3 -m capability_subnet.backend.api"
