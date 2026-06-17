"""The inventory simulator as a network service.

The tool logic is identical to the in-process simulator — it *is* the in-process
simulator, imported and wrapped — so a candidate cannot tell which deployment it
is running against, and a score obtained locally means the same thing as one
obtained in the containerised sandbox.

Running it as a separate container buys isolation: the inventory state lives
outside the process that talks to the model, so a prompt injection that
compromised the agent loop still could not rewrite the final state the stage is
scored on.

State is keyed by instance and is explicitly created and destroyed by the
orchestrator. There is no implicit creation: a request for an unknown instance is
a 404, not a fresh empty inventory, because silently inventing state would turn a
setup bug into a plausible-looking score.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel

from capability_subnet.workflows.industrial_maintenance_de_v1.instance import InventoryItem
from capability_subnet.workflows.industrial_maintenance_de_v1.inventory import InventorySimulator

log = logging.getLogger(__name__)


# Request models live at module scope so their forward references resolve. A
# model defined inside the factory cannot be rebuilt by the validator, and the
# failure only surfaces on the first request rather than at import.
class ItemPayload(BaseModel):
    part_number: str
    description_de: str
    on_hand: int
    reserved: int = 0
    location_de: str = ""
    obsolete: bool = False
    superseded_by: str | None = None


class SearchPayload(BaseModel):
    part_number: str


class ReservePayload(BaseModel):
    part_number: str
    quantity: int


def create_app():
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(
        title="Inventory simulator",
        description="Deterministic stock state for one workflow instance.",
    )
    simulators: dict[str, InventorySimulator] = {}

    def _get(instance_id: str) -> InventorySimulator:
        simulator = simulators.get(instance_id)
        if simulator is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no inventory state for instance {instance_id}. The orchestrator "
                    "must create it before the agent loop starts."
                ),
            )
        return simulator

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "instances": len(simulators)}

    @app.put("/instances/{instance_id}", status_code=201)
    def create(instance_id: str, items: list[ItemPayload] = Body(...)) -> dict[str, Any]:
        simulators[instance_id] = InventorySimulator.from_items(
            tuple(
                InventoryItem(
                    part_number=item.part_number,
                    description_de=item.description_de,
                    on_hand=item.on_hand,
                    reserved=item.reserved,
                    location_de=item.location_de,
                    obsolete=item.obsolete,
                    superseded_by=item.superseded_by,
                )
                for item in items
            )
        )
        return {"instance_id": instance_id, "items": len(items)}

    @app.post("/instances/{instance_id}/search")
    def search(instance_id: str, payload: SearchPayload) -> dict[str, Any]:
        return _get(instance_id).search(payload.part_number)

    @app.post("/instances/{instance_id}/reserve")
    def reserve(instance_id: str, payload: ReservePayload) -> dict[str, Any]:
        return _get(instance_id).reserve(payload.part_number, payload.quantity)

    @app.get("/instances/{instance_id}/final_state")
    def final_state(instance_id: str) -> dict[str, Any]:
        simulator = _get(instance_id)
        return {
            "reservations": simulator.final_state(),
            "attempts": simulator.attempts,
        }

    @app.delete("/instances/{instance_id}", status_code=204)
    def destroy(instance_id: str) -> None:
        # Unconditional: an orchestrator tearing down after a crash must not have
        # to know whether setup got far enough to create the state.
        simulators.pop(instance_id, None)

    return app


def main() -> int:
    import uvicorn

    from capability_subnet.common.logging import setup_logging

    setup_logging(os.environ.get("CAPSUB_LOG_LEVEL", "INFO"))
    uvicorn.run(
        create_app(),
        host=os.environ.get("CAPSUB_SERVICE_HOST", "0.0.0.0"),
        port=int(os.environ.get("CAPSUB_SERVICE_PORT", "8091")),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
