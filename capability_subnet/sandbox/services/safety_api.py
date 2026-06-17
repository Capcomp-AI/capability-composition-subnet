"""The safety rule engine as a network service.

A deterministic rule engine, not a model. It answers one question — does this
plan contain every required precaution and no prohibited action — by set
comparison against the instance's policy, and it answers the same way every time.

That determinism is what makes the safety axis usable in a paired statistical
comparison. A model-judged safety score would carry its own variance, and two
packages could differ on that axis without differing in behaviour at all.

The service reports which required steps are missing, because a technician who
forgot one should be told. It never reports anything about the fault, the
component or the part: the plan is evaluated against the policy alone.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel

from capability_subnet.workflows.industrial_maintenance_de_v1.instance import SafetyPolicy
from capability_subnet.workflows.industrial_maintenance_de_v1.policy import (
    evaluate_plan,
    render_policy_de,
)

log = logging.getLogger(__name__)


# Defined at module scope so the validator can resolve the forward references
# created by the postponed-annotations import.
class PolicyPayload(BaseModel):
    policy_id: str
    required_steps: list[str]
    forbidden_actions: list[str]
    step_descriptions_de: dict[str, str] = {}
    forbidden_descriptions_de: dict[str, str] = {}


def create_app():
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(
        title="Safety rule engine",
        description="Deterministic policy enforcement for one workflow instance.",
    )
    policies: dict[str, SafetyPolicy] = {}

    def _get(instance_id: str) -> SafetyPolicy:
        policy = policies.get(instance_id)
        if policy is None:
            raise HTTPException(
                status_code=404,
                detail=f"no policy loaded for instance {instance_id}",
            )
        return policy

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "instances": len(policies)}

    @app.put("/instances/{instance_id}", status_code=201)
    def create(instance_id: str, payload: PolicyPayload) -> dict[str, Any]:
        policies[instance_id] = SafetyPolicy(
            policy_id=payload.policy_id,
            required_steps=tuple(payload.required_steps),
            forbidden_actions=tuple(payload.forbidden_actions),
            step_descriptions_de=dict(payload.step_descriptions_de),
            forbidden_descriptions_de=dict(payload.forbidden_descriptions_de),
        )
        return {"instance_id": instance_id, "policy_id": payload.policy_id}

    @app.get("/instances/{instance_id}/policy")
    def policy_text(instance_id: str) -> dict[str, Any]:
        policy = _get(instance_id)
        return {"policy_id": policy.policy_id, "text": render_policy_de(policy)}

    @app.post("/instances/{instance_id}/check")
    def check(instance_id: str, plan: Any = Body(...)) -> dict[str, Any]:
        policy = _get(instance_id)
        verdict = evaluate_plan(policy, plan)
        payload = verdict.as_dict()
        payload["policy"] = render_policy_de(policy)
        return payload

    @app.delete("/instances/{instance_id}", status_code=204)
    def destroy(instance_id: str) -> None:
        policies.pop(instance_id, None)

    return app


def main() -> int:
    import uvicorn

    from capability_subnet.common.logging import setup_logging

    setup_logging(os.environ.get("CAPSUB_LOG_LEVEL", "INFO"))
    uvicorn.run(
        create_app(),
        host=os.environ.get("CAPSUB_SERVICE_HOST", "0.0.0.0"),
        port=int(os.environ.get("CAPSUB_SERVICE_PORT", "8092")),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
