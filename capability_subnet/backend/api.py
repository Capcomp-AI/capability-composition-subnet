"""The read-only engine API.

Everything the engine knows that is safe to publish, and nothing else. Validators
poll ``/weights``; miners read ``/contract``, ``/snapshot`` and their own
``/queue`` entry; anyone auditing the network reads ``/reports``.

The API is strictly read-only. It has no route that changes engine state, so
exposing it to the internet cannot affect an evaluation — the worst an attacker
can do is read what is already published.

What it will not serve, ever: hidden instances, their seeds, the secret seed root,
or ground truth. The endpoints below are an allow-list, not a filter over the
engine's state.
"""

from __future__ import annotations

import logging
from typing import Any

from capability_subnet import __spec_version__, __version__
from capability_subnet.backend.settings import BackendSettings, load_settings
from capability_subnet.backend.store import Store
from capability_subnet.registry.snapshot import load_snapshot
from capability_subnet.workflows import get_workflow

log = logging.getLogger(__name__)


def create_app(settings: BackendSettings | None = None):
    """Build the FastAPI application."""
    from fastapi import FastAPI, HTTPException, Query

    resolved = settings or load_settings()
    store = Store(resolved.database_path)
    snapshot = load_snapshot()
    workflow = get_workflow(resolved.workflow_id)

    app = FastAPI(
        title="Capability Composition Subnet",
        description=(
            "Read-only interface to the continuous champion-challenge evaluation "
            "engine. Validators fetch signed weight vectors here; miners read the "
            "workflow contract and the frozen adapter pool."
        ),
        version=__version__,
    )

    @app.get("/health", summary="Liveness and basic engine state")
    def health() -> dict[str, Any]:
        champion = store.get_champion()
        return {
            "status": "ok",
            "version": __version__,
            "spec_version": __spec_version__,
            "workflow_id": resolved.workflow_id,
            # Published so validators can judge a vector's freshness against the
            # window this deployment actually runs rather than a default.
            "window_blocks": resolved.window_blocks,
            "latest_window": store.latest_window_id(),
            "champion": champion.candidate_id if champion else None,
            "queued": len(store.list_queue("queued")),
        }

    @app.get("/contract", summary="The published workflow contract")
    def contract() -> dict[str, Any]:
        return workflow.build_contract(snapshot)

    @app.get("/snapshot", summary="The frozen certified adapter pool")
    def pool_snapshot() -> dict[str, Any]:
        return snapshot.document()

    @app.get("/weights", summary="The latest signed weight vector")
    def latest_weights() -> dict[str, Any]:
        vector = store.latest_weights()
        if vector is None:
            raise HTTPException(
                status_code=404,
                detail="no weight vector has been published yet",
            )
        return vector.model_dump(mode="json", exclude_none=True)

    @app.get("/weights/{window_id}", summary="The weight vector for one window")
    def weights_for_window(window_id: int) -> dict[str, Any]:
        vector = store.get_weights(window_id)
        if vector is None:
            raise HTTPException(status_code=404, detail=f"no weights for window {window_id}")
        return vector.model_dump(mode="json", exclude_none=True)

    @app.get("/champion", summary="The reigning champion")
    def champion() -> dict[str, Any]:
        record = store.get_champion()
        if record is None:
            return {"champion": None, "detail": "the throne is empty"}
        payload = record.model_dump(mode="json", exclude_none=True)
        payload["report_sha256"] = store.get_meta("champion_report_sha256")
        return payload

    @app.get("/queue", summary="Admitted challengers and their status")
    def queue(
        status: str | None = Query(default=None, description="Filter by status."),
    ) -> dict[str, Any]:
        entries = store.list_queue(status)
        return {
            "count": len(entries),
            "entries": [entry.model_dump(mode="json") for entry in entries],
        }

    @app.get("/queue/{hotkey}", summary="One miner's submission status")
    def queue_entry(hotkey: str) -> dict[str, Any]:
        entry = store.get_queue_entry(hotkey)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "no admitted submission for this hotkey. If a commitment was made, "
                    "it either has not been read yet or failed admission."
                ),
            )
        return entry.model_dump(mode="json")

    @app.get("/reports", summary="Published evaluation reports")
    def reports(
        window_id: int | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        rows = store.list_reports(window_id=window_id, limit=limit)
        return {
            "count": len(rows),
            "reports": [
                {"report_sha256": digest, **report.model_dump(mode="json", exclude_none=True)}
                for digest, report in rows
            ],
        }

    @app.get("/reports/{report_sha256}", summary="One evaluation report")
    def report(report_sha256: str) -> dict[str, Any]:
        record = store.get_report(report_sha256)
        if record is None:
            raise HTTPException(status_code=404, detail="no such report")
        return record.model_dump(mode="json", exclude_none=True)

    @app.get("/compatibility", summary="Adapter compatibility history")
    def compatibility(limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, Any]:
        rows = store.load_compatibility(limit=limit)
        return {"count": len(rows), "records": rows}

    @app.on_event("shutdown")
    def _close() -> None:
        store.close()

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    from capability_subnet.common.logging import setup_logging

    parser = argparse.ArgumentParser(
        prog="capability-backend-api",
        description="Read-only API for the evaluation engine.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    setup_logging(settings.log_level)

    uvicorn.run(
        create_app(settings),
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
