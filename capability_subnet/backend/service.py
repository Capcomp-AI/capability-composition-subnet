"""The engine process.

Wires the components together, then runs the loop: read the chain, admit new
commitments, evaluate the queue head, publish weights, wait, repeat.

Two operational rules are enforced here rather than left to a runbook. The
service refuses to start against a configuration that would make its own results
untrustworthy — an unpinned base model, a predictable hidden seed, a single
reconstruction worker — unless it is explicitly in dry-run mode. And a crash in
one pass never takes the process down: emission depends on validators being able
to fetch a weight vector, so the loop logs, backs off and continues.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from types import FrameType

from capability_subnet.backend.engine_loop import EngineLoop
from capability_subnet.backend.evaluation import Evaluator
from capability_subnet.backend.executor.reconstruction import ArtifactCache, Reconstructor
from capability_subnet.backend.executor.serving import ExternalServer
from capability_subnet.backend.monitor.admission import admit_new_commitments
from capability_subnet.backend.monitor.fetch import LocalRecipeSource
from capability_subnet.backend.reports.publisher import ReportPublisher
from capability_subnet.backend.settings import BackendSettings, load_settings
from capability_subnet.backend.store import Store
from capability_subnet.common.chain import current_block, read_commitments
from capability_subnet.common.logging import setup_logging
from capability_subnet.merge_engine.loader import SafetensorsAdapterSource
from capability_subnet.registry.base_model import require_pinned
from capability_subnet.registry.snapshot import load_snapshot
from capability_subnet.sandbox.orchestrator import SandboxConfig
from capability_subnet.workflows import get_workflow

log = logging.getLogger(__name__)


class EngineService:
    """The long-running engine."""

    def __init__(self, settings: BackendSettings) -> None:
        self.settings = settings
        self.should_exit = False

        settings.ensure_directories()

        self.snapshot = load_snapshot()
        self.workflow = get_workflow(settings.workflow_id)
        self.store = Store(settings.database_path)

        self.wallet = None
        self.subtensor = None
        keypair = None

        if not settings.dry_run:
            import bittensor as bt

            self.wallet = bt.wallet(name=settings.wallet_name, hotkey=settings.wallet_hotkey)
            self.subtensor = bt.Subtensor(
                network=settings.chain_endpoint or settings.network
            )
            keypair = self.wallet.hotkey

        self.publisher = ReportPublisher(settings.report_dir, keypair)
        self.recipe_store = LocalRecipeSource(settings.recipe_path)

        source = SafetensorsAdapterSource(settings.adapter_pool_dir)
        cache = ArtifactCache(settings.artifact_cache_dir)
        reconstructor = Reconstructor(
            self.snapshot, source, cache, workers=settings.reconstruction_workers
        )

        evaluator = Evaluator(
            reconstructor=reconstructor,
            server=ExternalServer(settings.serving_base_url, settings.serving_model_name),
            adapter_pool_dir=settings.adapter_pool_dir,
            sandbox_config=SandboxConfig(postgres_dsn=settings.postgres_dsn or None),
            stages=self.workflow.critical_axes,
            min_valid_samples=settings.min_axis_samples,
            require_vram_measurement=settings.require_vram_measurement,
        )

        self.loop = EngineLoop(
            settings=settings,
            store=self.store,
            snapshot=self.snapshot,
            workflow=self.workflow,
            evaluator=evaluator,
            publisher=self.publisher,
        )

    # -- preflight ----------------------------------------------------------

    def preflight(self) -> None:
        """Refuse to run a configuration whose results could not be defended.

        Raises:
            SystemExit: listing every problem, so an operator fixes them in one
                pass rather than restarting once per issue.
        """
        problems = self.settings.validate()

        try:
            require_pinned(self.snapshot.manifest)
        except Exception as exc:  # noqa: BLE001
            problems.append(str(exc))

        uncertified = [
            entry.adapter_id for entry in self.snapshot.registry.adapters if not entry.certified
        ]
        if uncertified:
            problems.append(
                f"{len(uncertified)} adapters have not completed certification: {uncertified}"
            )

        if self.publisher.keypair is None:
            # Unsigned reports and weight vectors are refused by any validator
            # enforcing an operator allow-list, so the engine would run, publish,
            # and quietly move no emission at all.
            problems.append(
                "no signing key is configured. Reports and weight vectors would be "
                "published unsigned, and validators would refuse them."
            )

        if not self.settings.recipe_path.is_dir():
            problems.append(
                f"the recipe store at {self.settings.recipe_path} does not exist. Admitted "
                "recipes could not be persisted, and no challenger could ever be evaluated."
            )

        if not problems:
            log.info("preflight passed")
            return

        if self.settings.dry_run:
            for problem in problems:
                log.warning("dry run, ignoring: %s", problem)
            return

        for problem in problems:
            log.error("preflight: %s", problem)
        raise SystemExit(
            f"refusing to start: {len(problems)} preflight problem(s). Fix them, or set "
            "dry_run to run against a development configuration."
        )

    # -- loop ---------------------------------------------------------------

    def block(self) -> int:
        if self.subtensor is None:
            # Dry runs advance a synthetic clock so window rotation can be
            # exercised without a chain.
            stored = int(self.store.get_meta("dry_run_block", "0") or 0)
            self.store.set_meta("dry_run_block", str(stored + 1))
            return stored + 1
        return current_block(self.subtensor)

    def admit(self, block: int) -> int:
        """Read the chain and admit anything new."""
        if self.subtensor is None:
            return 0

        metagraph = self.subtensor.metagraph(self.settings.netuid)
        commitments = read_commitments(
            self.subtensor,
            self.settings.netuid,
            metagraph=metagraph,
            min_block=self.settings.min_commit_block,
        )

        results = admit_new_commitments(
            commitments,
            snapshot=self.snapshot,
            store=self.store,
            registered_hotkeys=set(metagraph.hotkeys),
            current_block=block,
            recipe_store=self.recipe_store,
        )
        return sum(1 for result in results if result.admitted)

    def step(self) -> None:
        """One pass of the control loop."""
        block = self.block()

        admitted = self.admit(block)
        if admitted:
            log.info("admitted %d new submission(s)", admitted)

        self.loop.ensure_window(block)
        self.loop.evaluate_next_challenger(block)
        self.loop.publish_weights(block)

    def run(self) -> None:
        self.preflight()
        self._install_signal_handlers()

        log.info(
            "engine started: netuid %s, workflow %s, window %d blocks",
            self.settings.netuid,
            self.settings.workflow_id,
            self.settings.window_blocks,
        )

        consecutive_failures = 0
        while not self.should_exit:
            try:
                self.step()
                consecutive_failures = 0
            except KeyboardInterrupt:
                break
            except Exception:
                consecutive_failures += 1
                log.exception("control loop pass failed (%d in a row)", consecutive_failures)
                # Back off, but never give up: validators read the last published
                # weight vector, and a stopped engine eventually stops emission.
                time.sleep(min(300.0, self.settings.poll_seconds * consecutive_failures))
                continue

            time.sleep(self.settings.poll_seconds)

        log.info("engine stopped")
        self.store.close()

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, frame: FrameType | None) -> None:
            del frame
            log.info("received signal %d; finishing the current pass", signum)
            self.should_exit = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="capability-backend",
        description="Continuous champion-challenge evaluation engine.",
    )
    parser.add_argument("--config", default=None, help="Path to the settings YAML.")
    parser.add_argument("--once", action="store_true", help="Run one pass and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without a chain connection and downgrade preflight failures to warnings.",
    )
    parser.add_argument("--state-dir", default=None, help="Override the state directory.")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    if args.dry_run:
        settings.dry_run = True
    if args.state_dir:
        settings.state_dir = args.state_dir

    setup_logging(settings.log_level, log_file=f"{settings.state_dir}/engine.log")

    service = EngineService(settings)

    if args.once:
        service.preflight()
        service.step()
        service.store.close()
        return 0

    service.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
