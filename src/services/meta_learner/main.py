#!/usr/bin/env python3
"""
Main entry point for Meta Learner Service.

This service analyzes synthesis results to improve future code generation
by learning from patterns in successes and failures.
"""

import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

# Setup logging before anything else
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("meta_learner")

# Check for required ML dependencies
try:
    import numpy as np
    from sklearn.cluster import KMeans, DBSCAN

    HAVE_ML_DEPS = True
except ImportError:
    HAVE_ML_DEPS = False
    logger.warning("ML dependencies not available. Install numpy and scikit-learn for full functionality.")

# Import from shared modules
from src.services.shared.models.base import BaseEvent
from src.services.shared.models.enums import EventType, SynthesisStrategy
from src.services.shared.models.types import VerificationReport
from src.services.shared.models.synthesis import SynthesisResult
from src.services.shared.loggerService.loggingService import get_logger
from src.services.shared.pulsar.event_emitter import SecureEventEmitter
from src.services.shared.pulsar.event_listener import SecureEventListener

# Import service-specific modules
from src.services.meta_learner.meta_learner import MetaLearner


class MetaLearnerService:
    """Service wrapper for the Meta Learner component."""

    def __init__(self):
        """Initialize the Meta Learner service."""
        self.logger = get_logger("MetaLearnerService")
        self.running = False
        self.shutdown_event = threading.Event()

        # Get configuration from environment variables
        self.pulsar_service_url = os.getenv("PULSAR_SERVICE_URL", "pulsar://localhost:6650")
        self.storage_path = os.getenv("STORAGE_PATH", "meta_learning_data")
        self.enable_clustering = os.getenv("ENABLE_CLUSTERING", "true").lower() == "true"

        # Load secret key if available
        self.secret_key = self._load_secret("pulsar_secret_key")

        # Initialize components
        self.event_emitter = None
        self.event_listener = None
        self.meta_learner = None

        # Map of synthesis strategies for the meta_learner
        self.strategy_pool = {
            strategy.value: {
                "name": strategy.value,
                "description": f"Strategy using {strategy.value} approach",
                "success_rate": 0.0,
                "usage_count": 0,
            }
            for strategy in SynthesisStrategy
        }

    def _load_secret(self, secret_name: str) -> Optional[str]:
        """Load a secret from file if it exists."""
        secret_path = f"/run/secrets/{secret_name}"
        if os.path.exists(secret_path):
            with open(secret_path, "r") as f:
                return f.read().strip()
        return None

    def initialize(self):
        """Initialize all components of the service."""
        # Initialize event emitter (to send events to aggregator)
        self.event_emitter = SecureEventEmitter(
            service_url=self.pulsar_service_url,
            secret_key=self.secret_key,
            tenant="public",
            namespace="code-generator",
        )

        # Initialize event listener (to receive feedback & results)
        self.event_listener = SecureEventListener(
            service_url=self.pulsar_service_url,
            subscription_name="meta-learner",
            event_types=[
                EventType.CODE_GENERATION_COMPLETED,
                EventType.CODE_GENERATION_FAILED,
                EventType.SPEC_SHEET_CREATED,
                EventType.SPEC_SHEET_UPDATED,
                EventType.VERIFICATION_COMPLETED,
            ],
            secret_key=self.secret_key,
            tenant="public",
            namespace="code-generator",
        )

        # Initialize meta_learner
        self.meta_learner = MetaLearner(
            strategy_pool=self.strategy_pool,
            learning_rate=float(os.getenv("LEARNING_RATE", "0.1")),
            exploration_rate=float(os.getenv("EXPLORATION_RATE", "0.2")),
            storage_path=self.storage_path,
            perform_clustering=self.enable_clustering and HAVE_ML_DEPS,
            collect_performance_metrics=True,
        )

        # Register event handlers
        self.event_listener.register_handler(
            EventType.CODE_GENERATION_COMPLETED.value, self._handle_generation_completed
        )

        self.event_listener.register_handler(EventType.CODE_GENERATION_FAILED.value, self._handle_generation_failed)

        self.event_listener.register_handler(
            EventType.VERIFICATION_COMPLETED.value, self._handle_verification_completed
        )

        self.logger.info("Meta Learner service initialized")

    async def start(self):
        """Start the Meta Learner service."""
        if self.running:
            return

        self.logger.info("Starting Meta Learner service")
        self.running = True

        # Start the event listener
        await self.event_listener.start()

        # Main service loop
        while self.running and not self.shutdown_event.is_set():
            try:
                # Periodically perform pattern analysis
                if self.enable_clustering and HAVE_ML_DEPS:
                    # Analyze patterns in the collected data
                    # This is a lightweight operation done periodically
                    stats = self.meta_learner.analyze_strategy_patterns()
                    if stats["has_data"]:
                        self.logger.info(
                            f"Strategy pattern analysis completed: "
                            f"{len(stats['strategy_versatility'])} strategies analyzed"
                        )

                # Sleep to avoid busy wait
                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in main service loop: {e}")
                time.sleep(5)  # Shorter sleep on error

        self.logger.info("Meta Learner service main loop exited")

    async def stop(self):
        """Stop the Meta Learner service."""
        if not self.running:
            return

        self.logger.info("Stopping Meta Learner service")
        self.running = False
        self.shutdown_event.set()

        # Stop event listener
        if self.event_listener:
            await self.event_listener.stop()

        # Close emitter
        if self.event_emitter:
            self.event_emitter.close()

        self.logger.info("Meta Learner service stopped")

    async def _handle_generation_completed(self, event):
        """Handle successful code generation events."""
        try:
            self.logger.debug(f"Handling successful generation: {event.event_id}")

            # Extract data from event
            payload = event.payload
            spec_sheet_id = event.metadata.get("spec_sheet_id")

            generated_code = payload.get("generated_code", "")
            strategy_used = payload.get("strategy_used", "")
            confidence_score = payload.get("confidence_score", 0.0)
            time_taken = payload.get("time_taken", 0.0)

            # Extract specification from metadata if available
            specification = event.metadata.get("specification", "")

            # Create context for the meta_learner
            context = {
                "spec_sheet_id": spec_sheet_id,
                "correlation_id": event.correlation_id,
                "source_container": event.source_container,
            }

            # Create a synthesis result object
            synthesis_result = SynthesisResult(
                generated_code=generated_code,
                strategy=strategy_used,
                confidence_score=confidence_score,
                time_taken=time_taken,
            )

            # Record the success in the meta_learner
            self.meta_learner.record_success(specification, context, synthesis_result)

            # Emit event to the aggregator
            await self.event_emitter.emit_async(
                event=BaseEvent(
                    event_type=EventType.LEARNING_DATA_RECORDED,
                    source_container="meta_learner",
                    payload={
                        "spec_sheet_id": spec_sheet_id,
                        "event_type": "success",
                        "strategy": strategy_used,
                        "confidence": confidence_score,
                        "time_taken": time_taken,
                    },
                    correlation_id=event.correlation_id,
                )
            )

        except Exception as e:
            self.logger.error(f"Error handling generation completed event: {e}")

    async def _handle_generation_failed(self, event):
        """Handle failed code generation events."""
        try:
            self.logger.debug(f"Handling failed generation: {event.event_id}")

            # Extract data from event
            payload = event.payload
            spec_sheet_id = event.metadata.get("spec_sheet_id")

            error_message = payload.get("error_message", "")
            error_type = payload.get("error_type", "")
            strategy_used = event.metadata.get("strategy_used", "")
            time_taken = event.metadata.get("time_taken", 0.0)

            # Extract specification from metadata if available
            specification = event.metadata.get("specification", "")

            # Create context for the meta_learner
            context = {
                "spec_sheet_id": spec_sheet_id,
                "correlation_id": event.correlation_id,
                "source_container": event.source_container,
            }

            # Create a basic synthesis result object
            synthesis_result = SynthesisResult(
                generated_code="",
                strategy=strategy_used,
                confidence_score=0.0,
                time_taken=time_taken,
                error=error_message,
            )

            # Create a verification report with the error
            verification_report = VerificationReport(
                status="ERROR",
                reason=error_message,
                error_type=error_type,
                counterexamples=[],
                time_taken=0.0,
            )

            # Record the failure in the meta_learner
            self.meta_learner.record_failure(specification, context, synthesis_result, verification_report)

            # Emit event to the aggregator
            await self.event_emitter.emit_async(
                event=BaseEvent(
                    event_type=EventType.LEARNING_DATA_RECORDED,
                    source_container="meta_learner",
                    payload={
                        "spec_sheet_id": spec_sheet_id,
                        "event_type": "failure",
                        "strategy": strategy_used,
                        "error_type": error_type,
                        "time_taken": time_taken,
                    },
                    correlation_id=event.correlation_id,
                )
            )

        except Exception as e:
            self.logger.error(f"Error handling generation failed event: {e}")

    async def _handle_verification_completed(self, event):
        """Handle verification completed events."""
        try:
            self.logger.debug(f"Handling verification completed: {event.event_id}")

            # Extract data from event
            payload = event.payload
            spec_sheet_id = event.metadata.get("spec_sheet_id")

            verification_status = payload.get("status", "")
            reason = payload.get("reason", "")
            counterexamples = payload.get("counterexamples", [])
            time_taken = payload.get("time_taken", 0.0)

            # Extract synthesis info from metadata
            strategy_used = event.metadata.get("strategy_used", "")
            generated_code = event.metadata.get("generated_code", "")
            confidence_score = event.metadata.get("confidence_score", 0.0)
            generation_time = event.metadata.get("generation_time", 0.0)
            specification = event.metadata.get("specification", "")

            # Create context for the meta_learner
            context = {
                "spec_sheet_id": spec_sheet_id,
                "correlation_id": event.correlation_id,
                "source_container": event.source_container,
            }

            # Create synthesis result and verification report
            synthesis_result = SynthesisResult(
                generated_code=generated_code,
                strategy=strategy_used,
                confidence_score=confidence_score,
                time_taken=generation_time,
            )

            verification_report = VerificationReport(
                status=verification_status,
                reason=reason,
                counterexamples=counterexamples,
                time_taken=time_taken,
            )

            # Record success or failure based on verification status
            if verification_status == "VERIFIED":
                self.meta_learner.record_success(specification, context, synthesis_result)
            else:
                self.meta_learner.record_failure(specification, context, synthesis_result, verification_report)

            # Emit event to the aggregator
            await self.event_emitter.emit_async(
                event=BaseEvent(
                    event_type=EventType.LEARNING_DATA_RECORDED,
                    source_container="meta_learner",
                    payload={
                        "spec_sheet_id": spec_sheet_id,
                        "event_type": "verification",
                        "strategy": strategy_used,
                        "verification_status": verification_status,
                        "time_taken": time_taken + generation_time,
                    },
                    correlation_id=event.correlation_id,
                )
            )

        except Exception as e:
            self.logger.error(f"Error handling verification completed event: {e}")


def handle_signal(signum, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    if service:
        # Create event loop if it doesn't exist
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Stop the service
        loop.run_until_complete(service.stop())

    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Initialize service
    service = MetaLearnerService()
    service.initialize()

    # Run service
    import asyncio

    loop = asyncio.get_event_loop()

    try:
        logger.info("Starting Meta Learner service...")
        loop.run_until_complete(service.start())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        loop.run_until_complete(service.stop())
    finally:
        loop.close()
