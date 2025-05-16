"""
Main entry point for the Context Orchestrator service.

This module initializes and starts the Context Orchestrator
service, handling setup, shutdown, and signal handling.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional

from .config import load_config
from .orchestrator import NeuralContextOrchestrator
from .health_monitor import HealthMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
orchestrator: Optional[NeuralContextOrchestrator] = None
health_monitor: Optional[HealthMonitor] = None
shutdown_event = asyncio.Event()


async def setup_service() -> Dict[str, Any]:
    """
    Set up the service components.

    Returns:
        Dictionary of service components
    """
    global orchestrator, health_monitor

    # Load configuration
    config = load_config()

    # Set log level from configuration
    log_level = getattr(logging, config.get("log_level", "INFO"))
    logging.getLogger().setLevel(log_level)

    # Initialize health monitor
    health_monitor = HealthMonitor(config)
    await health_monitor.start()

    # Initialize orchestrator
    orchestrator = NeuralContextOrchestrator(config)

    # Register components with health monitor
    health_monitor.register_component("orchestrator", lambda: orchestrator is not None, is_critical=True)

    # Start orchestrator
    await orchestrator.start()

    logger.info("Context Orchestrator service started")

    return {
        "config": config,
        "orchestrator": orchestrator,
        "health_monitor": health_monitor,
    }


async def shutdown_service(components: Dict[str, Any]):
    """
    Shut down the service components.

    Args:
        components: Dictionary of service components
    """
    logger.info("Shutting down Context Orchestrator service")

    if "orchestrator" in components and components["orchestrator"]:
        await components["orchestrator"].stop()

    if "health_monitor" in components and components["health_monitor"]:
        await components["health_monitor"].stop()

    logger.info("Context Orchestrator service stopped")


def handle_signal(sig, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {sig}, initiating shutdown")
    shutdown_event.set()


async def main():
    """Main entry point for the service."""
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        # Set up service
        components = await setup_service()

        # Wait for shutdown event
        await shutdown_event.wait()

        # Shutdown service
        await shutdown_service(components)

    except Exception as e:
        logger.error(f"Error in Context Orchestrator service: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
