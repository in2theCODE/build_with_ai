"""
Main entry point for the Synapse Manager service.

This module initializes and starts the Synapse Manager
service, handling setup, shutdown, and signal handling.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional

from .config import load_config
from .synapse_service import SynapseService
from .health_monitor import SynapseHealthMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
synapse_service: Optional[SynapseService] = None
health_monitor: Optional[SynapseHealthMonitor] = None
shutdown_event = asyncio.Event()


async def setup_service() -> Dict[str, Any]:
    """
    Set up the service components.

    Returns:
        Dictionary of service components
    """
    global synapse_service, health_monitor

    # Load configuration
    config = load_config()

    # Set log level from configuration
    log_level = getattr(logging, config.get("log_level", "INFO"))
    logging.getLogger().setLevel(log_level)

    # Initialize synapse service
    synapse_service = SynapseService(config)

    # Initialize health monitor
    health_monitor = SynapseHealthMonitor(config, synapse_service)

    # Start health monitor first
    await health_monitor.start()

    # Start synapse service (which initializes and starts all sub-components)
    await synapse_service.start()

    logger.info("Synapse Manager service started")

    return {
        "config": config,
        "synapse_service": synapse_service,
        "health_monitor": health_monitor,
    }


async def shutdown_service(components: Dict[str, Any]):
    """
    Shut down the service components.

    Args:
        components: Dictionary of service components
    """
    logger.info("Shutting down Synapse Manager service")

    # Stop services in reverse order of initialization
    if "synapse_service" in components and components["synapse_service"]:
        await components["synapse_service"].stop()

    if "health_monitor" in components and components["health_monitor"]:
        await components["health_monitor"].stop()

    logger.info("Synapse Manager service stopped")


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
        logger.error(f"Error in Synapse Manager service: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
