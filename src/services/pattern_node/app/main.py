"""
Main entry point for the Pattern Node service.

This module initializes and starts the Pattern Node
service, handling setup, shutdown, and signal handling.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional

from .config import load_config
from .pattern_service import PatternNodeService
from .health_monitor import HealthMonitor
from .event_handlers import PatternNodeEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
pattern_service: Optional[PatternNodeService] = None
health_monitor: Optional[HealthMonitor] = None
event_handler: Optional[PatternNodeEventHandler] = None
shutdown_event = asyncio.Event()


async def setup_service() -> Dict[str, Any]:
    """
    Set up the service components.

    Returns:
        Dictionary of service components
    """
    global pattern_service, health_monitor, event_handler

    # Load configuration
    config = load_config()

    # Set log level from configuration
    log_level = getattr(logging, config.get("log_level", "INFO"))
    logging.getLogger().setLevel(log_level)

    # Initialize health monitor
    health_monitor = HealthMonitor(config)
    await health_monitor.start()

    # Initialize pattern service
    pattern_service = PatternNodeService(config)

    # Initialize event handler
    event_handler = PatternNodeEventHandler(pattern_service)

    # Register components with health monitor
    health_monitor.register_component("pattern_service", lambda: pattern_service is not None, is_critical=True)

    health_monitor.register_component(
        "event_handler",
        lambda: event_handler is not None and event_handler.running,
        recovery_func=lambda: asyncio.create_task(event_handler.start()),
        is_critical=True,
    )

    # Start pattern service
    await pattern_service.start()

    # Start event handler
    await event_handler.start()

    logger.info("Pattern Node service started")

    return {
        "config": config,
        "pattern_service": pattern_service,
        "health_monitor": health_monitor,
        "event_handler": event_handler,
    }


async def shutdown_service(components: Dict[str, Any]):
    """
    Shut down the service components.

    Args:
        components: Dictionary of service components
    """
    logger.info("Shutting down Pattern Node service")

    if "event_handler" in components and components["event_handler"]:
        await components["event_handler"].stop()

    if "pattern_service" in components and components["pattern_service"]:
        await components["pattern_service"].stop()

    if "health_monitor" in components and components["health_monitor"]:
        await components["health_monitor"].stop()

    logger.info("Pattern Node service stopped")


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
        logger.error(f"Error in Pattern Node service: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
