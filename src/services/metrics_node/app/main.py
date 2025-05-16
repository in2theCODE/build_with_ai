"""
Main entry point for the Metrics Node service.

This module initializes and starts the Metrics Node
service, handling setup, shutdown, and signal handling.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional

from .config import load_config
from .metrics_service import MetricsService
from .metrics_collector import MetricsCollector
from .metrics_analyzer import MetricsAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
metrics_service: Optional[MetricsService] = None
shutdown_event = asyncio.Event()


async def setup_service() -> Dict[str, Any]:
    """
    Set up the service components.

    Returns:
        Dictionary of service components
    """
    global metrics_service

    # Load configuration
    config = load_config()

    # Set log level from configuration
    log_level = getattr(logging, config.get("log_level", "INFO"))
    logging.getLogger().setLevel(log_level)

    # Initialize metrics collector
    metrics_collector = MetricsCollector(config)
    await metrics_collector.initialize()

    # Initialize metrics analyzer
    metrics_analyzer = MetricsAnalyzer(config)
    await metrics_analyzer.initialize()

    # Initialize metrics service
    metrics_service = MetricsService(config)
    metrics_service.metrics_collector = metrics_collector
    metrics_service.metrics_analyzer = metrics_analyzer

    # Start metrics service
    await metrics_service.start()

    logger.info("Metrics Node service started")

    return {
        "config": config,
        "metrics_service": metrics_service,
    }


async def shutdown_service(components: Dict[str, Any]):
    """
    Shut down the service components.

    Args:
        components: Dictionary of service components
    """
    logger.info("Shutting down Metrics Node service")

    if "metrics_service" in components and components["metrics_service"]:
        await components["metrics_service"].stop()

    logger.info("Metrics Node service stopped")


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
        logger.error(f"Error in Metrics Node service: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
