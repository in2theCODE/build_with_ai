#!/usr/bin/env python3
# main.py
import asyncio
import os
import logging
from src.services.spec_registry.app.spec_registry_event_adapter import SpecRegistryEventAdapter

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the Spec Registry service."""
    # Get configuration from environment
    pulsar_service_url = os.getenv("PULSAR_SERVICE_URL", "pulsar://localhost:6650")
    base_dir = os.getenv("BASE_DIR", "./storage")
    enable_events = os.getenv("ENABLE_EVENTS", "true").lower() == "true"

    # Read secret key from file if available
    secret_key = None
    secret_key_path = "/run/secrets/pulsar_secret_key"
    if os.path.exists(secret_key_path):
        with open(secret_key_path, "r") as f:
            secret_key = f.read().strip()

    # Initialize adapter
    adapter = SpecRegistryEventAdapter(
        pulsar_service_url=pulsar_service_url,
        base_dir=base_dir,
        enable_events=enable_events,
        secret_key=secret_key,
    )

    # Start adapter
    started = await adapter.start()
    if not started:
        logger.error("Failed to start Spec Registry service")
        return

    logger.info("Spec Registry service started")

    # Keep service running
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down Spec Registry service")
        await adapter.stop()


if __name__ == "__main__":
    asyncio.run(main())
