#!/usr/bin/env python3
"""Register Avro schemas with Pulsar."""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the schema registry client
from src.services.shared.models.schema_registry import SchemaRegistryClient

# Import the event models you want to register
from src.services.shared.models.event_avro import EventAvro


def main():
    """Register all schemas with Pulsar."""
    # Create the schema registry client
    client = SchemaRegistryClient("http://localhost:8080")

    # Register EventAvro schema
    try:
        schema_id = client.register_avro_base_model(EventAvro)
        logger.info(f"Registered EventAvro schema with ID: {schema_id}")
    except Exception as e:
        logger.error(f"Failed to register EventAvro schema: {e}")

    # Register other schemas here
    # ...


if __name__ == "__main__":
    main()
