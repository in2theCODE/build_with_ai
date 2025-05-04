"""Schema registration initialization for Apache Pulsar event models."""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

from infra.registration.schema_registry import SchemaRegistryClient

# Map of event types to their corresponding Avro schema files
EVENT_SCHEMA_MAPPING = {
    "Event": "Event.avsc",
    "ErrorPayload": "ErrorPayload.avsc",
    "Project": "Project.avsc",
    "SpecSheet": "SpecSheet.avsc",
    "Component": "Component.avsc",
    "Template": "Template.avsc",
    "Workflow": "Workflow.avsc",
    "AgentStatus": "AgentStatus.avsc",
    "AuthToken": "AuthToken.avsc"
}

# Registry client instance
_schema_registry = None


def load_avro_schema(schema_name: str) -> Dict[str, Any]:
    """Load an Avro schema from file."""
    # Determine schema directory - adjust path as needed for your project
    schema_dir = os.environ.get("AVRO_SCHEMA_DIR",
                                os.path.join(os.path.dirname(__file__), "../../shared/avro"))

    schema_path = os.path.join(schema_dir, EVENT_SCHEMA_MAPPING.get(schema_name, f"{schema_name}.avsc"))

    try:
        with open(schema_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Schema file not found: {schema_path}")


def init_schema_registry(url: str, auth_token: str = None):
    """Initialize the schema registry client."""
    global _schema_registry
    _schema_registry = SchemaRegistryClient(url=url, auth_token=auth_token)
    return register_event_schemas()


def register_event_schemas():
    """Register all event schemas with the schema registry."""
    if _schema_registry is None:
        raise RuntimeError("Schema registry client not initialized")

    registered_schemas = {}

    # Register all Avro schemas
    for schema_name in EVENT_SCHEMA_MAPPING.keys():
        try:
            # Load schema from file
            schema = load_avro_schema(schema_name)

            # Get the subject name from the schema name and namespace
            namespace = schema.get("namespace", "com.programsynthesis.schema")
            name = schema.get("name", schema_name)
            subject = f"{namespace}.{name}"

            # Register schema
            schema_id = _schema_registry.register_schema(subject, schema)
            registered_schemas[schema_name] = schema_id

        except Exception as e:
            print(f"Failed to register schema {schema_name}: {e}")

    return registered_schemas


def get_schema_registry():
    """Get the schema registry client instance."""
    if _schema_registry is None:
        raise RuntimeError("Schema registry not initialized")
    return _schema_registry