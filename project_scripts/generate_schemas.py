# app/services/shared/tools/generate_schemas.py

import json
import os
from typing import List, Type

from src.services.shared.models.schema_registry import (
    SchemaRegistryClient,
    register_all_models,
)
from src.services.shared.models.event_avro import EventAvro


# Import all other event app


def generate_schema_files(models: List[Type], output_dir="./schemas"):
    """Generate Avro schema files from model classes"""
    os.makedirs(output_dir, exist_ok=True)

    for model in models:
        # Get model name
        model_name = model.__name__

        # Generate schema
        schema = model.avro_schema()

        # Save to file
        schema_file = os.path.join(output_dir, f"{model_name}.avsc")
        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=2)

        print(f"Generated schema for {model_name} -> {schema_file}")


def register_schemas_with_pulsar(models: List[Type], pulsar_admin_url: str):
    """Register schemas with Pulsar schema registry"""
    client = SchemaRegistryClient(pulsar_admin_url)
    results = register_all_models(client, models)

    for subject, schema_id in results.items():
        print(f"Registered {subject} with schema ID {schema_id}")


if __name__ == "__main__":
    # List all app that need schemas
    models = [
        EventAvro,
        # Add other app here
    ]

    # Generate schema files
    generate_schema_files(models)

    # Register with Pulsar (assuming standalone instance)
    register_schemas_with_pulsar(models, "http://localhost:8080")
