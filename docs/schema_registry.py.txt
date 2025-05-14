"""
Apache Pulsar Schema Registry integration.

This module provides tools for registering Avro schemas with the Apache Pulsar
Schema Registry. It supports schema evolution, compatibility checking, and
versioning to ensure reliable communication between system components.

Classes:
    SchemaRegistryClient: Client for interacting with the Apache Pulsar Schema Registry

Functions:
    register_model_schema: Register a single model's schema with the registry
    register_all_models: Register multiple model schemas with the registry
"""

# src/services/shared/models/schema_registry.py
import json
from typing import Any, Dict, List, Tuple, Type

import requests

from pydantic_avro.base import AvroBase

from src.services.shared.models.base import AvroBaseModel


class SchemaRegistryClient:
    """Client for the Apache Pulsar Schema Registry."""

    def __init__(self, url: str, auth_token: str = None):
        """Initialize the schema registry client.

        Args:
            url: The base URL of the schema registry
            auth_token: Optional authentication token
        """
        self.url = url.rstrip("/")
        self.headers = {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

    def register_schema(self, subject: str, schema: Dict[str, Any]) -> int:
        """Register a schema with the registry.

        Args:
            subject: The subject name
            schema: The Avro schema as a dictionary

        Returns:
            The schema ID
        """
        url = f"{self.url}/subjects/{subject}/versions"
        response = requests.post(url, headers=self.headers, json={"schema": json.dumps(schema)})
        response.raise_for_status()
        return response.json()["id"]

    def get_schema(self, schema_id: int) -> Dict[str, Any]:
        """Get a schema by ID.

        Args:
            schema_id: The schema ID

        Returns:
            The Avro schema as a dictionary
        """
        url = f"{self.url}/schemas/ids/{schema_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return json.loads(response.json()["schema"])

    def get_latest_schema(self, subject: str) -> Tuple[int, Dict[str, Any]]:
        """Get the latest schema for a subject.

        Args:
            subject: The subject name

        Returns:
            A tuple of (schema_id, schema)
        """
        url = f"{self.url}/subjects/{subject}/versions/latest"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        return data["id"], json.loads(data["schema"])

    def check_schema_compatibility(self, subject: str, schema: Dict[str, Any]) -> bool:
        """Check if a schema is compatible with the latest registered schema.

        Args:
            subject: The subject name
            schema: The Avro schema as a dictionary

        Returns:
            True if compatible, False otherwise
        """
        url = f"{self.url}/compatibility/subjects/{subject}/versions/latest"
        response = requests.post(url, headers=self.headers, json={"schema": json.dumps(schema)})
        response.raise_for_status()
        return response.json()["is_compatible"]


def register_model_schema(
    registry_client: SchemaRegistryClient, model_class: Type[AvroBaseModel]
) -> int:
    """Register a model's schema with the registry.

    Args:
        registry_client: The schema registry client
        model_class: The model class

    Returns:
        The schema ID
    """
    subject = model_class.get_schema_subject()
    schema = model_class.avro_schema()
    schema_id = registry_client.register_schema(subject, schema)

    # Update the model class with the schema ID
    model_class.__avro_schema_id__ = schema_id

    return schema_id


def register_all_models(
    registry_client: SchemaRegistryClient, model_classes: List[Type[AvroBaseModel]]
) -> Dict[str, int]:
    """Register all model schemas with the registry.

    Args:
        registry_client: The schema registry client
        model_classes: A list of model classes

    Returns:
        A dictionary mapping subject names to schema IDs
    """
    results = {}

    for model_class in model_classes:
        subject = model_class.get_schema_subject()
        schema_id = register_model_schema(registry_client, model_class)
        results[subject] = schema_id

    return results
