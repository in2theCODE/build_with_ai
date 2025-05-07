"""
Avro Schema Registry Client for the Program Synthesis System.

This module provides a client for interacting with an Avro Schema Registry,
enabling schema registration, validation, and management. It integrates with
Pydantic v2 models to automatically generate Avro schemas.

The client implements caching mechanisms for performance and supports
authentication via bearer tokens for secure registry access.

Usage:
    client = SchemaRegistryClient("https://schema-registry-url")

    # Register a schema directly
    schema_id = client.register_schema("my-subject", avro_schema)

    # Register a Pydantic model
    schema_id = client.register_model_schema(MyPydanticModel)

    # Validate data against a schema
    is_valid = client.validate_event_against_schema(event_data, schema_id)
"""

import json
import logging
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel
import requests

logger = logging.getLogger(__name__)


class SchemaRegistryClient:
    """Client for interacting with an Avro Schema Registry."""

    def __init__(self, url: str, auth_token: Optional[str] = None):
        """Initialize the schema registry client."""
        self.url = url.rstrip("/")
        self.auth_token = auth_token
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

        # Cache of schema IDs by subject
        self.schema_id_cache: Dict[str, int] = {}
        # Cache of schemas by ID
        self.schema_cache: Dict[int, Dict[str, Any]] = {}

    def register_schema(self, subject: str, avro_schema: Dict[str, Any]) -> int:
        """Register a schema with the registry."""
        payload = json.dumps({"schema": json.dumps(avro_schema)})
        try:
            response = requests.post(
                f"{self.url}/subjects/{subject}/versions", headers=self.headers, data=payload
            )
            response.raise_for_status()
            schema_id = response.json()["id"]

            # Update caches
            self.schema_id_cache[subject] = schema_id
            self.schema_cache[schema_id] = avro_schema

            logger.info(f"Registered schema for subject {subject} with ID {schema_id}")
            return schema_id
        except Exception as e:
            logger.error(f"Failed to register schema for subject {subject}: {e}")
            raise

    def register_model_schema(
        self, model_class: Type[BaseModel], subject: Optional[str] = None
    ) -> int:
        """Register a model class with the schema registry."""
        return register_pydantic_model(model_class, self, subject)

    @property
    def get_model_schema_id(self) -> Optional[int]:
        """Get the schema ID for a model class."""
        return getattr(model_class, "__avro_schema_id__", None)

    @property
    def get_model_schema_subject(self, model_class=None) -> Optional[str]:
        """Get the schema subject for a model class."""
        return getattr(model_class, "__avro_schema_subject__", None)

    def validate_instance_against_schema(self, instance: BaseModel) -> bool:
        """Validate an instance against its schema in the registry."""
        schema_id = getattr(instance.__class__, "__avro_schema_id__", None)
        if schema_id is None:
            raise ValueError(f"No schema ID registered for {instance.__class__.__name__}")

        # Use Pydantic's native serialization
        event_data = instance.model_dump(exclude_unset=True, exclude_none=True)
        return self.validate_event_against_schema(event_data, schema_id)

    def get_schema_id(self, subject: str) -> Optional[int]:
        """Get the schema ID for a subject."""
        # Check cache first
        if subject in self.schema_id_cache:
            return self.schema_id_cache[subject]

        try:
            response = requests.get(
                f"{self.url}/subjects/{subject}/versions/latest", headers=self.headers
            )
            response.raise_for_status()
            schema_id = response.json()["id"]

            # Update cache
            self.schema_id_cache[subject] = schema_id

            return schema_id
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"No schema found for subject {subject}")
                return None
            else:
                logger.error(f"Failed to get schema ID for subject {subject}: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to get schema ID for subject {subject}: {e}")
            raise

    def get_schema(self, schema_id: int) -> Optional[Dict[str, Any]]:
        """Get a schema by ID."""
        # Check cache first
        if schema_id in self.schema_cache:
            return self.schema_cache[schema_id]

        try:
            response = requests.get(f"{self.url}/schemas/ids/{schema_id}", headers=self.headers)
            response.raise_for_status()
            schema = json.loads(response.json()["schema"])

            # Update cache
            self.schema_cache[schema_id] = schema

            return schema
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"No schema found for ID {schema_id}")
                return None
            else:
                logger.error(f"Failed to get schema for ID {schema_id}: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to get schema for ID {schema_id}: {e}")
            raise

    def validate_event_against_schema(self, event_data: Dict[str, Any], schema_id: int) -> bool:
        """Validate event data against a schema.

        Args:
            event_data: Event data to validate
            schema_id: Schema ID to validate against

        Returns:
            True if valid, False if not

        Raises:
            Exception: If validation fails due to server error
        """
        try:
            payload = json.dumps({"schema_id": schema_id, "payload": json.dumps(event_data)})
            response = requests.post(
                f"{self.url}/compatibility/subjects/{schema_id}/versions/latest",
                headers=self.headers,
                data=payload,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("is_compatible", False)
        except Exception as e:
            logger.error(f"Failed to validate event against schema {schema_id}: {e}")
            raise


def register_pydantic_model(
    model_class: Type[BaseModel],
    registry_client: SchemaRegistryClient,
    subject: Optional[str] = None,
    namespace: Optional[str] = None,
) -> int:
    """Register a Pydantic model with the schema registry."""
    # Generate Avro schema from Pydantic model using Pydantic v2's schema functionality
    schema_dict = model_class.model_json_schema()

    # Convert Pydantic JSON Schema to Avro schema format
    avro_schema = convert_pydantic_schema_to_avro(schema_dict, model_class.__name__, namespace)

    # If subject not provided, use model name
    if subject is None:
        subject = f"{model_class.__name__}-value"

    # Register schema
    schema_id = registry_client.register_schema(subject, avro_schema)

    # Set schema ID on model class
    setattr(model_class, "__avro_schema_id__", schema_id)
    setattr(model_class, "__avro_schema_subject__", subject)

    return schema_id


def convert_pydantic_schema_to_avro(
    pydantic_schema: Dict[str, Any], record_name: str, namespace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert a Pydantic JSON schema to Avro schema format.

    Args:
        pydantic_schema: Pydantic model schema
        record_name: Name for the Avro record
        namespace: Optional namespace for the Avro record

    Returns:
        Avro schema as a dictionary
    """
    avro_schema = {"type": "record", "name": record_name, "fields": []}

    if namespace:
        avro_schema["namespace"] = namespace

    # Extract required fields
    required_fields = pydantic_schema.get("required", [])
    properties = pydantic_schema.get("properties", {})

    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type")
        field_def = {"name": field_name}

        # Handle nullable fields
        if field_name not in required_fields:
            field_def["type"] = ["null", map_pydantic_type_to_avro(field_type, field_schema)]
            field_def["default"] = None
        else:
            field_def["type"] = map_pydantic_type_to_avro(field_type, field_schema)

        avro_schema["fields"].append(field_def)

    return avro_schema


def map_pydantic_type_to_avro(field_type: str, field_schema: Dict[str, Any]) -> Any:
    """
    Map Pydantic field types to Avro field types.

    Args:
        field_type: Pydantic field type
        field_schema: Complete field schema

    Returns:
        Avro type definition
    """
    type_mapping = {
        "string": "string",
        "integer": "int",
        "number": "double",
        "boolean": "boolean",
        "null": "null",
    }

    if field_type in type_mapping:
        return type_mapping[field_type]

    # Handle array types
    if field_type == "array":
        items = field_schema.get("items", {})
        item_type = items.get("type")
        return {"type": "array", "items": map_pydantic_type_to_avro(item_type, items)}

    # Handle object types
    if field_type == "object":
        properties = field_schema.get("properties", {})
        required = field_schema.get("required", [])

        fields = []
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type")
            field = {"name": prop_name}

            if prop_name not in required:
                field["type"] = ["null", map_pydantic_type_to_avro(prop_type, prop_schema)]
                field["default"] = None
            else:
                field["type"] = map_pydantic_type_to_avro(prop_type, prop_schema)

            fields.append(field)

        return {
            "type": "record",
            "name": field_schema.get("title", "NestedRecord"),
            "fields": fields,
        }

    # Handle enum types
    if field_type == "string" and "enum" in field_schema:
        return {
            "type": "enum",
            "name": field_schema.get("title", "Enum"),
            "symbols": field_schema["enum"],
        }

    # Default to string for unknown types
    return "string"
