# schema_registry.py
import requests
import json
from typing import Dict, Any, Optional, Type
import logging
from pydantic import BaseModel

from infra.registration.event_converter import EventConverter

logger = logging.getLogger(__name__)


class SchemaRegistryClient:
    """Client for interacting with an Avro Schema Registry."""

    def __init__(self, url: str, auth_token: Optional[str] = None):
        """Initialize the schema registry client."""
        self.url = url.rstrip('/')
        self.auth_token = auth_token
        self.headers = {
            'Content-Type': 'application/json'
        }
        if auth_token:
            self.headers['Authorization'] = f'Bearer {auth_token}'

        # Cache of schema IDs by subject
        self.schema_id_cache: Dict[str, int] = {}
        # Cache of schemas by ID
        self.schema_cache: Dict[int, Dict[str, Any]] = {}



    def register_schema(self, subject: str, avro_schema: Dict[str, Any]) -> int:
        """Register a schema with the registry."""
        payload = json.dumps({
            'schema': json.dumps(avro_schema)
        })
        try:
            response = requests.post(
                f'{self.url}/subjects/{subject}/versions',
                headers=self.headers,
                data=payload
            )
            response.raise_for_status()
            schema_id = response.json()['id']

            # Update caches
            self.schema_id_cache[subject] = schema_id
            self.schema_cache[schema_id] = avro_schema

            logger.info(f"Registered schema for subject {subject} with ID {schema_id}")
            return schema_id
        except Exception as e:
            logger.error(f"Failed to register schema for subject {subject}: {e}")
            raise

    def register_model_schema(self, model_class: Type[BaseModel], subject: Optional[str] = None) -> int:
        """Register a model class with the schema registry."""

        return register_pydantic_model(model_class, self, subject)

    def get_model_schema_id(self, model_class: Type[BaseModel]) -> Optional[int]:
        """Get the schema ID for a model class."""
        return getattr(model_class, "__avro_schema_id__", None)

    def get_model_schema_subject(self, model_class: Type[BaseModel]) -> Optional[str]:
        """Get the schema subject for a model class."""
        return getattr(model_class, "__avro_schema_subject__", None)

    def validate_instance_against_schema(self, instance: BaseModel) -> bool:
        """Validate an instance against its schema in the registry."""
        schema_id = getattr(instance.__class__, "__avro_schema_id__", None)
        if schema_id is None:
            raise ValueError(f"No schema ID registered for {instance.__class__.__name__}")

        # Assuming instance has a to_dict method or similar
        event_data = instance.model_dump(exclude_unset=True, exclude_none=True)
        return self.validate_event_against_schema(event_data, schema_id)

    def get_schema_id(self, subject: str) -> Optional[int]:
        """Get the schema ID for a subject."""
        # Check cache first
        if subject in self.schema_id_cache:
            return self.schema_id_cache[subject]

        try:
            response = requests.get(
                f'{self.url}/subjects/{subject}/versions/latest',
                headers=self.headers
            )
            response.raise_for_status()
            schema_id = response.json()['id']

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
            response = requests.get(
                f'{self.url}/schemas/ids/{schema_id}',
                headers=self.headers
            )
            response.raise_for_status()
            schema = json.loads(response.json()['schema'])

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
            payload = json.dumps({
                'schema_id': schema_id,
                'payload': json.dumps(event_data)
            })
            response = requests.post(
                f'{self.url}/compatibility/subjects/{schema_id}/versions/latest',
                headers=self.headers,
                data=payload
            )
            response.raise_for_status()
            result = response.json()
            return result.get('is_compatible', False)
        except Exception as e:
            logger.error(f"Failed to validate event against schema {schema_id}: {e}")
            raise


def register_pydantic_model(
        model_class: Type[BaseModel],
        registry_client: SchemaRegistryClient,
        subject: Optional[str] = None,
        namespace: Optional[str] = None
) -> int:
    """Register a Pydantic model with the schema registry."""
    # Generate schema
    schema = EventConverter.generate_avro_schema(model_class, namespace)

    # If subject not provided, use model name
    if subject is None:
        subject = f"{model_class.__name__}-value"

    # Register schema
    schema_id = registry_client.register_schema(subject, schema)

    # Set schema ID on model class
    setattr(model_class, "__avro_schema_id__", schema_id)
    setattr(model_class, "__avro_schema_subject__", subject)

    return schema_id