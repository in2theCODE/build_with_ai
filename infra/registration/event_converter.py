# event_converter.py
from datetime import datetime, timezone
from typing import Dict, Any, Union, List, Optional, Type, get_args, get_origin, Literal
from enum import Enum

from pydantic import BaseModel

from src.services.shared.models.events import BaseEvent
from src.services.shared.models.events import (
    EventType, EventPriority,
    CodeGenerationRequestedEvent, CodeGenerationCompletedEvent,
    KnowledgeQueryRequestedEvent, KnowledgeQueryCompletedEvent
)
from src.services.shared.models.event_avro import EventAvro


class EventConverter:
    """Utility to convert between internal event models and Avro-compatible models."""

    @staticmethod
    def _format_datetime(dt: datetime) -> str:
        """Format datetime to Avro-compatible ISO 8601 UTC string with Z suffix."""
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_datetime(timestamp_str: str) -> datetime:
        """Parse ISO 8601 datetime string with Z suffix to UTC datetime object."""
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(
                f"Invalid timestamp format: {timestamp_str}. Expected ISO 8601 format with UTC 'Z' suffix."
            )

    @staticmethod
    def to_avro(event: BaseEvent) -> EventAvro:
        """Convert application event model to Avro-compatible model."""
        # Validate required fields
        EventConverter._validate_required_fields(event)

        return EventAvro(
            event_id=event.event_id,
            event_type=event.event_type.value,
            source_container=event.source_container,
            payload=EventConverter._ensure_avro_compatible(event.payload),
            timestamp=EventConverter._format_datetime(event.timestamp)
            if isinstance(event.timestamp, datetime) else str(event.timestamp),
            priority=event.priority.value,
            correlation_id=event.correlation_id,
            metadata=EventConverter._ensure_avro_compatible(event.metadata),
            version=event.version
        )

    @staticmethod
    def from_avro(avro_event: EventAvro) -> BaseEvent:
        """Convert Avro model back to internal BaseEvent."""
        # Validate required fields
        EventConverter._validate_required_fields(avro_event)

        timestamp = avro_event.timestamp
        if isinstance(timestamp, str):
            timestamp = EventConverter._parse_datetime(timestamp)

        try:
            event_type = EventType(avro_event.event_type)
            priority = EventPriority(avro_event.priority)
        except ValueError as e:
            raise ValueError(f"Invalid enum value in Avro event: {e}")

        return BaseEvent(
            event_id=avro_event.event_id,
            event_type=event_type,
            source_container=avro_event.source_container,
            payload=avro_event.payload,
            timestamp=timestamp,
            priority=priority,
            correlation_id=avro_event.correlation_id,
            metadata=avro_event.metadata,
            version=avro_event.version
        )

    @staticmethod
    def _validate_required_fields(event: Union[BaseEvent, EventAvro]) -> None:
        """Validate that all required fields are present and valid."""
        required_fields = ["event_id", "event_type", "source_container", "timestamp"]
        for field in required_fields:
            if getattr(event, field, None) is None:
                raise ValueError(f"Required field {field} is missing or None")

    @staticmethod
    def _ensure_avro_compatible(obj: Any) -> Any:
        """Recursively ensure all objects in a structure are Avro-compatible."""
        if isinstance(obj, dict):
            return {k: EventConverter._ensure_avro_compatible(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [EventConverter._ensure_avro_compatible(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return EventConverter._format_datetime(obj)
        elif hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            return EventConverter._ensure_avro_compatible(obj.model_dump())
        else:
            # Convert other objects to strings
            return str(obj)

    @staticmethod
    def specialized_event_to_avro(event: BaseEvent, expected_event_type: EventType = None) -> EventAvro:
        """Convert any specialized event to Avro model.

        Args:
            event: The event to convert
            expected_event_type: Optional event type to validate against

        Returns:
            EventAvro representation of the event

        Raises:
            ValueError: If event_type doesn't match expected type
        """
        # Validate event type if specified
        if expected_event_type and event.event_type != expected_event_type:
            raise ValueError(f"Expected event_type {expected_event_type}, got {event.event_type}")

        # Convert payload if it's a Pydantic model
        payload = event.payload.model_dump() if hasattr(event.payload, "model_dump") else event.payload

        return EventConverter.to_avro(BaseEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            source_container=event.source_container,
            payload=payload,
            timestamp=event.timestamp,
            priority=event.priority,
            correlation_id=event.correlation_id,
            metadata=event.metadata,
            version=event.version
        ))

    # Type-specific converters that utilize the specialized converter
    @staticmethod
    def code_generation_requested_to_avro(event: CodeGenerationRequestedEvent) -> EventAvro:
        """Convert CodeGenerationRequestedEvent to Avro model."""
        return EventConverter.specialized_event_to_avro(
            event, expected_event_type=EventType.CODE_GENERATION_REQUESTED
        )

    @staticmethod
    def code_generation_completed_to_avro(event: CodeGenerationCompletedEvent) -> EventAvro:
        """Convert CodeGenerationCompletedEvent to Avro model."""
        return EventConverter.specialized_event_to_avro(
            event, expected_event_type=EventType.CODE_GENERATION_COMPLETED
        )

    @staticmethod
    def knowledge_query_requested_to_avro(event: KnowledgeQueryRequestedEvent) -> EventAvro:
        """Convert KnowledgeQueryRequestedEvent to Avro model."""
        return EventConverter.specialized_event_to_avro(
            event, expected_event_type=EventType.KNOWLEDGE_QUERY_REQUESTED
        )

    @staticmethod
    def knowledge_query_completed_to_avro(event: KnowledgeQueryCompletedEvent) -> EventAvro:
        """Convert KnowledgeQueryCompletedEvent to Avro model."""
        return EventConverter.specialized_event_to_avro(
            event, expected_event_type=EventType.KNOWLEDGE_QUERY_COMPLETED
        )

    @staticmethod
    def pydantic_to_avro_type(python_type: Type) -> Union[str, Dict[str, Any], List[Any]]:
        """Convert a Python type to an Avro type.

        Args:
            python_type: Python type to convert

        Returns:
            Avro type as a string or complex type as a dictionary
        """
        # Handle None/NoneType explicitly
        if python_type is None or python_type == type(None):
            return "null"

        # Handle Union types (Optional is a Union with None)
        origin = get_origin(python_type)
        args = get_args(python_type)

        if origin is Union:
            if type(None) in args:
                # Optional type - get the non-None type
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    avro_type = EventConverter.pydantic_to_avro_type(non_none_args[0])
                    # Always ensure "null" is the first type in the union
                    return ["null", avro_type]
                else:
                    # Complex union type with null
                    return ["null"] + [EventConverter.pydantic_to_avro_type(arg) for arg in non_none_args]
            else:
                # Union type without null
                return [EventConverter.pydantic_to_avro_type(arg) for arg in args]

        # Handle Literal types
        elif origin is Literal:
            # For Literal types, we'll use the type of the first argument
            if args:
                first_arg = args[0]
                # Determine the type of the literal value
                return EventConverter.pydantic_to_avro_type(type(first_arg))
            else:
                return "string"  # Default if no arguments

        # Handle basic types
        elif python_type == str:
            return "string"
        elif python_type == int:
            return "int"
        elif python_type == float:
            return "double"
        elif python_type == bool:
            return "boolean"
        elif python_type == datetime:
            return "string"  # ISO format string
        elif origin is dict:
            # Dict[str, Any] case
            key_type, value_type = args
            if key_type != str:
                raise ValueError(f"Avro only supports string keys in maps, got {key_type}")

            return {
                "type": "map",
                "values": EventConverter.pydantic_to_avro_type(value_type)
            }
        elif origin is list:
            # List[T] case
            item_type = args[0] if args else Any

            # Create a separate variable for the Avro item type
            item_avro_type = EventConverter.pydantic_to_avro_type(item_type)

            # Construct and return the array schema with explicit items field
            array_schema = {
                "type": "array",
                "items": item_avro_type
            }

            return array_schema
        elif isinstance(python_type, type) and issubclass(python_type, Enum):
            # For Enums, convert to string
            return "string"  # Enums are serialized as strings
        elif isinstance(python_type, type) and issubclass(python_type, BaseModel):
            # For nested models, generate schema recursively
            return EventConverter.generate_avro_schema(python_type)
        else:
            return "string"  # Default to string for unknown types

    @staticmethod
    def generate_avro_schema(model_class: Type[BaseModel], namespace: Optional[str] = None) -> Dict[str, Any]:
        """Generate an Avro schema from a Pydantic model.

        Args:
            model_class: Pydantic model class
            namespace: Optional namespace for the Avro schema

        Returns:
            Avro schema as a dictionary
        """
        model_name = model_class.__name__
        fields = []

        # Get field types from model
        for field_name, field_info in model_class.__pydantic_fields__.items():
            field_type = field_info.annotation

            # Convert to Avro type
            avro_type = EventConverter.pydantic_to_avro_type(field_type)

            # Determine if field has a default value
            has_default = field_info.default is not None and field_info.default is not Ellipsis

            field_def = {
                "name": field_name,
                "type": avro_type
            }

            # Add default value if present with proper conversion
            if has_default:
                default_value = field_info.default
                # Convert default value to Avro-compatible format
                field_def["default"] = EventConverter._ensure_avro_compatible(default_value)

            fields.append(field_def)

        # Build schema
        schema = {
            "type": "record",
            "name": model_name,
            "fields": fields
        }

        if namespace:
            schema["namespace"] = namespace

        return schema

    @staticmethod
    def roundtrip_test(event: BaseEvent) -> BaseEvent:
        """Convert event to Avro and back to verify roundtrip integrity."""
        # Convert to Avro
        avro_event = EventConverter.to_avro(event)

        # Convert to dict to simulate serialization
        avro_dict = avro_event.model_dump()

        # Recreate Avro event from dict
        recreated_avro = EventAvro(**avro_dict)

        # Convert back to BaseEvent
        return EventConverter.from_avro(recreated_avro)