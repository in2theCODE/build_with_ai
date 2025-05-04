import uuid
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Dict, Any, ClassVar, Optional, List, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator, PrivateAttr

from src.services.shared.models import EventType, EventPriority


class BaseEvent(BaseModel):
    """Base event class for all events in the system."""
    event_type: EventType = Field(..., description="The type of event")
    source_container: str = Field(..., description="The container that emitted the event")
    payload: Dict[str, Union[None, str, int, bool, float, List[Any], Dict[str, Any]]] = Field(
        ..., description="The event data payload")
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the event")
    priority: EventPriority = Field(default=EventPriority.NORMAL, description="Event priority level")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),
                                description="Timestamp of the event")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID for tracing")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata for the event")
    version: str = Field(default="1.0", description="Version of the event")
    __avro_schema_id__: ClassVar[Optional[int]] = None
    __avro_schema_subject__: ClassVar[Optional[str]] = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode='before')
    @classmethod
    def validate_event_type(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and convert the event_type field if it's a string."""
        if isinstance(data, dict) and "event_type" in data and isinstance(data["event_type"], str):
            try:
                data["event_type"] = EventType(data["event_type"])
            except ValueError:
                raise ValueError(f"Invalid event_type: {data['event_type']}")
        return data

    @model_validator(mode='before')
    @classmethod
    def validate_priority(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and convert the priority field if it's an integer."""
        if isinstance(data, dict) and "priority" in data and isinstance(data["priority"], int):
            try:
                data["priority"] = EventPriority(data["priority"])
            except ValueError:
                data["priority"] = EventPriority.NORMAL
        return data

    @model_validator(mode='before')
    @classmethod
    def validate_timestamp(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and convert timestamp if it's a string."""
        if isinstance(data, dict) and "timestamp" in data and isinstance(data["timestamp"], str):
            try:
                data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
            except ValueError:
                # If we can't parse it, leave it as is for Pydantic to handle
                pass
        return data

    @classmethod
    def create_from_legacy(cls, legacy_data: Dict[str, Any]) -> 'BaseEvent':
        """Create an instance from legacy event format"""
        # Map legacy fields to current fields
        if 'old_field_name' in legacy_data:
            legacy_data['new_field_name'] = legacy_data.pop('old_field_name')

        # Handle missing required fields with defaults
        for field_name, field in cls.model_fields.items():
            if field_name not in legacy_data and field.is_required():
                if field.default_factory:
                    legacy_data[field_name] = field.default_factory()
                else:
                    legacy_data[field_name] = field.default

        return cls.model_validate(legacy_data)

    @property
    def schema_version(self) -> str:
        """Return schema version for compatibility checks"""
        return getattr(self, '_schema_version', '1.0.0')

    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary with Avro-compatible format."""
        result = self.model_dump()
        result["event_type"] = self.event_type.value
        result["priority"] = self.priority.value
        # Convert datetime to ISO string with Z format for Avro compatibility
        result["timestamp"] = self.timestamp.isoformat().replace("+00:00", "Z") if self.timestamp else None
        return result

    def to_json(self) -> str:
        """Convert the event to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEvent":
        """Create an event from a dictionary."""
        # Handle datetime conversion if timestamp is a string
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
            try:
                # Handle 'Z' in ISO format which Python doesn't parse directly
                data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
            except ValueError:
                # If we can't parse it, leave it as is for Pydantic to handle
                pass
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseEvent":
        """Create an event from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class BaseMessage(BaseModel):
    """Base class for all message schemas."""

    # Avro schema metadata (to be populated by subclasses)
    __avro_schema_id__: ClassVar[Optional[int]] = None
    __avro_schema_subject__: ClassVar[Optional[str]] = None
    __schema_version__: ClassVar[str] = "1.0.0"

    model_config = ConfigDict(
        frozen=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")
        }
    )

    @classmethod
    def get_schema_subject(cls) -> str:
        """Get the schema subject for Avro registry."""
        if cls.__avro_schema_subject__ is None:
            return f"{cls.__module__}.{cls.__name__}"
        return cls.__avro_schema_subject__

    @property
    def schema_version(self) -> str:
        """Return schema version for compatibility checks."""
        return self.__class__.__schema_version__

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary with Avro-compatible format."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseMessage":
        """Create an instance from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseMessage":
        """Create an instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def serialize(self) -> bytes:
        """Serialize for Pulsar message."""
        return self.to_json().encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "BaseMessage":
        """Deserialize from Pulsar message."""
        return cls.from_json(data.decode("utf-8"))

class AvroBaseModel(BaseModel):
    """Base model with Avro schema support for all models in the system."""

    # Avro schema metadata (to be populated by subclasses)
    __avro_schema_id__: ClassVar[Optional[int]] = None
    __avro_schema_subject__: ClassVar[Optional[str]] = None
    __schema_version__: ClassVar[str] = "1.0.0"

    model_config = ConfigDict(
        frozen=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")
        }
    )

    @classmethod
    def get_schema_subject(cls) -> str:
        """Get the schema subject for Avro registry."""
        if cls.__avro_schema_subject__ is None:
            return f"{cls.__module__}.{cls.__name__}"
        return cls.__avro_schema_subject__

    @property
    def schema_version(self) -> str:
        """Return schema version for compatibility checks."""
        return self.__class__.__schema_version__

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary with Avro-compatible format."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AvroBaseModel":
        """Create an instance from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "AvroBaseModel":
        """Create an instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def serialize(self) -> bytes:
        """Serialize for Pulsar message."""
        return self.to_json().encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "AvroBaseModel":
        """Deserialize from Pulsar message."""
        return cls.from_json(data.decode("utf-8"))

class BaseComponent(BaseModel):
    """Base class for all system services built on Pydantic.

    This class provides a foundation for component configuration using Pydantic's
    validation while maintaining compatibility with component-based architecture.
    """
    # Private attribute to store additional params not captured in the model fields
    _params: Dict[str, Any] = PrivateAttr(default_factory=dict)

    # Allow extra parameters that aren't defined as fields
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True
    )

    def __init__(self, **data):
        """Initialize the component with parameters.

        This special initialization handles both Pydantic fields and
        custom parameters that aren't defined in the model.
        """
        super().__init__(**data)
        # Store any extra parameters not in the model fields
        self._params = {k: v for k, v in data.items() if k not in self.model_fields}
        # Call initialization hook
        self.initialize()

    def initialize(self):
        """Additional initialization logic after parameters are set.

        Override this method to perform setup operations after initialization.
        """
        pass

    @property
    def component_name(self) -> str:
        """Get the name of this component."""
        return self.__class__.__name__

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with a default.

        Checks both model fields and extra parameters.
        """
        # First check model fields
        if key in self.model_fields and hasattr(self, key):
            return getattr(self, key)
        # Then check extra params
        return self._params.get(key, default)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include extra params."""
        data = super().model_dump(**kwargs)
        # Include extra params in the dump
        data.update(self._params)
        return data


class ConfigurableComponent(BaseComponent):
    """A component that can be configured dynamically."""

    def configure(self, config: Dict[str, Any]):
        """Update the component configuration.

        This method updates both model fields and extra parameters.
        """
        # Update model fields
        for key, value in config.items():
            if key in self.model_fields:
                setattr(self, key, value)
            else:
                # Update extra params
                self._params[key] = value

        # Re-initialize with new parameters
        self.initialize()