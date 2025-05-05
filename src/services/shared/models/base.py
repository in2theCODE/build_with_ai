from __future__ import annotations  # This must be the first import
import json
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Union, TypeVar, Generic
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, model_validator, PrivateAttr
from .enums import EventType, EventPriority

T = TypeVar('T')


class EventPayload(BaseModel):
    """Base class for all event payloads."""
    model_config = ConfigDict(frozen=True)


class BaseEvent(BaseModel):
    """Base event class for all events in the system."""

    # Required fields
    event_type: EventType = Field(..., description="The type of event")
    source_container: str = Field(..., description="The container that emitted the event")
    payload: Dict[str, Union[None, str, int, bool, float, List[Any], Dict[str, Any]]] = Field(
        ..., description="The event data payload"
    )

    # Fields with defaults
    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the event")
    priority: EventPriority = Field(default=EventPriority.NORMAL, description="Event priority level")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID for tracing")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata for the event")

    # Version tracking
    version: str = Field(default="1.0", description="Version of the event format")

    # Avro schema identifiers to be filled by subclasses
    __avro_schema_id__: ClassVar[Optional[int]] = None
    __avro_schema_subject__: ClassVar[Optional[str]] = None
    __schema_version__: ClassVar[str] = "1.0.0"

    model_config = ConfigDict(
        frozen=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")
        }
    )

    # ---------------------- Validators ----------------------

    @model_validator(mode='before')
    @classmethod
    def validate_event_type(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict) and isinstance(data.get("event_type"), str):
            try:
                data["event_type"] = EventType(data["event_type"])
            except ValueError:
                raise ValueError(f"Invalid event_type: {data['event_type']}")
        return data

    @model_validator(mode='before')
    @classmethod
    def validate_priority(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict) and isinstance(data.get("priority"), int):
            try:
                data["priority"] = EventPriority(data["priority"])
            except ValueError:
                data["priority"] = EventPriority.NORMAL
        return data

    @model_validator(mode='before')
    @classmethod
    def validate_timestamp(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict) and isinstance(data.get("timestamp"), str):
            try:
                data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
            except ValueError:
                pass  # Let Pydantic raise if it fails
        return data

    # ---------------------- Schema Methods ----------------------



    @classmethod
    def get_schema_subject(cls) -> str:
        """Get the schema subject for Avro registry."""
        if cls.__avro_schema_subject__ is None:
            return f"{cls.__module__}.{cls.__name__}"
        return cls.__avro_schema_subject__

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary with Avro-compatible format."""
        data = self.model_dump()
        # Convert enums to their string values
        if isinstance(data.get("event_type"), EventType):
            data["event_type"] = data["event_type"].value
        if isinstance(data.get("priority"), EventPriority):
            data["priority"] = data["priority"].value
        # Format datetime
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat().replace("+00:00", "Z")
        return data

    def to_json(self) -> str:
        """Convert to a JSON string."""
        from json import dumps
        return dumps(self.to_dict())

    def to_avro(self) -> Dict[str, Any]:
        """Convert to Avro-compatible dict."""
        return self._make_avro_compatible(self.to_dict())

    @staticmethod
    def _make_avro_compatible(value: Any) -> Any:
        """Convert Python values to Avro-compatible values."""
        if isinstance(value, dict):
            return {k: BaseEvent._make_avro_compatible(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [BaseEvent._make_avro_compatible(v) for v in value]
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, datetime):
            return value.isoformat().replace("+00:00", "Z")
        elif isinstance(value, Enum):
            return value.value
        else:
            return str(value)  # Convert other types to string

    # ---------------------- Legacy Support ----------------------

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
    # Version tracking for schema evolution
    __schema_version__: ClassVar[str] = "1.0.0"

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

    @property
    def schema_version(self) -> str:
        """Return schema version for compatibility checks."""
        return self.__class__.__schema_version__

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
    # Inherit schema version or override with specific version
    __schema_version__: ClassVar[str] = "1.0.0"

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