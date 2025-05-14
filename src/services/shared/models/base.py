"""
Base models for Apache Pulsar integration with Avro schema serialization.

This module provides the foundation for all models in the system, with full
support for Pydantic v2, Fast Avro serialization, and Apache Pulsar schema
registry integration. The models in this module serve as base classes that
other models should inherit from to ensure consistent serialization and
schema generation.

Classes:
    AvroBaseModel: Base model with Avro schema support
    AvroBase: Enhanced model with Pulsar integration
    EventPayload: Base class for all event payloads
    BaseEvent: Base class for all system events
    BaseMessage: Base class for all message schemas
    BaseComponent: Base class for all system components
    ConfigurableComponent: Component that can be dynamically configured
"""

from __future__ import annotations

import fastavro
from datetime import datetime
from datetime import timezone
import io
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)
from uuid import uuid4
from pydantic_avro import AvroBase
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from .enums import EventPriority
from .enums import EventType


T = TypeVar("T")


class AvroBaseModel(AvroBase):
    """Base model with Avro and Pulsar support for all models in the system.

    This model inherits from AvroBaseModel which provides Avro schema generation,
    serialization, and deserialization functionality from dataclasses-avroschema.
    It adds additional methods for Pulsar integration and schema registry support.
    """

    # Schema versioning and tracking
    __avro_schema_id__: ClassVar[Optional[int]] = None
    __avro_schema_subject__: ClassVar[Optional[str]] = None
    __schema_version__: ClassVar[str] = "1.0.0"

    model_config = ConfigDict(
        frozen=True, json_encoders={datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")}
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

    def to_avro_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary with Avro-compatible format."""
        # This internally handles enum and datetime conversion
        return self.model_dump()

    def serialize(self) -> bytes:
        """Serialize to binary using Fast Avro."""
        schema = self.__class__.avro_schema_to_python()
        buffer = io.BytesIO()
        fastavro.schemaless_writer(buffer, schema, self.to_avro_dict())
        return buffer.getvalue()

    @classmethod
    def deserialize(cls, data: bytes) -> "AvroBase":
        """Deserialize from binary using Fast Avro."""
        schema = cls.avro_schema_to_python()
        buffer = io.BytesIO(data)
        record = fastavro.schemaless_reader(buffer, schema)
        return cls.model_validate(record)

    @classmethod
    def from_avro_dict(cls, data: Dict[str, Any]) -> "AvroBase":
        """Create an instance from an Avro dictionary."""
        return cls.model_validate(data)


class EventPayload(AvroBase):
    """Base class for all event payloads."""

    pass


class BaseEvent(AvroBase):
    """Base event class for all events in the system."""

    # Required fields
    event_type: EventType = Field(..., description="The type of event")
    source_container: str = Field(..., description="The container that emitted the event")
    payload: Dict[str, Union[None, str, int, bool, float, List[Any], Dict[str, Any]]] = Field(
        ..., description="The event data payload"
    )

    # Fields with defaults
    event_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique identifier for the event"
    )
    priority: EventPriority = Field(
        default=EventPriority.NORMAL, description="Event priority level"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp"
    )
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID for tracing")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata for the event"
    )

    # Version tracking
    version: str = Field(default="1.0", description="Version of the event format")

    class Meta:
        """Avro schema metadata"""

        namespace = "events"

    # ---------------------- Validators ----------------------

    @classmethod
    def validate_event_type(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict) and isinstance(data.get("event_type"), str):
            try:
                data["event_type"] = EventType(data["event_type"])
            except ValueError:
                raise ValueError(f"Invalid event_type: {data['event_type']}")
        return data

    @classmethod
    def validate_priority(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict) and isinstance(data.get("priority"), int):
            try:
                data["priority"] = EventPriority(data["priority"])
            except ValueError:
                data["priority"] = EventPriority.NORMAL
        return data

    @classmethod
    def validate_timestamp(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict) and isinstance(data.get("timestamp"), str):
            try:
                data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            except ValueError:
                pass  # Let Pydantic raise if it fails
        return data


class BaseMessage(AvroBase):
    """Base class for all message schemas."""

    pass


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
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, validate_assignment=True)

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
