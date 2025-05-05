from __future__ import annotations
import json
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Union, TypeVar
from uuid import uuid4
from datetime import datetime, timezone

from pydantic import BaseModel, Field, ConfigDict, model_validator, model
from pydantic_core import PydanticUndefined


# --- Example Enums (replace with your own) ---
class EventType(str, Enum):
    EXAMPLE = "example"


class EventPriority(int, Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2


T = TypeVar("T")


class LegacySupportMixin:
    @classmethod
    def create_from_legacy(cls, legacy_data: Dict[str, Any]) -> BaseModel:
        data = legacy_data.copy()

        # Get model fields safely using getattr with fallback to empty dict
        fields_dict = getattr(cls, "model_fields", {})

        for field_name, field_info in fields_dict.items():
            if field_name not in data:
                if hasattr(field_info, "is_required") and field_info.is_required():
                    if (
                        hasattr(field_info, "default_factory")
                        and field_info.default_factory is not None
                    ):
                        data[field_name] = field_info.default_factory()
                    elif (
                        hasattr(field_info, "default")
                        and field_info.default is not PydanticUndefined
                    ):
                        data[field_name] = field_info.default
                    else:
                        raise ValueError(f"Missing required field: {field_name}")

        # Create a new instance without validation using object.__new__
        instance = object.__new__(cls)

        # Set __dict__ directly - bypassing validation
        object.__setattr__(instance, "__dict__", {})

        # Handle each field manually
        for field_name, value in data.items():
            # Set the attribute directly
            object.__setattr__(instance, field_name, value)

        # Ensure model private attributes are set
        object.__setattr__(instance, "__pydantic_fields_set__", set(data.keys()))
        object.__setattr__(instance, "__pydantic_extra__", {})

        return instance


# --- BaseEvent ---
class BaseEvent(BaseModel, LegacySupportMixin):
    event_type: EventType
    source_container: str
    payload: Dict[str, Union[None, str, int, bool, float, List[Any], Dict[str, Any]]]
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    priority: EventPriority = Field(default=EventPriority.NORMAL)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0"

    __avro_schema_id__: ClassVar[Optional[int]] = None
    __avro_schema_subject__: ClassVar[Optional[str]] = None
    __schema_version__: ClassVar[str] = "1.0.0"

    model_config = ConfigDict(
        frozen=True,
        json_encoders={datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")},
    )

    @model_validator(mode="before")
    @classmethod
    def validate_event_type(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data.get("event_type"), str):
            data["event_type"] = EventType(data["event_type"])
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_priority(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data.get("priority"), int):
            data["priority"] = EventPriority(data["priority"])
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_timestamp(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(
                data["timestamp"].replace("Z", "+00:00")
            )
        return data

    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["event_type"] = self.event_type.value
        data["priority"] = self.priority.value
        data["timestamp"] = self.timestamp.isoformat().replace("+00:00", "Z")
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def serialize(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> BaseEvent:
        return cls.model_validate(json.loads(data.decode("utf-8")))
