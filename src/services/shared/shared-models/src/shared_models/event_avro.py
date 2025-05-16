"""
Avro-compatible event model for Apache Pulsar integration.

This module provides a simplified event model that is fully
compatible with Apache Pulsar's Avro schema registry. It is designed
to ensure reliable serialization and deserialization of event data
across system boundaries.

Classes:
    EventAvro: Avro-compatible event model for Apache Pulsar
"""

# refactored_event_avro.py

from datetime import datetime
from datetime import timezone
from typing import Any, Dict, Optional
from uuid import uuid4
from pydantic_avro.base import AvroBase
from pydantic import Field
from .enums import EventPriority


class EventAvro(AvroBase):
    """Avro-compatible event model for Apache Pulsar."""

    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the event")
    event_type: str = Field(..., description="The type of event")
    source_container: str = Field(..., description="The container that emitted the event")
    payload: Dict[str, Any] = Field(..., description="The event data payload")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        description="UTC timestamp in ISO format",
    )
    priority: int = Field(default=EventPriority.NORMAL.value, description="Event priority level")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID for tracing")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata for the event")
    version: str = Field(default="1.0", description="Version of the event format")

    class Meta:
        """Avro schema metadata."""

        namespace = "events"
        schema_name = "EventAvro"
