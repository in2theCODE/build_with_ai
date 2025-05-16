# test_event_converter.py
from datetime import datetime
from datetime import timezone
from enum import Enum

from infra.registration.event_converter import EventConverter
from src.services.shared.models.event_avro import EventAvro
from src.services.shared.models.events import BaseEvent
from src.services.shared.models.events import EventPriority
from src.services.shared.models.events import EventType


class TestEnum(str, Enum):
    VALUE1 = "value1"
    VALUE2 = "value2"


def test_to_avro_basic():
    """Test basic conversion to Avro."""
    event = BaseEvent(
        event_type=EventType.CODE_GENERATION_REQUESTED,
        source_container="test_container",
        payload={"test": "value"},
        priority=EventPriority.NORMAL,
        correlation_id="test-correlation-id",
        metadata={"meta": "data"},
    )

    avro_event = EventConverter.to_avro(event)

    assert avro_event.event_type == EventType.CODE_GENERATION_REQUESTED.value
    assert avro_event.priority == EventPriority.NORMAL.value
    assert "Z" in avro_event.timestamp


def test_from_avro_basic():
    """Test basic conversion from Avro."""
    avro_event = EventAvro(
        event_id="test-id",
        event_type=EventType.CODE_GENERATION_REQUESTED.value,
        source_container="test_container",
        payload={"test": "value"},
        timestamp="2023-01-01T12:00:00Z",
        priority=EventPriority.NORMAL.value,
        correlation_id="test-correlation-id",
        metadata={"meta": "data"},
    )

    event = EventConverter.from_avro(avro_event)

    assert event.event_type == EventType.CODE_GENERATION_REQUESTED
    assert event.priority == EventPriority.NORMAL
    assert event.timestamp.isoformat().replace("+00:00", "Z") == "2023-01-01T12:00:00Z"


def test_ensure_avro_compatible():
    """Test conversion of complex structures to Avro-compatible types."""
    complex_data = {
        "enum": TestEnum.VALUE1,
        "date": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "nested": {"list": [TestEnum.VALUE2, datetime(2023, 1, 2, tzinfo=timezone.utc)]},
    }

    avro_data = EventConverter._ensure_avro_compatible(complex_data)

    assert avro_data["enum"] == "value1"
    assert avro_data["date"] == "2023-01-01T12:00:00Z"
    assert avro_data["nested"]["list"][0] == "value2"
    assert avro_data["nested"]["list"][1] == "2023-01-02T00:00:00Z"


def test_roundtrip():
    """Test roundtrip conversion."""
    original_event = BaseEvent(
        event_type=EventType.CODE_GENERATION_REQUESTED,
        source_container="test_container",
        payload={"test": "value", "nested": {"value": 123}},
        priority=EventPriority.NORMAL,
        correlation_id="test-correlation-id",
        metadata={"meta": "data"},
    )

    avro_event = EventConverter.to_avro(original_event)
    recreated_event = EventConverter.from_avro(avro_event)

    assert recreated_event.event_type == original_event.event_type
    assert recreated_event.source_container == original_event.source_container
    assert recreated_event.payload == original_event.payload
    assert recreated_event.priority == original_event.priority
    assert recreated_event.correlation_id == original_event.correlation_id
    assert recreated_event.metadata == original_event.metadata
