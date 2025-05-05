import pytest
from pydantic import ValidationError

from src.services.shared.models.event_avro import EventAvro
from src.services.shared.models.event_avro_converter import event_to_avro, avro_to_event
from src.services.shared.models.events.events import BaseEvent as Event

# Import all your models


def test_event_model_validation():
    """Test validation of Event model."""
    # Valid case
    event = Event(
        event_type="CODE_GENERATION_REQUESTED",
        source_container="test-container",
        payload={"key": "value"},
    )
    assert event.event_id is not None

    # Invalid case
    with pytest.raises(ValidationError):
        Event(
            event_type="INVALID_TYPE",  # Invalid enum value
            source_container="test-container",
            payload={"key": "value"},
        )


def test_avro_serialization():
    """Test serialization to/from Avro."""
    original_event = Event(
        event_type="CODE_GENERATION_REQUESTED",
        source_container="test-container",
        payload={"key": "value"},
    )

    # Convert to Avro
    avro_event = event_to_avro(original_event)
    assert avro_event.event_type == original_event.event_type.value

    # Convert back to application model
    reconstructed_event = avro_to_event(avro_event)
    assert reconstructed_event.event_type == original_event.event_type

    # Test full serialization
    avro_dict = avro_event.to_avro()
    reconstructed_avro = EventAvro.from_avro(avro_dict)
    assert reconstructed_avro.event_id == avro_event.event_id
