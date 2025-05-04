from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict


class EventAvro(BaseModel):
    event_id: str
    event_type: str  # String, not enum for Avro compatibility
    source_container: str
    payload: Dict[str, Any]  # Simplified, conversion handled in EventConverter
    timestamp: str  # String for ISO format
    priority: int = 1
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    version: str = "1.0"

    model_config = ConfigDict(frozen=True)

    def to_avro(self) -> Dict[str, Any]:
        """Convert to Avro-compatible dict."""
        # Make sure all values are Avro compatible
        return {k: self._make_avro_compatible(v) for k, v in self.model_dump().items()}

    @staticmethod
    def _make_avro_compatible(value):
        """Convert Python values to Avro-compatible values."""
        if isinstance(value, dict):
            return {k: EventAvro._make_avro_compatible(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [EventAvro._make_avro_compatible(v) for v in value]
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        else:
            return str(value)  # Convert other types to string

    @classmethod
    def from_avro(cls, data: Dict) -> "EventAvro":
        """Create from Avro record."""
        return cls.model_validate(data)