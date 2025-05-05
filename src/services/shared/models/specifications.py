# src/services/shared/models/specifications.py
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Literal
import uuid

from pydantic import BaseModel, Field, ConfigDict

from src.services.shared.models.base import BaseMessage
from src.services.shared.models.enums import EventType  # Import from your new enums.py


class FieldDefinition(BaseModel):
    """Field definition in a spec sheet."""
    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type")
    description: str = Field(..., description="Field description")
    required: bool = Field(False, description="Whether the field is required")
    default_value: Any = Field(None, description="Default value")
    validation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Validation rules")

    model_config = ConfigDict(frozen=True)


class SectionDefinition(BaseModel):
    """Section definition in a spec sheet."""
    name: str = Field(..., description="Section name")
    description: str = Field(..., description="Section description")
    fields: List[FieldDefinition] = Field(default_factory=list, description="Section fields")

    model_config = ConfigDict(frozen=True)


class SpecSheetDefinition(BaseMessage):
    """Template for a specification sheet."""
    id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    version: str = Field(..., description="Template version")
    category: str = Field(..., description="Template category")
    sections: List[SectionDefinition] = Field(default_factory=list, description="Template sections")
    __schema_version__: str = "1.0.0"  # Schema version for evolution tracking

    model_config = ConfigDict(frozen=True)

    def to_avro(self) -> Dict[str, Any]:
        """Convert to Avro-compatible dict."""
        data = self.model_dump()
        # Ensure Avro compatibility
        return self._make_avro_compatible(data)

    @staticmethod
    def _make_avro_compatible(value):
        """Convert Python values to Avro-compatible values."""
        if isinstance(value, dict):
            return {k: SpecSheetDefinition._make_avro_compatible(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [SpecSheetDefinition._make_avro_compatible(v) for v in value]
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, datetime):
            # Convert datetime to ISO format with Z suffix
            return value.isoformat().replace("+00:00", "Z")
        else:
            return str(value)  # Convert other types to string


class FieldValue(BaseModel):
    """Field value in a completed spec sheet."""
    name: str = Field(..., description="Field name")
    value: Any = Field(None, description="Field value")

    model_config = ConfigDict(frozen=True)


class SectionValues(BaseModel):
    """Section values in a completed spec sheet."""
    name: str = Field(..., description="Section name")
    fields: List[FieldValue] = Field(default_factory=list, description="Section field values")

    model_config = ConfigDict(frozen=True)


class SpecSheet(BaseMessage):
    """Completed specification sheet."""
    id: str = Field(..., description="Spec sheet identifier")
    template_id: str = Field(..., description="Template identifier")
    project_id: str = Field(..., description="Project identifier")
    name: str = Field(..., description="Spec sheet name")
    sections: List[SectionValues] = Field(default_factory=list, description="Spec sheet sections")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                            description="Creation timestamp")
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                            description="Update timestamp")
    completed: bool = Field(False, description="Whether the spec sheet is completed")
    validated: bool = Field(False, description="Whether the spec sheet is validated")
    __schema_version__: str = "1.0.0"  # Schema version for evolution tracking

    model_config = ConfigDict(frozen=True)

    def to_avro(self) -> Dict[str, Any]:
        """Convert to Avro-compatible dict."""
        data = self.model_dump()
        # Special handling for datetime fields if needed
        return self._make_avro_compatible(data)

    @staticmethod
    def _make_avro_compatible(value):
        """Convert Python values to Avro-compatible values."""
        if isinstance(value, dict):
            return {k: SpecSheet._make_avro_compatible(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [SpecSheet._make_avro_compatible(v) for v in value]
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, datetime):
            # Convert datetime to ISO format with Z suffix
            return value.isoformat().replace("+00:00", "Z")
        else:
            return str(value)  # Convert other types to string


class SpecSheetGenerationRequestMessage(BaseMessage):
    """Message requesting spec sheet generation."""
    message_type: Literal["SPEC_SHEET_GENERATION_REQUEST"] = "SPEC_SHEET_GENERATION_REQUEST"
    project_id: str = Field(..., description="Project identifier")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                           description="Request timestamp in ISO 8601 UTC format with Z suffix")
    __schema_version__: str = "1.0.0"

    model_config = ConfigDict(frozen=True)


class SpecSheetCompletionRequestMessage(BaseMessage):
    """Message requesting AI completion of a spec sheet."""
    spec_sheet_id: str = Field(..., description="Spec sheet identifier")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request identifier")
    section_name: Optional[str] = Field(None, description="Section name to complete")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                           description="Request timestamp")
    __schema_version__: str = "1.0.0"

    model_config = ConfigDict(frozen=True)


class SpecSheetDefinitionRequest(BaseMessage):
    """Request for a spec sheet definition matching specific criteria."""
    request_id: str = Field(..., description="Request identifier")
    generator_type: str = Field(..., description="Generator type")
    formal_spec: Dict[str, Any] = Field(..., description="Formal specification")
    domain: Optional[str] = Field(None, description="Domain")
    language: Optional[str] = Field(None, description="Programming language")
    framework: Optional[str] = Field(None, description="Framework")
    component: Optional[str] = Field(None, description="Component")
    pattern: Optional[str] = Field(None, description="Pattern")
    additional_criteria: Dict[str, Any] = Field(default_factory=dict, description="Additional criteria")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                            description="Creation timestamp")
    __schema_version__: str = "1.0.0"

    model_config = ConfigDict(frozen=True)


class SpecSheetDefinitionResponse(BaseMessage):
    """Response with the best matching spec sheet definition."""
    request_id: str = Field(..., description="Request identifier")
    spec_sheet_definition_id: Optional[str] = Field(None, description="Spec sheet definition identifier")
    spec_sheet_definition_path: Optional[str] = Field(None, description="Spec sheet definition path")
    match_confidence: float = Field(0.0, description="Match confidence")
    spec_sheet_definition_metadata: Dict[str, Any] = Field(default_factory=dict,
                                                           description="Spec sheet definition metadata")
    error: Optional[str] = Field(None, description="Error message")
    __schema_version__: str = "1.0.0"

    model_config = ConfigDict(frozen=True)