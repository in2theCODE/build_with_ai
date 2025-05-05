# src/services/shared/models/events/domain.py

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Literal, Union, ClassVar
from pydantic import Field, model_validator

from src.services.shared.models.enums import EventType, EventPriority
from .base import BaseEvent, EventPayload


# ======================= Payload Models =======================

class CodeGenerationRequestPayload(EventPayload):
    """Payload model for code generation requests."""
    spec_sheet: Dict[str, Any] = Field(..., description="Specification sheet")
    target_language: str = Field("python", description="Target programming language")


class CodeGenerationCompletedPayload(EventPayload):
    """Payload model for completed code generation."""
    generated_code: str = Field(..., description="The generated code")
    program_ast: Dict[str, Any] = Field(..., description="The AST of the generated program")
    confidence_score: float = Field(..., description="Confidence score of the generation")
    strategy_used: str = Field(..., description="Generation strategy used")
    time_taken: float = Field(..., description="Time taken to generate code in seconds")


class CodeGenerationFailedPayload(EventPayload):
    """Payload model for failed code generation."""
    error_message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    partial_result: Optional[Dict[str, Any]] = Field(None, description="Partial result if available")


class KnowledgeQueryPayload(EventPayload):
    """Payload model for knowledge query requests."""
    query: str = Field(..., description="The search query string")
    limit: int = Field(5, description="Maximum number of results to return")


class KnowledgeQueryCompletedPayload(EventPayload):
    """Payload model for completed knowledge queries."""
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    query: str = Field(..., description="Original query")
    time_taken: float = Field(..., description="Time taken to execute query in seconds")


class KnowledgeUpdatedPayload(EventPayload):
    """Payload model for knowledge update events."""
    key: str = Field(..., description="Key of the updated knowledge")
    update_type: str = Field(..., description="Type of update")


# ======================= Base Event Classes =======================

class SpecSheetEvent(BaseEvent):
    """Base class for spec sheet-related events"""
    spec_sheet_id: str = Field(..., description="Spec sheet identifier")
    spec_sheet_version: Optional[str] = Field(None, description="Spec sheet version")
    spec_sheet_name: Optional[str] = Field(None, description="Spec sheet name")
    spec_sheet_category: Optional[str] = Field(None, description="Spec sheet category")

    @model_validator(mode='before')
    @classmethod
    def ensure_payload_consistency(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure that spec sheet data is consistent with payload."""
        if not isinstance(data, dict):
            return data

        if "payload" not in data:
            data["payload"] = {}

        payload = data["payload"]
        spec_sheet_fields = ["spec_sheet_id", "spec_sheet_version", "spec_sheet_name", "spec_sheet_category"]

        for field in spec_sheet_fields:
            if field in data and data[field] is not None:
                payload[field] = data[field]

        return data


class SpecInstanceEvent(BaseEvent):
    """Base class for spec sheet instance-related events"""
    instance_id: str = Field(..., description="Instance identifier")
    spec_sheet_id: str = Field(..., description="Spec sheet identifier")
    spec_sheet_version: str = Field(..., description="Spec sheet version")
    project_id: str = Field("", description="Project identifier")

    @model_validator(mode='before')
    @classmethod
    def ensure_payload_consistency(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure that instance data is consistent with payload."""
        if not isinstance(data, dict):
            return data

        if "payload" not in data:
            data["payload"] = {}

        payload = data["payload"]
        instance_fields = ["instance_id", "spec_sheet_id", "spec_sheet_version", "project_id"]

        for field in instance_fields:
            if field in data and data[field] is not None:
                payload[field] = data[field]

        return data


# ======================= Concrete Event Classes =======================

class CodeGenerationRequestedEvent(BaseEvent):
    """Event for requesting code generation."""
    event_type: Literal[EventType.CODE_GENERATION_REQUESTED] = EventType.CODE_GENERATION_REQUESTED
    payload: Union[CodeGenerationRequestPayload, Dict[str, Any]] = Field(
        ..., description="Event payload containing spec sheet and target language"
    )

    __schema_version__: ClassVar[str] = "1.0.0"

    @classmethod
    def create(cls,
               source_container: str,
               spec_sheet: Dict[str, Any],
               target_language: str = "python",
               correlation_id: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> "CodeGenerationRequestedEvent":
        """Factory method to create a code generation request event."""
        return cls(
            source_container=source_container,
            payload=CodeGenerationRequestPayload(
                spec_sheet=spec_sheet,
                target_language=target_language
            ),
            correlation_id=correlation_id,
            metadata=metadata or {}
        )


class CodeGenerationCompletedEvent(BaseEvent):
    """Event for completed code generation."""
    event_type: Literal[EventType.CODE_GENERATION_COMPLETED] = EventType.CODE_GENERATION_COMPLETED
    payload: Union[CodeGenerationCompletedPayload, Dict[str, Any]] = Field(
        ..., description="Event payload containing generated code and results"
    )

    __schema_version__: ClassVar[str] = "1.0.0"

    @classmethod
    def create(cls,
               source_container: str,
               generated_code: str,
               program_ast: Dict[str, Any],
               confidence_score: float,
               strategy_used: str,
               time_taken: float,
               correlation_id: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> "CodeGenerationCompletedEvent":
        """Factory method to create a code generation completed event."""
        return cls(
            source_container=source_container,
            payload=CodeGenerationCompletedPayload(
                generated_code=generated_code,
                program_ast=program_ast,
                confidence_score=confidence_score,
                strategy_used=strategy_used,
                time_taken=time_taken
            ),
            correlation_id=correlation_id,
            metadata=metadata or {}
        )


class CodeGenerationFailedEvent(BaseEvent):
    """Event for failed code generation."""
    event_type: Literal[EventType.CODE_GENERATION_FAILED] = EventType.CODE_GENERATION_FAILED
    payload: Union[CodeGenerationFailedPayload, Dict[str, Any]] = Field(
        ..., description="Event payload containing error details"
    )
    priority: Literal[EventPriority.HIGH] = EventPriority.HIGH

    __schema_version__: ClassVar[str] = "1.0.0"

    @classmethod
    def create(cls,
               source_container: str,
               error_message: str,
               error_type: str,
               partial_result: Optional[Dict[str, Any]] = None,
               correlation_id: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> "CodeGenerationFailedEvent":
        """Factory method to create a code generation failed event."""
        return cls(
            source_container=source_container,
            payload=CodeGenerationFailedPayload(
                error_message=error_message,
                error_type=error_type,
                partial_result=partial_result
            ),
            correlation_id=correlation_id,
            metadata=metadata or {}
        )


class KnowledgeQueryRequestedEvent(BaseEvent):
    """Event for requesting a knowledge base query."""
    event_type: Literal[EventType.KNOWLEDGE_QUERY_REQUESTED] = EventType.KNOWLEDGE_QUERY_REQUESTED
    payload: Union[KnowledgeQueryPayload, Dict[str, Any]] = Field(
        ..., description="Event payload containing query parameters"
    )

    __schema_version__: ClassVar[str] = "1.0.0"

    @classmethod
    def create(cls,
               source_container: str,
               query: str,
               limit: int = 5,
               correlation_id: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> "KnowledgeQueryRequestedEvent":
        """Factory method to create a knowledge query request event."""
        return cls(
            source_container=source_container,
            payload=KnowledgeQueryPayload(
                query=query,
                limit=limit
            ),
            correlation_id=correlation_id,
            metadata=metadata or {}
        )


class KnowledgeQueryCompletedEvent(BaseEvent):
    """Event for completed knowledge base query."""
    event_type: Literal[EventType.KNOWLEDGE_QUERY_COMPLETED] = EventType.KNOWLEDGE_QUERY_COMPLETED
    payload: Union[KnowledgeQueryCompletedPayload, Dict[str, Any]] = Field(
        ..., description="Event payload containing query results"
    )

    __schema_version__: ClassVar[str] = "1.0.0"

    @classmethod
    def create(cls,
               source_container: str,
               results: List[Dict[str, Any]],
               query: str,
               time_taken: float,
               correlation_id: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> "KnowledgeQueryCompletedEvent":
        """Factory method to create a knowledge query completed event."""
        return cls(
            source_container=source_container,
            payload=KnowledgeQueryCompletedPayload(
                results=results,
                query=query,
                time_taken=time_taken
            ),
            correlation_id=correlation_id,
            metadata=metadata or {}
        )


class KnowledgeUpdatedEvent(BaseEvent):
    """Event for knowledge base updates."""
    event_type: Literal[EventType.KNOWLEDGE_UPDATED] = EventType.KNOWLEDGE_UPDATED
    payload: Union[KnowledgeUpdatedPayload, Dict[str, Any]] = Field(
        ..., description="Event payload containing update details"
    )

    __schema_version__: ClassVar[str] = "1.0.0"

    @classmethod
    def create(cls,
               source_container: str,
               key: str,
               update_type: str,
               metadata: Optional[Dict[str, Any]] = None) -> "KnowledgeUpdatedEvent":
        """Factory method to create a knowledge updated event."""
        return cls(
            source_container=source_container,
            payload=KnowledgeUpdatedPayload(
                key=key,
                update_type=update_type
            ),
            metadata=metadata or {}
        )


# ======================= Spec Sheet Events =======================

class SpecSheetCreatedEvent(SpecSheetEvent):
    """Event for spec sheet creation."""
    event_type: Literal[EventType.SPEC_SHEET_CREATED] = EventType.SPEC_SHEET_CREATED
    __schema_version__: ClassVar[str] = "1.0.0"


class SpecSheetUpdatedEvent(SpecSheetEvent):
    """Event for spec sheet updates."""
    event_type: Literal[EventType.SPEC_SHEET_UPDATED] = EventType.SPEC_SHEET_UPDATED
    __schema_version__: ClassVar[str] = "1.0.0"


class SpecSheetDeletedEvent(SpecSheetEvent):
    """Event for spec sheet deletion."""
    event_type: Literal[EventType.SPEC_SHEET_DELETED] = EventType.SPEC_SHEET_DELETED
    __schema_version__: ClassVar[str] = "1.0.0"


class SpecSheetPublishedEvent(SpecSheetEvent):
    """Event for spec sheet publication."""
    event_type: Literal[EventType.SPEC_SHEET_PUBLISHED] = EventType.SPEC_SHEET_PUBLISHED
    __schema_version__: ClassVar[str] = "1.0.0"


class SpecSheetDeprecatedEvent(SpecSheetEvent):
    """Event for spec sheet deprecation."""
    event_type: Literal[EventType.SPEC_SHEET_DEPRECATED] = EventType.SPEC_SHEET_DEPRECATED
    __schema_version__: ClassVar[str] = "1.0.0"


class SpecSheetArchivedEvent(SpecSheetEvent):
    """Event for spec sheet archival."""
    event_type: Literal[EventType.SPEC_SHEET_ARCHIVED] = EventType.SPEC_SHEET_ARCHIVED
    __schema_version__: ClassVar[str] = "1.0.0"


# ======================= Spec Instance Events =======================

class SpecInstanceCreatedEvent(SpecInstanceEvent):
    """Event for spec instance creation."""
    event_type: Literal[EventType.SPEC_INSTANCE_CREATED] = EventType.SPEC_INSTANCE_CREATED
    __schema_version__: ClassVar[str] = "1.0.0"


class SpecInstanceUpdatedEvent(SpecInstanceEvent):
    """Event for spec instance updates."""
    event_type: Literal[EventType.SPEC_INSTANCE_UPDATED] = EventType.SPEC_INSTANCE_UPDATED
    __schema_version__: ClassVar[str] = "1.0.0"


class SpecInstanceCompletedEvent(SpecInstanceEvent):
    """Event for spec instance completion."""
    event_type: Literal[EventType.SPEC_INSTANCE_COMPLETED] = EventType.SPEC_INSTANCE_COMPLETED
    __schema_version__: ClassVar[str] = "1.0.0"


class SpecInstanceValidatedEvent(SpecInstanceEvent):
    """Event for spec instance validation."""
    event_type: Literal[EventType.SPEC_INSTANCE_VALIDATED] = EventType.SPEC_INSTANCE_VALIDATED
    __schema_version__: ClassVar[str] = "1.0.0"


class SpecInstanceDeletedEvent(SpecInstanceEvent):
    """Event for spec instance deletion."""
    event_type: Literal[EventType.SPEC_INSTANCE_DELETED] = EventType.SPEC_INSTANCE_DELETED
    __schema_version__: ClassVar[str] = "1.0.0"