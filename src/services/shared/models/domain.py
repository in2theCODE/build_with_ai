"""
This is the models for codegen domains
"""

from typing import Any, ClassVar, Dict, List, Literal, Optional, Union
from pydantic import Field
from . import CodeGenerationCompletedPayload
from . import CodeGenerationFailedPayload
from . import CodeGenerationRequestPayload
from . import KnowledgeQueryCompletedPayload
from . import KnowledgeQueryPayload
from . import KnowledgeUpdatedPayload
from . import SpecInstanceEvent
from . import SpecSheetEvent
from .base import BaseEvent
from .enums import EventPriority
from .enums import EventType


#

# ======================= Concrete Event Classes =======================


class CodeGenerationRequestedEvent(BaseEvent):
    """Event for requesting code generation."""

    event_type: Literal[EventType.CODE_GENERATION_REQUESTED] = EventType.CODE_GENERATION_REQUESTED
    payload: Union[CodeGenerationRequestPayload, Dict[str, Any]] = Field(
        ..., description="Event payload containing spec sheet and target language"
    )

    __schema_version__: ClassVar[str] = "1.0.0"

    @classmethod
    def create(
        cls,
        source_container: str,
        spec_sheet: Dict[str, Any],
        target_language: str = "python",
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CodeGenerationRequestedEvent":
        """Factory method to create a code generation request event."""
        return cls(
            source_container=source_container,
            payload=CodeGenerationRequestPayload(
                spec_sheet=spec_sheet, target_language=target_language
            ),
            correlation_id=correlation_id,
            metadata=metadata or {},
        )


class CodeGenerationCompletedEvent(BaseEvent):
    """Event for completed code generation."""

    event_type: Literal[EventType.CODE_GENERATION_COMPLETED] = EventType.CODE_GENERATION_COMPLETED
    payload: Union[CodeGenerationCompletedPayload, Dict[str, Any]] = Field(
        ..., description="Event payload containing generated code and results"
    )

    __schema_version__: ClassVar[str] = "1.0.0"

    @classmethod
    def create(
        cls,
        source_container: str,
        generated_code: str,
        program_ast: Dict[str, Any],
        confidence_score: float,
        strategy_used: str,
        time_taken: float,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CodeGenerationCompletedEvent":
        """Factory method to create a code generation completed event."""
        return cls(
            source_container=source_container,
            payload=CodeGenerationCompletedPayload(
                generated_code=generated_code,
                program_ast=program_ast,
                confidence_score=confidence_score,
                strategy_used=strategy_used,
                time_taken=time_taken,
            ),
            correlation_id=correlation_id,
            metadata=metadata or {},
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
    def create(
        cls,
        source_container: str,
        error_message: str,
        error_type: str,
        partial_result: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CodeGenerationFailedEvent":
        """Factory method to create a code generation failed event."""
        return cls(
            source_container=source_container,
            payload=CodeGenerationFailedPayload(
                error_message=error_message, error_type=error_type, partial_result=partial_result
            ),
            correlation_id=correlation_id,
            metadata=metadata or {},
        )


class KnowledgeQueryRequestedEvent(BaseEvent):
    """Event for requesting a knowledge base query."""

    event_type: Literal[EventType.KNOWLEDGE_QUERY_REQUESTED] = EventType.KNOWLEDGE_QUERY_REQUESTED
    payload: Union[KnowledgeQueryPayload, Dict[str, Any]] = Field(
        ..., description="Event payload containing query parameters"
    )

    __schema_version__: ClassVar[str] = "1.0.0"

    @classmethod
    def create(
        cls,
        source_container: str,
        query: str,
        limit: int = 5,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "KnowledgeQueryRequestedEvent":
        """Factory method to create a knowledge query request event."""
        return cls(
            source_container=source_container,
            payload=KnowledgeQueryPayload(query=query, limit=limit),
            correlation_id=correlation_id,
            metadata=metadata or {},
        )


class KnowledgeQueryCompletedEvent(BaseEvent):
    """Event for completed knowledge base query."""

    event_type: Literal[EventType.KNOWLEDGE_QUERY_COMPLETED] = EventType.KNOWLEDGE_QUERY_COMPLETED
    payload: Union[KnowledgeQueryCompletedPayload, Dict[str, Any]] = Field(
        ..., description="Event payload containing query results"
    )

    __schema_version__: ClassVar[str] = "1.0.0"

    @classmethod
    def create(
        cls,
        source_container: str,
        results: List[Dict[str, Any]],
        query: str,
        time_taken: float,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "KnowledgeQueryCompletedEvent":
        """Factory method to create a knowledge query completed event."""
        return cls(
            source_container=source_container,
            payload=KnowledgeQueryCompletedPayload(
                results=results, query=query, time_taken=time_taken
            ),
            correlation_id=correlation_id,
            metadata=metadata or {},
        )


class KnowledgeUpdatedEvent(BaseEvent):
    """Event for knowledge base updates."""

    event_type: Literal[EventType.KNOWLEDGE_UPDATED] = EventType.KNOWLEDGE_UPDATED
    payload: Union[KnowledgeUpdatedPayload, Dict[str, Any]] = Field(
        ..., description="Event payload containing update details"
    )

    __schema_version__: ClassVar[str] = "1.0.0"

    @classmethod
    def create(
        cls,
        source_container: str,
        key: str,
        update_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "KnowledgeUpdatedEvent":
        """Factory method to create a knowledge updated event."""
        return cls(
            source_container=source_container,
            payload=KnowledgeUpdatedPayload(key=key, update_type=update_type),
            metadata=metadata or {},
        )


# ======================= Spec Sheet Events =======================


class SpecSheetHasBeenCreated(SpecSheetEvent):
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


class SpecSheetArchivedEvent(SpecSheetEvent):
    """Event for spec sheet archiving."""

    event_type: Literal[EventType.SPEC_SHEET_ARCHIVED]
    __schema_version__: ClassVar[str] = "1.0.0"


class SpecSheetDeprecatedEvent(SpecSheetEvent):
    """Event for spec sheet deprecation."""

    event_type: Literal[EventType.SPEC_SHEET_DEPRECATED] = EventType
