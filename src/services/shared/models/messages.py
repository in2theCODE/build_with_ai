"""
Messaging models for communication between system components.

This module defines the message schemas used for communication between
the system's components. These messages are designed to be serializable
with Avro and compatible with Apache Pulsar as the messaging transport.

Messages in this module are typically used for request/response patterns,
while events (defined in events.py) are used for broadcasting changes
in system state.

Classes:
    Pattern: Represents a pattern match result
    IntentAnalysis: Results of query intent analysis
    ErrorResponse: Standard error response format
    QueryResponse: Response to a query request
    HealthResponse: Health check response
    PatternCreateRequest: Request to create a new pattern
    PatternResponse: Response to pattern creation
    PatternListResponse: Response with a list of patterns
    QueryRequest: Query request from user or system
    SymbolicTestResult: Result of symbolic execution testing
    InterfaceVerificationResult: Result of interface contract verification
    Task: Task object with metadata for processing
"""

# src/services/shared/models/messages.py
from datetime import datetime
from datetime import timezone
from typing import Any, Dict, List, Optional
import uuid

from pydantic import Field

from .base import AvroBaseModel
from .base import BaseMessage
from .enums import ProcessingMode
from .enums import TaskPriority
from .enums import TaskStatus


class Pattern(AvroBaseModel):
    """Pattern match result"""

    id: str = Field(..., description="Pattern identifier")
    text: str = Field(..., description="Pattern text")
    score: float = Field(..., description="Match confidence score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Pattern metadata")


class IntentAnalysis(AvroBaseModel):
    """Intent analysis results"""

    processing_mode: ProcessingMode = Field(..., description="Selected processing mode")
    complexity_score: float = Field(..., description="Query complexity score")
    confidence: float = Field(..., description="Analysis confidence score")
    pattern_match: Optional[Pattern] = Field(None, description="Matching pattern if found")
    elapsed_ms: int = Field(..., description="Analysis time in milliseconds")


class ErrorResponse(AvroBaseModel):
    """Error response"""

    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    request_id: Optional[str] = Field(None, description="Request identifier")


class QueryResponse(BaseMessage):
    """Response to query request"""

    request_id: str = Field(..., description="Request identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    intent_analysis: IntentAnalysis = Field(..., description="Query intent analysis")
    status: str = Field("success", description="Response status")
    events_published: List[str] = Field(
        default_factory=list, description="List of published event topics"
    )


class HealthResponse(AvroBaseModel):
    """Health check response"""

    status: str = Field("healthy", description="Service status")
    version: str = Field(..., description="Service version")
    uptime_seconds: int = Field(..., description="Service uptime in seconds")
    database_status: str = Field(..., description="Database connection status")
    embedding_service_status: str = Field(..., description="Embedding service status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp"
    )


class PatternCreateRequest(BaseMessage):
    """Request to create a new pattern"""

    text: str = Field(..., description="Pattern text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pattern metadata")
    embedding: Optional[List[float]] = Field(None, description="Optional pre-computed embedding")


class PatternResponse(BaseMessage):
    """Response after pattern creation"""

    id: str = Field(..., description="Pattern identifier")
    text: str = Field(..., description="Pattern text")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    metadata: Dict[str, Any] = Field(..., description="Pattern metadata")


class PatternListResponse(BaseMessage):
    """Response with list of patterns"""

    patterns: List[PatternResponse] = Field(..., description="List of patterns")
    count: int = Field(..., description="Total pattern count")
    page: Optional[int] = Field(None, description="Current page number")
    total_pages: Optional[int] = Field(None, description="Total number of pages")


class QueryRequest(BaseMessage):
    """Query request from user or system"""

    prompt: str = Field(..., description="User query text")
    system_message: Optional[str] = Field(None, description="System context message")
    require_deliberation: bool = Field(False, description="Force query to deliberative path")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    session_id: Optional[str] = Field(None, description="Session identifier")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class SymbolicTestResult(AvroBaseModel):
    """Result of symbolic execution testing."""

    passed: bool = Field(..., description="Whether all symbolic tests passed")
    failing_tests: List[Dict[str, Any]] = Field(
        default_factory=list, description="Failing test cases"
    )
    total_tests: int = Field(0, description="Total number of tests executed")
    time_taken: float = Field(0.0, description="Time taken in seconds")


class InterfaceVerificationResult(AvroBaseModel):
    """Result of interface contract verification."""

    is_valid: bool = Field(..., description="Whether the interface contracts are satisfied")
    failures: List[Dict[str, Any]] = Field(default_factory=list, description="Contract violations")
    time_taken: float = Field(0.0, description="Time taken in seconds")


class Task(BaseMessage):
    """Task object containing all metadata needed for processing"""

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Task creation timestamp"
    )
    prompt: str = Field("", description="Task prompt or description")
    system_message: Optional[str] = Field(None, description="System context message")
    processing_mode: ProcessingMode = Field(ProcessingMode.REACTIVE, description="Processing mode")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Task priority")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Task status")
    estimated_processing_time: float = Field(
        0.0, description="Estimated processing time in seconds"
    )
    parent_task_id: Optional[str] = Field(None, description="Parent task ID for subtasks")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task context")
    user_id: Optional[str] = Field(None, description="User identifier")
    service_route: str = Field("", description="The service that should handle this task")
    workflow_id: Optional[str] = Field(None, description="Workflow identifier")
    phase: Optional[str] = Field(None, description="Workflow phase")
    project_id: Optional[str] = Field(None, description="Project identifier")

    class Meta:
        """Avro schema metadata."""

        namespace = "tasks"
        schema_name = "Task"
