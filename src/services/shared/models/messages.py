from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum, auto
import uuid


from pydantic import BaseModel, Field,  ConfigDict
from pydantic.generics import GenericModel

from src.services.shared.models.base import BaseMessage, ProcessingMode, TaskStatus, TaskPriority


class ProjectType(str, Enum):
    """Project types."""
    WEB_APP = "WEB_APP"
    MOBILE_APP = "MOBILE_APP"
    API_SERVICE = "API_SERVICE"
    LIBRARY = "LIBRARY"
    CLI_TOOL = "CLI_TOOL"


class ProjectStatus(str, Enum):
    """Project status."""
    INITIALIZING = "INITIALIZING"
    ANALYZING = "ANALYZING"
    SPEC_SHEETS_GENERATED = "SPEC_SHEETS_GENERATED"
    SPEC_SHEETS_COMPLETED = "SPEC_SHEETS_COMPLETED"
    GENERATING_CODE = "GENERATING_CODE"
    CODE_GENERATED = "CODE_GENERATED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"




class SynthesisStrategy(str, Enum):
    """Types of synthesis strategies."""
    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    ENUMERATIVE = "enumerative"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    CONSTRAINT_BASED = "constraint_based"
    EXAMPLE_GUIDED = "example_guided"
    NEURAL_GUIDED = "neural_guided"


class DisclosureLevel(Enum):
    """Progressive disclosure levels for code synthesis."""
    HIGH_LEVEL = auto()  # Only signatures and high-level descriptions
    MID_LEVEL = auto()  # Implementation with simplified details
    DETAILED = auto()  # Complete implementation with all details


class Pattern(BaseModel):
    """Pattern match result"""
    id: str = Field(..., description="Pattern identifier")
    text: str = Field(..., description="Pattern text")
    score: float = Field(..., description="Match confidence score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Pattern metadata")

    model_config = ConfigDict(frozen=True)


class IntentAnalysis(BaseModel):
    """Intent analysis results"""
    processing_mode: ProcessingMode = Field(..., description="Selected processing mode")
    complexity_score: float = Field(..., description="Query complexity score")
    confidence: float = Field(..., description="Analysis confidence score")
    pattern_match: Optional[Pattern] = Field(None, description="Matching pattern if found")
    elapsed_ms: int = Field(..., description="Analysis time in milliseconds")

    model_config = ConfigDict(frozen=True)


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    request_id: Optional[str] = Field(None, description="Request identifier")

    model_config = ConfigDict(frozen=True)


class QueryResponse(BaseModel):
    """Response to query request"""
    request_id: str = Field(..., description="Request identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),
                                description="Response timestamp"
    )
    intent_analysis: IntentAnalysis = Field(..., description="Query intent analysis")
    status: str = Field("success", description="Response status")
    events_published: List[str] = Field(default_factory=list, description="List of published event topics")

    model_config = ConfigDict(frozen=True)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field("healthy", description="Service status")
    version: str = Field(..., description="Service version")
    uptime_seconds: int = Field(..., description="Service uptime in seconds")
    database_status: str = Field(..., description="Database connection status")
    embedding_service_status: str = Field(..., description="Embedding service status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )

    model_config = ConfigDict(frozen=True)


class PatternCreateRequest(BaseModel):
    """Request to create a new pattern"""
    text: str = Field(..., description="Pattern text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pattern metadata")
    embedding: Optional[List[float]] = Field(None, description="Optional pre-computed embedding")

    model_config = ConfigDict(frozen=True)


class PatternResponse(BaseModel):
    """Response after pattern creation"""
    id: str = Field(..., description="Pattern identifier")
    text: str = Field(..., description="Pattern text")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),)
    metadata: Dict[str, Any] = Field(..., description="Pattern metadata")

    model_config = ConfigDict(frozen=True)


class PatternListResponse(BaseModel):
    """Response with list of patterns"""
    patterns: List[PatternResponse] = Field(..., description="List of patterns")
    count: int = Field(..., description="Total pattern count")
    page: Optional[int] = Field(None, description="Current page number")
    total_pages: Optional[int] = Field(None, description="Total number of pages")

    model_config = ConfigDict(frozen=True)


class QueryRequest(BaseModel):
    """Query request from user or system"""
    prompt: str = Field(..., description="User query text")
    system_message: Optional[str] = Field(None, description="System context message")
    require_deliberation: bool = Field(False, description="Force query to deliberative path")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    session_id: Optional[str] = Field(None, description="Session identifier")
    request_id: Optional[str] = Field(None, description="Unique request identifier")

    model_config = ConfigDict(frozen=True)


class SymbolicTestResult(BaseModel):
    """Result of symbolic execution testing."""
    passed: bool = Field(..., description="Whether all symbolic tests passed")
    failing_tests: List[Dict[str, Any]] = Field(default_factory=list, description="Failing test cases")
    total_tests: int = Field(0, description="Total number of tests executed")
    time_taken: float = Field(0.0, description="Time taken in seconds")

    model_config = ConfigDict(frozen=True)


class InterfaceVerificationResult(BaseModel):
    """Result of interface contract verification."""
    is_valid: bool = Field(..., description="Whether the interface contracts are satisfied")
    failures: List[Dict[str, Any]] = Field(default_factory=list, description="Contract violations")
    time_taken: float = Field(0.0, description="Time taken in seconds")

    model_config = ConfigDict(frozen=True)


class Task(BaseMessage):
    """Task object containing all metadata needed for processing"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Task creation timestamp")
    prompt: str = Field("", description="Task prompt or description")
    system_message: Optional[str] = Field(None, description="System context message")
    processing_mode: ProcessingMode = Field(ProcessingMode.REACTIVE, description="Processing mode")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Task priority")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Task status")
    estimated_processing_time: float = Field(0.0, description="Estimated processing time in seconds")
    parent_task_id: Optional[str] = Field(None, description="Parent task ID for subtasks")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task context")
    user_id: Optional[str] = Field(None, description="User identifier")
    service_route: str = Field("", description="The service that should handle this task")
    workflow_id: Optional[str] = Field(None, description="Workflow identifier")
    phase: Optional[str] = Field(None, description="Workflow phase")
    project_id: Optional[str] = Field(None, description="Project identifier")

    model_config = ConfigDict(
        frozen=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat(),
            Enum: lambda v: v.value
        }
    )