"""
Neural Code Generator - Spec Sheet to Code Generation Platform

This package provides a comprehensive framework for generating code from specification
sheets, with full support for Apache Pulsar event-driven architecture, Avro schema
serialization, and advanced neural code generation techniques.

The system uses a modular design with event-driven communication between components,
allowing for scalable and extensible code generation capabilities.

Core Components:
- Avro-compatible event and message models
- Apache Pulsar integration with schema registry
- Specification sheet parsing and processing
- Neural code generation with various strategies
- Knowledge base integration for improved code synthesis
- Constraint relaxation for handling complex requirements

This package serves as the shared model library used across all system components.
"""

# Version information
__version__ = "1.0.0"

# First, import base classes and enums as they are dependencies for other modules
from .base import (
    AvroBaseModel,
    AvroBase,
    EventPayload,
    BaseEvent,
    BaseMessage,
    BaseComponent,
    ConfigurableComponent,
    )

from .enums import (
    Components,
    Events,
    EventPriority,
    EventType,
    Database,
    Techniques,
    ModelConfig,
    Metrics,
    ErrorCodes,
    Constants,
    ProcessingMode,
    TaskStatus,
    TaskPriority,
    ProjectType,
    ProjectStatus,
    SynthesisStrategy,
    DisclosureLevel,
    HealthStatus,
    VerificationResult,
    ContextType,
    SynapseState,
    ActivationFunction,
    LearningStrategy,
    EvolutionMechanism,
    is_failure_event,
    get_retry_event_type,
    )

# Import constants
from .constants import (
    DatabaseConfig,
    Paths,
    ModelConfig,
    MetricsConfig,
    DeploymentConfig,
    Constants,
    )

# Import event-related classes
from .event_avro import EventAvro

# Import event payload models
from .events import (
    CodeGenerationRequestPayload,
    CodeGenerationCompletedPayload,
    CodeGenerationFailedPayload,
    KnowledgeQueryPayload,
    KnowledgeQueryCompletedPayload,
    KnowledgeUpdatedPayload,
    SpecSheetEvent,
    SpecInstanceEvent,
    CodeGenerationRequestedEvent,
    CodeGenerationCompletedEvent,
    CodeGenerationFailedEvent,
    KnowledgeQueryRequestedEvent,
    KnowledgeQueryCompletedEvent,
    KnowledgeUpdatedEvent,
    ContextNodeActivatedPayload,
    SynapseStateChangedPayload,
    EvolutionEventPayload,
    )

# Import domain-specific events
from .domain import (
    CodeGenerationRequestedEvent,
    CodeGenerationCompletedEvent,
    CodeGenerationFailedEvent,
    KnowledgeQueryRequestedEvent,
    KnowledgeQueryCompletedEvent,
    KnowledgeUpdatedEvent,
    SpecSheetHasBeenCreated,
    SpecSheetUpdatedEvent,
    SpecSheetDeletedEvent,
    SpecSheetPublishedEvent,
    SpecSheetArchivedEvent,
    SpecSheetDeprecatedEvent,
    )

# Import message schemas
from .messages import (
    Pattern,
    IntentAnalysis,
    ErrorResponse,
    QueryResponse,
    HealthResponse,
    PatternCreateRequest,
    PatternResponse,
    PatternListResponse,
    QueryRequest,
    SymbolicTestResult,
    InterfaceVerificationResult,
    Task,
    )

# Import neural mesh models
from .neural_mesh import (
    ContextNode,
    CodePatternNode,
    KnowledgeNode,
    MetricNode,
    EvolutionNode,
    GlobalContextState,
    ContextActivation,
    Synapse,
    PathwaySegment,
    Pathway,
    SynapseActivity,
    Template,
    EvolutionEvent,
    EmergentPattern,
    FitnessEvaluation,
    )

# Import project management models
from .projects import (
    TechnologyStack,
    Requirement,
    ProjectCreatedMessage,
    ProjectAnalysisRequestMessage,
    )

# Import schema registry client
from .schema_registry import (
    SchemaRegistryClient,
    register_pydantic_model,
    convert_pydantic_schema_to_avro,
    map_pydantic_type_to_avro,
    )

# Import specification models
from .specifications import (
    FieldDefinition,
    SectionDefinition,
    SpecSheetDefinition,
    FieldValue,
    SectionValues,
    SpecSheet,
    SpecSheetGenerationRequestMessage,
    SpecSheetCompletionRequestMessage,
    SpecSheetDefinitionRequest,
    SpecSheetDefinitionResponse,
    )

# Import synthesis models
from .synthesis import SynthesisResult, SynthesisStrategyType

# Import telemetry manager
from .telemetry import TelemetryManager

# Import types for constraint relaxation
from .types import (
    VerificationResult,
    FormalSpecification,
    VerificationReport,
    ConstraintRelaxationRequest,
    ConstraintRelaxationResponse,
    )

# Import Z3 utilities
from .z3_utils import (
    get_model,
    extract_variables,
    is_satisfiable,
    get_unsat_core,
    optimize_constraints,
    relax_constraint,
    )

# Import validation utilities
from .validator import (
    ValidationResult,
    ValidationError,
    Validator,
    TypeValidator,
    StringValidator,
    NumberValidator,
    validate_input,
    )


# Define what gets imported with `from package import *`
__all__ = [
    # Version info
    "__version__",

    # Base classes
    "AvroBaseModel", "AvroBase", "EventPayload", "BaseEvent", "BaseMessage",
    "BaseComponent", "ConfigurableComponent",

    # Enums
    "Components", "Events", "EventPriority", "EventType", "Database",
    "Techniques", "ModelConfig", "Metrics", "ErrorCodes", "Constants",
    "ProcessingMode", "TaskStatus", "TaskPriority", "ProjectType",
    "ProjectStatus", "SynthesisStrategy", "DisclosureLevel", "HealthStatus",
    "VerificationResult", "ContextType", "SynapseState", "ActivationFunction",
    "LearningStrategy", "EvolutionMechanism", "is_failure_event", "get_retry_event_type",

    # Constants
    "DatabaseConfig", "Paths", "ModelConfig", "MetricsConfig", "DeploymentConfig",

    # Events
    "EventAvro", "CodeGenerationRequestPayload", "CodeGenerationCompletedPayload",
    "CodeGenerationFailedPayload", "KnowledgeQueryPayload", "KnowledgeQueryCompletedPayload",
    "KnowledgeUpdatedPayload", "SpecSheetEvent", "SpecInstanceEvent",
    "CodeGenerationRequestedEvent", "CodeGenerationCompletedEvent", "CodeGenerationFailedEvent",
    "KnowledgeQueryRequestedEvent", "KnowledgeQueryCompletedEvent", "KnowledgeUpdatedEvent",
    "SpecSheetHasBeenCreated", "SpecSheetUpdatedEvent", "SpecSheetDeletedEvent",
    "SpecSheetPublishedEvent", "SpecSheetArchivedEvent", "SpecSheetDeprecatedEvent",
    "ContextNodeActivatedPayload", "SynapseStateChangedPayload", "EvolutionEventPayload",

    # Messages
    "Pattern", "IntentAnalysis", "ErrorResponse", "QueryResponse", "HealthResponse",
    "PatternCreateRequest", "PatternResponse", "PatternListResponse", "QueryRequest",
    "SymbolicTestResult", "InterfaceVerificationResult", "Task",

    # Neural mesh
    "ContextNode", "CodePatternNode", "KnowledgeNode", "MetricNode", "EvolutionNode",
    "GlobalContextState", "ContextActivation", "Synapse", "PathwaySegment", "Pathway",
    "SynapseActivity", "Template", "EvolutionEvent", "EmergentPattern", "FitnessEvaluation",

    # Projects
    "TechnologyStack", "Requirement", "ProjectCreatedMessage", "ProjectAnalysisRequestMessage",

    # Schema registry
    "SchemaRegistryClient", "register_pydantic_model", "convert_pydantic_schema_to_avro",
    "map_pydantic_type_to_avro",

    # Specifications
    "FieldDefinition", "SectionDefinition", "SpecSheetDefinition", "FieldValue",
    "SectionValues", "SpecSheet", "SpecSheetGenerationRequestMessage",
    "SpecSheetCompletionRequestMessage", "SpecSheetDefinitionRequest",
    "SpecSheetDefinitionResponse",

    # Synthesis
    "SynthesisResult", "SynthesisStrategyType",

    # Telemetry
    "TelemetryManager",

    # Types
    "FormalSpecification", "VerificationReport", "ConstraintRelaxationRequest",
    "ConstraintRelaxationResponse",

    # Z3 utils
    "get_model", "extract_variables", "is_satisfiable", "get_unsat_core",
    "optimize_constraints", "relax_constraint",

    # Validation
    "ValidationResult", "ValidationError", "Validator", "TypeValidator",
    "StringValidator", "NumberValidator", "validate_input",
    ]