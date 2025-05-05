"""Event models and schema registration for Apache Pulsar."""
import os
# Import existing models
from src.services.shared.models.events.events import (
    BaseEvent,
    CodeGenerationRequestedEvent,
    CodeGenerationCompletedEvent,
    CodeGenerationFailedEvent,
    KnowledgeQueryRequestedEvent,
    KnowledgeQueryCompletedEvent,
    KnowledgeUpdatedEvent,
    SpecSheetCreatedEvent,
    SpecSheetUpdatedEvent,
    SpecSheetDeletedEvent,
    # Add other event types
    EventType,
    EventPriority,
    SpecSheetEvent,
    SpecInstanceEvent,
    # Add payload models
    CodeGenerationRequestPayload,
    CodeGenerationCompletedPayload,
    CodeGenerationFailedPayload,
    KnowledgeQueryPayload,
    KnowledgeQueryCompletedPayload,
    KnowledgeUpdatedPayload
)
from src.services.shared.models.event_avro import EventAvro
from infra.registration.event_converter import EventConverter
from infra.registration.schema_registry import SchemaRegistryClient, register_pydantic_model
from src.services.shared.models.schema_init import load_avro_schema, get_schema_registry
from src.services.shared.models.base import (
    BaseMessage,
    ProcessingMode,
    TaskStatus,
    TaskPriority,
    EventPriority as BaseEventPriority
)
from src.services.shared.models.base import BaseComponent, ConfigurableComponent
from src.services.shared.models.enums import (
    Components,
    Events,
    Database,
    DatabaseConfig,
    Paths,
    Techniques,
    ModelConfig,
    Metrics,
    MetricsConfig,
    DeploymentConfig,
    ErrorCodes,
    Constants
)
from src.services.shared.models.messages import (
    SynthesisStrategy,
    DisclosureLevel,
    ProjectStatus,
    ProjectType,
    Pattern,
    IntentAnalysis,
    ErrorResponse,
    QueryResponse,
    QueryRequest,
    PatternResponse,
    HealthResponse,
    PatternListResponse,
    PatternCreateRequest,
    Task,
    SymbolicTestResult,
    InterfaceVerificationResult
)

# Import from types.py
from src.services.shared.models.types import (
    VerificationResult,
    FormalSpecification,
    VerificationReport,
    ConstraintRelaxationRequest,
    ConstraintRelaxationResponse
)

# Import from specifications.py
from src.services.shared.models.specifications import (
    FieldDefinition,
    SectionDefinition,
    SpecSheetDefinition,
    FieldValue,
    SectionValues,
    SpecSheet,
    SpecSheetGenerationRequestMessage,
    SpecSheetCompletionRequestMessage,
    TemplateRequest,
    TemplateResponse
)

# Import from projects.py
from src.services.shared.models.projects import (
    TechnologyStack,
    Requirement,
    ProjectCreatedMessage,
    ProjectAnalysisRequestMessage
)

# Import telemetry manager if needed
from src.services.shared.models.telemetry import TelemetryManager

# Registry client instance
_schema_registry = None


def init_schema_registry(url: str, auth_token: str = None):
    """Initialize the schema registry client."""
    global _schema_registry
    _schema_registry = SchemaRegistryClient(url=url, auth_token=auth_token)
    return register_event_schemas()


def register_event_schemas():
    """Register all event schemas with the schema registry."""
    if _schema_registry is None:
        raise RuntimeError("Schema registry client not initialized")

    registered_schemas = {}

    # List of event models to register
    event_models = [
        BaseEvent,
        CodeGenerationRequestedEvent,
        CodeGenerationCompletedEvent,
        CodeGenerationFailedEvent,
        KnowledgeQueryRequestedEvent,
        KnowledgeQueryCompletedEvent,
        KnowledgeUpdatedEvent,
        SpecSheetCreatedEvent,
        SpecSheetUpdatedEvent,
        SpecSheetDeletedEvent
    ]

    for model_class in event_models:
        try:
            subject = f"{model_class.__name__}-value"
            schema_id = register_pydantic_model(model_class, _schema_registry, subject)
            registered_schemas[model_class.__name__] = schema_id
        except Exception as e:
            print(f"Failed to register schema for {model_class.__name__}: {e}")

    return registered_schemas


# Convenience serialization functions
def serialize_event(event: BaseEvent) -> bytes:
    """Serialize an event to Avro format."""
    avro_event = EventConverter.to_avro(event)
    return avro_event.model_dump_json().encode('utf-8')


def deserialize_event(data: bytes) -> BaseEvent:
    """Deserialize an event from Avro format."""
    avro_data = EventAvro.model_validate_json(data.decode('utf-8'))
    return EventConverter.from_avro(avro_data)


# Export needed symbols
__all__ = [
    # Models
    'BaseEvent',
    'EventType',
    'EventPriority',
    'BaseEventPriority',  # From base.py
    'CodeGenerationRequestedEvent',
    'CodeGenerationCompletedEvent',
    'CodeGenerationFailedEvent',
    'KnowledgeQueryRequestedEvent',
    'KnowledgeQueryCompletedEvent',
    'KnowledgeUpdatedEvent',
    'SpecSheetCreatedEvent',
    'SpecSheetUpdatedEvent',
    'SpecSheetDeletedEvent',
    'SpecSheetEvent',
    'SpecInstanceEvent',
    'EventAvro',
    'BaseMessage',
    'VerificationResult',

    # From types.py
    'FormalSpecification',
    'VerificationReport',
    'ConstraintRelaxationRequest',
    'ConstraintRelaxationResponse',

    # From specifications.py
    'FieldDefinition',
    'SectionDefinition',
    'SpecSheetDefinition',
    'FieldValue',
    'SectionValues',
    'SpecSheet',
    'SpecSheetGenerationRequestMessage',
    'SpecSheetCompletionRequestMessage',
    'TemplateRequest',
    'TemplateResponse',

    # From projects.py
    'TechnologyStack',
    'Requirement',
    'ProjectCreatedMessage',
    'ProjectAnalysisRequestMessage',

    # Payload models
    'CodeGenerationRequestPayload',
    'CodeGenerationCompletedPayload',
    'CodeGenerationFailedPayload',
    'KnowledgeQueryPayload',
    'KnowledgeQueryCompletedPayload',
    'KnowledgeUpdatedPayload',

    # From base.py
    'ProcessingMode',
    'TaskStatus',
    'TaskPriority',

    # From base_component.py
    'BaseComponent',
    'ConfigurableComponent',

    # From enums.py
    'Components',
    'Events',
    'Database',
    'DatabaseConfig',
    'Paths',
    'Techniques',
    'ModelConfig',
    'Metrics',
    'MetricsConfig',
    'DeploymentConfig',
    'ErrorCodes',
    'Constants',

    # From messages.py
    'SynthesisStrategy',
    'DisclosureLevel',
    'ProjectStatus',
    'ProjectType',
    'Pattern',
    'IntentAnalysis',
    'ErrorResponse',
    'QueryResponse',
    'QueryRequest',
    'PatternResponse',
    'HealthResponse',
    'PatternListResponse',
    'PatternCreateRequest',
    'Task',
    'SymbolicTestResult',
    'InterfaceVerificationResult',

    # From telemetry.py
    'TelemetryManager',

    # Functions
    'init_schema_registry',
    'register_event_schemas',
    'get_schema_registry',
    'load_avro_schema',
    'serialize_event',
    'deserialize_event'
]

# Auto-initialize if environment variables are set
if os.environ.get("SCHEMA_REGISTRY_URL") and os.environ.get("AUTO_REGISTER_SCHEMAS", "").lower() == "true":
    try:
        init_schema_registry(
            os.environ.get("SCHEMA_REGISTRY_URL"),
            os.environ.get("SCHEMA_REGISTRY_TOKEN")
        )
        print("Successfully registered event schemas with Schema Registry")
    except Exception as e:
        print(f"Failed to initialize schema registry: {e}")