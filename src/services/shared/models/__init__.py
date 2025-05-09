"""
Event models and schema registration for Apache Pulsar.

This module provides a centralized import point for all models used in the system,
as well as functionality for registering schemas with Apache Pulsar's schema registry.
It handles serialization and deserialization of events and messages using Avro.
"""

import os
from typing import Any, Dict, Type

from infra.registration.schema_registry import convert_pydantic_schema_to_avro
from infra.registration.schema_registry import register_pydantic_model
from infra.registration.schema_registry import SchemaRegistryClient

# Base models
from src.services.shared.models.base import BaseComponent
from src.services.shared.models.base import BaseMessage
from src.services.shared.models.base import ConfigurableComponent
from src.services.shared.models.base import EventPriority as BaseEventPriority
from src.services.shared.models.base import EventType as BaseEventType
from src.services.shared.models.base import
from src.services.shared.models.enums import  Aggregation_Completed_Partial as EventType
# Enumerations
from src.services.shared.models.enums import Components
from src.services.shared.models.enums import Constants
from src.services.shared.models.enums import Database
from src.services.shared.models.enums import DatabaseConfig
from src.services.shared.models.enums import DeploymentConfig
from src.services.shared.models.enums import DisclosureLevel
from src.services.shared.models.enums import ErrorCodes
from src.services.shared.models.enums import Events
from src.services.shared.models.enums import Metrics
from src.services.shared.models.enums import MetricsConfig
from src.services.shared.models.enums import ModelConfig
from src.services.shared.models.enums import Paths
from src.services.shared.models.enums import ProcessingMode
from src.services.shared.models.enums import ProjectStatus
from src.services.shared.models.enums import ProjectType
from src.services.shared.models.enums import SynthesisStrategy
from src.services.shared.models.enums import TaskPriority
from src.services.shared.models.enums import TaskStatus
from src.services.shared.models.enums import Techniques

# Avro event model
from src.services.shared.models.event_avro import EventAvro

# Event models
from src.services.shared.models.events import (
    BaseEvent,  # Base event classes; Event implementations; Payload models
)
from src.services.shared.models.events import CodeGenerationCompletedEvent
from src.services.shared.models.events import CodeGenerationCompletedPayload
from src.services.shared.models.events import CodeGenerationFailedEvent
from src.services.shared.models.events import CodeGenerationFailedPayload
from src.services.shared.models.events import CodeGenerationRequestedEvent
from src.services.shared.models.events import CodeGenerationRequestPayload
from src.services.shared.models.events import EventPriority
from src.services.shared.models.events import EventType
from src.services.shared.models.events import KnowledgeQueryCompletedEvent
from src.services.shared.models.events import KnowledgeQueryCompletedPayload
from src.services.shared.models.events import KnowledgeQueryPayload
from src.services.shared.models.events import KnowledgeQueryRequestedEvent
from src.services.shared.models.events import KnowledgeUpdatedEvent
from src.services.shared.models.events import KnowledgeUpdatedPayload
from src.services.shared.models.events import SpecInstanceEvent
from src.services.shared.models.events import SpecSheetCreatedEvent
from src.services.shared.models.events import SpecSheetDeletedEvent
from src.services.shared.models.events import SpecSheetEvent
from src.services.shared.models.events import SpecSheetUpdatedEvent

# Message models
from src.services.shared.models.messages import ErrorResponse
from src.services.shared.models.messages import HealthResponse
from src.services.shared.models.messages import IntentAnalysis
from src.services.shared.models.messages import InterfaceVerificationResult
from src.services.shared.models.messages import Pattern
from src.services.shared.models.messages import PatternCreateRequest
from src.services.shared.models.messages import PatternListResponse
from src.services.shared.models.messages import PatternResponse
from src.services.shared.models.messages import QueryRequest
from src.services.shared.models.messages import QueryResponse
from src.services.shared.models.messages import SymbolicTestResult
from src.services.shared.models.messages import Task

# Project models
from src.services.shared.models.projects import ProjectAnalysisRequestMessage
from src.services.shared.models.projects import ProjectCreatedMessage
from src.services.shared.models.projects import Requirement
from src.services.shared.models.projects import TechnologyStack
from src.services.shared.models.specifications import (
    SpecSheetCompletionRequestMessage,
)
from src.services.shared.models.specifications import (
    SpecSheetGenerationRequestMessage,
)

# Specifications
from src.services.shared.models.specifications import FieldDefinition
from src.services.shared.models.specifications import FieldValue
from src.services.shared.models.specifications import SectionDefinition
from src.services.shared.models.specifications import SectionValues
from src.services.shared.models.specifications import SpecSheet
from src.services.shared.models.specifications import SpecSheetDefinition

# Telemetry
from src.services.shared.models.telemetry import TelemetryManager

# Types
from src.services.shared.models.types import ConstraintRelaxationRequest
from src.services.shared.models.types import ConstraintRelaxationResponse
from src.services.shared.models.types import FormalSpecification
from src.services.shared.models.types import VerificationReport
from src.services.shared.models.types import VerificationResult


# Schema registry


# Registry client instance
_schema_registry = None


def init_schema_registry(url: str, auth_token: str = None):
    """
    Initialize the schema registry client.

    Args:
        url: The base URL of the schema registry
        auth_token: Optional authentication token

    Returns:
        A dictionary mapping schema names to their IDs
    """
    global _schema_registry
    _schema_registry = SchemaRegistryClient(url=url, auth_token=auth_token)
    return register_event_schemas()


def register_event_schemas():
    """
    Register all event schemas with the schema registry.

    Returns:
        A dictionary mapping schema names to their IDs

    Raises:
        RuntimeError: If the schema registry is not initialized
    """
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
        SpecSheetDeletedEvent,
    ]

    for model_class in event_models:
        try:
            subject = f"{model_class.__name__}-value"
            schema_id = register_pydantic_model(model_class, SchemaRegistryClient, subject)
            registered_schemas[model_class.__name__] = schema_id
        except Exception as e:
            print(f"Failed to register schema for {model_class.__name__}: {e}")

    return registered_schemas


def serialize_event(event: BaseEvent) -> bytes:
    """
    Serialize an event to Avro format.

    Args:
        event: The event to serialize

    Returns:
        The serialized event as bytes
    """
    avro_event = convert_pydantic_schema_to_avro(pydantic_schema=).to_avro(event)
    return avro_event.model_dump_json().encode("utf-8")


def deserialize_event(data: bytes) -> BaseEvent:
    """
    Deserialize an event from Avro format.

    Args:
        data: The serialized event data

    Returns:
        The deserialized event
    """
    avro_data = EventAvro.model_validate_json(data.decode("utf-8"))
    return convert_pydantic_schema_to_avro(   ).from_avro(avro_data)


# Export symbols
__all__ = [
    # Models
    "BaseEvent",
    "EventType",
    "EventPriority",
    "BaseComponent",
    "DatabaseConfig",
    "DeploymentConfig",
    "DisclosureLevel",
    "ErrorCodes",
    "Events",
    "Metrics",
    "MetricsConfig",
    "ModelConfig",
    "Paths",

    "ProcessingMode",
    "ProjectStatus",
    "ProjectType",
    "SynthesisStrategy",
    "TaskPriority",
    "TaskStatus",
    "Techniques",
    "CodeGenerationRequestPayload",
    "CodeGenerationCompletedPayload",
    "CodeGenerationFailedPayload",
    "KnowledgeQueryPayload",
    "KnowledgeUpdatedPayload",
    "QueryRequest",
    "QueryResponse",
    "PatternCreateRequest",
    "PatternListResponse",
    "PatternResponse",
    "ConfigurableComponent",
    "TelemetryManager",
    "BaseEventPriority",
    "CodeGenerationRequestedEvent",
    "CodeGenerationCompletedEvent",
    "CodeGenerationFailedEvent",
    "KnowledgeQueryRequestedEvent",
    "KnowledgeQueryCompletedEvent",
    "KnowledgeUpdatedEvent",
    "SpecSheetCreatedEvent",
    "SpecSheetUpdatedEvent",
    "SpecSheetDeletedEvent",
    "SpecSheetEvent",
    "SpecInstanceEvent",
    "EventAvro",
    "BaseMessage",
    "VerificationResult",
    # From types.py
    "FormalSpecification",
    "VerificationReport",
    "ConstraintRelaxationRequest",
    "ConstraintRelaxationResponse",
    # From specifications.py
    "FieldDefinition",
    "SectionDefinition",
    "SpecSheetDefinition",
    "FieldValue",
    "SectionValues",
    "SpecSheet",
    "SpecSheetGenerationRequestMessage",
    "SpecSheetCompletionRequestMessage",
    # From projects.py
    "TechnologyStack",
    "Requirement",
    "ProjectCreatedMessage",
]
