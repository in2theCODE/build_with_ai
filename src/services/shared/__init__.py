"""
Shared package for common models, utilities, and services.
"""

from validation import

# Version information
__version__ = "1.0.0"

# Import directly from within the shared package
from src.services.shared.models.base import (
    BaseComponent,
    ConfigurableComponent,
)


from src.services.shared.models import (
    ProjectAnalysisRequestMessage,
    VerificationResult,
    SynthesisStrategy,
    DisclosureLevel,
    Components as ComponentType, ProjectCreatedMessage
)


from src.services.shared.models.base import  BaseMessage
from src.services.shared.models.messages import  (
    ProjectType,
    ProjectStatus,
    SymbolicTestResult,
    InterfaceVerificationResult,
)


from src.services.shared.models.specifications import (
    TemplateRequest,
    TemplateResponse,
    FieldDefinition,
    SectionDefinition,
    SpecSheetDefinition,
    FieldValue,
    SectionValues,
    SpecSheet,
    SpecSheetGenerationRequestMessage,
    SpecSheetCompletionRequestMessage,
)


from src.services.shared.models.manager_models import (
    TechnologyStack,
    Requirement,
)


from src.services.shared.models.types import (
    FormalSpecification
)


from src.services.shared.models.synthesis import (
    SynthesisResult, SynthesisStrategyType
)

from src.services.shared.models.projects import ProjectCreatedMessage, ProjectType, ProjectAnalysisRequestMessage

# Expose all schemas for easy importing
__all__ = [
    # Base services
    "BaseComponent", "ConfigurableComponent",

    # Base
    "BaseMessage", "dataclass_to_dict",

    # Enums
    "ProjectType", "ProjectStatus", "VerificationResult",
    "SynthesisStrategy", "DisclosureLevel", "ComponentType",

    # Core models
    "FormalSpecification", "SynthesisResult",
    "SymbolicTestResult", "InterfaceVerificationResult",

    # Project
    "TechnologyStack", "Requirement", "ProjectCreatedMessage",
    "ProjectAnalysisRequestMessage",

    # Templates
    "TemplateRequest", "TemplateResponse",

    # Spec sheets
    "FieldDefinition", "SectionDefinition", "SpecSheetDefinition",
    "FieldValue", "SectionValues", "SpecSheet",
    "SpecSheetGenerationRequestMessage", "SpecSheetCompletionRequestMessage",

    # Code generation
    "CodeGenerationRequestMessage", "CodeGenerationUpdateMessage",

    # System
    "MetricsMessage", "SystemEventMessage",
]



    CodeGenerationRequestMessage,
    CodeGenerationUpdateMessage,
    MetricsMessage,
    SystemEventMessage