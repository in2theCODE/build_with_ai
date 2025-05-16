"""
Shared package for common app, utilities, and services.
"""

import validation

# Version information
__version__ = "1.0.0"

from src.services.shared.models import Components as ComponentType
from src.services.shared.models import DisclosureLevel
from src.services.shared.models import ProjectAnalysisRequestMessage
from src.services.shared.models import ProjectCreatedMessage
from src.services.shared.models import SynthesisStrategy
from src.services.shared.models import VerificationResult

# Import directly from within the shared package
from src.services.shared.models.base import BaseComponent
from src.services.shared.models.base import BaseMessage
from src.services.shared.models.base import ConfigurableComponent
from src.services.shared.models.manager_models import Requirement
from src.services.shared.models.manager_models import TechnologyStack
from src.services.shared.models.messages import InterfaceVerificationResult
from src.services.shared.models.messages import ProjectStatus
from src.services.shared.models.messages import ProjectType
from src.services.shared.models.messages import SymbolicTestResult
from src.services.shared.models.projects import ProjectAnalysisRequestMessage
from src.services.shared.models.projects import ProjectCreatedMessage
from src.services.shared.models.projects import ProjectType
from src.services.shared.models.specifications import (
    SpecSheetCompletionRequestMessage,
)
from src.services.shared.models.specifications import (
    SpecSheetGenerationRequestMessage,
)
from src.services.shared.models.specifications import FieldDefinition
from src.services.shared.models.specifications import FieldValue
from src.services.shared.models.specifications import SectionDefinition
from src.services.shared.models.specifications import SectionValues
from src.services.shared.models.specifications import SpecSheet
from src.services.shared.models.specifications import SpecSheetDefinition
from src.services.shared.models.specifications import TemplateRequest
from src.services.shared.models.specifications import TemplateResponse
from src.services.shared.models.synthesis import SynthesisResult
from src.services.shared.models.synthesis import SynthesisStrategyType
from src.services.shared.models.types import FormalSpecification


# Expose all schemas for easy importing
__all__ = [
    # Base services
    "BaseComponent", "ConfigurableComponent",

    # Base
    "BaseMessage", "dataclass_to_dict",

    # Enums
    "ProjectType", "ProjectStatus", "VerificationResult",
    "SynthesisStrategy", "DisclosureLevel", "ComponentType",

    # Core app
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