"""
Project management models for the code generation system.

This module defines the data models related to projects in the system.
A project represents a collection of related code generation tasks and
their specifications.

Classes:
    TechnologyStack: Technology stack for a project
    Requirement: Project requirement
    ProjectCreatedMessage: Message sent when a project is created
    ProjectAnalysisRequestMessage: Message requesting project requirements analysis
"""

from datetime import datetime
from datetime import timezone
from enum import Enum
from typing import List
import uuid

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .base import BaseMessage
from .enums import ProjectType


class TechnologyStack(BaseModel):
    """Technology stack for a project."""

    languages: List[str] = Field(default_factory=list, description="Programming languages")
    frameworks: List[str] = Field(default_factory=list, description="Frameworks")
    databases: List[str] = Field(default_factory=list, description="Databases")
    frontend: List[str] = Field(default_factory=list, description="Frontend technologies")
    backend: List[str] = Field(default_factory=list, description="Backend technologies")
    infrastructure: List[str] = Field(
        default_factory=list, description="Infrastructure technologies"
    )

    model_config = ConfigDict(frozen=True, json_encoders={Enum: lambda v: v.value})


class Requirement(BaseModel):
    """Project requirement."""

    id: str = Field(..., description="Requirement identifier")
    description: str = Field(..., description="Requirement description")
    category: str = Field("FUNCTIONAL", description="Requirement category")
    priority: str = Field("MEDIUM", description="Requirement priority")

    model_config = ConfigDict(frozen=True, json_encoders={Enum: lambda v: v.value})


class ProjectCreatedMessage(BaseMessage):
    """Message sent when a project is created."""

    project_id: str = Field(..., description="Project identifier")
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    project_type: ProjectType = Field(..., description="Project type")
    technology_stack: TechnologyStack = Field(..., description="Technology stack")
    requirements: List[Requirement] = Field(
        default_factory=list, description="Project requirements"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp",
    )

    model_config = ConfigDict(frozen=True)


class ProjectAnalysisRequestMessage(BaseMessage):
    """Message requesting project requirements analysis."""

    project_id: str = Field(..., description="Project identifier")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Request identifier"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Request timestamp",
    )

    model_config = ConfigDict(frozen=True)
