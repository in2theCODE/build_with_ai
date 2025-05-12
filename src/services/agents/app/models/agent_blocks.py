# agent_template_service/models/agent_blocks.py
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class BlockType(str, Enum):
    PERCEPTION = "perception"
    MEMORY = "memory"
    REASONING = "reasoning"
    ACTION = "action"
    COORDINATION = "coordination"
    INTEGRATION = "integration"
    EVALUATION = "evaluation"


class InputParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


class OutputParameter(BaseModel):
    name: str
    type: str
    description: str


class BlockMetadata(BaseModel):
    name: str
    description: str
    version: str
    author: Optional[str] = None
    tags: List[str] = []
    language: str = "python"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AgentBlock(BaseModel):
    id: str
    type: BlockType
    metadata: BlockMetadata
    inputs: List[InputParameter]
    outputs: List[OutputParameter]
    implementation: str
    dependencies: List[str] = []
    requirements: List[str] = []
    events_emitted: List[str] = []
    events_handled: List[str] = []


class AgentTemplate(BaseModel):
    id: str
    name: str
    description: str
    blocks: List[str]  # References to block IDs
    connections: List[Dict[str, str]]  # Source block to target block
    metadata: Dict[str, Any] = {}
    version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AgentInstance(BaseModel):
    id: str
    template_id: str
    name: str
    description: str
    configuration: Dict[str, Any] = {}
    status: str = "created"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
