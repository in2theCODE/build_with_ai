# agent_template_service/app/events.py
from typing import Dict, List, Any

from src.services.shared.models.base import BaseEvent
from src.services.shared.models.enums import EventType


# Agent Block Events
class AgentBlockCreatedEvent(BaseEvent):
    event_type: str = EventType.SPEC_SHEET_CREATED.value
    block_id: str
    block_type: str
    block_name: str


class AgentBlockUpdatedEvent(BaseEvent):
    event_type: str = EventType.SPEC_SHEET_UPDATED.value
    block_id: str
    block_type: str
    block_name: str


# Agent Template Events
class AgentTemplateCreatedEvent(BaseEvent):
    event_type: str = "agent_template.created"
    template_id: str
    template_name: str
    blocks: List[str]


class AgentTemplateUpdatedEvent(BaseEvent):
    event_type: str = "agent_template.updated"
    template_id: str
    template_name: str
    blocks: List[str]


# Agent Instance Events
class AgentInstanceCreatedEvent(BaseEvent):
    event_type: str = "agent_instance.created"
    instance_id: str
    template_id: str
    instance_name: str


class AgentInstanceStartedEvent(BaseEvent):
    event_type: str = "agent_instance.started"
    instance_id: str
    template_id: str


class AgentInstanceCompletedEvent(BaseEvent):
    event_type: str = "agent_instance.completed"
    instance_id: str
    template_id: str
    results: Dict[str, Any] = {}


class AgentInstanceFailedEvent(BaseEvent):
    event_type: str = "agent_instance.failed"
    instance_id: str
    template_id: str
    error: str
