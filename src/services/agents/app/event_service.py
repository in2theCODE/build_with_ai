# agent_template_service/services/event_service.py
import os
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from src.services.shared.client_factory import create_pulsar_client
from src.services.shared.models.base import BaseEvent
from src.services.shared.event_emitter import SecureEventEmitter
from src.services.shared.event_listener import SecureEventListener, EventHandlerType

logger = logging.getLogger(__name__)


class AgentEventService:
    """Service for handling event communication for the agent template system."""

    def __init__(self, pulsar_url: str = "pulsar://pulsar:6650"):
        self.pulsar_url = pulsar_url
        self.secret_key = os.environ.get("PULSAR_SECRET_KEY", "")

        # Initialize event emitter
        self.event_emitter = SecureEventEmitter(
            service_url=pulsar_url,
            secret_key=self.secret_key,
            tenant="public",
            namespace="agent-template",
        )

        # Initialize event listener
        self.event_listener = None
        self.handlers = {}

    async def initialize(self):
        """Initialize event listeners."""
        # Define event types to listen for
        event_types = [
            "agent_block.created",
            "agent_block.updated",
            "agent_template.created",
            "agent_template.updated",
            "agent_instance.created",
            "agent_instance.started",
            "agent_instance.completed",
            "agent_instance.failed",
            "code_generation.completed",  # Listen for code generation events from main system
        ]

        # Create event listener
        self.event_listener = SecureEventListener(
            service_url=self.pulsar_url,
            subscription_name="agent-template-service",
            event_types=event_types,
            secret_key=self.secret_key,
            tenant="public",
            namespace="agent-template",
        )

        # Start listening
        await self.event_listener.start()
        logger.info("Agent event service initialized and listening for events")

    async def emit_event(self, event: BaseEvent) -> None:
        """Emit an event to the event bus."""
        try:
            await self.event_emitter.emit_async(event)
            logger.debug(f"Emitted event {event.event_id} of type {event.event_type}")
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")

    def register_handler(self, event_type: str, handler: EventHandlerType) -> None:
        """Register a handler for a specific event type."""
        if self.event_listener:
            self.event_listener.register_handler(event_type, handler)
            self.handlers[event_type] = handler
            logger.info(f"Registered handler for event type {event_type}")
        else:
            logger.warning(
                f"Event listener not initialized, couldn't register handler for {event_type}"
            )

    async def close(self):
        """Close connections and clean up resources."""
        if self.event_listener:
            await self.event_listener.stop()

        if self.event_emitter:
            self.event_emitter.close()

        logger.info("Agent event service shut down")
