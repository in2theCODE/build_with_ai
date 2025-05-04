#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spec Registry Event Adapter Module.

This module adapts the SpecRegistry to work with the event-driven architecture,
enabling event-driven communication between services using Apache Pulsar.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
import os
import uuid

# Import the event bus implementation
from src.services.shared.pulsar.event_emitter import SecureEventEmitter
from src.services.shared.pulsar.event_listener import SecureEventListener, EventHandlerType
from src.services.shared.models.events import (
    BaseEvent, EventType, EventPriority,
    SpecSheetEvent, SpecInstanceEvent
)
from src.services.shared.models.enums import Components
from src.services.shared.models.base import BaseComponent

# Import the spec registry
from src.services.spec_registry.app.spec_registry import SpecRegistry

logger = logging.getLogger(__name__)


class SpecRegistryEventAdapter(BaseComponent):
    """
    Adapter that connects the SpecRegistry system to an event-driven architecture,
    enabling event-driven communication and supporting workflow orchestration.
    """

    def __init__(self, **params):
        """
        Initialize the Spec Registry Event Adapter with parameters.

        Args:
            **params: Configuration parameters
        """
        super().__init__(**params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Extract configuration parameters
        self.pulsar_service_url = self.get_param("pulsar_service_url", "pulsar://localhost:6650")
        self.base_dir = self.get_param("base_dir", "./spec_registry_data")
        self.enable_events = self.get_param("enable_events", True)
        self.secret_key = self.get_param("secret_key", None)
        self.tenant = self.get_param("tenant", "public")
        self.namespace = self.get_param("namespace", "code-generator")

        # Create storage directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)

        # Initialize event system components
        if self.enable_events:
            self._init_event_components()

        # Initialize spec registry
        self._init_spec_registry()

        self.logger.info("SpecRegistryEventAdapter initialized")

    def _init_event_components(self):
        """Initialize event emitter and listener."""
        # Initialize event emitter for sending events
        self.event_emitter = SecureEventEmitter(
            service_url=self.pulsar_service_url,
            secret_key=self.secret_key,
            tenant=self.tenant,
            namespace=self.namespace
        )

        # Initialize event listener for receiving events
        self.event_listener = SecureEventListener(
            service_url=self.pulsar_service_url,
            subscription_name="spec-registry-adapter",
            event_types=[
                # Spec sheet events
                EventType.SPEC_SHEET_CREATED,
                EventType.SPEC_SHEET_UPDATED,
                EventType.SPEC_SHEET_DELETED,
                EventType.SPEC_SHEET_PUBLISHED,
                EventType.SPEC_SHEET_DEPRECATED,
                EventType.SPEC_SHEET_ARCHIVED,

                # Spec instance events
                EventType.SPEC_INSTANCE_CREATED,
                EventType.SPEC_INSTANCE_UPDATED,
                EventType.SPEC_INSTANCE_COMPLETED,
                EventType.SPEC_INSTANCE_VALIDATED,
                EventType.SPEC_INSTANCE_DELETED,

                # Spec evolution events
                EventType.SPEC_SHEET_ANALYSIS_REQUESTED,
                EventType.SPEC_SHEET_ANALYSIS_COMPLETED,
                EventType.SPEC_SHEET_EVOLUTION_SUGGESTED,
                EventType.SPEC_SHEET_EVOLUTION_APPLIED,

                # Code generation events
                EventType.CODE_GENERATION_REQUESTED,
                EventType.CODE_GENERATION_COMPLETED,
                EventType.CODE_GENERATION_FAILED
            ],
            secret_key=self.secret_key,
            tenant=self.tenant,
            namespace=self.namespace
        )

        self.logger.info("Event components initialized")

    def _init_spec_registry(self):
        """Initialize the spec registry."""
        # Create storage repository
        # In a real implementation, this would be a proper repository
        # For now, we'll use a simple in-memory storage
        self.storage_repository = None

        # Initialize the spec registry
        self.registry = SpecRegistry(storage_repository=self.storage_repository)

        self.logger.info("SpecRegistry initialized")

    async def start(self) -> bool:
        """
        Start the event adapter and register event handlers.

        Returns:
            bool: True if successfully started, False otherwise
        """
        if not self.enable_events:
            self.logger.info("Events disabled, not starting event listeners")
            return True

        try:
            # Start event listener
            await self.event_listener.start()

            # Register event handlers
            self._register_event_handlers()

            self.logger.info("SpecRegistryEventAdapter started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start SpecRegistryEventAdapter: {e}")
            return False

    async def stop(self) -> bool:
        """
        Stop the event adapter and clean up resources.

        Returns:
            bool: True if successfully stopped, False otherwise
        """
        if not self.enable_events:
            return True

        try:
            # Stop event listener
            await self.event_listener.stop()

            # Close event emitter
            if hasattr(self, 'event_emitter'):
                self.event_emitter.close()

            self.logger.info("SpecRegistryEventAdapter stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping SpecRegistryEventAdapter: {e}")
            return False

    def _register_event_handlers(self):
        """Register handlers for different event types."""
        self.logger.info("Registering event handlers")

        # Register spec sheet event handlers
        self.event_listener.register_handler(
            EventType.SPEC_SHEET_CREATED,
            self._handle_spec_sheet_created
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_UPDATED,
            self._handle_spec_sheet_updated
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_DELETED,
            self._handle_spec_sheet_deleted
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_PUBLISHED,
            self._handle_spec_sheet_published
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_DEPRECATED,
            self._handle_spec_sheet_deprecated
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_ARCHIVED,
            self._handle_spec_sheet_archived
        )

        # Register spec instance event handlers
        self.event_listener.register_handler(
            EventType.SPEC_INSTANCE_CREATED,
            self._handle_spec_instance_created
        )

        self.event_listener.register_handler(
            EventType.SPEC_INSTANCE_UPDATED,
            self._handle_spec_instance_updated
        )

        self.event_listener.register_handler(
            EventType.SPEC_INSTANCE_COMPLETED,
            self._handle_spec_instance_completed
        )

        self.event_listener.register_handler(
            EventType.SPEC_INSTANCE_VALIDATED,
            self._handle_spec_instance_validated
        )

        self.event_listener.register_handler(
            EventType.SPEC_INSTANCE_DELETED,
            self._handle_spec_instance_deleted
        )

        # Register spec evolution event handlers
        self.event_listener.register_handler(
            EventType.SPEC_SHEET_ANALYSIS_REQUESTED,
            self._handle_spec_sheet_analysis_requested
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_ANALYSIS_COMPLETED,
            self._handle_spec_sheet_analysis_completed
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_EVOLUTION_SUGGESTED,
            self._handle_spec_sheet_evolution_suggested
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_EVOLUTION_APPLIED,
            self._handle_spec_sheet_evolution_applied
        )

        # Register code generation event handlers
        self.event_listener.register_handler(
            EventType.CODE_GENERATION_REQUESTED,
            self._handle_code_generation_requested
        )

        self.event_listener.register_handler(
            EventType.CODE_GENERATION_COMPLETED,
            self._handle_code_generation_completed
        )

        self.event_listener.register_handler(
            EventType.CODE_GENERATION_FAILED,
            self._handle_code_generation_failed
        )

        self.logger.info("Event handlers registered")

    # Spec sheet event handlers
    async def _handle_spec_sheet_created(self, event: BaseEvent):
        """Handle spec sheet created events."""
        try:
            self.logger.info(f"Handling spec sheet creation: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecSheetEvent):
                spec_sheet_id = event.spec_sheet_id
                project_id = event.payload.get("project_id")
                spec_sheet_name = event.spec_sheet_name
            else:
                spec_sheet_id = event.payload.get("spec_sheet_id")
                project_id = event.payload.get("project_id")
                spec_sheet_name = event.payload.get("spec_sheet_name")

            # Determine spec type
            spec_type = event.payload.get("spec_type", "container")

            # Check if spec already exists
            existing_spec = await self.registry.get_spec(spec_sheet_id)
            if existing_spec:
                self.logger.info(f"Spec sheet {spec_sheet_id} already exists")
                return

            # Create spec in registry
            await self.registry.create_empty_spec(spec_type, project_id)

            self.logger.info(f"Created spec sheet: {spec_sheet_id} of type {spec_type}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet creation: {e}")

    async def _handle_spec_sheet_updated(self, event: BaseEvent):
        """Handle spec sheet updated events."""
        try:
            self.logger.info(f"Handling spec sheet update: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecSheetEvent):
                spec_sheet_id = event.spec_sheet_id
                field_updates = event.payload.get("field_updates", {})
            else:
                spec_sheet_id = event.payload.get("spec_sheet_id")
                field_updates = event.payload.get("field_updates", {})

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, field_updates)

            self.logger.info(f"Updated spec sheet: {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet update: {e}")

    async def _handle_spec_sheet_deleted(self, event: BaseEvent):
        """Handle spec sheet deleted events."""
        try:
            self.logger.info(f"Handling spec sheet deletion: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecSheetEvent):
                spec_sheet_id = event.spec_sheet_id
            else:
                spec_sheet_id = event.payload.get("spec_sheet_id")

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return

            # Delete spec from registry
            await self.registry.delete_spec(spec_sheet_id)

            self.logger.info(f"Deleted spec sheet: {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet deletion: {e}")

    async def _handle_spec_sheet_published(self, event: BaseEvent):
        """Handle spec sheet published events."""
        try:
            self.logger.info(f"Handling spec sheet publication: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecSheetEvent):
                spec_sheet_id = event.spec_sheet_id
            else:
                spec_sheet_id = event.payload.get("spec_sheet_id")

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return

            # Mark as published (via update)
            spec["status"] = "published"
            spec["published_at"] = datetime.now().isoformat()

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Published spec sheet: {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet publication: {e}")

    async def _handle_spec_sheet_deprecated(self, event: BaseEvent):
        """Handle spec sheet deprecated events."""
        try:
            self.logger.info(f"Handling spec sheet deprecation: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecSheetEvent):
                spec_sheet_id = event.spec_sheet_id
            else:
                spec_sheet_id = event.payload.get("spec_sheet_id")

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return

            # Mark as deprecated (via update)
            spec["status"] = "deprecated"
            spec["deprecated_at"] = datetime.now().isoformat()

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Deprecated spec sheet: {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet deprecation: {e}")

    async def _handle_spec_sheet_archived(self, event: BaseEvent):
        """Handle spec sheet archived events."""
        try:
            self.logger.info(f"Handling spec sheet archival: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecSheetEvent):
                spec_sheet_id = event.spec_sheet_id
            else:
                spec_sheet_id = event.payload.get("spec_sheet_id")

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return

            # Mark as archived (via update)
            spec["status"] = "archived"
            spec["archived_at"] = datetime.now().isoformat()

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Archived spec sheet: {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet archival: {e}")

    # Spec instance event handlers
    async def _handle_spec_instance_created(self, event: BaseEvent):
        """Handle spec instance created events."""
        try:
            self.logger.info(f"Handling spec instance creation: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecInstanceEvent):
                instance_id = event.instance_id
                spec_sheet_id = event.spec_sheet_id
                project_id = event.project_id
            else:
                instance_id = event.payload.get("instance_id")
                spec_sheet_id = event.payload.get("spec_sheet_id")
                project_id = event.payload.get("project_id")

            if not instance_id or not spec_sheet_id:
                self.logger.error("Missing instance_id or spec_sheet_id in event")
                return

            # For now, we'll just log this (spec instances are handled differently)
            self.logger.info(f"Received spec instance creation: {instance_id} for spec {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec instance creation: {e}")

    async def _handle_spec_instance_updated(self, event: BaseEvent):
        """Handle spec instance updated events."""
        try:
            self.logger.info(f"Handling spec instance update: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecInstanceEvent):
                instance_id = event.instance_id
                spec_sheet_id = event.spec_sheet_id
            else:
                instance_id = event.payload.get("instance_id")
                spec_sheet_id = event.payload.get("spec_sheet_id")

            if not instance_id or not spec_sheet_id:
                self.logger.error("Missing instance_id or spec_sheet_id in event")
                return

            # For now, we'll just log this (spec instances are handled differently)
            self.logger.info(f"Received spec instance update: {instance_id} for spec {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec instance update: {e}")

    async def _handle_spec_instance_completed(self, event: BaseEvent):
        """Handle spec instance completed events."""
        try:
            self.logger.info(f"Handling spec instance completion: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecInstanceEvent):
                instance_id = event.instance_id
                spec_sheet_id = event.spec_sheet_id
            else:
                instance_id = event.payload.get("instance_id")
                spec_sheet_id = event.payload.get("spec_sheet_id")

            if not instance_id or not spec_sheet_id:
                self.logger.error("Missing instance_id or spec_sheet_id in event")
                return

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return

            # Mark as completed (via update)
            spec["status"] = "completed"
            spec["completed_at"] = datetime.now().isoformat()

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Marked spec sheet as completed via instance: {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec instance completion: {e}")

    async def _handle_spec_instance_validated(self, event: BaseEvent):
        """Handle spec instance validated events."""
        try:
            self.logger.info(f"Handling spec instance validation: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecInstanceEvent):
                instance_id = event.instance_id
                spec_sheet_id = event.spec_sheet_id
                is_valid = event.payload.get("is_valid", False)
            else:
                instance_id = event.payload.get("instance_id")
                spec_sheet_id = event.payload.get("spec_sheet_id")
                is_valid = event.payload.get("is_valid", False)

            if not instance_id or not spec_sheet_id:
                self.logger.error("Missing instance_id or spec_sheet_id in event")
                return

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return

            # Mark as validated (via update)
            if is_valid:
                spec["status"] = "validated"
                spec["validation_errors"] = []
            else:
                spec["status"] = "validation_failed"
                spec["validation_errors"] = event.payload.get("validation_errors", [])

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Updated spec sheet validation status via instance: {spec_sheet_id} - valid: {is_valid}")

        except Exception as e:
            self.logger.error(f"Error handling spec instance validation: {e}")

    async def _handle_spec_instance_deleted(self, event: BaseEvent):
        """Handle spec instance deleted events."""
        try:
            self.logger.info(f"Handling spec instance deletion: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecInstanceEvent):
                instance_id = event.instance_id
                spec_sheet_id = event.spec_sheet_id
            else:
                instance_id = event.payload.get("instance_id")
                spec_sheet_id = event.payload.get("spec_sheet_id")

            if not instance_id:
                self.logger.error("Missing instance_id in event")
                return

            # For now, we'll just log this (spec instances are handled differently)
            self.logger.info(f"Received spec instance deletion: {instance_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec instance deletion: {e}")

    # Spec evolution event handlers
    async def _handle_spec_sheet_analysis_requested(self, event: BaseEvent):
        """Handle spec sheet analysis requested events."""
        try:
            self.logger.info(f"Handling spec sheet analysis request: {event.event_id}")

            # Extract spec details from event
            spec_sheet_id = event.payload.get("spec_sheet_id")

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return

            # Mark as being analyzed (via update)
            spec["status"] = "analyzing"

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Marked spec sheet for analysis: {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet analysis request: {e}")

    async def _handle_spec_sheet_analysis_completed(self, event: BaseEvent):
        """Handle spec sheet analysis completed events."""
        try:
            self.logger.info(f"Handling spec sheet analysis completion: {event.event_id}")

            # Extract spec details from event
            spec_sheet_id = event.payload.get("spec_sheet_id")
            analysis_results = event.payload.get("analysis_results", {})

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return

            # Update spec with analysis results
            spec["status"] = "analyzed"
            spec["analysis_results"] = analysis_results

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Updated spec sheet with analysis results: {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet analysis completion: {e}")

    async def _handle_spec_sheet_evolution_suggested(self, event: BaseEvent):
        """Handle spec sheet evolution suggested events."""
        try:
            self.logger.info(f"Handling spec sheet evolution suggestion: {event.event_id}")

            # Extract spec details from event
            spec_sheet_id = event.payload.get("spec_sheet_id")
            suggestions = event.payload.get("suggestions", [])

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return

            # Store suggestions
            spec["evolution_suggestions"] = suggestions

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Updated spec sheet with evolution suggestions: {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet evolution suggestion: {e}")

    async def _handle_spec_sheet_evolution_applied(self, event: BaseEvent):
        """Handle spec sheet evolution applied events."""
        try:
            self.logger.info(f"Handling spec sheet evolution application: {event.event_id}")

            # Extract spec details from event
            spec_sheet_id = event.payload.get("spec_sheet_id")
            applied_changes = event.payload.get("applied_changes", [])

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return

            # Store applied changes
            spec["applied_evolution"] = applied_changes
            spec["status"] = "evolved"

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Updated spec sheet with applied evolution: {spec_sheet_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet evolution application: {e}")

    # Code generation event handlers
    async def _handle_code_generation_requested(self, event: BaseEvent):
        """Handle code generation requested events."""
        try:
            self.logger.info(f"Handling code generation request: {event.event_id}")

            # Extract spec details from event
            if hasattr(event.payload, 'spec_sheet'):
                spec_sheet = event.payload.spec_sheet
                spec_id = spec_sheet.get("id") if isinstance(spec_sheet, dict) else None
            else:
                spec_sheet = event.payload.get("spec_sheet", {})
                spec_id = spec_sheet.get("id") if isinstance(spec_sheet, dict) else None

            if not spec_id:
                self.logger.error("Missing spec_id in code generation request")
                return

            # Update spec status
            spec = await self.registry.get_spec(spec_id)
            if spec:
                spec["status"] = "generating"
                spec["generation_id"] = event.event_id

                # Update spec in registry
                await self.registry.update_spec(spec_id, {})

            self.logger.info(f"Updated spec sheet for code generation: {spec_id}")

        except Exception as e:
            self.logger.error(f"Error handling code generation request: {e}")

    async def _handle_code_generation_completed(self, event: BaseEvent):
        """Handle code generation completed events."""
        try:
            self.logger.info(f"Handling code generation completion: {event.event_id}")

            # Extract spec ID from metadata
            spec_id = event.metadata.get("spec_id")

            if not spec_id:
                self.logger.error("Missing spec_id in code generation completion")
                return

            # Update spec status
            spec = await self.registry.get_spec(spec_id)
            if spec:
                spec["status"] = "generated"

                # Store the generated code
                if hasattr(event.payload, 'generated_code'):
                    generated_code = event.payload.generated_code
                else:
                    generated_code = event.payload.get("generated_code", "")

                # Update fields if they exist
                if "fields" in spec and "generated_code" in spec["fields"]:
                    spec["fields"]["generated_code"]["value"] = generated_code

                # Update spec in registry
                await self.registry.update_spec(spec_id, {})

            self.logger.info(f"Updated spec sheet after code generation: {spec_id}")

        except Exception as e:
            self.logger.error(f"Error handling code generation completion: {e}")

    async def _handle_code_generation_failed(self, event: BaseEvent):
        """Handle code generation failed events."""
        try:
            self.logger.info(f"Handling code generation failure: {event.event_id}")

            # Extract spec ID from metadata
            spec_id = event.metadata.get("spec_id")

            if not spec_id:
                self.logger.error("Missing spec_id in code generation failure")
                return

            # Extract error details
            if hasattr(event.payload, 'error_message'):
                error_message = event.payload.error_message
                error_type = event.payload.error_type
            else:
                error_message = event.payload.get("error_message", "Unknown error")
                error_type = event.payload.get("error_type", "unknown")

            # Update spec status
            spec = await self.registry.get_spec(spec_id)
            if spec:
                spec["status"] = "generation_failed"
                spec["generation_error"] = error_message
                spec["generation_error_type"] = error_type

                # Update spec in registry
                await self.registry.update_spec(spec_id, {})

            self.logger.info(f"Updated spec sheet after failed generation: {spec_id}")

        except Exception as e:
            self.logger.error(f"Error handling code generation failure: {e}")

    # Public API methods
    async def create_spec(self, spec_type: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a spec sheet and emit an event.

        Args:
            spec_type: Type of spec to create
            project_id: Optional project ID

        Returns:
            Created spec sheet
        """
        # Create spec in registry
        spec = await self.registry.create_empty_spec(spec_type, project_id)

        if self.enable_events and hasattr(self, 'event_emitter'):
            # Create and emit event
            event = BaseEvent(
                event_type=EventType.SPEC_SHEET_CREATED,
                source_container=Components.SYNTHESIS_ENGINE,
                payload={
                    "spec_sheet_id": spec["id"],
                    "spec_type": spec_type,
                    "project_id": project_id
                }
            )

            await self.event_emitter.emit_async(event)

        return spec

    async def update_spec(self, spec_id: str, field_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a spec sheet and emit an event.

        Args:
            spec_id: ID of the spec to update
            field_updates: Field updates to apply

        Returns:
            Updated spec sheet
        """
        # Update spec in registry
        spec = await self.registry.update_spec(spec_id, field_updates)

        if self.enable_events and hasattr(self, 'event_emitter'):
            # Create and emit event
            event = BaseEvent(
                event_type=EventType.SPEC_SHEET_UPDATED,
                source_container=Components.SYNTHESIS_ENGINE,
                payload={
                    "spec_sheet_id": spec_id,
                    "field_updates": field_updates
                }
            )

            await self.event_emitter.emit_async(event)

        return spec

    async def delete_spec(self, spec_id: str) -> bool:
        """
        Delete a spec sheet and emit an event.

        Args:
            spec_id: ID of the spec to delete

        Returns:
            True if deleted successfully
        """
        # Delete spec from registry
        result = await self.registry.delete_spec(spec_id)

        if result and self.enable_events and hasattr(self, 'event_emitter'):
            # Create and emit event
            event = BaseEvent(
                event_type=EventType.SPEC_SHEET_DELETED,
                source_container=Components.SYNTHESIS_ENGINE,
                payload={
                    "spec_sheet_id": spec_id
                }
            )

            await self.event_emitter.emit_async(event)

        return result

    async def complete_spec(self, spec_id: str) -> Dict[str, Any]:
        """
        Mark a spec sheet as completed and emit an event.

        Args:
            spec_id: ID of the spec to mark as completed

        Returns:
            Updated spec sheet
        """
        # Get spec from registry
        spec = await self.registry.get_spec(spec_id)
        if not spec:
            raise ValueError(f"Spec with ID {spec_id} not found")

        # Update status
        spec["status"] = "completed"

        # Update spec in registry
        updated_spec = await self.registry.update_spec(spec_id, {})

        if self.enable_events and hasattr(self, 'event_emitter'):
            # Create and emit event for instance completion
            event = BaseEvent(
                event_type=EventType.SPEC_INSTANCE_COMPLETED,
                source_container=Components.SYNTHESIS_ENGINE,
                payload={
                    "instance_id": str(uuid.uuid4()),  # Generate a temporary instance ID
                    "spec_sheet_id": spec_id,
                    "project_id": spec.get("project_id")
                }
            )

            await self.event_emitter.emit_async(event)

        return updated_spec

    async def validate_spec(self, spec_id: str) -> Dict[str, Any]:
        """
        Validate a spec sheet and emit an event.

        Args:
            spec_id: ID of the spec to validate

        Returns:
            Validation result
        """
        # Get spec from registry
        spec = await self.registry.get_spec(spec_id)
        if not spec:
            raise ValueError(f"Spec with ID {spec_id} not found")

        # Validate the spec
        validation_errors = self.registry._validate_spec(spec)
        is_valid = len(validation_errors) == 0

        # Update spec status
        if is_valid:
            spec["status"] = "validated"
            spec["validation_errors"] = []
        else:
            spec["status"] = "validation_failed"
            spec["validation_errors"] = validation_errors

        # Update spec in registry
        await self.registry.update_spec(spec_id, {})

        # Create result
        result = {
            "spec_id": spec_id,
            "is_valid": is_valid,
            "validation_errors": validation_errors
        }

        if self.enable_events and hasattr(self, 'event_emitter'):
            # Create and emit event for instance validation
            event = BaseEvent(
                event_type=EventType.SPEC_INSTANCE_VALIDATED,
                source_container=Components.SYNTHESIS_ENGINE,
                payload={
                    "instance_id": str(uuid.uuid4()),  # Generate a temporary instance ID
                    "spec_sheet_id": spec_id,
                    "is_valid": is_valid,
                    "validation_errors": validation_errors,
                    "project_id": spec.get("project_id")
                }
            )

            await self.event_emitter.emit_async(event)

        return result

    async def get_spec(self, spec_id: str) -> Dict[str, Any]:
        """
        Get a spec sheet by ID.

        Args:
            spec_id: ID of the spec to get

        Returns:
            Spec sheet or None if not found
        """
        return await self.registry.get_spec(spec_id)

    async def list_specs(self, project_id: Optional[str] = None, spec_type: Optional[str] = None) -> List[
        Dict[str, Any]]:
        """
        List specs, optionally filtered by project ID or type.

        Args:
            project_id: Optional project ID to filter by
            spec_type: Optional spec type to filter by

        Returns:
            List of matching spec sheets
        """
        return await self.registry.list_specs(project_id, spec_type)