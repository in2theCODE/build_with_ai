# src/services/spec_registry/app/spec_registry_event_adapter.py
import asyncio
import datetime
import os
import uuid
from typing import List, Dict, Any, Optional, Union

from src.services.shared import BaseComponent, logging
from src.services.shared.models import EventType, Components, BaseEvent, SpecInstanceEvent
from src.services.spec_registry.app.models import SpecTemplate, SpecStatus, ValidationResult
from src.services.spec_registry.app.spec_registry import SpecRegistry


class InMemorySpecRepository:
    """Simple in-memory repository for spec sheets when PostgreSQL is not available."""

    def __init__(self):
        self._specs = {}
        self._templates = {}
        self._relations = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    async def initialize(self):
        """Initialize the repository."""
        self.logger.info("Initializing in-memory repository")
        return True

    async def close(self):
        """Close the repository (no-op for in-memory)."""
        self.logger.info("Closing in-memory repository")
        return True

    async def store_spec(self, spec: Dict[str, Any]) -> bool:
        """Store a spec in memory."""
        try:
            self._specs[spec["id"]] = spec.copy()
            return True
        except Exception as e:
            self.logger.error(f"Error storing spec {spec.get('id')}: {e}")
            return False

    async def update_spec(self, spec: Dict[str, Any]) -> bool:
        """Update a spec in memory."""
        return await self.store_spec(spec)

    async def get_spec(self, spec_id: str) -> Optional[Dict[str, Any]]:
        """Get a spec by ID."""
        spec = self._specs.get(spec_id)
        if spec:
            return spec.copy()
        return None

    async def delete_spec(self, spec_id: str) -> bool:
        """Delete a spec from memory."""
        if spec_id in self._specs:
            del self._specs[spec_id]
            return True
        return False

    async def list_specs(
            self,
            project_id: Optional[str] = None,
            spec_type: Optional[str] = None,
            status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List specs with optional filtering."""
        specs = list(self._specs.values())

        if project_id:
            specs = [s for s in specs if s.get("project_id") == project_id]

        if spec_type:
            specs = [s for s in specs if s.get("type") == spec_type]

        if status:
            specs = [s for s in specs if s.get("status") == status]

        return specs

    async def store_spec_relation(
            self, spec_id: str, related_spec_id: str, relation_type: str
    ) -> bool:
        """Store a relation between specs."""
        try:
            if spec_id not in self._relations:
                self._relations[spec_id] = []

            relation = {"spec_id": related_spec_id, "type": relation_type}
            if relation not in self._relations[spec_id]:
                self._relations[spec_id].append(relation)
            return True
        except Exception as e:
            self.logger.error(f"Error storing relation: {e}")
            return False

    async def get_related_specs(self, spec_id: str) -> List[Dict[str, Any]]:
        """Get specs related to the given spec."""
        if spec_id not in self._relations:
            return []

        related_specs = []
        for relation in self._relations[spec_id]:
            spec = await self.get_spec(relation["spec_id"])
            if spec:
                related_specs.append({"spec": spec, "relation_type": relation["type"]})

        return related_specs

    async def store_template(
            self,
            template_type: str,
            fields: Dict[str, Any],
            version: str = "1.0",
            metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a template."""
        try:
            key = f"{template_type}:{version}"
            self._templates[key] = {
                "type": template_type,
                "version": version,
                "fields": fields,
                "metadata": metadata or {},
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "is_active": True,
            }
            return True
        except Exception as e:
            self.logger.error(f"Error storing template {template_type}: {e}")
            return False

    async def get_template(
            self, template_type: str, version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a template by type and optional version."""
        if version:
            key = f"{template_type}:{version}"
            return self._templates.get(key)

        # Get latest version
        matching_templates = [
            t
            for k, t in self._templates.items()
            if k.startswith(f"{template_type}:") and t.get("is_active", True)
        ]
        if not matching_templates:
            return None

        # Sort by version (simple string sort for now)
        matching_templates.sort(key=lambda t: t["version"], reverse=True)
        return matching_templates[0]

    async def list_templates(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """List all templates."""
        templates = list(self._templates.values())
        if not include_inactive:
            templates = [t for t in templates if t.get("is_active", True)]
        return templates

    async def delete_template(
            self, template_type: str, version: Optional[str] = None, hard_delete: bool = False
    ) -> bool:
        """Delete a template."""
        try:
            if version:
                key = f"{template_type}:{version}"
                if hard_delete:
                    if key in self._templates:
                        del self._templates[key]
                else:
                    if key in self._templates:
                        self._templates[key]["is_active"] = False
            else:
                # Delete all versions
                keys_to_delete = [
                    k for k in self._templates.keys() if k.startswith(f"{template_type}:")
                ]
                for key in keys_to_delete:
                    if hard_delete:
                        del self._templates[key]
                    else:
                        self._templates[key]["is_active"] = False
            return True
        except Exception as e:
            self.logger.error(f"Error deleting template {template_type}: {e}")
            return False


class SecureEventEmitter:
    """
    Secure event emitter for sending events.
    This is a placeholder implementation.
    """

    def __init__(self, service_url, secret_key=None, tenant="public", namespace="default"):
        self.service_url = service_url
        self.secret_key = secret_key
        self.tenant = tenant
        self.namespace = namespace
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized SecureEventEmitter with service URL: {service_url}")

    async def emit_async(self, event: BaseEvent) -> bool:
        """Emit an event asynchronously."""
        try:
            self.logger.info(f"Emitting event: {event.event_type}")
            # This is a placeholder - in a real implementation, this would send the event
            # to a message broker like Pulsar or Kafka
            return True
        except Exception as e:
            self.logger.error(f"Error emitting event: {e}")
            return False

    def close(self):
        """Close the emitter and release resources."""
        self.logger.info("Closing event emitter")


class SecureEventListener:
    """
    Secure event listener for receiving events.
    This is a placeholder implementation.
    """

    def __init__(
            self,
            service_url,
            subscription_name,
            event_types,
            secret_key=None,
            tenant="public",
            namespace="default",
    ):
        self.service_url = service_url
        self.subscription_name = subscription_name
        self.event_types = event_types
        self.secret_key = secret_key
        self.tenant = tenant
        self.namespace = namespace
        self.handlers = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initialized SecureEventListener with service URL: {service_url}, "
            f"subscription: {subscription_name}"
        )

    async def start(self) -> bool:
        """Start listening for events."""
        self.logger.info("Starting event listener")
        return True

    async def stop(self) -> bool:
        """Stop listening for events."""
        self.logger.info("Stopping event listener")
        return True

    def register_handler(self, event_type: str, handler):
        """Register a handler for a specific event type."""
        self.handlers[event_type] = handler
        self.logger.info(f"Registered handler for event type: {event_type}")


class TemplateAnalysisRequest:
    """Request for template analysis."""

    def __init__(
            self,
            spec_id: str,
            template_type: str,
            template_fields: Dict[str, Any],
            current_fields: Dict[str, Any],
            project_context: Optional[Dict[str, Any]] = None,
    ):
        self.spec_id = spec_id
        self.template_type = template_type
        self.template_fields = template_fields
        self.current_fields = current_fields
        self.project_context = project_context or {}

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spec_id": self.spec_id,
            "template_type": self.template_type,
            "template_fields": self.template_fields,
            "current_fields": self.current_fields,
            "project_context": self.project_context,
        }


class SpecSheetEvent(BaseEvent):
    """Event for spec sheet operations."""

    def __init__(
            self,
            event_type: str,
            source_container: str,
            spec_sheet_id: str,
            spec_sheet_name: Optional[str] = None,
            spec_sheet_version: Optional[str] = None,
            project_id: Optional[str] = None,
            payload: Optional[Dict[str, Any]] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            event_type=event_type,
            source_container=source_container,
            payload=payload or {},
            metadata=metadata or {},
        )
        self.spec_sheet_id = spec_sheet_id
        self.spec_sheet_name = spec_sheet_name
        self.spec_sheet_version = spec_sheet_version
        self.project_id = project_id


class SpecRegistryEventAdapter(BaseComponent):
    """
    Adapter that connects the SpecRegistry system to an event-driven architecture,
    enabling event-driven communication and supporting workflow orchestration.
    """

    def __init__(self, **params):
        """
        Initialize the Spec Registry Event Adapter with parameters.

        Args:
            **params: Configuration parameters including:
                pulsar_service_url: URL for the Pulsar service
                base_dir: Base directory for storage
                enable_events: Whether to enable events
                secret_key: Secret key for secure event transmission
                tenant: Pulsar tenant
                namespace: Pulsar namespace
                db_connection_string: Database connection string
                db_schema: Database schema
        """
        super().__init__(**params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Extract configuration parameters
        self.pulsar_service_url = self.get_param("pulsar_service_url", "pulsar://localhost:6650")
        self.base_dir = self.get_param("base_dir", "./spec_registry_data")
        self.enable_events = self.get_param("enable_events", True)
        self.secret_key = self.get_param("secret_key")
        self.tenant = self.get_param("tenant", "public")
        self.namespace = self.get_param("namespace", "code-generator")

        # Extract database configuration
        self.db_connection_string = self.get_param("db_connection_string", None)
        self.db_schema = self.get_param("db_schema", "public")

        # Create storage directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)

        # Initialize storage repository
        self._init_storage_repository()

        # Initialize event system components
        if self.enable_events:
            self._init_event_components()

        # Initialize spec registry
        self._init_spec_registry()

        self.logger.info("SpecRegistryEventAdapter initialized")

    def _init_storage_repository(self):
        """Initialize the storage repository."""
        try:
            if self.db_connection_string:
                try:
                    # Try to import and initialize PostgreSQL repository
                    from src.services.spec_registry.app.postgresql_spec_repository import (
                        PostgreSQLSpecRepository,
                    )

                    self.storage_repository = PostgreSQLSpecRepository(
                        connection_string=self.db_connection_string,
                        schema=self.db_schema,
                    )
                    self.logger.info("PostgreSQL storage repository initialized")
                except ImportError:
                    self.logger.error("PostgreSQLSpecRepository module not found")
                    self.logger.warning("Falling back to in-memory storage")
                    self.storage_repository = InMemorySpecRepository()
                except Exception as e:
                    self.logger.error(f"Failed to init PostgreSQL repository: {e}")
                    self.logger.warning("Falling back to in-memory storage")
                    self.storage_repository = InMemorySpecRepository()
            else:
                self.logger.warning("No DB config, using in-memory storage")
                self.storage_repository = InMemorySpecRepository()
        except Exception as e:
            self.logger.error(f"Error initializing storage repository: {e}", exc_info=True)
            # Create a fallback repository to ensure operation
            self.storage_repository = InMemorySpecRepository()

    def _init_event_components(self):
        """Initialize event emitter and listener."""
        try:
            # Initialize event emitter for sending events
            self.event_emitter = SecureEventEmitter(
                service_url=self.pulsar_service_url,
                secret_key=self.secret_key,
                tenant=self.tenant,
                namespace=self.namespace,
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
                    # Template events
                    EventType.TEMPLATE_CREATED,
                    EventType.TEMPLATE_UPDATED,
                    EventType.TEMPLATE_DELETED,
                    # AI analysis events
                    EventType.TEMPLATE_ANALYSIS_REQUESTED,
                    EventType.TEMPLATE_ANALYSIS_COMPLETED,
                    # Code generation events
                    EventType.CODE_GENERATION_REQUESTED,
                    EventType.CODE_GENERATION_COMPLETED,
                    EventType.CODE_GENERATION_FAILED,
                ],
                secret_key=self.secret_key,
                tenant=self.tenant,
                namespace=self.namespace,
            )

            self.logger.info("Event components initialized")
        except Exception as e:
            self.logger.error(f"Error initializing event components: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize event components: {e}")

    def _init_spec_registry(self):
        """Initialize the spec registry."""
        try:
            # Initialize the spec registry with the storage repository
            self.registry = SpecRegistry(storage_repository=self.storage_repository)
            self.logger.info("SpecRegistry initialized")
        except Exception as e:
            self.logger.error(f"Error initializing spec registry: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize spec registry: {e}")

    async def start(self) -> bool:
        """
        Start the event adapter and register event handlers.

        Returns:
            bool: True if successfully started, False otherwise
        """
        try:
            # Initialize database if we're using a repository that needs initialization
            if hasattr(self, "storage_repository") and self.storage_repository:
                if hasattr(self.storage_repository, "initialize"):
                    await self.storage_repository.initialize()
                    self.logger.info("Storage repository initialized")

            if not self.enable_events:
                self.logger.info("Events disabled, not starting event listeners")
                return True

            # Start event listener
            await self.event_listener.start()

            # Register event handlers
            self._register_event_handlers()

            self.logger.info("SpecRegistryEventAdapter started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start SpecRegistryEventAdapter: {e}", exc_info=True)
            return False

    async def stop(self) -> bool:
        """
        Stop the event adapter and clean up resources.

        Returns:
            bool: True if successfully stopped, False otherwise
        """
        try:
            # Close database connection if we're using a repository with a close method
            if hasattr(self, "storage_repository") and self.storage_repository:
                if hasattr(self.storage_repository, "close"):
                    await self.storage_repository.close()
                    self.logger.info("Storage repository connection closed")

            if not self.enable_events:
                return True

            # Stop event listener
            if hasattr(self, "event_listener"):
                await self.event_listener.stop()

            # Close event emitter
            if hasattr(self, "event_emitter"):
                self.event_emitter.close()

            self.logger.info("SpecRegistryEventAdapter stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping SpecRegistryEventAdapter: {e}", exc_info=True)
            return False

    def _register_event_handlers(self):
        """Register handlers for different event types."""
        self.logger.info("Registering event handlers")

        # Register spec sheet event handlers
        self.event_listener.register_handler(
            EventType.SPEC_SHEET_CREATED, self._handle_spec_sheet_created
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_UPDATED, self._handle_spec_sheet_updated
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_DELETED, self._handle_spec_sheet_deleted
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_PUBLISHED, self._handle_spec_sheet_published
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_DEPRECATED, self._handle_spec_sheet_deprecated
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_ARCHIVED, self._handle_spec_sheet_archived
        )

        # Register spec instance event handlers
        self.event_listener.register_handler(
            EventType.SPEC_INSTANCE_CREATED, self._handle_spec_instance_created
        )

        self.event_listener.register_handler(
            EventType.SPEC_INSTANCE_UPDATED, self._handle_spec_instance_updated
        )

        self.event_listener.register_handler(
            EventType.SPEC_INSTANCE_COMPLETED, self._handle_spec_instance_completed
        )

        self.event_listener.register_handler(
            EventType.SPEC_INSTANCE_VALIDATED, self._handle_spec_instance_validated
        )

        self.event_listener.register_handler(
            EventType.SPEC_INSTANCE_DELETED, self._handle_spec_instance_deleted
        )

        # Register spec evolution event handlers
        self.event_listener.register_handler(
            EventType.SPEC_SHEET_ANALYSIS_REQUESTED,
            self._handle_spec_sheet_analysis_requested,
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_ANALYSIS_COMPLETED,
            self._handle_spec_sheet_analysis_completed,
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_EVOLUTION_SUGGESTED,
            self._handle_spec_sheet_evolution_suggested,
        )

        self.event_listener.register_handler(
            EventType.SPEC_SHEET_EVOLUTION_APPLIED,
            self._handle_spec_sheet_evolution_applied,
        )

        # Register template event handlers
        self.event_listener.register_handler(
            EventType.TEMPLATE_CREATED,
            self._handle_template_created,
        )

        self.event_listener.register_handler(
            EventType.TEMPLATE_UPDATED,
            self._handle_template_updated,
        )

        self.event_listener.register_handler(
            EventType.TEMPLATE_DELETED,
            self._handle_template_deleted,
        )

        # Register AI analysis event handlers
        self.event_listener.register_handler(
            EventType.TEMPLATE_ANALYSIS_REQUESTED,
            self._handle_template_analysis_requested,
        )

        self.event_listener.register_handler(
            EventType.TEMPLATE_ANALYSIS_COMPLETED,
            self._handle_template_analysis_completed,
        )

        # Register code generation event handlers
        self.event_listener.register_handler(
            EventType.CODE_GENERATION_REQUESTED, self._handle_code_generation_requested
        )

        self.event_listener.register_handler(
            EventType.CODE_GENERATION_COMPLETED, self._handle_code_generation_completed
        )

        self.event_listener.register_handler(
            EventType.CODE_GENERATION_FAILED, self._handle_code_generation_failed
        )

        self.logger.info("Event handlers registered")

    # Spec sheet event handlers
    async def _handle_spec_sheet_created(self, event: Union[BaseEvent, SpecSheetEvent]):
        """
        Handle spec sheet created events.

        Args:
            event: Event containing spec sheet creation details
        """
        try:
            self.logger.info(f"Handling spec sheet creation: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecSheetEvent):
                spec_sheet_id = event.spec_sheet_id
                project_id = event.project_id
                spec_sheet_name = event.spec_sheet_name
                spec_sheet_version = event.spec_sheet_version
            else:
                spec_sheet_id = event.payload.get("spec_sheet_id")
                project_id = event.payload.get("project_id")
                spec_sheet_name = event.payload.get("spec_sheet_name")
                spec_sheet_version = event.payload.get("spec_sheet_version")

            # Determine spec type
            spec_type = event.payload.get("spec_type", "container")

            # Check if spec already exists
            existing_spec = await self.registry.get_spec(spec_sheet_id)
            if existing_spec:
                self.logger.info(f"Spec sheet {spec_sheet_id} already exists")
                return

            # Create spec in registry
            spec = await self.registry.create_empty_spec(
                spec_type, project_id=project_id, template_version=spec_sheet_version
            )

            # Update spec name if provided
            if spec_sheet_name:
                await self.registry.update_spec(spec["id"], {"name": spec_sheet_name})

            self.logger.info(f"Created spec sheet: {spec_sheet_id} of type {spec_type}")

            # Return success response if needed
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec sheet creation: {e}", exc_info=True)
            # Emit error event if event system is enabled
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_sheet_creation",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_sheet_updated(self, event: Union[BaseEvent, SpecSheetEvent]):
        """
        Handle spec sheet updated events.

        Args:
            event: Event containing spec sheet update details
        """
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
                return False

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, field_updates)

            self.logger.info(f"Updated spec sheet: {spec_sheet_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec sheet update: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_sheet_update",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_sheet_deleted(self, event: Union[BaseEvent, SpecSheetEvent]):
        """
        Handle spec sheet deleted events.

        Args:
            event: Event containing spec sheet deletion details
        """
        try:
            self.logger.info(f"Handling spec sheet deletion: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecSheetEvent):
                spec_sheet_id = event.spec_sheet_id
            else:
                spec_sheet_id = event.payload.get("spec_sheet_id")

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return False

            # Delete spec from registry
            success = await self.registry.delete_spec(spec_sheet_id)

            if success:
                self.logger.info(f"Deleted spec sheet: {spec_sheet_id}")
            else:
                self.logger.warning(f"Failed to delete spec sheet: {spec_sheet_id}")

            return success
        except Exception as e:
            self.logger.error(f"Error handling spec sheet deletion: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_sheet_deletion",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_sheet_published(self, event: Union[BaseEvent, SpecSheetEvent]):
        """
        Handle spec sheet published events.

        Args:
            event: Event containing spec sheet publication details
        """
        try:
            self.logger.info(f"Handling spec sheet publication: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecSheetEvent):
                spec_sheet_id = event.spec_sheet_id
            else:
                spec_sheet_id = event.payload.get("spec_sheet_id")

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return False

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return False

            # Mark as published (via update)
            spec["status"] = SpecStatus.PUBLISHED
            spec["published_at"] = datetime.datetime.now().isoformat()

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Published spec sheet: {spec_sheet_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec sheet publication: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_sheet_publication",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_sheet_deprecated(self, event: Union[BaseEvent, SpecSheetEvent]):
        """
        Handle spec sheet deprecated events.

        Args:
            event: Event containing spec sheet deprecation details
        """
        try:
            self.logger.info(f"Handling spec sheet deprecation: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecSheetEvent):
                spec_sheet_id = event.spec_sheet_id
            else:
                spec_sheet_id = event.payload.get("spec_sheet_id")

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return False

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return False

            # Mark as deprecated (via update)
            spec["status"] = SpecStatus.DEPRECATED
            spec["deprecated_at"] = datetime.datetime.now().isoformat()

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Deprecated spec sheet: {spec_sheet_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec sheet deprecation: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_sheet_deprecation",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_sheet_archived(self, event: Union[BaseEvent, SpecSheetEvent]):
        """
        Handle spec sheet archived events.

        Args:
            event: Event containing spec sheet archival details
        """
        try:
            self.logger.info(f"Handling spec sheet archival: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecSheetEvent):
                spec_sheet_id = event.spec_sheet_id
            else:
                spec_sheet_id = event.payload.get("spec_sheet_id")

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return False

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return False

            # Mark as archived (via update)
            spec["status"] = SpecStatus.ARCHIVED
            spec["archived_at"] = datetime.datetime.now().isoformat()

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Archived spec sheet: {spec_sheet_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec sheet archival: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_sheet_archival",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    # Template event handlers
    async def _handle_template_created(self, event: BaseEvent):
        """
        Handle template created events.

        Args:
            event: Event containing template creation details
        """
        try:
            self.logger.info(f"Handling template creation: {event.event_id}")

            # Extract template details
            template_type = event.payload.get("template_type")
            template_fields = event.payload.get("fields", {})
            template_version = event.payload.get("version", "1.0")

            if not template_type or not template_fields:
                self.logger.error("Missing template_type or fields in event")
                return False

            # Store the template
            if self.storage_repository:
                success = await self.storage_repository.store_template(
                    template_type=template_type,
                    fields=template_fields,
                    version=template_version,
                    metadata=event.payload.get("metadata", {}),
                )

                if success:
                    self.logger.info(f"Created template: {template_type} v{template_version}")
                else:
                    self.logger.warning(f"Failed to create template: {template_type}")
                return success
            else:
                self.logger.warning("No storage repository available for storing template")
                return False
        except Exception as e:
            self.logger.error(f"Error handling template creation: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "template_creation",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_template_updated(self, event: BaseEvent):
        """
        Handle template updated events.

        Args:
            event: Event containing template update details
        """
        try:
            self.logger.info(f"Handling template update: {event.event_id}")

            # Extract template details
            template_type = event.payload.get("template_type")
            template_fields = event.payload.get("fields", {})
            template_version = event.payload.get("version", "1.0")

            if not template_type or not template_fields:
                self.logger.error("Missing template_type or fields in event")
                return False

            # Update the template
            if self.storage_repository:
                success = await self.storage_repository.store_template(
                    template_type=template_type,
                    fields=template_fields,
                    version=template_version,
                    metadata=event.payload.get("metadata", {}),
                )

                if success:
                    self.logger.info(f"Updated template: {template_type} v{template_version}")
                else:
                    self.logger.warning(f"Failed to update template: {template_type}")
                return success
            else:
                self.logger.warning("No storage repository available for updating template")
                return False
        except Exception as e:
            self.logger.error(f"Error handling template update: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "template_update",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_template_deleted(self, event: BaseEvent):
        """
        Handle template deleted events.

        Args:
            event: Event containing template deletion details
        """
        try:
            self.logger.info(f"Handling template deletion: {event.event_id}")

            # Extract template details
            template_type = event.payload.get("template_type")
            template_version = event.payload.get("version")
            hard_delete = event.payload.get("hard_delete", False)

            if not template_type:
                self.logger.error("Missing template_type in event")
                return False

            # Delete the template
            if self.storage_repository:
                success = await self.storage_repository.delete_template(
                    template_type=template_type,
                    version=template_version,
                    hard_delete=hard_delete
                )

                if success:
                    self.logger.info(
                        f"Deleted template: {template_type}"
                        + (f" v{template_version}" if template_version else "")
                    )
                else:
                    self.logger.warning(f"Failed to delete template: {template_type}")
                return success
            else:
                self.logger.warning("No storage repository available for deleting template")
                return False
        except Exception as e:
            self.logger.error(f"Error handling template deletion: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "template_deletion",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    # AI-powered template analysis handlers
    async def _handle_template_analysis_requested(self, event: BaseEvent):
        """
        Handle template analysis requested events.

        Args:
            event: Event containing template analysis request details
        """
        try:
            self.logger.info(f"Handling template analysis request: {event.event_id}")

            # Extract analysis details
            spec_id = event.payload.get("spec_id")
            template_type = event.payload.get("template_type")

            if not spec_id:
                self.logger.error("Missing spec_id in event")
                return False

            # Get the spec
            spec = await self.registry.get_spec(spec_id)
            if not spec:
                self.logger.error(f"Spec {spec_id} not found")
                return False

            # Get the template type if not specified
            if not template_type:
                template_type = spec.get("type")

            # Get the template
            template = None
            if self.storage_repository:
                template = await self.storage_repository.get_template(template_type)

            if not template:
                # Use the public method to get fields instead of accessing protected member
                template_fields = self.registry.get_fields_for_spec_type(template_type)
                if not template_fields:
                    self.logger.error(f"Template {template_type} not found")
                    return False

                template = {"type": template_type, "fields": template_fields}

            # Create analysis request
            analysis_request = TemplateAnalysisRequest(
                spec_id=spec_id,
                template_type=template_type,
                template_fields=template.get("fields", {}),
                current_fields=spec.get("fields", {}),
                project_context=event.payload.get("project_context", {}),
            )

            # Mark spec as analyzing
            spec["status"] = SpecStatus.ANALYZING
            await self.registry.update_spec(spec_id, {})

            # Send analysis request to neural service
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type=EventType.TEMPLATE_ANALYSIS_REQUESTED,
                        source_container=Components.SPEC_REGISTRY,
                        payload=analysis_request.dict(),
                        metadata={"spec_id": spec_id, "template_type": template_type},
                    )
                )

                self.logger.info(f"Sent template analysis request for spec {spec_id}")
                return True
            else:
                self.logger.warning("Events disabled, cannot send template analysis request")
                return False
        except Exception as e:
            self.logger.error(f"Error handling template analysis request: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "template_analysis_request",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_template_analysis_completed(self, event: BaseEvent):
        """
        Handle template analysis completed events.

        Args:
            event: Event containing template analysis completion details
        """
        try:
            self.logger.info(f"Handling template analysis completion: {event.event_id}")

            # Extract analysis results
            spec_id = event.metadata.get("spec_id") or event.payload.get("spec_id")
            template_type = event.metadata.get("template_type") or event.payload.get(
                "template_type"
            )
            analysis_results = event.payload.get("analysis_results", {})

            if not spec_id:
                self.logger.error("Missing spec_id in event")
                return False

            # Get the spec
            spec = await self.registry.get_spec(spec_id)
            if not spec:
                self.logger.error(f"Spec {spec_id} not found")
                return False

            # Update spec with analysis results
            spec["status"] = SpecStatus.ANALYZED
            spec["analysis_results"] = analysis_results

            # Apply suggested fields if requested
            if event.payload.get("apply_suggestions", False):
                try:
                    suggested_fields = analysis_results.get("suggested_fields", {})
                    if suggested_fields:
                        # Update fields that match template fields
                        for field_name, field_data in suggested_fields.items():
                            if field_name in spec.get("fields", {}):
                                # Update existing field with suggested values and metadata
                                spec["fields"][field_name].update(field_data)
                            else:
                                # Add new field
                                spec["fields"][field_name] = field_data

                        self.logger.info(f"Applied suggested fields to spec {spec_id}")
                except Exception as field_error:
                    self.logger.error(
                        f"Error applying suggested fields: {field_error}", exc_info=True
                    )

            # Update spec in registry
            await self.registry.update_spec(spec_id, {})

            self.logger.info(f"Updated spec {spec_id} with analysis results")
            return True
        except Exception as e:
            self.logger.error(f"Error handling template analysis completion: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "template_analysis_completion",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    # Spec instance event handlers
    async def _handle_spec_instance_created(self, event: Union[BaseEvent, SpecInstanceEvent]):
        """
        Handle spec instance created events.

        Args:
            event: Event containing spec instance creation details
        """
        try:
            self.logger.info(f"Handling spec instance creation: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecInstanceEvent):
                instance_id = event.instance_id
                spec_sheet_id = event.spec_sheet_id
                project_id = event.project_id
                spec_sheet_version = event.spec_sheet_version
            else:
                instance_id = event.payload.get("instance_id")
                spec_sheet_id = event.payload.get("spec_sheet_id")
                project_id = event.payload.get("project_id")
                spec_sheet_version = event.payload.get("spec_sheet_version")

            if not instance_id or not spec_sheet_id:
                self.logger.error("Missing instance_id or spec_sheet_id in event")
                return False

            # For now, we'll just log this (spec instances are handled differently)
            self.logger.info(
                f"Received spec instance creation: {instance_id} for spec {spec_sheet_id}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec instance creation: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_instance_creation",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_instance_updated(self, event: Union[BaseEvent, SpecInstanceEvent]):
        """
        Handle spec instance updated events.

        Args:
            event: Event containing spec instance update details
        """
        try:
            self.logger.info(f"Handling spec instance update: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecInstanceEvent):
                instance_id = event.instance_id
                spec_sheet_id = event.spec_sheet_id
                field_updates = event.payload.get("field_updates", {})
            else:
                instance_id = event.payload.get("instance_id")
                spec_sheet_id = event.payload.get("spec_sheet_id")
                field_updates = event.payload.get("field_updates", {})

            if not instance_id or not spec_sheet_id:
                self.logger.error("Missing instance_id or spec_sheet_id in event")
                return False

            # Update the spec with the instance's field updates if any
            if field_updates:
                await self.registry.update_spec(spec_sheet_id, field_updates)
                self.logger.info(f"Updated spec {spec_sheet_id} with instance field updates")
                return True
            else:
                self.logger.info(f"No field updates for spec instance: {instance_id}")
                return True
        except Exception as e:
            self.logger.error(f"Error handling spec instance update: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_instance_update",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_instance_completed(self, event: Union[BaseEvent, SpecInstanceEvent]):
        """
        Handle spec instance completed events.

        Args:
            event: Event containing spec instance completion details
        """
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
                return False

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return False

            # Mark as completed (via update)
            spec["status"] = SpecStatus.VALIDATED
            spec["completed_at"] = datetime.datetime.now().isoformat()

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Marked spec sheet as completed via instance: {spec_sheet_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec instance completion: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_instance_completion",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_instance_validated(self, event: Union[BaseEvent, SpecInstanceEvent]):
        """
        Handle spec instance validated events.

        Args:
            event: Event containing spec instance validation details
        """
        try:
            self.logger.info(f"Handling spec instance validation: {event.event_id}")

            # Extract spec details from event
            if isinstance(event, SpecInstanceEvent):
                instance_id = event.instance_id
                spec_sheet_id = event.spec_sheet_id
                is_valid = event.payload.get("is_valid", False)
                validation_errors = event.payload.get("validation_errors", [])
            else:
                instance_id = event.payload.get("instance_id")
                spec_sheet_id = event.payload.get("spec_sheet_id")
                is_valid = event.payload.get("is_valid", False)
                validation_errors = event.payload.get("validation_errors", [])

            if not instance_id or not spec_sheet_id:
                self.logger.error("Missing instance_id or spec_sheet_id in event")
                return False

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return False

            # Mark as validated (via update)
            if is_valid:
                spec["status"] = SpecStatus.VALIDATED
                spec["validation_errors"] = []
            else:
                spec["status"] = SpecStatus.VALIDATION_FAILED
                spec["validation_errors"] = validation_errors

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(
                f"Updated spec sheet validation status via instance: {spec_sheet_id} - valid: {is_valid}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec instance validation: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_instance_validation",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_instance_deleted(self, event: Union[BaseEvent, SpecInstanceEvent]):
        """
        Handle spec instance deleted events.

        Args:
            event: Event containing spec instance deletion details
        """
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
                return False

            # For now, we'll just log this (spec instances are handled differently)
            self.logger.info(f"Received spec instance deletion: {instance_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec instance deletion: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_instance_deletion",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    # Spec evolution event handlers
    async def _handle_spec_sheet_analysis_requested(self, event: BaseEvent):
        """
        Handle spec sheet analysis requested events.

        Args:
            event: Event containing spec sheet analysis request details
        """
        try:
            self.logger.info(f"Handling spec sheet analysis request: {event.event_id}")

            # Extract spec details from event
            spec_sheet_id = event.payload.get("spec_sheet_id")
            new_template_type = event.payload.get("new_template_type")
            new_template_version = event.payload.get("new_template_version")

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return False

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return False

            # Mark as being analyzed (via update)
            spec["status"] = SpecStatus.ANALYZING

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            # Analyze template compatibility
            analysis_result = await self.registry.analyze_template_compatibility(
                spec_id=spec_sheet_id,
                new_template_type=new_template_type,
                new_template_version=new_template_version,
            )

            # Store analysis results in metadata
            spec["analysis_result"] = analysis_result
            await self.registry.update_spec(spec_sheet_id, {})

            # Emit analysis completed event
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type=EventType.SPEC_SHEET_ANALYSIS_COMPLETED,
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "spec_sheet_id": spec_sheet_id,
                            "analysis_results": analysis_result,
                        },
                    )
                )

            self.logger.info(f"Marked spec sheet for analysis: {spec_sheet_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec sheet analysis request: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_sheet_analysis_request",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_sheet_analysis_completed(self, event: BaseEvent):
        """
        Handle spec sheet analysis completed events.

        Args:
            event: Event containing spec sheet analysis completion details
        """
        try:
            self.logger.info(f"Handling spec sheet analysis completion: {event.event_id}")

            # Extract spec details from event
            spec_sheet_id = event.payload.get("spec_sheet_id")
            analysis_results = event.payload.get("analysis_results", {})

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return False

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return False

            # Update spec with analysis results
            spec["status"] = SpecStatus.ANALYZED
            spec["analysis_results"] = analysis_results

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Updated spec sheet with analysis results: {spec_sheet_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec sheet analysis completion: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_sheet_analysis_completion",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_sheet_evolution_suggested(self, event: BaseEvent):
        """
        Handle spec sheet evolution suggested events.

        Args:
            event: Event containing spec sheet evolution suggestion details
        """
        try:
            self.logger.info(f"Handling spec sheet evolution suggestion: {event.event_id}")

            # Extract spec details from event
            spec_sheet_id = event.payload.get("spec_sheet_id")
            suggestions = event.payload.get("suggestions", [])

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return False

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")
                return False

            # Store suggestions
            spec["evolution_suggestions"] = suggestions

            # Update spec in registry
            await self.registry.update_spec(spec_sheet_id, {})

            self.logger.info(f"Updated spec sheet with evolution suggestions: {spec_sheet_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error handling spec sheet evolution suggestion: {e}", exc_info=True)
            if self.enable_events and hasattr(self, "event_emitter"):
                await self.event_emitter.emit_async(
                    BaseEvent(
                        event_type="ERROR",
                        source_container=Components.SPEC_REGISTRY,
                        payload={
                            "error": str(e),
                            "operation": "spec_sheet_evolution_suggestion",
                            "event_id": getattr(event, "event_id", None),
                        },
                    )
                )
            return False

    async def _handle_spec_sheet_evolution_applied(self, event: BaseEvent):
        """
        Handle spec sheet evolution applied events.

        Args:
            event: Event containing spec sheet evolution application details
        """
        try:
            self.logger.info(f"Handling spec sheet evolution application: {event.event_id}")

            # Extract spec details from event
            spec_sheet_id = event.payload.get("spec_sheet_id")
            new_template_type = event.payload.get("new_template_type")
            new_template_version = event.payload.get("new_template_version")
            keep_extra_fields = event.payload.get("keep_extra_fields", True)

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return False

            # Get the spec from registry
            spec = await self.registry.get_spec(spec_sheet_id)
            if not spec:
                self.logger.error(f"Spec sheet {spec_sheet_id} not found")