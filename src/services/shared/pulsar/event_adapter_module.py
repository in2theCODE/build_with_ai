"""
Spec Registry Event Adapter Module.

This module adapts the Pulsar Event Bus to work with the Spec Registry System,
enabling event-driven communication between services and support for the
spec sheet workflow.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
import os
import uuid

# Import the event bus implementation
from src.services.shared.pulsar.event_bus import PulsarEventBus

# Import the spec registry system instead of template registry
from src.services.spec_registry import SpecRegistry

# Import the spec sheet workflow - might need to create this if it doesn't exist
from src.services.shared.workflows.spec_sheet_workflow import SpecSheetWorkflow, SpecSheet

logger = logging.getLogger(__name__)


class SpecRegistryEventAdapter:
    """
    Adapter that connects the Spec Registry system to an event bus,
    enabling event-driven communication and supporting the spec sheet workflow.
    """

    def __init__(
            self,
            base_dir: str = "./spec_registry_data",
            pulsar_service_url: str = "pulsar://localhost:6650",
            topic_prefix: str = "persistent://public/default/spec-",
            enable_events: bool = True
    ):
        """
        Initialize the Spec Registry Event Adapter.

        Args:
            base_dir: Base directory for spec registry data
            pulsar_service_url: Pulsar service URL
            topic_prefix: Prefix for Pulsar topics
            enable_events: Whether to enable event-driven communication
        """
        self.base_dir = base_dir
        self.pulsar_service_url = pulsar_service_url
        self.topic_prefix = topic_prefix
        self.enable_events = enable_events

        # Initialize event bus
        self.event_bus = PulsarEventBus(
            service_url=pulsar_service_url,
            topic_prefix=topic_prefix
        )

        # Initialize spec registry
        self.registry = SpecRegistry()

        # Initialize event handlers
        self._setup_event_handlers()

        # Initialize spec sheet workflow
        self.workflow = SpecSheetWorkflow(
            base_dir=os.path.join(base_dir, "workflow"),
            spec_registry=self.registry
        )

    async def start(self) -> bool:
        """
        Start the event adapter and underlying services.

        Returns:
            bool: True if successfully started, False otherwise
        """
        try:
            # Start event bus
            started = await self.event_bus.start()
            if not started:
                logger.error("Failed to start event bus")
                return False

            logger.info("Started Spec Registry Event Adapter")
            return True

        except Exception as e:
            logger.error(f"Error starting Spec Registry Event Adapter: {e}")
            await self.stop()
            return False

    async def stop(self) -> None:
        """
        Stop the event adapter and underlying services.
        """
        try:
            # Stop event bus
            await self.event_bus.stop()

            logger.info("Stopped Spec Registry Event Adapter")

        except Exception as e:
            logger.error(f"Error stopping Spec Registry Event Adapter: {e}")

    def _setup_event_handlers(self) -> None:
        """
        Set up event handlers for various event types.
        """
        if not self.enable_events:
            return

        # Subscribe to request_spec_sheets events
        self.event_bus.subscribe(
            event_types=["request_spec_sheets"],
            handler=self._handle_request_spec_sheets,
            subscription_name="spec_sheets_handler",
            subscription_type="exclusive"
        )

        # Subscribe to fill_spec_sheet events
        self.event_bus.subscribe(
            event_types=["fill_spec_sheet"],
            handler=self._handle_fill_spec_sheet,
            subscription_name="fill_spec_handler",
            subscription_type="exclusive"
        )

        # Subscribe to generate_code events
        self.event_bus.subscribe(
            event_types=["generate_code"],
            handler=self._handle_generate_code,
            subscription_name="generate_code_handler",
            subscription_type="exclusive"
        )

        # Subscribe to assemble_application events
        self.event_bus.subscribe(
            event_types=["assemble_application"],
            handler=self._handle_assemble_application,
            subscription_name="assemble_app_handler",
            subscription_type="exclusive"
        )

        # Subscribe to spec_feedback events
        self.event_bus.subscribe(
            event_types=["spec_feedback"],
            handler=self._handle_spec_feedback,
            subscription_name="feedback_handler",
            subscription_type="shared"
        )

    async def _handle_request_spec_sheets(self, event: Dict[str, Any]) -> None:
        """
        Handle request_spec_sheets events.

        Args:
            event: Event data
        """
        try:
            payload = event.get("payload", {})
            project_requirements = payload.get("project_requirements", {})

            # Generate spec sheets
            spec_sheets = self.workflow.request_spec_sheets(project_requirements)

            # Publish response event
            await self.event_bus.publish_event(
                event_type="spec_sheets_response",
                payload={
                    "success": True,
                    "spec_sheet_count": len(spec_sheets),
                    "spec_sheets": [sheet.id for sheet in spec_sheets],
                    "request_id": event.get("event_id")
                },
                correlation_id=event.get("event_id")
            )

            logger.info(f"Handled request_spec_sheets event {event.get('event_id')}")

        except Exception as e:
            logger.error(f"Error handling request_spec_sheets event: {e}")

            # Publish error event
            await self.event_bus.publish_event(
                event_type="error",
                payload={
                    "success": False,
                    "error": str(e),
                    "request_id": event.get("event_id")
                },
                correlation_id=event.get("event_id")
            )

    async def _handle_fill_spec_sheet(self, event: Dict[str, Any]) -> None:
        """
        Handle fill_spec_sheet events.

        Args:
            event: Event data
        """
        try:
            payload = event.get("payload", {})
            spec_id = payload.get("spec_id")
            values = payload.get("values", {})

            if not spec_id:
                raise ValueError("Missing spec_id in payload")

            # Fill spec sheet
            spec_sheet = await self.registry.update_spec(spec_id, values)

            # Publish response event
            await self.event_bus.publish_event(
                event_type="fill_spec_response",
                payload={
                    "success": bool(spec_sheet),
                    "spec_id": spec_id,
                    "filled": bool(spec_sheet),
                    "request_id": event.get("event_id")
                },
                correlation_id=event.get("event_id")
            )

            logger.info(f"Handled fill_spec_sheet event {event.get('event_id')} for spec {spec_id}")

        except Exception as e:
            logger.error(f"Error handling fill_spec_sheet event: {e}")

            # Publish error event
            await self.event_bus.publish_event(
                event_type="error",
                payload={
                    "success": False,
                    "error": str(e),
                    "request_id": event.get("event_id")
                },
                correlation_id=event.get("event_id")
            )

    async def _handle_generate_code(self, event: Dict[str, Any]) -> None:
        """
        Handle generate_code events.

        Args:
            event: Event data
        """
        try:
            payload = event.get("payload", {})
            spec_id = payload.get("spec_id")

            if not spec_id:
                # Check if we need to generate all code
                if payload.get("generate_all", False):
                    # Generate code for all completed specs
                    results = self.workflow.generate_all_code()

                    # Publish response event
                    await self.event_bus.publish_event(
                        event_type="generate_code_response",
                        payload={
                            "success": True,
                            "generate_all": True,
                            "results": results,
                            "request_id": event.get("event_id")
                        },
                        correlation_id=event.get("event_id")
                    )

                    logger.info(f"Handled generate_all_code event {event.get('event_id')}")
                    return
                else:
                    raise ValueError("Missing spec_id in payload")

            # Generate code for specific spec
            result = self.workflow.generate_code_from_spec(spec_id)

            # Publish response event
            await self.event_bus.publish_event(
                event_type="generate_code_response",
                payload={
                    "success": result.get("success", False),
                    "spec_id": spec_id,
                    "result": result,
                    "request_id": event.get("event_id")
                },
                correlation_id=event.get("event_id")
            )

            logger.info(f"Handled generate_code event {event.get('event_id')} for spec {spec_id}")

        except Exception as e:
            logger.error(f"Error handling generate_code event: {e}")

            # Publish error event
            await self.event_bus.publish_event(
                event_type="error",
                payload={
                    "success": False,
                    "error": str(e),
                    "request_id": event.get("event_id")
                },
                correlation_id=event.get("event_id")
            )

    async def _handle_assemble_application(self, event: Dict[str, Any]) -> None:
        """
        Handle assemble_application events.

        Args:
            event: Event data
        """
        try:
            payload = event.get("payload", {})
            output_dir = payload.get("output_dir", "./assembled_application")

            # Assemble application
            result = self.workflow.assemble_application(output_dir)

            # Publish response event
            await self.event_bus.publish_event(
                event_type="assemble_application_response",
                payload={
                    "success": result.get("success", False),
                    "output_dir": output_dir,
                    "result": result,
                    "request_id": event.get("event_id")
                },
                correlation_id=event.get("event_id")
            )

            logger.info(f"Handled assemble_application event {event.get('event_id')}")

        except Exception as e:
            logger.error(f"Error handling assemble_application event: {e}")

            # Publish error event
            await self.event_bus.publish_event(
                event_type="error",
                payload={
                    "success": False,
                    "error": str(e),
                    "request_id": event.get("event_id")
                },
                correlation_id=event.get("event_id")
            )

    async def _handle_spec_feedback(self, event: Dict[str, Any]) -> None:
        """
        Handle spec_feedback events.

        Args:
            event: Event data
        """
        try:
            payload = event.get("payload", {})
            spec_id = payload.get("spec_id")
            feedback_data = payload.get("feedback", {})

            if not spec_id:
                raise ValueError("Missing spec_id in payload")

            # Store feedback - in a real implementation, you'd want to add this to the spec registry
            # For now, we'll just log it
            logger.info(f"Received feedback for spec {spec_id}: {feedback_data}")

            # Publish response event
            await self.event_bus.publish_event(
                event_type="spec_feedback_response",
                payload={
                    "success": True,
                    "spec_id": spec_id,
                    "request_id": event.get("event_id")
                },
                correlation_id=event.get("event_id")
            )

            logger.info(f"Handled spec_feedback event {event.get('event_id')} for spec {spec_id}")

        except Exception as e:
            logger.error(f"Error handling spec_feedback event: {e}")

            # Publish error event
            await self.event_bus.publish_event(
                event_type="error",
                payload={
                    "success": False,
                    "error": str(e),
                    "request_id": event.get("event_id")
                },
                correlation_id=event.get("event_id")
            )

    async def request_spec_sheets(self, project_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request spec sheets for a project.

        Args:
            project_requirements: Project requirements

        Returns:
            Response from the workflow
        """
        if not self.enable_events:
            # Call workflow directly
            spec_sheets = self.workflow.request_spec_sheets(project_requirements)
            return {
                "success": True,
                "spec_sheet_count": len(spec_sheets),
                "spec_sheets": [sheet.id for sheet in spec_sheets]
            }

        # Generate event ID
        event_id = str(uuid.uuid4())

        # Publish event
        await self.event_bus.publish_event(
            event_type="request_spec_sheets",
            payload={
                "project_requirements": project_requirements
            },
            event_id=event_id
        )

        logger.info(f"Published request_spec_sheets event {event_id}")

        return {
            "event_id": event_id,
            "pending": True
        }

    async def fill_spec_sheet(self, spec_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill a spec sheet with values.

        Args:
            spec_id: ID of the spec sheet
            values: Values to fill in

        Returns:
            Response from the workflow
        """
        if not self.enable_events:
            # Call registry directly
            spec_sheet = await self.registry.update_spec(spec_id, values)
            return {
                "success": bool(spec_sheet),
                "spec_id": spec_id,
                "filled": bool(spec_sheet)
            }

        # Generate event ID
        event_id = str(uuid.uuid4())

        # Publish event
        await self.event_bus.publish_event(
            event_type="fill_spec_sheet",
            payload={
                "spec_id": spec_id,
                "values": values
            },
            event_id=event_id
        )

        logger.info(f"Published fill_spec_sheet event {event_id} for spec {spec_id}")

        return {
            "event_id": event_id,
            "pending": True
        }

    async def generate_code(self, spec_id: Optional[str] = None, generate_all: bool = False) -> Dict[str, Any]:
        """
        Generate code from spec sheets.

        Args:
            spec_id: ID of the spec sheet (or None to generate all)
            generate_all: Whether to generate code for all completed specs

        Returns:
            Response from the workflow
        """
        if not self.enable_events:
            # Call workflow directly
            if generate_all or not spec_id:
                results = self.workflow.generate_all_code()
                return {
                    "success": True,
                    "generate_all": True,
                    "results": results
                }
            else:
                result = self.workflow.generate_code_from_spec(spec_id)
                return {
                    "success": result.get("success", False),
                    "spec_id": spec_id,
                    "result": result
                }

        # Generate event ID
        event_id = str(uuid.uuid4())

        # Publish event
        await self.event_bus.publish_event(
            event_type="generate_code",
            payload={
                "spec_id": spec_id,
                "generate_all": generate_all
            },
            event_id=event_id
        )

        if spec_id:
            logger.info(f"Published generate_code event {event_id} for spec {spec_id}")
        else:
            logger.info(f"Published generate_all_code event {event_id}")

        return {
            "event_id": event_id,
            "pending": True
        }

    async def assemble_application(self, output_dir: str = "./assembled_application") -> Dict[str, Any]:
        """
        Assemble the application from generated code.

        Args:
            output_dir: Directory to output the assembled application

        Returns:
            Response from the workflow
        """
        if not self.enable_events:
            # Call workflow directly
            result = self.workflow.assemble_application(output_dir)
            return result

        # Generate event ID
        event_id = str(uuid.uuid4())

        # Publish event
        await self.event_bus.publish_event(
            event_type="assemble_application",
            payload={
                "output_dir": output_dir
            },
            event_id=event_id
        )

        logger.info(f"Published assemble_application event {event_id}")

        return {
            "event_id": event_id,
            "pending": True
        }

    async def submit_spec_feedback(self, spec_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit feedback for a spec.

        Args:
            spec_id: ID of the spec
            feedback: Feedback data

        Returns:
            Response
        """
        if not self.enable_events:
            # Just log the feedback for now
            logger.info(f"Received feedback for spec {spec_id}: {feedback}")
            return {
                "success": True,
                "spec_id": spec_id
            }

        # Generate event ID
        event_id = str(uuid.uuid4())

        # Publish event
        await self.event_bus.publish_event(
            event_type="spec_feedback",
            payload={
                "spec_id": spec_id,
                "feedback": feedback
            },
            event_id=event_id
        )

        logger.info(f"Published spec_feedback event {event_id} for spec {spec_id}")

        return {
            "event_id": event_id,
            "pending": True
        }