#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Workflow Orchestrator Service - Manages the workflow phases for the spec-driven code generation system

This service tracks project progress through different phases and coordinates the overall workflow.
"""

import asyncio
from datetime import datetime
from datetime import timezone
from enum import Enum
import json
import logging
import os
from typing import Any, Dict, List, Optional

import pulsar
from src.services.shared.constants.models import (
    SpecSheetCompletionRequestMessage,
)
from src.services.shared.constants.models import (
    SpecSheetGenerationRequestMessage,
)

# Import shared schemas
from src.services.shared.constants.models import CodeGenerationRequestMessage
from src.services.shared.constants.models import ProjectAnalysisRequestMessage
from src.services.shared.constants.models import ProjectCreatedMessage
from src.services.shared.constants.models import ProjectStatus
from src.services.shared.constants.models import SystemEventMessage


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WorkflowPhase(str, Enum):
    """Workflow phases for the spec-driven code generation system"""

    INITIALIZATION = "initialization"
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    SPEC_SHEET_GENERATION = "spec_sheet_generation"
    SPEC_SHEET_COMPLETION = "spec_sheet_completion"
    SPEC_VALIDATION = "spec_validation"
    CODE_GENERATION = "code_generation"
    CODE_VERIFICATION = "code_verification"
    CODE_OPTIMIZATION = "code_optimization"
    INTEGRATION = "integration"
    TESTING = "testing"
    DEBUGGING = "debugging"
    FINALIZATION = "finalization"


class WorkflowEventType(str, Enum):
    """Event types for the workflow orchestrator"""

    # Phase 1: Spec Sheet Generation
    PROJECT_CREATED = "project.created"
    PROJECT_ANALYSIS_STARTED = "project.analysis.started"
    PROJECT_ANALYSIS_COMPLETED = "project.analysis.completed"
    SPEC_SHEETS_GENERATION_STARTED = "spec_sheets.generation.started"
    SPEC_SHEETS_GENERATED = "spec_sheets.generated"

    # Phase 2: Spec Sheet Completion
    SPEC_SHEET_COMPLETION_STARTED = "spec_sheet.completion.started"
    SPEC_SHEET_COMPLETED = "spec_sheet.completed"
    ALL_SPEC_SHEETS_COMPLETED = "spec_sheets.all_completed"
    SPEC_VALIDATION_STARTED = "spec.validation.started"
    SPEC_VALIDATION_COMPLETED = "spec.validation.completed"

    # Phase 3: Code Generation
    CODE_GENERATION_STARTED = "code.generation.started"
    COMPONENT_GENERATION_COMPLETED = "component.generation.completed"
    CODE_VERIFICATION_STARTED = "code.verification.started"
    CODE_VERIFICATION_COMPLETED = "code.verification.completed"
    CODE_OPTIMIZATION_STARTED = "code.optimization.started"
    CODE_OPTIMIZATION_COMPLETED = "code.optimization.completed"
    CODE_GENERATION_COMPLETED = "code.generation.completed"

    # Phase 4: Integration & Testing
    INTEGRATION_STARTED = "integration.started"
    INTEGRATION_COMPLETED = "integration.completed"
    TEST_GENERATION_STARTED = "test.generation.started"
    TEST_GENERATION_COMPLETED = "test.generation.completed"
    TEST_EXECUTION_STARTED = "test.execution.started"
    TEST_EXECUTION_COMPLETED = "test.execution.completed"
    DEBUGGING_REQUESTED = "debugging.requested"
    DEBUGGING_COMPLETED = "debugging.completed"
    APPLICATION_FINALIZED = "application.finalized"

    # Error and Support
    ERROR_OCCURRED = "error.occurred"
    ASSISTANCE_REQUESTED = "assistance.requested"
    ASSISTANCE_PROVIDED = "assistance.provided"


class WorkflowOrchestrator:
    """
    Service that orchestrates the workflow of projects through different phases.
    """

    def __init__(
        self,
        pulsar_url: str = "pulsar://pulsar:6650",
        storage_dir: str = "/app/storage",
        project_events_topic: str = "persistent://public/default/project_events",
        spec_sheet_events_topic: str = "persistent://public/default/spec_sheet_events",
        code_gen_events_topic: str = "persistent://public/default/code_generation_events",
        integration_events_topic: str = "persistent://public/default/integration_events",
        test_events_topic: str = "persistent://public/default/test_events",
        workflow_commands_topic: str = "persistent://public/default/workflow_commands",
        assistance_events_topic: str = "persistent://public/default/assistance_events",
    ):
        """
        Initialize the workflow orchestrator.

        Args:
            pulsar_url: Pulsar service URL
            storage_dir: Directory for storing project state
            project_events_topic: Topic for project-related events
            spec_sheet_events_topic: Topic for spec sheet-related events
            code_gen_events_topic: Topic for code generation events
            integration_events_topic: Topic for integration events
            test_events_topic: Topic for test events
            workflow_commands_topic: Topic for workflow commands
            assistance_events_topic: Topic for assistance events
        """
        self.pulsar_url = pulsar_url
        self.storage_dir = storage_dir
        self.project_events_topic = project_events_topic
        self.spec_sheet_events_topic = spec_sheet_events_topic
        self.code_gen_events_topic = code_gen_events_topic
        self.integration_events_topic = integration_events_topic
        self.test_events_topic = test_events_topic
        self.workflow_commands_topic = workflow_commands_topic
        self.assistance_events_topic = assistance_events_topic

        # Pulsar connections
        self.client = None
        self.producers = {}
        self.consumers = {}

        # Project state tracking
        self.project_states: Dict[str, Dict[str, Any]] = {}
        self.project_state_file = os.path.join(storage_dir, "workflow_state.json")

        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)

        # Load existing state if available
        self._load_project_states()

    def _load_project_states(self):
        """Load project states from disk if available."""
        if os.path.exists(self.project_state_file):
            try:
                with open(self.project_state_file, "r") as f:
                    self.project_states = json.load(f)
                logger.info(f"Loaded state for {len(self.project_states)} projects")
            except Exception as e:
                logger.error(f"Failed to load project states: {e}")

    def _save_project_states(self):
        """Save project states to disk."""
        try:
            with open(self.project_state_file, "w") as f:
                json.dump(self.project_states, f, indent=2)
            logger.info(f"Saved state for {len(self.project_states)} projects")
        except Exception as e:
            logger.error(f"Failed to save project states: {e}")

    async def initialize(self):
        """Initialize Pulsar connections."""
        try:
            # Create Pulsar client
            self.client = pulsar.Client(self.pulsar_url)

            # Create producers
            self.producers["project_commands"] = self.client.create_producer(
                topic="persistent://public/default/project_commands",
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=10,
            )

            self.producers["spec_sheet_commands"] = self.client.create_producer(
                topic="persistent://public/default/spec_sheet_commands",
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=10,
            )

            self.producers["code_gen_commands"] = self.client.create_producer(
                topic="persistent://public/default/code_gen_commands",
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=10,
            )

            self.producers["integration_commands"] = self.client.create_producer(
                topic="persistent://public/default/integration_commands",
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=10,
            )

            self.producers["test_commands"] = self.client.create_producer(
                topic="persistent://public/default/test_commands",
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=10,
            )

            self.producers["system_events"] = self.client.create_producer(
                topic="persistent://public/default/system_events",
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=10,
            )

            # Create consumers
            await self.create_consumer(
                "project_events", self.project_events_topic, self.handle_project_event
            )

            await self.create_consumer(
                "spec_sheet_events", self.spec_sheet_events_topic, self.handle_spec_sheet_event
            )

            await self.create_consumer(
                "code_gen_events", self.code_gen_events_topic, self.handle_code_gen_event
            )

            await self.create_consumer(
                "integration_events", self.integration_events_topic, self.handle_integration_event
            )

            await self.create_consumer(
                "test_events", self.test_events_topic, self.handle_test_event
            )

            await self.create_consumer(
                "workflow_commands", self.workflow_commands_topic, self.handle_workflow_command
            )

            await self.create_consumer(
                "assistance_events", self.assistance_events_topic, self.handle_assistance_event
            )

            logger.info(f"Initialized Pulsar connections to {self.pulsar_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Pulsar client: {e}")
            await self.close()
            raise

    async def create_consumer(self, name: str, topic: str, callback):
        """Create a consumer for a topic with a callback."""
        self.consumers[name] = self.client.subscribe(
            topic=topic,
            subscription_name=f"workflow-orchestrator-{name}",
            consumer_type=pulsar.ConsumerType.Shared,
        )

        # Start consumer task
        asyncio.create_task(self.run_consumer(name, callback))
        logger.info(f"Started consumer for {name}")

    async def run_consumer(self, name: str, callback):
        """Run a consumer in a loop."""
        consumer = self.consumers[name]

        while True:
            try:
                # Receive message with timeout
                msg = consumer.receive(timeout_millis=1000)
                try:
                    # Process message
                    await callback(msg)

                    # Acknowledge message
                    consumer.acknowledge(msg)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    consumer.negative_acknowledge(msg)
            except Exception as e:
                # Handle timeout or other errors
                if "timeout" not in str(e).lower():
                    logger.error(f"Error receiving messages: {e}")
                    await asyncio.sleep(1)  # Avoid tight loop if there's an error

    async def close(self):
        """Close Pulsar connections."""
        try:
            # Close producers
            for producer_name, producer in self.producers.items():
                try:
                    producer.close()
                except Exception as e:
                    logger.error(f"Error closing producer {producer_name}: {e}")

            # Close consumers
            for consumer_name, consumer in self.consumers.items():
                try:
                    consumer.close()
                except Exception as e:
                    logger.error(f"Error closing consumer {consumer_name}: {e}")

            # Close client
            if self.client:
                self.client.close()

            # Clear references
            self.producers = {}
            self.consumers = {}
            self.client = None

            logger.info("Closed all Pulsar connections")
        except Exception as e:
            logger.error(f"Error closing Pulsar client: {e}")

    async def handle_project_event(self, msg):
        """Handle project-related events."""
        try:
            # Parse message data
            data = msg.data().decode("utf-8")
            payload = json.loads(data)
            event_type = payload.get("event_type")

            logger.info(f"Received project event: {event_type}")

            # Handle different event types
            if event_type == WorkflowEventType.PROJECT_CREATED.value:
                # Use the ProjectCreatedMessage to properly deserialize
                project_message = ProjectCreatedMessage.from_json(data)
                await self.handle_project_created(project_message)
            elif event_type == WorkflowEventType.PROJECT_ANALYSIS_STARTED.value:
                await self.handle_project_analysis_started(payload)
            elif event_type == WorkflowEventType.PROJECT_ANALYSIS_COMPLETED.value:
                await self.handle_project_analysis_completed(payload)
            elif event_type == "PROJECT_UPDATED":
                await self.handle_project_updated(payload)
            elif event_type == "PROJECT_DELETED":
                await self.handle_project_deleted(payload)
            else:
                logger.warning(f"Unknown project event type: {event_type}")

        except Exception as e:
            logger.error(f"Error handling project event: {e}")
            raise

    async def handle_spec_sheet_event(self, msg):
        """Handle spec sheet-related events."""
        try:
            # Parse message data
            payload = json.loads(msg.data().decode("utf-8"))
            event_type = payload.get("event_type")
            project_id = payload.get("project_id")

            logger.info(f"Received spec sheet event: {event_type} for project {project_id}")

            # Handle different event types
            if event_type == WorkflowEventType.SPEC_SHEETS_GENERATION_STARTED.value:
                await self.handle_spec_sheets_generation_started(payload)
            elif event_type == WorkflowEventType.SPEC_SHEETS_GENERATED.value:
                await self.handle_spec_sheets_generated(payload)
            elif event_type == WorkflowEventType.SPEC_SHEET_COMPLETION_STARTED.value:
                await self.handle_spec_sheet_completion_started(payload)
            elif event_type == WorkflowEventType.SPEC_SHEET_COMPLETED.value:
                await self.handle_spec_sheet_completed(payload)
            elif event_type == WorkflowEventType.ALL_SPEC_SHEETS_COMPLETED.value:
                await self.handle_all_spec_sheets_completed(payload)
            elif event_type == WorkflowEventType.SPEC_VALIDATION_STARTED.value:
                await self.handle_spec_validation_started(payload)
            elif event_type == WorkflowEventType.SPEC_VALIDATION_COMPLETED.value:
                await self.handle_spec_validation_completed(payload)
            else:
                logger.warning(f"Unknown spec sheet event type: {event_type}")

        except Exception as e:
            logger.error(f"Error handling spec sheet event: {e}")
            raise

    async def handle_code_gen_event(self, msg):
        """Handle code generation events."""
        try:
            # Parse message data
            payload = json.loads(msg.data().decode("utf-8"))
            event_type = payload.get("event_type")
            project_id = payload.get("project_id")

            logger.info(f"Received code generation event: {event_type} for project {project_id}")

            # Handle different event types
            if event_type == WorkflowEventType.CODE_GENERATION_STARTED.value:
                await self.handle_code_generation_started(payload)
            elif event_type == WorkflowEventType.COMPONENT_GENERATION_COMPLETED.value:
                await self.handle_component_generation_completed(payload)
            elif event_type == WorkflowEventType.CODE_VERIFICATION_STARTED.value:
                await self.handle_code_verification_started(payload)
            elif event_type == WorkflowEventType.CODE_VERIFICATION_COMPLETED.value:
                await self.handle_code_verification_completed(payload)
            elif event_type == WorkflowEventType.CODE_OPTIMIZATION_STARTED.value:
                await self.handle_code_optimization_started(payload)
            elif event_type == WorkflowEventType.CODE_OPTIMIZATION_COMPLETED.value:
                await self.handle_code_optimization_completed(payload)
            elif event_type == WorkflowEventType.CODE_GENERATION_COMPLETED.value:
                await self.handle_code_generation_completed(payload)
            elif event_type == "CODE_GENERATION_FAILED":
                await self.handle_code_generation_failed(payload)
            else:
                logger.warning(f"Unknown code generation event type: {event_type}")

        except Exception as e:
            logger.error(f"Error handling code generation event: {e}")
            raise

    async def handle_integration_event(self, msg):
        """Handle integration events."""
        try:
            # Parse message data
            payload = json.loads(msg.data().decode("utf-8"))
            event_type = payload.get("event_type")
            project_id = payload.get("project_id")

            logger.info(f"Received integration event: {event_type} for project {project_id}")

            # Handle different event types
            if event_type == WorkflowEventType.INTEGRATION_STARTED.value:
                await self.handle_integration_started(payload)
            elif event_type == WorkflowEventType.INTEGRATION_COMPLETED.value:
                await self.handle_integration_completed(payload)
            else:
                logger.warning(f"Unknown integration event type: {event_type}")

        except Exception as e:
            logger.error(f"Error handling integration event: {e}")
            raise

    async def handle_test_event(self, msg):
        """Handle test events."""
        try:
            # Parse message data
            payload = json.loads(msg.data().decode("utf-8"))
            event_type = payload.get("event_type")
            project_id = payload.get("project_id")

            logger.info(f"Received test event: {event_type} for project {project_id}")

            # Handle different event types
            if event_type == WorkflowEventType.TEST_GENERATION_STARTED.value:
                await self.handle_test_generation_started(payload)
            elif event_type == WorkflowEventType.TEST_GENERATION_COMPLETED.value:
                await self.handle_test_generation_completed(payload)
            elif event_type == WorkflowEventType.TEST_EXECUTION_STARTED.value:
                await self.handle_test_execution_started(payload)
            elif event_type == WorkflowEventType.TEST_EXECUTION_COMPLETED.value:
                await self.handle_test_execution_completed(payload)
            elif event_type == WorkflowEventType.DEBUGGING_REQUESTED.value:
                await self.handle_debugging_requested(payload)
            elif event_type == WorkflowEventType.DEBUGGING_COMPLETED.value:
                await self.handle_debugging_completed(payload)
            elif event_type == WorkflowEventType.APPLICATION_FINALIZED.value:
                await self.handle_application_finalized(payload)
            else:
                logger.warning(f"Unknown test event type: {event_type}")

        except Exception as e:
            logger.error(f"Error handling test event: {e}")
            raise

    async def handle_assistance_event(self, msg):
        """Handle assistance events."""
        try:
            # Parse message data
            payload = json.loads(msg.data().decode("utf-8"))
            event_type = payload.get("event_type")
            project_id = payload.get("project_id")

            logger.info(f"Received assistance event: {event_type} for project {project_id}")

            # Handle different event types
            if event_type == WorkflowEventType.ASSISTANCE_REQUESTED.value:
                await self.handle_assistance_requested(payload)
            elif event_type == WorkflowEventType.ASSISTANCE_PROVIDED.value:
                await self.handle_assistance_provided(payload)
            else:
                logger.warning(f"Unknown assistance event type: {event_type}")

        except Exception as e:
            logger.error(f"Error handling assistance event: {e}")
            raise

    async def handle_workflow_command(self, msg):
        """Handle workflow commands."""
        try:
            # Parse message data
            payload = json.loads(msg.data().decode("utf-8"))
            command = payload.get("command")
            project_id = payload.get("project_id")

            logger.info(f"Received workflow command: {command} for project {project_id}")

            # Handle different commands
            if command == "ANALYZE_PROJECT":
                await self.start_project_analysis(project_id)
            elif command == "GENERATE_SPEC_SHEETS":
                await self.start_spec_sheet_generation(project_id)
            elif command == "COMPLETE_SPEC_SHEETS":
                await self.start_spec_sheet_completion(
                    project_id, payload.get("spec_sheet_ids", [])
                )
            elif command == "VALIDATE_SPEC_SHEETS":
                await self.start_spec_sheet_validation(
                    project_id, payload.get("spec_sheet_ids", [])
                )
            elif command == "GENERATE_CODE":
                await self.start_code_generation(project_id, payload.get("spec_sheet_ids", []))
            elif command == "VERIFY_CODE":
                await self.start_code_verification(project_id)
            elif command == "OPTIMIZE_CODE":
                await self.start_code_optimization(project_id)
            elif command == "START_INTEGRATION":
                await self.start_integration(project_id)
            elif command == "GENERATE_TESTS":
                await self.start_test_generation(project_id)
            elif command == "EXECUTE_TESTS":
                await self.start_test_execution(project_id)
            elif command == "REQUEST_DEBUGGING":
                await self.request_debugging(project_id, payload.get("issue_description"))
            elif command == "FINALIZE_APPLICATION":
                await self.finalize_application(project_id)
            else:
                logger.warning(f"Unknown workflow command: {command}")

        except Exception as e:
            logger.error(f"Error handling workflow command: {e}")
            raise

    async def handle_project_created(self, message: ProjectCreatedMessage):
        """Handle project created event."""
        project_id = message.project_id

        # Initialize project state
        self.project_states[project_id] = {
            "status": ProjectStatus.INITIALIZING.value,
            "created_at": message.created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "spec_sheet_ids": [],
            "code_generation_ids": [],
            "current_phase": WorkflowPhase.INITIALIZATION.value,
        }

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} created and initialized")

        # Automatically start project analysis (optional)
        # await self.start_project_analysis(project_id)

    async def handle_project_analysis_started(self, payload):
        """Handle project analysis started event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(f"Received analysis start for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.ANALYZING.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.REQUIREMENTS_ANALYSIS.value,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} analysis started")

    async def handle_project_analysis_completed(self, payload):
        """Handle project analysis completed event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(f"Received analysis completion for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.ANALYZING.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.REQUIREMENTS_ANALYSIS.value,
                "analysis_results": payload.get("analysis_results", {}),
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} analysis completed")

        # Automatically start spec sheet generation
        await self.start_spec_sheet_generation(project_id)

    async def handle_project_updated(self, payload):
        """Handle project updated event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(f"Received update for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": payload.get("status", self.project_states[project_id]["status"]),
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} updated")

    async def handle_project_deleted(self, payload):
        """Handle project deleted event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(f"Received deletion for unknown project: {project_id}")
            return

        # Remove project from state
        del self.project_states[project_id]

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} deleted")

    async def handle_spec_sheets_generation_started(self, payload):
        """Handle spec sheets generation started event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(
                f"Received spec sheets generation start for unknown project: {project_id}"
            )
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.INITIALIZING.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.SPEC_SHEET_GENERATION.value,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} spec sheets generation started")

    async def handle_spec_sheets_generated(self, payload):
        """Handle spec sheets generated event."""
        project_id = payload.get("project_id")
        spec_sheet_ids = payload.get("spec_sheet_ids", [])

        if project_id not in self.project_states:
            logger.warning(f"Received spec sheets generated for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.SPEC_SHEETS_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.SPEC_SHEET_GENERATION.value,
                "spec_sheet_ids": spec_sheet_ids,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} spec sheets generated: {len(spec_sheet_ids)} sheets")

        # Automatically start spec sheet completion
        await self.start_spec_sheet_completion(project_id, spec_sheet_ids)

    async def handle_spec_sheet_completion_started(self, payload):
        """Handle spec sheet completion started event."""
        project_id = payload.get("project_id")
        spec_sheet_id = payload.get("spec_sheet_id")

        if project_id not in self.project_states:
            logger.warning(
                f"Received spec sheet completion start for unknown project: {project_id}"
            )
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.SPEC_SHEETS_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.SPEC_SHEET_COMPLETION.value,
                "current_spec_sheet": spec_sheet_id,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} spec sheet {spec_sheet_id} completion started")

    async def handle_spec_sheet_completed(self, payload):
        """Handle individual spec sheet completed event."""
        project_id = payload.get("project_id")
        spec_sheet_id = payload.get("spec_sheet_id")

        if project_id not in self.project_states:
            logger.warning(f"Received spec sheet completion for unknown project: {project_id}")
            return

        # Add completed spec sheet
        if "completed_spec_sheets" not in self.project_states[project_id]:
            self.project_states[project_id]["completed_spec_sheets"] = []

        if spec_sheet_id not in self.project_states[project_id]["completed_spec_sheets"]:
            self.project_states[project_id]["completed_spec_sheets"].append(spec_sheet_id)

        # Update timestamp
        self.project_states[project_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} spec sheet {spec_sheet_id} completed")

        # Check if all spec sheets are completed
        completed_sheets = self.project_states[project_id].get("completed_spec_sheets", [])
        all_sheets = self.project_states[project_id].get("spec_sheet_ids", [])

        if all_sheets and completed_sheets and set(completed_sheets) == set(all_sheets):
            # All spec sheets completed, update project status
            self.project_states[project_id]["status"] = ProjectStatus.SPEC_SHEETS_COMPLETED.value
            self.project_states[project_id][
                "current_phase"
            ] = WorkflowPhase.SPEC_SHEET_COMPLETION.value
            self._save_project_states()

            # Publish event
            await self.publish_system_event(
                WorkflowEventType.ALL_SPEC_SHEETS_COMPLETED.value,
                "workflow_orchestrator",
                f"All spec sheets completed for project {project_id}",
                {"project_id": project_id, "spec_sheet_ids": all_sheets},
            )

            logger.info(f"All spec sheets completed for project {project_id}")

            # Automatically start spec sheet validation
            await self.start_spec_sheet_validation(project_id, all_sheets)

    async def handle_all_spec_sheets_completed(self, payload):
        """Handle all spec sheets completed event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(f"Received all spec sheets completed for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.SPEC_SHEETS_COMPLETED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.SPEC_SHEET_COMPLETION.value,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} all spec sheets completed")

    async def handle_spec_validation_started(self, payload):
        """Handle spec validation started event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(f"Received spec validation start for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.SPEC_SHEETS_COMPLETED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.SPEC_VALIDATION.value,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} spec validation started")

    async def handle_spec_validation_completed(self, payload):
        """Handle spec validation completed event."""
        project_id = payload.get("project_id")
        validation_result = payload.get("validation_result", {})
        is_valid = validation_result.get("is_valid", True)

        if project_id not in self.project_states:
            logger.warning(f"Received spec validation completion for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.SPEC_SHEETS_COMPLETED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.SPEC_VALIDATION.value,
                "validation_result": validation_result,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} spec validation completed, is_valid: {is_valid}")

        # If validation was successful, start code generation
        if is_valid:
            spec_sheet_ids = self.project_states[project_id].get("completed_spec_sheets", [])
            await self.start_code_generation(project_id, spec_sheet_ids)

    async def handle_code_generation_started(self, payload):
        """Handle code generation started event."""
        project_id = payload.get("project_id")
        generation_id = payload.get("generation_id")

        if project_id not in self.project_states:
            logger.warning(f"Received code generation start for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.GENERATING_CODE.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.CODE_GENERATION.value,
                "current_generation_id": generation_id,
            }
        )

        # Add to code generation IDs
        if "code_generation_ids" not in self.project_states[project_id]:
            self.project_states[project_id]["code_generation_ids"] = []

        if generation_id not in self.project_states[project_id]["code_generation_ids"]:
            self.project_states[project_id]["code_generation_ids"].append(generation_id)

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} code generation started with ID {generation_id}")

    async def handle_component_generation_completed(self, payload):
        """Handle component generation completed event."""
        project_id = payload.get("project_id")
        generation_id = payload.get("generation_id")
        component_id = payload.get("component_id")

        if project_id not in self.project_states:
            logger.warning(
                f"Received component generation completion for unknown project: {project_id}"
            )
            return

        # Add completed component
        if "completed_components" not in self.project_states[project_id]:
            self.project_states[project_id]["completed_components"] = []

        self.project_states[project_id]["completed_components"].append(component_id)

        # Update timestamp
        self.project_states[project_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} component {component_id} generation completed")

    async def handle_code_verification_started(self, payload):
        """Handle code verification started event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(f"Received code verification start for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.GENERATING_CODE.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.CODE_VERIFICATION.value,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} code verification started")

    async def handle_code_verification_completed(self, payload):
        """Handle code verification completed event."""
        project_id = payload.get("project_id")
        verification_result = payload.get("verification_result", {})
        is_valid = verification_result.get("is_valid", True)

        if project_id not in self.project_states:
            logger.warning(
                f"Received code verification completion for unknown project: {project_id}"
            )
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.GENERATING_CODE.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.CODE_VERIFICATION.value,
                "verification_result": verification_result,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} code verification completed, is_valid: {is_valid}")

        # If verification was successful, start code optimization
        if is_valid:
            await self.start_code_optimization(project_id)

    async def handle_code_optimization_started(self, payload):
        """Handle code optimization started event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(f"Received code optimization start for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.GENERATING_CODE.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.CODE_OPTIMIZATION.value,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} code optimization started")

    async def handle_code_optimization_completed(self, payload):
        """Handle code optimization completed event."""
        project_id = payload.get("project_id")
        optimization_result = payload.get("optimization_result", {})

        if project_id not in self.project_states:
            logger.warning(
                f"Received code optimization completion for unknown project: {project_id}"
            )
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.GENERATING_CODE.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.CODE_OPTIMIZATION.value,
                "optimization_result": optimization_result,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} code optimization completed")

    async def handle_code_generation_completed(self, payload):
        """Handle code generation completed event."""
        project_id = payload.get("project_id")
        generation_id = payload.get("generation_id")

        if project_id not in self.project_states:
            logger.warning(f"Received code generation completion for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.CODE_GENERATION.value,
                "last_completed_generation_id": generation_id,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} code generation completed for ID {generation_id}")

        # Automatically start integration
        await self.start_integration(project_id)

    async def handle_code_generation_failed(self, payload):
        """Handle code generation failed event."""
        project_id = payload.get("project_id")
        generation_id = payload.get("generation_id")
        error = payload.get("error")

        if project_id not in self.project_states:
            logger.warning(f"Received code generation failure for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.FAILED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": error,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.error(f"Project {project_id} code generation failed for ID {generation_id}: {error}")

    async def handle_integration_started(self, payload):
        """Handle integration started event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(f"Received integration start for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.INTEGRATION.value,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} integration started")

    async def handle_integration_completed(self, payload):
        """Handle integration completed event."""
        project_id = payload.get("project_id")
        integration_result = payload.get("integration_result", {})

        if project_id not in self.project_states:
            logger.warning(f"Received integration completion for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.INTEGRATION.value,
                "integration_result": integration_result,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} integration completed")

        # Automatically start test generation
        await self.start_test_generation(project_id)

    async def handle_test_generation_started(self, payload):
        """Handle test generation started event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(f"Received test generation start for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.TESTING.value,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} test generation started")

    async def handle_test_generation_completed(self, payload):
        """Handle test generation completed event."""
        project_id = payload.get("project_id")
        test_result = payload.get("test_result", {})

        if project_id not in self.project_states:
            logger.warning(f"Received test generation completion for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.TESTING.value,
                "test_result": test_result,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} test generation completed")

        # Automatically start test execution
        await self.start_test_execution(project_id)

    async def handle_test_execution_started(self, payload):
        """Handle test execution started event."""
        project_id = payload.get("project_id")

        if project_id not in self.project_states:
            logger.warning(f"Received test execution start for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.TESTING.value,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} test execution started")

    async def handle_test_execution_completed(self, payload):
        """Handle test execution completed event."""
        project_id = payload.get("project_id")
        execution_result = payload.get("execution_result", {})
        all_tests_passed = execution_result.get("all_passed", False)

        if project_id not in self.project_states:
            logger.warning(f"Received test execution completion for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.TESTING.value,
                "execution_result": execution_result,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(
            f"Project {project_id} test execution completed, all_passed: {all_tests_passed}"
        )

        # If all tests passed, finalize the application
        if all_tests_passed:
            await self.finalize_application(project_id)
        else:
            # If tests failed, request debugging
            failed_tests = execution_result.get("failed_tests", [])
            issue_description = f"Failed tests: {', '.join(failed_tests)}"
            await self.request_debugging(project_id, issue_description)

    async def handle_debugging_requested(self, payload):
        """Handle debugging requested event."""
        project_id = payload.get("project_id")
        issue_description = payload.get("issue_description")

        if project_id not in self.project_states:
            logger.warning(f"Received debugging request for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.DEBUGGING.value,
                "issue_description": issue_description,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} debugging requested: {issue_description}")

    async def handle_debugging_completed(self, payload):
        """Handle debugging completed event."""
        project_id = payload.get("project_id")
        debugging_result = payload.get("debugging_result", {})
        issue_resolved = debugging_result.get("issue_resolved", False)

        if project_id not in self.project_states:
            logger.warning(f"Received debugging completion for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.DEBUGGING.value,
                "debugging_result": debugging_result,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} debugging completed, issue_resolved: {issue_resolved}")

        # If the issue was resolved, run tests again
        if issue_resolved:
            await self.start_test_execution(project_id)
        else:
            # If issue not resolved, request further debugging
            await self.request_debugging(project_id, "Previous debugging attempt unsuccessful")

    async def handle_application_finalized(self, payload):
        """Handle application finalized event."""
        project_id = payload.get("project_id")
        finalization_result = payload.get("finalization_result", {})

        if project_id not in self.project_states:
            logger.warning(f"Received application finalization for unknown project: {project_id}")
            return

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.COMPLETED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.FINALIZATION.value,
                "finalization_result": finalization_result,
            }
        )

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} application finalized")

    async def handle_assistance_requested(self, payload):
        """Handle assistance requested event."""
        project_id = payload.get("project_id")
        query = payload.get("query")

        if project_id not in self.project_states:
            logger.warning(f"Received assistance request for unknown project: {project_id}")
            return

        # Update project state to track assistance requests
        if "assistance_requests" not in self.project_states[project_id]:
            self.project_states[project_id]["assistance_requests"] = []

        self.project_states[project_id]["assistance_requests"].append(
            {"timestamp": datetime.now(timezone.utc).isoformat(), "query": query}
        )

        # Update timestamp
        self.project_states[project_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} assistance requested: {query}")

    async def handle_assistance_provided(self, payload):
        """Handle assistance provided event."""
        project_id = payload.get("project_id")
        response = payload.get("response")

        if project_id not in self.project_states:
            logger.warning(f"Received assistance provision for unknown project: {project_id}")
            return

        # Update project state to track assistance responses
        if "assistance_responses" not in self.project_states[project_id]:
            self.project_states[project_id]["assistance_responses"] = []

        self.project_states[project_id]["assistance_responses"].append(
            {"timestamp": datetime.now(timezone.utc).isoformat(), "response": response}
        )

        # Update timestamp
        self.project_states[project_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Save state
        self._save_project_states()

        # Log event
        logger.info(f"Project {project_id} assistance provided")

    async def start_project_analysis(self, project_id):
        """Start project analysis."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot start analysis for unknown project: {project_id}")
            return

        logger.info(f"Starting analysis for project {project_id}")

        # Publish project analysis started event
        await self.publish_system_event(
            WorkflowEventType.PROJECT_ANALYSIS_STARTED.value,
            "workflow_orchestrator",
            f"Starting analysis for project {project_id}",
            {"project_id": project_id},
        )

        # Create analysis request
        request = ProjectAnalysisRequestMessage(project_id=project_id)

        # Publish to project commands topic
        await self.publish_message(
            "project_commands",
            json.loads(request.to_json()),
            {"action": "analyze", "project_id": project_id},
        )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.ANALYZING.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.REQUIREMENTS_ANALYSIS.value,
            }
        )

        # Save state
        self._save_project_states()

    async def start_spec_sheet_generation(self, project_id):
        """Start spec sheet generation."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot start spec sheet generation for unknown project: {project_id}")
            return

        logger.info(f"Starting spec sheet generation for project {project_id}")

        # Publish spec sheets generation started event
        await self.publish_system_event(
            WorkflowEventType.SPEC_SHEETS_GENERATION_STARTED.value,
            "workflow_orchestrator",
            f"Starting spec sheet generation for project {project_id}",
            {"project_id": project_id},
        )

        # Create spec sheet generation request
        request = SpecSheetGenerationRequestMessage(project_id=project_id)

        # Publish to spec sheet commands topic
        await self.publish_message(
            "spec_sheet_commands",
            json.loads(request.to_json()),
            {"action": "generate", "project_id": project_id},
        )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.INITIALIZING.value,  # Will be updated when generation completes
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.SPEC_SHEET_GENERATION.value,
            }
        )

        # Save state
        self._save_project_states()

    async def start_spec_sheet_completion(self, project_id, spec_sheet_ids=None):
        """Start spec sheet completion."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot start spec sheet completion for unknown project: {project_id}")
            return

        # If no spec_sheet_ids provided, use all from project state
        if not spec_sheet_ids:
            spec_sheet_ids = self.project_states[project_id].get("spec_sheet_ids", [])

        if not spec_sheet_ids:
            logger.warning(f"No spec sheets found for project {project_id}")
            return

        logger.info(
            f"Starting spec sheet completion for project {project_id}, sheets: {spec_sheet_ids}"
        )

        # Publish spec sheet completion started event
        await self.publish_system_event(
            WorkflowEventType.SPEC_SHEET_COMPLETION_STARTED.value,
            "workflow_orchestrator",
            f"Starting spec sheet completion for project {project_id}",
            {"project_id": project_id, "spec_sheet_ids": spec_sheet_ids},
        )

        # For each spec sheet, create a completion request
        for spec_sheet_id in spec_sheet_ids:
            request = SpecSheetCompletionRequestMessage(spec_sheet_id=spec_sheet_id)

            # Publish to spec sheet commands topic
            await self.publish_message(
                "spec_sheet_commands",
                json.loads(request.to_json()),
                {"action": "complete", "project_id": project_id, "spec_sheet_id": spec_sheet_id},
            )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.SPEC_SHEETS_GENERATED.value,  # Will be updated as sheets complete
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.SPEC_SHEET_COMPLETION.value,
            }
        )

        # Save state
        self._save_project_states()

    async def start_spec_sheet_validation(self, project_id, spec_sheet_ids=None):
        """Start spec sheet validation."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot start spec sheet validation for unknown project: {project_id}")
            return

        # If no spec_sheet_ids provided, use completed spec sheets from project state
        if not spec_sheet_ids:
            spec_sheet_ids = self.project_states[project_id].get("completed_spec_sheets", [])

        if not spec_sheet_ids:
            logger.warning(f"No completed spec sheets found for project {project_id}")
            return

        logger.info(
            f"Starting spec sheet validation for project {project_id}, sheets: {spec_sheet_ids}"
        )

        # Publish spec validation started event
        await self.publish_system_event(
            WorkflowEventType.SPEC_VALIDATION_STARTED.value,
            "workflow_orchestrator",
            f"Starting spec validation for project {project_id}",
            {"project_id": project_id, "spec_sheet_ids": spec_sheet_ids},
        )

        # Create spec validation request
        # This would be your actual message structure for validation
        validation_request = {
            "project_id": project_id,
            "spec_sheet_ids": spec_sheet_ids,
            "validation_type": "comprehensive",
        }

        # Publish to spec sheet commands topic
        await self.publish_message(
            "spec_sheet_commands",
            validation_request,
            {"action": "validate", "project_id": project_id},
        )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.SPEC_SHEETS_COMPLETED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.SPEC_VALIDATION.value,
            }
        )

        # Save state
        self._save_project_states()

    async def start_code_generation(self, project_id, spec_sheet_ids=None):
        """Start code generation."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot start code generation for unknown project: {project_id}")
            return

        # If no spec_sheet_ids provided, use completed spec sheets from project state
        if not spec_sheet_ids:
            spec_sheet_ids = self.project_states[project_id].get("completed_spec_sheets", [])

        if not spec_sheet_ids:
            logger.warning(f"No completed spec sheets found for project {project_id}")
            return

        logger.info(f"Starting code generation for project {project_id}, sheets: {spec_sheet_ids}")

        # Publish code generation started event
        await self.publish_system_event(
            WorkflowEventType.CODE_GENERATION_STARTED.value,
            "workflow_orchestrator",
            f"Starting code generation for project {project_id}",
            {"project_id": project_id, "spec_sheet_ids": spec_sheet_ids},
        )

        # Create code generation request
        request = CodeGenerationRequestMessage(project_id=project_id, spec_sheet_ids=spec_sheet_ids)

        # Publish to code gen commands topic
        await self.publish_message(
            "code_gen_commands",
            json.loads(request.to_json()),
            {"action": "generate", "project_id": project_id},
        )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.GENERATING_CODE.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.CODE_GENERATION.value,
            }
        )

        # Save state
        self._save_project_states()

    async def start_code_verification(self, project_id):
        """Start code verification."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot start code verification for unknown project: {project_id}")
            return

        logger.info(f"Starting code verification for project {project_id}")

        # Publish code verification started event
        await self.publish_system_event(
            WorkflowEventType.CODE_VERIFICATION_STARTED.value,
            "workflow_orchestrator",
            f"Starting code verification for project {project_id}",
            {"project_id": project_id},
        )

        # Create verification request
        verification_request = {"project_id": project_id, "verification_type": "static_analysis"}

        # Publish to code gen commands topic
        await self.publish_message(
            "code_gen_commands",
            verification_request,
            {"action": "verify", "project_id": project_id},
        )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.GENERATING_CODE.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.CODE_VERIFICATION.value,
            }
        )

        # Save state
        self._save_project_states()

    async def start_code_optimization(self, project_id):
        """Start code optimization."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot start code optimization for unknown project: {project_id}")
            return

        logger.info(f"Starting code optimization for project {project_id}")

        # Publish code optimization started event
        await self.publish_system_event(
            WorkflowEventType.CODE_OPTIMIZATION_STARTED.value,
            "workflow_orchestrator",
            f"Starting code optimization for project {project_id}",
            {"project_id": project_id},
        )

        # Create optimization request
        optimization_request = {"project_id": project_id, "optimization_level": "medium"}

        # Publish to code gen commands topic
        await self.publish_message(
            "code_gen_commands",
            optimization_request,
            {"action": "optimize", "project_id": project_id},
        )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.GENERATING_CODE.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.CODE_OPTIMIZATION.value,
            }
        )

        # Save state
        self._save_project_states()

    async def start_integration(self, project_id):
        """Start application integration."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot start integration for unknown project: {project_id}")
            return

    async def start_integration(self, project_id):
        """Start application integration."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot start integration for unknown project: {project_id}")
            return

        logger.info(f"Starting integration for project {project_id}")

        # Publish integration started event
        await self.publish_system_event(
            WorkflowEventType.INTEGRATION_STARTED.value,
            "workflow_orchestrator",
            f"Starting integration for project {project_id}",
            {"project_id": project_id},
        )

        # Create integration request
        integration_request = {"project_id": project_id, "integration_type": "component_assembly"}

        # Publish to integration commands topic
        await self.publish_message(
            "integration_commands",
            integration_request,
            {"action": "integrate", "project_id": project_id},
        )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.INTEGRATION.value,
            }
        )

        # Save state
        self._save_project_states()

    async def start_test_generation(self, project_id):
        """Start test generation."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot start test generation for unknown project: {project_id}")
            return

        logger.info(f"Starting test generation for project {project_id}")

        # Publish test generation started event
        await self.publish_system_event(
            WorkflowEventType.TEST_GENERATION_STARTED.value,
            "workflow_orchestrator",
            f"Starting test generation for project {project_id}",
            {"project_id": project_id},
        )

        # Create test generation request
        test_request = {
            "project_id": project_id,
            "test_types": ["unit", "integration", "functional"],
        }

        # Publish to test commands topic
        await self.publish_message(
            "test_commands", test_request, {"action": "generate_tests", "project_id": project_id}
        )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.TESTING.value,
            }
        )

        # Save state
        self._save_project_states()

    async def start_test_execution(self, project_id):
        """Start test execution."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot start test execution for unknown project: {project_id}")
            return

        logger.info(f"Starting test execution for project {project_id}")

        # Publish test execution started event
        await self.publish_system_event(
            WorkflowEventType.TEST_EXECUTION_STARTED.value,
            "workflow_orchestrator",
            f"Starting test execution for project {project_id}",
            {"project_id": project_id},
        )

        # Create test execution request
        execution_request = {"project_id": project_id, "execution_mode": "all"}

        # Publish to test commands topic
        await self.publish_message(
            "test_commands",
            execution_request,
            {"action": "execute_tests", "project_id": project_id},
        )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.TESTING.value,
            }
        )

        # Save state
        self._save_project_states()

    async def request_debugging(self, project_id, issue_description):
        """Request debugging assistance."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot request debugging for unknown project: {project_id}")
            return

        logger.info(f"Requesting debugging for project {project_id}: {issue_description}")

        # Publish debugging requested event
        await self.publish_system_event(
            WorkflowEventType.DEBUGGING_REQUESTED.value,
            "workflow_orchestrator",
            f"Requesting debugging for project {project_id}",
            {"project_id": project_id, "issue_description": issue_description},
        )

        # Create debugging request
        debugging_request = {"project_id": project_id, "issue_description": issue_description}

        # Publish to test commands topic (or could be a dedicated debugging topic)
        await self.publish_message(
            "test_commands", debugging_request, {"action": "debug", "project_id": project_id}
        )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.CODE_GENERATED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.DEBUGGING.value,
                "issue_description": issue_description,
            }
        )

        # Save state
        self._save_project_states()

    async def finalize_application(self, project_id):
        """Finalize the application."""
        if project_id not in self.project_states:
            logger.warning(f"Cannot finalize unknown project: {project_id}")
            return

        logger.info(f"Finalizing application for project {project_id}")

        # Publish application finalized event
        await self.publish_system_event(
            WorkflowEventType.APPLICATION_FINALIZED.value,
            "workflow_orchestrator",
            f"Finalizing application for project {project_id}",
            {"project_id": project_id},
        )

        # Create finalization request
        finalization_request = {"project_id": project_id, "package_type": "production"}

        # Publish to integration commands topic
        await self.publish_message(
            "integration_commands",
            finalization_request,
            {"action": "finalize", "project_id": project_id},
        )

        # Update project state
        self.project_states[project_id].update(
            {
                "status": ProjectStatus.COMPLETED.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_phase": WorkflowPhase.FINALIZATION.value,
            }
        )

        # Save state
        self._save_project_states()

    async def publish_message(self, producer_name, payload, properties=None):
        """Publish a message to a topic."""
        if producer_name not in self.producers:
            raise RuntimeError(f"Producer {producer_name} not initialized")

        # Convert payload to JSON
        if not isinstance(payload, str):
            payload = json.dumps(payload)

        # Publish to topic
        self.producers[producer_name].send(payload.encode("utf-8"), properties=properties or {})

        logger.info(f"Published message to {producer_name}")

    async def publish_system_event(self, event_type, component, message, details=None):
        """Publish a system event."""
        # Create system event message
        event = SystemEventMessage(
            event_type=event_type, component=component, message=message, details=details or {}
        )

        # Publish to system events topic
        await self.publish_message(
            "system_events",
            json.loads(event.to_json()),
            {"event_type": event_type, "component": component},
        )

    async def run_service():
        """Run the workflow orchestrator service."""
        orchestrator = WorkflowOrchestrator(
            pulsar_url=os.environ.get("PULSAR_URL", "pulsar://pulsar:6650"),
            storage_dir=os.environ.get("STORAGE_DIR", "/app/storage"),
            project_events_topic=os.environ.get(
                "PROJECT_EVENTS_TOPIC", "persistent://public/default/project_events"
            ),
            spec_sheet_events_topic=os.environ.get(
                "SPEC_SHEET_EVENTS_TOPIC", "persistent://public/default/spec_sheet_events"
            ),
            code_gen_events_topic=os.environ.get(
                "CODE_GEN_EVENTS_TOPIC", "persistent://public/default/code_generation_events"
            ),
            integration_events_topic=os.environ.get(
                "INTEGRATION_EVENTS_TOPIC", "persistent://public/default/integration_events"
            ),
            test_events_topic=os.environ.get(
                "TEST_EVENTS_TOPIC", "persistent://public/default/test_events"
            ),
            workflow_commands_topic=os.environ.get(
                "WORKFLOW_COMMANDS_TOPIC", "persistent://public/default/workflow_commands"
            ),
            assistance_events_topic=os.environ.get(
                "ASSISTANCE_EVENTS_TOPIC", "persistent://public/default/assistance_events"
            ),
        )

        try:
            # Initialize the orchestrator
            await orchestrator.initialize()

            # Just keep running, processing events
            while True:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutting down Workflow Orchestrator Service...")
        finally:
            # Clean up
            await orchestrator.close()

    if __name__ == "__main__":
        # Run the service
        asyncio.run(run_service())
