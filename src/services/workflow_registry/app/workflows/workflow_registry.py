"""
WorkflowRegistry - Centralized registry for managing and customizing event-driven workflows.

This module provides a robust, production-ready implementation of a workflow registry
that integrates with the existing event-based architecture while allowing for
customization and extension of workflows without modifying core components.
"""

import asyncio
import aiofiles
import logging
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowPhase(str, Enum):
    """Workflow phases that represent the high-level stages of a workflow."""
    INITIALIZING = "initializing"
    SPEC_SHEETS_GENERATED = "spec_sheets_generated"
    SPEC_SHEETS_COMPLETING = "spec_sheets_completing"
    SPEC_SHEETS_COMPLETED = "spec_sheets_completed"
    CODE_GENERATING = "code_generating"
    CODE_GENERATED = "code_generated"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    ERROR = "error"


class WorkflowEventType(str, Enum):
    """Event types used within workflows."""
    PROJECT_CREATED = "project_created"
    SPEC_SHEETS_GENERATED = "spec_sheets_generated"
    SPEC_SHEET_COMPLETED = "spec_sheet_completed"
    CODE_GENERATION_REQUESTED = "code_generation_requested"
    CODE_GENERATED = "code_generated"
    APPLICATION_ASSEMBLED = "application_assembled"
    ERROR = "error"
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_STEP_STARTED = "workflow_step_started"
    WORKFLOW_STEP_COMPLETED = "workflow_step_completed"
    WORKFLOW_STEP_FAILED = "workflow_step_failed"
    WORKFLOW_EXTENSION_TRIGGERED = "workflow_extension_triggered"
    WORKFLOW_EXTENSION_COMPLETED = "workflow_extension_completed"
    WORKFLOW_EXTENSION_FAILED = "workflow_extension_failed"


class ExtensionPoint(str, Enum):
    """Standard extension points within workflows."""
    BEFORE_WORKFLOW = "before_workflow"
    AFTER_WORKFLOW = "after_workflow"
    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_ERROR = "on_error"
    CUSTOM = "custom"


class WorkflowStep:
    """Represents a step in a workflow."""

    def __init__(
            self,
            step_id: str,
            name: str,
            handler: Callable,
            next_steps: Optional[List[str]] = None,
            condition: Optional[Callable] = None,
            retry_policy: Optional[Dict[str, Any]] = None,
            extension_points: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a workflow step.

        Args:
            step_id: Unique identifier for the step
            name: Display name for the step
            handler: Function that executes the step logic
            next_steps: IDs of steps to execute after this one
            condition: Function that determines if this step should execute
            retry_policy: Configuration for retry behavior
            extension_points: Custom extension points for this step
            metadata: Additional information about the step
        """
        self.step_id = step_id
        self.name = name
        self.handler = handler
        self.next_steps = next_steps or []
        self.condition = condition
        self.retry_policy = retry_policy or {
            "max_retries": 3,
            "retry_delay": 1.0,
            "backoff_factor": 2.0,
            "max_delay": 60.0
        }
        self.extension_points = extension_points or []
        self.metadata = metadata or {}

    async def execute(self, context: Dict[str, Any], workflow_registry: "WorkflowRegistry") -> Dict[str, Any]:
        """
        Execute the step with the given context.

        Args:
            context: The workflow context
            workflow_registry: Reference to the workflow registry

        Returns:
            Updated context after step execution
        """
        workflow_id = context.get("workflow_id", "unknown")
        step_context = context.copy()
        step_context["step_id"] = self.step_id
        step_context["step_name"] = self.name

        # Check condition if provided
        if self.condition and not await self._call_async(self.condition, step_context):
            logger.info(f"Skipping step {self.step_id} in workflow {workflow_id} due to condition")
            return context

        # Execute before_step extensions
        step_context = await workflow_registry.execute_extensions(
            workflow_id,
            ExtensionPoint.BEFORE_STEP,
            step_context,
            step_id=self.step_id
        )

        # Execute custom extensions for this step
        for ext_point in self.extension_points:
            step_context = await workflow_registry.execute_extensions(
                workflow_id,
                ext_point,
                step_context,
                step_id=self.step_id
            )

        # Execute the step handler with retries
        retry_count = 0
        max_retries = self.retry_policy["max_retries"]
        retry_delay = self.retry_policy["retry_delay"]
        backoff_factor = self.retry_policy["backoff_factor"]
        max_delay = self.retry_policy["max_delay"]

        while True:
            try:
                # Publish step started event
                await workflow_registry.event_bus.publish_event(
                    WorkflowEventType.WORKFLOW_STEP_STARTED,
                    {
                        "workflow_id": workflow_id,
                        "step_id": self.step_id,
                        "step_name": self.name,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )

                # Execute the step handler
                logger.info(f"Executing step {self.step_id} in workflow {workflow_id}")
                result = await self._call_async(self.handler, step_context)

                # Update context with result
                if isinstance(result, dict):
                    step_context.update(result)

                # Publish step completed event
                await workflow_registry.event_bus.publish_event(
                    WorkflowEventType.WORKFLOW_STEP_COMPLETED,
                    {
                        "workflow_id": workflow_id,
                        "step_id": self.step_id,
                        "step_name": self.name,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )

                break

            except Exception as e:
                retry_count += 1
                logger.error(f"Error executing step {self.step_id} in workflow {workflow_id}: {str(e)}")

                # Publish step failed event
                await workflow_registry.event_bus.publish_event(
                    WorkflowEventType.WORKFLOW_STEP_FAILED,
                    {
                        "workflow_id": workflow_id,
                        "step_id": self.step_id,
                        "step_name": self.name,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )

                # Check if we can retry
                if retry_count <= max_retries:
                    delay = min(retry_delay * (backoff_factor ** (retry_count - 1)), max_delay)
                    logger.info(
                        f"Retrying step {self.step_id} in {delay} seconds (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    # Execute on_error extensions
                    error_context = step_context.copy()
                    error_context["error"] = str(e)
                    error_context = await workflow_registry.execute_extensions(
                        workflow_id,
                        ExtensionPoint.ON_ERROR,
                        error_context,
                        step_id=self.step_id
                    )

                    # Re-raise the exception if we've exhausted retries
                    raise

        # Execute after_step extensions
        step_context = await workflow_registry.execute_extensions(
            workflow_id,
            ExtensionPoint.AFTER_STEP,
            step_context,
            step_id=self.step_id
        )

        return step_context

    async def _call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function, handling both synchronous and asynchronous functions.

        Args:
            func: The function to call
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result of the function call
        """
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


class WorkflowDefinition:
    """Defines a workflow with steps, transitions, and extension points."""

    def __init__(
            self,
            workflow_id: str,
            name: str,
            description: Optional[str] = None,
            version: str = "1.0.0",
            steps: Optional[Dict[str, WorkflowStep]] = None,
            initial_step_id: Optional[str] = None,
            extension_points: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            parent_workflow_id: Optional[str] = None
    ):
        """
        Initialize a workflow definition.

        Args:
            workflow_id: Unique identifier for the workflow
            name: Display name for the workflow
            description: Detailed description of the workflow
            version: Version string for the workflow
            steps: Dictionary of workflow steps keyed by step_id
            initial_step_id: ID of the first step to execute
            extension_points: Custom extension points for this workflow
            metadata: Additional information about the workflow
            parent_workflow_id: ID of parent workflow if this is an extension
        """
        self.workflow_id = workflow_id
        self.name = name
        self.description = description or ""
        self.version = version
        self.steps = steps or {}
        self.initial_step_id = initial_step_id
        self.extension_points = extension_points or []
        self.metadata = metadata or {}
        self.parent_workflow_id = parent_workflow_id

    def add_step(self, step: WorkflowStep) -> None:
        """
        Add a step to the workflow.

        Args:
            step: The workflow step to add
        """
        self.steps[step.step_id] = step

        # If this is the first step, set it as the initial step
        if not self.initial_step_id and len(self.steps) == 1:
            self.initial_step_id = step.step_id

    def add_extension_point(self, extension_point: str) -> None:
        """
        Add a custom extension point to the workflow.

        Args:
            extension_point: The extension point name
        """
        if extension_point not in self.extension_points:
            self.extension_points.append(extension_point)

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """
        Get a step by ID.

        Args:
            step_id: The step ID to retrieve

        Returns:
            The workflow step or None if not found
        """
        return self.steps.get(step_id)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert workflow definition to a dictionary.

        Returns:
            Dictionary representation of the workflow
        """
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": [step.step_id for step in self.steps.values()],
            "initial_step_id": self.initial_step_id,
            "extension_points": self.extension_points,
            "metadata": self.metadata,
            "parent_workflow_id": self.parent_workflow_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], steps: Dict[str, WorkflowStep]) -> "WorkflowDefinition":
        """
        Create a workflow definition from a dictionary.

        Args:
            data: Dictionary representation of the workflow
            steps: Dictionary of workflow steps keyed by step_id

        Returns:
            A workflow definition
        """
        workflow = cls(
            workflow_id=data["workflow_id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            steps={},
            initial_step_id=data.get("initial_step_id"),
            extension_points=data.get("extension_points", []),
            metadata=data.get("metadata", {}),
            parent_workflow_id=data.get("parent_workflow_id")
        )

        # Add steps
        for step_id in data.get("steps", []):
            if step_id in steps:
                workflow.add_step(steps[step_id])

        return workflow


class WorkflowExtension:
    """Defines an extension to a workflow at a specific extension point."""

    def __init__(
            self,
            extension_id: str,
            name: str,
            handler: Callable,
            workflow_id: str,
            extension_point: str,
            condition: Optional[Callable] = None,
            priority: int = 0,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a workflow extension.

        Args:
            extension_id: Unique identifier for the extension
            name: Display name for the extension
            handler: Function that executes the extension logic
            workflow_id: ID of the workflow to extend
            extension_point: Point in the workflow to execute the extension
            condition: Function that determines if this extension should execute
            priority: Order of execution when multiple extensions target the same point
            metadata: Additional information about the extension
        """
        self.extension_id = extension_id
        self.name = name
        self.handler = handler
        self.workflow_id = workflow_id
        self.extension_point = extension_point
        self.condition = condition
        self.priority = priority
        self.metadata = metadata or {}

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the extension with the given context.

        Args:
            context: The workflow context

        Returns:
            Updated context after extension execution
        """
        # Check condition if provided
        if self.condition and not await self._call_async(self.condition, context):
            logger.debug(f"Skipping extension {self.extension_id} due to condition")
            return context

        logger.info(
            f"Executing extension {self.extension_id} at {self.extension_point} for workflow {self.workflow_id}")
        result = await self._call_async(self.handler, context)

        # Update context with result
        if isinstance(result, dict):
            context.update(result)

        return context

    async def _call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function, handling both synchronous and asynchronous functions.

        Args:
            func: The function to call
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result of the function call
        """
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


class WorkflowInstance:
    """Represents a running instance of a workflow."""

    def __init__(
            self,
            instance_id: str,
            workflow_id: str,
            context: Dict[str, Any],
            created_at: Optional[datetime] = None,
            current_step_id: Optional[str] = None,
            status: str = "created",
            error: Optional[str] = None
    ):
        """
        Initialize a workflow instance.

        Args:
            instance_id: Unique identifier for the instance
            workflow_id: ID of the workflow definition
            context: The workflow context
            created_at: When the instance was created
            current_step_id: ID of the current step
            status: Current status of the instance
            error: Error message if the instance failed
        """
        self.instance_id = instance_id
        self.workflow_id = workflow_id
        self.context = context
        self.created_at = created_at or datetime.utcnow()
        self.current_step_id = current_step_id
        self.status = status
        self.error = error
        self.completed_steps: List[str] = []
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert workflow instance to a dictionary.

        Returns:
            Dictionary representation of the instance
        """
        return {
            "instance_id": self.instance_id,
            "workflow_id": self.workflow_id,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "current_step_id": self.current_step_id,
            "status": self.status,
            "error": self.error,
            "completed_steps": self.completed_steps,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowInstance":
        """
        Create a workflow instance from a dictionary.

        Args:
            data: Dictionary representation of the instance

        Returns:
            A workflow instance
        """
        instance = cls(
            instance_id=data["instance_id"],
            workflow_id=data["workflow_id"],
            context=data["context"],
            created_at=datetime.fromisoformat(data["created_at"]),
            current_step_id=data.get("current_step_id"),
            status=data.get("status", "created"),
            error=data.get("error")
        )

        instance.completed_steps = data.get("completed_steps", [])

        if data.get("started_at"):
            instance.started_at = datetime.fromisoformat(data["started_at"])

        if data.get("completed_at"):
            instance.completed_at = datetime.fromisoformat(data["completed_at"])

        return instance


class WorkflowPersistenceProvider:
    """Interface for workflow persistence providers."""

    async def save_workflow_definition(self, workflow: WorkflowDefinition) -> None:
        """
        Save a workflow definition.

        Args:
            workflow: The workflow definition to save
        """
        raise NotImplementedError

    async def load_workflow_definition(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a workflow definition.

        Args:
            workflow_id: ID of the workflow to load

        Returns:
            Dictionary representation of the workflow or None if not found
        """
        raise NotImplementedError

    async def list_workflow_definitions(self) -> List[Dict[str, Any]]:
        """
        List all workflow definitions.

        Returns:
            List of workflow definition dictionaries
        """
        raise NotImplementedError

    async def save_workflow_instance(self, instance: WorkflowInstance) -> None:
        """
        Save a workflow instance.

        Args:
            instance: The workflow instance to save
        """
        raise NotImplementedError

    async def load_workflow_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a workflow instance.

        Args:
            instance_id: ID of the instance to load

        Returns:
            Dictionary representation of the instance or None if not found
        """
        raise NotImplementedError

    async def list_workflow_instances(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List workflow instances.

        Args:
            workflow_id: Optional workflow ID to filter by

        Returns:
            List of workflow instance dictionaries
        """
        raise NotImplementedError


class FileSystemPersistenceProvider(WorkflowPersistenceProvider):
    """File system-based persistence provider for workflows."""

    def __init__(self, data_dir: str = "./data/workflows"):
        """
        Initialize the provider.

        Args:
            data_dir: Directory to store workflow data
        """
        import os
        self.data_dir = data_dir
        self.definitions_dir = os.path.join(data_dir, "definitions")
        self.instances_dir = os.path.join(data_dir, "instances")

        # Create directories if they don't exist
        os.makedirs(self.definitions_dir, exist_ok=True)
        os.makedirs(self.instances_dir, exist_ok=True)

    async def save_workflow_definition(self, workflow: WorkflowDefinition) -> None:
        """
        Save a workflow definition.

        Args:
            workflow: The workflow definition to save
        """
        import os
        import json

        path = os.path.join(self.definitions_dir, f"{workflow.workflow_id}.json")
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(workflow.to_dict(), indent=2))

    async def load_workflow_definition(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a workflow definition.

        Args:
            workflow_id: ID of the workflow to load

        Returns:
            Dictionary representation of the workflow or None if not found
        """
        import os
        import json

        path = os.path.join(self.definitions_dir, f"{workflow_id}.json")
        if not os.path.exists(path):
            return None

        async with aiofiles.open(path, "r") as f:
            content = await f.read()
            return json.loads(content)

    async def list_workflow_definitions(self) -> List[Dict[str, Any]]:
        """
        List all workflow definitions.

        Returns:
            List of workflow definition dictionaries
        """
        import os
        import json

        result = []
        for filename in os.listdir(self.definitions_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.definitions_dir, filename)
                async with aiofiles.open(path, "r") as f:
                    content = await f.read()
                    result.append(json.loads(content))

        return result

    async def save_workflow_instance(self, instance: WorkflowInstance) -> None:
        """
        Save a workflow instance.

        Args:
            instance: The workflow instance to save
        """
        import os
        import json

        path = os.path.join(self.instances_dir, f"{instance.instance_id}.json")
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(instance.to_dict(), indent=2))

    async def load_workflow_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a workflow instance.

        Args:
            instance_id: ID of the instance to load

        Returns:
            Dictionary representation of the instance or None if not found
        """
        import os
        import json

        path = os.path.join(self.instances_dir, f"{instance_id}.json")
        if not os.path.exists(path):
            return None

        async with aiofiles.open(path, "r") as f:
            content = await f.read()
            return json.loads(content)

    async def list_workflow_instances(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List workflow instances.

        Args:
            workflow_id: Optional workflow ID to filter by

        Returns:
            List of workflow instance dictionaries
        """
        import os
        import json

        result = []
        for filename in os.listdir(self.instances_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.instances_dir, filename)
                async with aiofiles.open(path, "r") as f:
                    content = await f.read()
                    instance_data = json.loads(content)

                    if workflow_id is None or instance_data.get("workflow_id") == workflow_id:
                        result.append(instance_data)

        return result


class WorkflowRegistry:
    """
    Centralized registry for managing workflows, extensions, and instances.

    This class provides a robust, production-ready implementation for registering,
    executing, and extending workflows in an event-driven architecture.
    """

    def __init__(
            self,
            event_bus,
            persistence_provider: Optional[WorkflowPersistenceProvider] = None,
            auto_register_events: bool = True
    ):
        """
        Initialize the workflow registry.

        Args:
            event_bus: Event bus for publishing and subscribing to events
            persistence_provider: Provider for persisting workflow data
            auto_register_events: Whether to automatically register event handlers
        """
        self.event_bus = event_bus
        self.persistence_provider = persistence_provider
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.extensions: Dict[str, Dict[str, List[WorkflowExtension]]] = {}
        self.instances: Dict[str, WorkflowInstance] = {}
        self.custom_extension_points: Set[str] = set()

        # Standard extension points
        for ext_point in ExtensionPoint:
            self.custom_extension_points.add(ext_point.value)

        # Auto-register event handlers
        if auto_register_events:
            self._register_event_handlers()

    async def initialize(self) -> None:
        """
        Initialize the registry, loading persisted workflows and instances.
        """
        if not self.persistence_provider:
            return

        # Load workflow definitions
        workflow_defs = await self.persistence_provider.list_workflow_definitions()
        for workflow_def in workflow_defs:
            # Note: We need to load steps separately and reconstruct the workflow
            self.workflows[workflow_def["workflow_id"]] = WorkflowDefinition.from_dict(workflow_def, {})

        logger.info(f"Loaded {len(workflow_defs)} workflow definitions")

    def _register_event_handlers(self) -> None:
        """
        Register event handlers for workflow-related events.
        """
        # Register handlers for standard workflow events
        self.event_bus.subscribe(
            [evt.value for evt in WorkflowEventType],
            self._handle_workflow_event,
            "workflow_registry"
        )

        logger.info("Registered workflow event handlers")

    async def _handle_workflow_event(self, event: Dict[str, Any]) -> None:
        """
        Handle workflow-related events.

        Args:
            event: The event to handle
        """
        event_type = event.get("event_type")
        payload = event.get("payload", {})

        if event_type == WorkflowEventType.PROJECT_CREATED.value:
            # Create a new workflow instance for the project
            project_id = payload.get("project_id")
            if project_id:
                await self.start_workflow("standard_workflow", {
                    "project_id": project_id,
                    "correlation_id": event.get("correlation_id"),
                    "source_event": event
                })

        elif event_type == WorkflowEventType.ERROR.value:
            # Handle error events
            workflow_id = payload.get("workflow_id")
            instance_id = payload.get("instance_id")
            error = payload.get("error")

            if instance_id and instance_id in self.instances:
                instance = self.instances[instance_id]
                instance.status = "error"
                instance.error = error

                # Save the instance
                if self.persistence_provider:
                    await self.persistence_provider.save_workflow_instance(instance)

    async def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """
        Register a workflow definition.

        Args:
            workflow: The workflow definition to register
        """
        self.workflows[workflow.workflow_id] = workflow

        # Register custom extension points
        for ext_point in workflow.extension_points:
            self.custom_extension_points.add(ext_point)

        # Save the workflow
        if self.persistence_provider:
            await self.persistence_provider.save_workflow_definition(workflow)

        logger.info(f"Registered workflow {workflow.workflow_id}")

    async def register_extension(self, extension: WorkflowExtension) -> None:
        """
        Register a workflow extension.

        Args:
            extension: The workflow extension to register
        """
        workflow_id = extension.workflow_id
        ext_point = extension.extension_point

        # Initialize dictionaries if they don't exist
        if workflow_id not in self.extensions:
            self.extensions[workflow_id] = {}

        if ext_point not in self.extensions[workflow_id]:
            self.extensions[workflow_id][ext_point] = []

        # Add the extension
        self.extensions[workflow_id][ext_point].append(extension)

        # Sort extensions by priority
        self.extensions[workflow_id][ext_point].sort(key=lambda x: x.priority)

        logger.info(f"Registered extension {extension.extension_id} for workflow {workflow_id} at {ext_point}")

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """
        Get a workflow definition by ID.

        Args:
            workflow_id: The workflow ID to retrieve

        Returns:
            The workflow definition or None if not found
        """
        return self.workflows.get(workflow_id)

    async def list_workflows(self) -> List[WorkflowDefinition]:
        """
        List all registered workflows.

        Returns:
            List of workflow definitions
        """
        return list(self.workflows.values())

    async def get_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """
        Get a workflow instance by ID.

        Args:
            instance_id: The instance ID to retrieve

        Returns:
            The workflow instance or None if not found
        """
        # Check in-memory cache first
        if instance_id in self.instances:
            return self.instances[instance_id]

        # Try to load from persistence provider
        if self.persistence_provider:
            instance_data = await self.persistence_provider.load_workflow_instance(instance_id)
            if instance_data:
                instance = WorkflowInstance.from_dict(instance_data)
                self.instances[instance_id] = instance
                return instance

        return None

    async def list_instances(self, workflow_id: Optional[str] = None) -> List[WorkflowInstance]:
        """
        List workflow instances.

        Args:
            workflow_id: Optional workflow ID to filter by

        Returns:
            List of workflow instances
        """
        result = []

        # Combine in-memory instances with persisted instances
        if self.persistence_provider:
            instance_datas = await self.persistence_provider.list_workflow_instances(workflow_id)
            for instance_data in instance_datas:
                instance_id = instance_data.get("instance_id")

                # Use in-memory instance if available (more up-to-date)
                if instance_id in self.instances:
                    result.append(self.instances[instance_id])
                else:
                    instance = WorkflowInstance.from_dict(instance_data)
                    result.append(instance)
        else:
            # Just use in-memory instances
            for instance in self.instances.values():
                if workflow_id is None or instance.workflow_id == workflow_id:
                    result.append(instance)

        return result

    async def start_workflow(
            self,
            workflow_id: str,
            context: Dict[str, Any],
            instance_id: Optional[str] = None
    ) -> Optional[WorkflowInstance]:
        """
        Start a new workflow instance.

        Args:
            workflow_id: ID of the workflow to start
            context: Initial context for the workflow
            instance_id: Optional ID for the instance (generated if not provided)

        Returns:
            The created workflow instance or None if the workflow was not found
        """
        # Check if workflow exists
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Workflow {workflow_id} not found")
            return None

        # Generate instance ID if not provided
        if not instance_id:
            instance_id = str(uuid.uuid4())

        # Create context with instance ID
        full_context = context.copy()
        full_context["instance_id"] = instance_id
        full_context["workflow_id"] = workflow_id

        # Create the instance
        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow_id=workflow_id,
            context=full_context,
            status="running"
        )

        instance.started_at = datetime.utcnow()
        instance.current_step_id = workflow.initial_step_id

        # Store the instance
        self.instances[instance_id] = instance

        # Save the instance
        if self.persistence_provider:
            await self.persistence_provider.save_workflow_instance(instance)

        # Publish workflow created event
        await self.event_bus.publish_event(
            WorkflowEventType.WORKFLOW_CREATED,
            {
                "instance_id": instance_id,
                "workflow_id": workflow_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Execute before_workflow extensions
        try:
            instance.context = await self.execute_extensions(
                workflow_id,
                ExtensionPoint.BEFORE_WORKFLOW,
                instance.context
            )
        except Exception as e:
            logger.error(f"Error executing before_workflow extensions: {str(e)}")
            instance.status = "error"
            instance.error = f"Error in before_workflow extensions: {str(e)}"

            if self.persistence_provider:
                await self.persistence_provider.save_workflow_instance(instance)

            await self.event_bus.publish_event(
                WorkflowEventType.WORKFLOW_FAILED,
                {
                    "instance_id": instance_id,
                    "workflow_id": workflow_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            return instance

        # Start executing the workflow asynchronously
        asyncio.create_task(self._execute_workflow(instance))

        logger.info(f"Started workflow {workflow_id} with instance {instance_id}")
        return instance

    async def execute_extensions(
            self,
            workflow_id: str,
            extension_point: Union[str, ExtensionPoint],
            context: Dict[str, Any],
            **kwargs
    ) -> Dict[str, Any]:
        """
        Execute extensions for a workflow at a specific extension point.

        Args:
            workflow_id: ID of the workflow
            extension_point: Extension point to execute
            context: Context to pass to extensions
            **kwargs: Additional parameters for extension filtering

        Returns:
            Updated context after extension execution
        """
        # Convert enum to string if necessary
        if isinstance(extension_point, ExtensionPoint):
            extension_point = extension_point.value

        # Get extensions for this workflow and extension point
        workflow_extensions = self.extensions.get(workflow_id, {})
        extensions = workflow_extensions.get(extension_point, [])

        # Get extensions for ALL workflows at this extension point
        all_extensions = self.extensions.get("*", {}).get(extension_point, [])
        extensions.extend(all_extensions)

        # No extensions to execute
        if not extensions:
            return context

        # Filter extensions based on additional parameters
        if kwargs:
            filtered_extensions = []
            for ext in extensions:
                matches = True
                for key, value in kwargs.items():
                    if key in ext.metadata and ext.metadata[key] != value:
                        matches = False
                        break

                if matches:
                    filtered_extensions.append(ext)

            extensions = filtered_extensions

        # Execute each extension
        updated_context = context.copy()
        for ext in extensions:
            try:
                # Publish extension triggered event
                await self.event_bus.publish_event(
                    WorkflowEventType.WORKFLOW_EXTENSION_TRIGGERED,
                    {
                        "workflow_id": workflow_id,
                        "extension_id": ext.extension_id,
                        "extension_point": extension_point,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )

                # Execute the extension
                updated_context = await ext.execute(updated_context)

                # Publish extension completed event
                await self.event_bus.publish_event(
                    WorkflowEventType.WORKFLOW_EXTENSION_COMPLETED,
                    {
                        "workflow_id": workflow_id,
                        "extension_id": ext.extension_id,
                        "extension_point": extension_point,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"Error executing extension {ext.extension_id}: {str(e)}")

                # Publish extension failed event
                await self.event_bus.publish_event(
                    WorkflowEventType.WORKFLOW_EXTENSION_FAILED,
                    {
                        "workflow_id": workflow_id,
                        "extension_id": ext.extension_id,
                        "extension_point": extension_point,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )

        return updated_context

    async def _execute_workflow(self, instance: WorkflowInstance) -> None:
        """
        Execute a workflow instance.

        Args:
            instance: The workflow instance to execute
        """
        workflow_id = instance.workflow_id
        instance_id = instance.instance_id

        try:
            # Get the workflow
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")

            # Start from the current step
            current_step_id = instance.current_step_id

            while current_step_id:
                # Get the current step
                current_step = workflow.get_step(current_step_id)
                if not current_step:
                    raise ValueError(f"Step {current_step_id} not found in workflow {workflow_id}")

                # Execute the step
                instance.current_step_id = current_step_id

                # Save the instance
                if self.persistence_provider:
                    await self.persistence_provider.save_workflow_instance(instance)

                # Execute the step
                try:
                    instance.context = await current_step.execute(instance.context, self)

                    # Add to completed steps
                    instance.completed_steps.append(current_step_id)

                    # Determine next step
                    if current_step.next_steps:
                        # If there are multiple next steps, we take the first one for now
                        # More complex branching logic would go here
                        current_step_id = current_step.next_steps[0]
                    else:
                        # No more steps
                        current_step_id = None
                except Exception as e:
                    logger.error(f"Error executing step {current_step_id} in workflow {workflow_id}: {str(e)}")
                    instance.status = "error"
                    instance.error = f"Error in step {current_step_id}: {str(e)}"

                    # Save the instance
                    if self.persistence_provider:
                        await self.persistence_provider.save_workflow_instance(instance)

                    # Publish workflow failed event
                    await self.event_bus.publish_event(
                        WorkflowEventType.WORKFLOW_FAILED,
                        {
                            "instance_id": instance_id,
                            "workflow_id": workflow_id,
                            "step_id": current_step_id,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )

                    return

            # Workflow completed successfully
            instance.status = "completed"
            instance.completed_at = datetime.utcnow()

            # Execute after_workflow extensions
            try:
                instance.context = await self.execute_extensions(
                    workflow_id,
                    ExtensionPoint.AFTER_WORKFLOW,
                    instance.context
                )
            except Exception as e:
                logger.error(f"Error executing after_workflow extensions: {str(e)}")
                instance.status = "completed_with_errors"
                instance.error = f"Error in after_workflow extensions: {str(e)}"

            # Save the instance
            if self.persistence_provider:
                await self.persistence_provider.save_workflow_instance(instance)

            # Publish workflow completed event
            await self.event_bus.publish_event(
                WorkflowEventType.WORKFLOW_COMPLETED,
                {
                    "instance_id": instance_id,
                    "workflow_id": workflow_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            logger.info(f"Workflow {workflow_id} instance {instance_id} completed successfully")

        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id} instance {instance_id}: {str(e)}")
            instance.status = "error"
            instance.error = str(e)

            # Save the instance
            if self.persistence_provider:
                await self.persistence_provider.save_workflow_instance(instance)

            # Publish workflow failed event
            await self.event_bus.publish_event(
                WorkflowEventType.WORKFLOW_FAILED,
                {
                    "instance_id": instance_id,
                    "workflow_id": workflow_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )


# Register our existing workflows

async def register_standard_workflow(workflow_registry: WorkflowRegistry) -> None:
    """
    Register the standard program synthesis workflow.

    Args:
        workflow_registry: The workflow registry
    """
    # Create workflow definition
    standard_workflow = WorkflowDefinition(
        workflow_id="standard_workflow",
        name="Standard Program Synthesis Workflow",
        description="The standard workflow for program synthesis from specification to code generation",
        version="1.0.0",
        extension_points=[
            "before_analysis",
            "after_analysis",
            "before_spec_generation",
            "after_spec_generation",
            "before_spec_completion",
            "after_spec_completion",
            "before_code_generation",
            "after_code_generation",
            "before_integration",
            "after_integration"
        ]
    )

    # Add steps

    # Project Analysis step
    async def analyze_project(context):
        project_id = context.get("project_id")
        if not project_id:
            raise ValueError("Project ID is required")

        # Get the project manager from context or service registry
        project_manager = context.get("project_manager")
        if not project_manager:
            # In a real implementation, we would get this from a service registry
            raise ValueError("Project manager is required")

        # Analyze project requirements
        analysis_result = await project_manager.analyze_project_requirements(project_id)

        # Publish event
        await workflow_registry.event_bus.publish_event(
            WorkflowEventType.WORKFLOW_STEP_COMPLETED,
            {
                "step": "project_analysis",
                "project_id": project_id,
                "analysis_result": analysis_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        return {
            "analysis_result": analysis_result
        }

    standard_workflow.add_step(
        WorkflowStep(
            step_id="project_analysis",
            name="Project Analysis",
            handler=analyze_project,
            next_steps=["spec_sheet_generation"],
            extension_points=["before_analysis", "after_analysis"]
        )
    )

    # Spec Sheet Generation step
    async def generate_spec_sheets(context):
        project_id = context.get("project_id")
        if not project_id:
            raise ValueError("Project ID is required")

        # Get the project manager from context or service registry
        project_manager = context.get("project_manager")
        if not project_manager:
            # In a real implementation, we would get this from a service registry
            raise ValueError("Project manager is required")

        # Generate spec sheets
        spec_sheets, errors = await project_manager.generate_spec_sheets(project_id)

        # Publish event
        await workflow_registry.event_bus.publish_event(
            WorkflowEventType.SPEC_SHEETS_GENERATED,
            {
                "project_id": project_id,
                "spec_sheet_ids": [sheet.id for sheet in spec_sheets],
                "errors": errors,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        return {
            "spec_sheets": spec_sheets,
            "errors": errors
        }

    standard_workflow.add_step(
        WorkflowStep(
            step_id="spec_sheet_generation",
            name="Specification Sheet Generation",
            handler=generate_spec_sheets,
            next_steps=["spec_sheet_completion"],
            extension_points=["before_spec_generation", "after_spec_generation"]
        )
    )

    # Spec Sheet Completion step
    async def complete_spec_sheets(context):
        project_id = context.get("project_id")
        if not project_id:
            raise ValueError("Project ID is required")

        spec_sheets = context.get("spec_sheets")
        if not spec_sheets:
            raise ValueError("Spec sheets are required")

        # Get the spec generator from context or service registry
        spec_generator = context.get("spec_generator")
        if not spec_generator:
            # In a real implementation, we would get this from a service registry
            raise ValueError("Spec generator is required")

        # Get the project manager from context or service registry
        project_manager = context.get("project_manager")
        if not project_manager:
            # In a real implementation, we would get this from a service registry
            raise ValueError("Project manager is required")

        # Get the project
        project = await project_manager.get_project(project_id)

        # Complete each spec sheet
        completed_sheets = []
        for sheet in spec_sheets:
            completed_sheet = await spec_generator.complete_spec_sheet(sheet, project)
            completed_sheets.append(completed_sheet)

            # Publish event for each completed sheet
            await workflow_registry.event_bus.publish_event(
                WorkflowEventType.SPEC_SHEET_COMPLETED,
                {
                    "project_id": project_id,
                    "spec_sheet_id": completed_sheet.id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        return {
            "completed_sheets": completed_sheets
        }

    standard_workflow.add_step(
        WorkflowStep(
            step_id="spec_sheet_completion",
            name="Specification Sheet Completion",
            handler=complete_spec_sheets,
            next_steps=["code_generation"],
            extension_points=["before_spec_completion", "after_spec_completion"]
        )
    )

    # Code Generation step
    async def generate_code(context):
        project_id = context.get("project_id")
        if not project_id:
            raise ValueError("Project ID is required")

        completed_sheets = context.get("completed_sheets")
        if not completed_sheets:
            raise ValueError("Completed spec sheets are required")

        # Get the synthesis engine from context or service registry
        synthesis_engine = context.get("synthesis_engine")
        if not synthesis_engine:
            # In a real implementation, we would get this from a service registry
            raise ValueError("Synthesis engine is required")

        # Generate code for each spec sheet
        code_results = []
        for sheet in completed_sheets:
            # Convert spec sheet to formal specification
            formal_spec = await convert_to_formal_spec(sheet)

            # Generate code
            synthesis_result = await synthesis_engine.synthesize(formal_spec)
            code_results.append(synthesis_result)

            # Publish event for each generated code
            await workflow_registry.event_bus.publish_event(
                WorkflowEventType.CODE_GENERATED,
                {
                    "project_id": project_id,
                    "spec_sheet_id": sheet.id,
                    "code_generation_id": synthesis_result.id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        return {
            "code_results": code_results
        }

    standard_workflow.add_step(
        WorkflowStep(
            step_id="code_generation",
            name="Code Generation",
            handler=generate_code,
            next_steps=["integration"],
            extension_points=["before_code_generation", "after_code_generation"]
        )
    )

    # Integration step
    async def integrate_code(context):
        project_id = context.get("project_id")
        if not project_id:
            raise ValueError("Project ID is required")

        code_results = context.get("code_results")
        if not code_results:
            raise ValueError("Code results are required")

        # Get the assembler service from context or service registry
        assembler_service = context.get("assembler_service")
        if not assembler_service:
            # In a real implementation, we would get this from a service registry
            raise ValueError("Assembler service is required")

        # Assemble the application
        assembly_result = await assembler_service.assemble(project_id, code_results)

        # Publish event
        await workflow_registry.event_bus.publish_event(
            WorkflowEventType.APPLICATION_ASSEMBLED,
            {
                "project_id": project_id,
                "assembly_result": assembly_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        return {
            "assembly_result": assembly_result
        }

    standard_workflow.add_step(
        WorkflowStep(
            step_id="integration",
            name="Integration",
            handler=integrate_code,
            next_steps=[],  # End of workflow
            extension_points=["before_integration", "after_integration"]
        )
    )

    # Register the workflow
    await workflow_registry.register_workflow(standard_workflow)
    logger.info("Registered standard program synthesis workflow")


async def convert_to_formal_spec(spec_sheet):
    """
    Convert a completed spec sheet to a formal specification.

    In a real implementation, this would parse the spec sheet and create a formal specification
    for the synthesis engine.

    Args:
        spec_sheet: The spec sheet to convert

    Returns:
        A formal specification
    """
    # Simplified implementation for demonstration
    return {
        "function_name": spec_sheet.get_field_value("function_name"),
        "description": spec_sheet.get_field_value("description"),
        "parameters": [],
        "return_type": spec_sheet.get_field_value("return_type"),
        "constraints": [],
        "examples": []
    }


async def register_specialized_workflow(workflow_registry: WorkflowRegistry) -> None:
    """
    Register a specialized program synthesis workflow.

    Args:
        workflow_registry: The workflow registry
    """
    # Create workflow definition
    specialized_workflow = WorkflowDefinition(
        workflow_id="specialized_workflow",
        name="Specialized Program Synthesis Workflow",
        description="A specialized workflow for program synthesis with additional quality checks",
        version="1.0.0",
        parent_workflow_id="standard_workflow",  # Inherit from standard workflow
        extension_points=[
            "before_quality_check",
            "after_quality_check"
        ]
    )

    # Reuse steps from the standard workflow
    standard_workflow = await workflow_registry.get_workflow("standard_workflow")
    if not standard_workflow:
        logger.error("Standard workflow not found")
        return

    # Copy steps from standard workflow
    for step_id, step in standard_workflow.steps.items():
        specialized_workflow.add_step(step)

    # Add quality check step after code generation
    async def check_code_quality(context):
        project_id = context.get("project_id")
        if not project_id:
            raise ValueError("Project ID is required")

        code_results = context.get("code_results")
        if not code_results:
            raise ValueError("Code results are required")

        # Get the quality checker from context or service registry
        quality_checker = context.get("quality_checker")
        if not quality_checker:
            # In a real implementation, we would get this from a service registry
            raise ValueError("Quality checker is required")

        # Check quality for each result
        quality_results = []
        for result in code_results:
            quality_result = await quality_checker.check(result)
            quality_results.append(quality_result)

            # If quality is below threshold, trigger improvement
            if quality_result.score < 0.8:
                improved_result = await quality_checker.improve(result)
                # Replace the original result with the improved one
                code_results[code_results.index(result)] = improved_result

        return {
            "code_results": code_results,
            "quality_results": quality_results
        }

    quality_check_step = WorkflowStep(
        step_id="quality_check",
        name="Code Quality Check",
        handler=check_code_quality,
        next_steps=["integration"],
        extension_points=["before_quality_check", "after_quality_check"]
    )

    specialized_workflow.add_step(quality_check_step)

    # Modify the step connections
    for step in specialized_workflow.steps.values():
        if "code_generation" in step.next_steps:
            # Redirect to quality check instead of integration
            step.next_steps = ["quality_check"]

    # Register the workflow
    await workflow_registry.register_workflow(specialized_workflow)
    logger.info("Registered specialized program synthesis workflow")


async def register_extensions(workflow_registry: WorkflowRegistry) -> None:
    """
    Register workflow extensions.

    Args:
        workflow_registry: The workflow registry
    """

    # Register an extension for collecting metrics before code generation
    async def collect_metrics_before_code_generation(context):
        project_id = context.get("project_id")
        if not project_id:
            return context

        logger.info(f"Collecting metrics before code generation for project {project_id}")

        # In a real implementation, this would collect metrics
        context["metrics"] = {
            "spec_count": len(context.get("completed_sheets", [])),
            "timestamp": datetime.utcnow().isoformat()
        }

        return context

    metrics_extension = WorkflowExtension(
        extension_id="collect_metrics_before_code_generation",
        name="Collect Metrics Before Code Generation",
        handler=collect_metrics_before_code_generation,
        workflow_id="standard_workflow",
        extension_point="before_code_generation",
        priority=0
    )

    await workflow_registry.register_extension(metrics_extension)

    # Register an extension for sending notifications when a workflow completes
    async def send_completion_notification(context):
        project_id = context.get("project_id")
        if not project_id:
            return context

        logger.info(f"Sending completion notification for project {project_id}")

        # In a real implementation, this would send a notification
        context["notification_sent"] = True
        context["notification_timestamp"] = datetime.utcnow().isoformat()

        return context

    notification_extension = WorkflowExtension(
        extension_id="send_completion_notification",
        name="Send Completion Notification",
        handler=send_completion_notification,
        workflow_id="*",  # Apply to all workflows
        extension_point=ExtensionPoint.AFTER_WORKFLOW,
        priority=0
    )

    await workflow_registry.register_extension(notification_extension)

    logger.info("Registered workflow extensions")


async def initialize_workflow_registry(event_bus) -> WorkflowRegistry:
    """
    Initialize the workflow registry with our standard workflows.

    Args:
        event_bus: The event bus to use

    Returns:
        The initialized workflow registry
    """
    # Create the workflow registry
    persistence_provider = FileSystemPersistenceProvider()
    workflow_registry = WorkflowRegistry(event_bus, persistence_provider)

    # Initialize the registry
    await workflow_registry.initialize()

    # Register our standard workflow
    await register_standard_workflow(workflow_registry)

    # Register our specialized workflow
    await register_specialized_workflow(workflow_registry)

    # Register extensions
    await register_extensions(workflow_registry)

    return workflow_registry

# Example usage:

# async def main():
#     event_bus = PulsarEventBus(service_url="pulsar://localhost:6650")
#     await event_bus.start()
#
#     workflow_registry = await initialize_workflow_registry(event_bus)
#
#     # Start a standard workflow
#     instance = await workflow_registry.start_workflow(
#         "standard_workflow",
#         {"project_id": "project-123"}
#     )
#
#     # Start a specialized workflow
#     specialized_instance = await workflow_registry.start_workflow(
#         "specialized_workflow",
#         {"project_id": "project-456"}
#     )
#
#     await event_bus.stop()

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())