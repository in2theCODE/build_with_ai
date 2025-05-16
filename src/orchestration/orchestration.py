#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code Generation Orchestration System

This module orchestrates the full code generation pipeline using an event-driven architecture.
It coordinates the flow between spec sheet creation, template selection, code generation,
and integration phases while using the existing event system.
"""

import asyncio
from datetime import datetime
from datetime import timezone
from enum import Enum
import logging
import time
from typing import Any, Dict, Optional
import uuid

from src.services.shared.models.base import BaseComponent
from src.services.shared.models.enums import Components

# Import your existing event system components
from src.services.shared.models.events import BaseEvent
from src.services.shared.models.events import CodeGenerationCompletedEvent
from src.services.shared.models.events import CodeGenerationFailedEvent
from src.services.shared.models.events import CodeGenerationRequestedEvent
from src.services.shared.models.events import EventType
from src.services.shared.pulsar.event_emitter import SecureEventEmitter
from src.services.shared.pulsar.event_listener import SecureEventListener


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CodeGenPhase(str, Enum):
    """Orchestration phases for the code generation system"""

    SPEC_ANALYSIS = "spec_analysis"
    TEMPLATE_SELECTION = "template_selection"
    CODE_GENERATION = "code_generation"
    AST_GENERATION = "ast_generation"
    CODE_VERIFICATION = "code_verification"
    CODE_OPTIMIZATION = "code_optimization"
    INTEGRATION = "integration"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    COMPLETION = "completion"


class CodeGenOrchestrator(BaseComponent):
    """
    Orchestrates the code generation pipeline using the event system.

    This service coordinates the flow between the various components involved
    in code generation, ensuring data flows correctly through each phase.
    """

    def __init__(self, **params):
        """Initialize the code generation orchestrator."""
        super().__init__(**params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration parameters
        self.pulsar_service_url = self.get_param("pulsar_service_url", "pulsar://localhost:6650")
        self.secret_key = self.get_param("secret_key", None)
        self.tenant = self.get_param("tenant", "public")
        self.namespace = self.get_param("namespace", "code-generator")
        self.enable_metrics = self.get_param("enable_metrics", True)

        # Event system components
        self.event_emitter = None
        self.event_listener = None

        # Project state tracking
        self.project_states = {}
        self.spec_states = {}
        self.generation_states = {}

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize event system components."""
        self.logger.info("Initializing CodeGenOrchestrator components")

        # Initialize event emitter
        self.event_emitter = SecureEventEmitter(
            service_url=self.pulsar_service_url,
            secret_key=self.secret_key,
            tenant=self.tenant,
            namespace=self.namespace,
        )

        # Initialize event listener
        self.event_listener = SecureEventListener(
            service_url=self.pulsar_service_url,
            subscription_name="code-gen-orchestrator",
            event_types=[
                EventType.CODE_GENERATION_REQUESTED,
                EventType.CODE_GENERATION_COMPLETED,
                EventType.CODE_GENERATION_FAILED,
                EventType.SPEC_SHEET_CREATED,
                EventType.SPEC_SHEET_UPDATED,
                EventType.SPEC_SHEET_COMPLETED,
                EventType.SPEC_SHEET_VALIDATED,
            ],
            secret_key=self.secret_key,
            tenant=self.tenant,
            namespace=self.namespace,
        )

        self.logger.info("CodeGenOrchestrator components initialized")

    async def initialize(self):
        """Initialize the orchestrator and start event listeners."""
        try:
            self.logger.info("Starting CodeGenOrchestrator")

            # Start event listener
            await self.event_listener.start()

            # Register event handlers
            self._register_event_handlers()

            self.logger.info("CodeGenOrchestrator started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start CodeGenOrchestrator: {e}")
            return False

    async def shutdown(self):
        """Shutdown the orchestrator and cleanup resources."""
        try:
            self.logger.info("Shutting down CodeGenOrchestrator")

            # Stop event listener
            await self.event_listener.stop()

            # Close event emitter
            self.event_emitter.close()

            self.logger.info("CodeGenOrchestrator shutdown complete")
            return True
        except Exception as e:
            self.logger.error(f"Error during CodeGenOrchestrator shutdown: {e}")
            return False

    def _register_event_handlers(self):
        """Register handlers for different event types."""
        self.logger.info("Registering event handlers")

        # Register handlers for code generation events
        self.event_listener.register_handler(
            EventType.CODE_GENERATION_REQUESTED, self._handle_code_generation_requested
        )

        self.event_listener.register_handler(
            EventType.CODE_GENERATION_COMPLETED, self._handle_code_generation_completed
        )

        self.event_listener.register_handler(EventType.CODE_GENERATION_FAILED, self._handle_code_generation_failed)

        # Register handlers for spec sheet events
        self.event_listener.register_handler(EventType.SPEC_SHEET_CREATED, self._handle_spec_sheet_created)

        self.event_listener.register_handler(EventType.SPEC_SHEET_UPDATED, self._handle_spec_sheet_updated)

        self.event_listener.register_handler(EventType.SPEC_SHEET_COMPLETED, self._handle_spec_sheet_completed)

        self.event_listener.register_handler(EventType.SPEC_SHEET_VALIDATED, self._handle_spec_sheet_validated)

        self.logger.info("Event handlers registered")

    async def _handle_code_generation_requested(self, event: BaseEvent):
        """Handle CODE_GENERATION_REQUESTED events."""
        try:
            self.logger.info(f"Handling code generation request: {event.event_id}")

            # Extract relevant data from event
            if isinstance(event.payload, dict):
                spec_sheet = event.payload.get("spec_sheet", {})
                spec_id = spec_sheet.get("id") if isinstance(spec_sheet, dict) else None
                target_language = event.payload.get("target_language", "python")
            else:
                # If it's a typed payload class
                spec_sheet = event.payload.spec_sheet
                spec_id = spec_sheet.get("id") if hasattr(spec_sheet, "get") else None
                target_language = event.payload.target_language

            if not spec_id:
                self.logger.error("Missing spec_id in code generation request")
                return

            # Create generation state entry
            generation_id = event.event_id
            self.generation_states[generation_id] = {
                "spec_id": spec_id,
                "status": "requested",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "target_language": target_language,
                "current_phase": CodeGenPhase.SPEC_ANALYSIS,
                "phases_completed": [],
                "correlation_id": event.correlation_id,
            }

            # Update spec state
            if spec_id in self.spec_states:
                self.spec_states[spec_id]["generation_id"] = generation_id
                self.spec_states[spec_id]["status"] = "generating"

            # Start the orchestration pipeline
            await self._orchestrate_code_generation(generation_id, spec_sheet, target_language)

        except Exception as e:
            self.logger.error(f"Error handling code generation request: {e}")
            # Emit failure event if possible
            if "generation_id" in locals() and "spec_id" in locals():
                await self._emit_generation_failed_event(
                    generation_id,
                    spec_id,
                    f"Orchestration error: {str(e)}",
                    "orchestration_error",
                )

    async def _handle_code_generation_completed(self, event: BaseEvent):
        """Handle CODE_GENERATION_COMPLETED events."""
        try:
            self.logger.info(f"Handling code generation completion: {event.event_id}")

            # Extract generation ID from the event
            generation_id = event.correlation_id

            if not generation_id or generation_id not in self.generation_states:
                self.logger.warning(f"Unknown generation ID in completion event: {generation_id}")
                return

            # Update generation state
            generation_state = self.generation_states[generation_id]
            generation_state["status"] = "completed"
            generation_state["completion_time"] = datetime.now(timezone.utc).isoformat()
            generation_state["current_phase"] = CodeGenPhase.COMPLETION
            generation_state["phases_completed"].append(CodeGenPhase.COMPLETION)

            # Update spec state
            spec_id = generation_state["spec_id"]
            if spec_id in self.spec_states:
                self.spec_states[spec_id]["status"] = "generated"
                self.spec_states[spec_id]["generation_completed"] = True

            # Extract generated code and results
            if isinstance(event.payload, dict):
                generated_code = event.payload.get("generated_code", "")
                confidence_score = event.payload.get("confidence_score", 0.0)
                program_ast = event.payload.get("program_ast", {})
            else:
                # If it's a typed payload class
                generated_code = event.payload.generated_code
                confidence_score = event.payload.confidence_score
                program_ast = event.payload.program_ast

            # Store the results
            generation_state["generated_code"] = generated_code
            generation_state["confidence_score"] = confidence_score
            generation_state["program_ast"] = program_ast

            # Notify any waiting components
            project_id = self.spec_states.get(spec_id, {}).get("project_id")
            if project_id and project_id in self.project_states:
                await self._update_project_state(project_id)

            self.logger.info(f"Code generation completed for {generation_id} with confidence {confidence_score}")

        except Exception as e:
            self.logger.error(f"Error handling code generation completion: {e}")

    async def _handle_code_generation_failed(self, event: BaseEvent):
        """Handle CODE_GENERATION_FAILED events."""
        try:
            self.logger.info(f"Handling code generation failure: {event.event_id}")

            # Extract generation ID from the event
            generation_id = event.correlation_id

            if not generation_id or generation_id not in self.generation_states:
                self.logger.warning(f"Unknown generation ID in failure event: {generation_id}")
                return

            # Update generation state
            generation_state = self.generation_states[generation_id]
            generation_state["status"] = "failed"
            generation_state["failure_time"] = datetime.now(timezone.utc).isoformat()

            # Extract error details
            if isinstance(event.payload, dict):
                error_message = event.payload.get("error_message", "Unknown error")
                error_type = event.payload.get("error_type", "unknown")
                partial_result = event.payload.get("partial_result")
            else:
                # If it's a typed payload class
                error_message = event.payload.error_message
                error_type = event.payload.error_type
                partial_result = event.payload.partial_result

            # Store the error information
            generation_state["error_message"] = error_message
            generation_state["error_type"] = error_type

            if partial_result:
                generation_state["partial_result"] = partial_result

            # Update spec state
            spec_id = generation_state["spec_id"]
            if spec_id in self.spec_states:
                self.spec_states[spec_id]["status"] = "generation_failed"
                self.spec_states[spec_id]["generation_error"] = error_message

            # Notify any waiting components
            project_id = self.spec_states.get(spec_id, {}).get("project_id")
            if project_id and project_id in self.project_states:
                await self._update_project_state(project_id)

            self.logger.error(f"Code generation failed for {generation_id}: {error_message}")

        except Exception as e:
            self.logger.error(f"Error handling code generation failure: {e}")

    async def _handle_spec_sheet_created(self, event: BaseEvent):
        """Handle SPEC_SHEET_CREATED events."""
        try:
            # Extract spec sheet ID and project ID
            spec_sheet_id = (
                event.spec_sheet_id if hasattr(event, "spec_sheet_id") else event.payload.get("spec_sheet_id")
            )
            project_id = event.payload.get("project_id")

            if not spec_sheet_id:
                self.logger.error("Missing spec_sheet_id in event")
                return

            # Add to spec states
            self.spec_states[spec_sheet_id] = {
                "id": spec_sheet_id,
                "project_id": project_id,
                "status": "created",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "completed": False,
                "validated": False,
                "generation_completed": False,
            }

            # Update project state
            if project_id and project_id in self.project_states:
                if "spec_sheet_ids" not in self.project_states[project_id]:
                    self.project_states[project_id]["spec_sheet_ids"] = []

                if spec_sheet_id not in self.project_states[project_id]["spec_sheet_ids"]:
                    self.project_states[project_id]["spec_sheet_ids"].append(spec_sheet_id)

                await self._update_project_state(project_id)

            self.logger.info(f"Spec sheet {spec_sheet_id} created for project {project_id}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet creation: {e}")

    async def _handle_spec_sheet_updated(self, event: BaseEvent):
        """Handle SPEC_SHEET_UPDATED events."""
        try:
            # Extract spec sheet ID
            spec_sheet_id = (
                event.spec_sheet_id if hasattr(event, "spec_sheet_id") else event.payload.get("spec_sheet_id")
            )

            if not spec_sheet_id or spec_sheet_id not in self.spec_states:
                self.logger.warning(f"Unknown spec sheet ID in update event: {spec_sheet_id}")
                return

            # Update spec state
            self.spec_states[spec_sheet_id]["status"] = "updated"
            self.spec_states[spec_sheet_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

            self.logger.info(f"Spec sheet {spec_sheet_id} updated")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet update: {e}")

    async def _handle_spec_sheet_completed(self, event: BaseEvent):
        """Handle SPEC_SHEET_COMPLETED events."""
        try:
            # Extract spec sheet ID
            spec_sheet_id = (
                event.spec_sheet_id if hasattr(event, "spec_sheet_id") else event.payload.get("spec_sheet_id")
            )

            if not spec_sheet_id or spec_sheet_id not in self.spec_states:
                self.logger.warning(f"Unknown spec sheet ID in completion event: {spec_sheet_id}")
                return

            # Update spec state
            self.spec_states[spec_sheet_id]["status"] = "completed"
            self.spec_states[spec_sheet_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.spec_states[spec_sheet_id]["completed"] = True

            # Check if this spec needs automatic validation
            project_id = self.spec_states[spec_sheet_id].get("project_id")
            if project_id and project_id in self.project_states:
                auto_validate = self.project_states[project_id].get("auto_validate_specs", False)
                if auto_validate:
                    # Request validation
                    await self._request_spec_validation(spec_sheet_id)

            self.logger.info(f"Spec sheet {spec_sheet_id} completed")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet completion: {e}")

    async def _handle_spec_sheet_validated(self, event: BaseEvent):
        """Handle SPEC_SHEET_VALIDATED events."""
        try:
            # Extract spec sheet ID and validation result
            spec_sheet_id = (
                event.spec_sheet_id if hasattr(event, "spec_sheet_id") else event.payload.get("spec_sheet_id")
            )
            if isinstance(event.payload, dict):
                is_valid = event.payload.get("is_valid", False)
                validation_errors = event.payload.get("validation_errors", [])
            else:
                # If it's a typed payload class
                is_valid = getattr(event.payload, "is_valid", False)
                validation_errors = getattr(event.payload, "validation_errors", [])

            if not spec_sheet_id or spec_sheet_id not in self.spec_states:
                self.logger.warning(f"Unknown spec sheet ID in validation event: {spec_sheet_id}")
                return

            # Update spec state
            self.spec_states[spec_sheet_id]["validated"] = is_valid
            self.spec_states[spec_sheet_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

            if is_valid:
                self.spec_states[spec_sheet_id]["status"] = "validated"

                # Check if this spec needs automatic code generation
                project_id = self.spec_states[spec_sheet_id].get("project_id")
                if project_id and project_id in self.project_states:
                    auto_generate = self.project_states[project_id].get("auto_generate_code", False)
                    if auto_generate:
                        # Request code generation
                        target_language = self.project_states[project_id].get("target_language", "python")
                        await self._request_code_generation(spec_sheet_id, target_language)
            else:
                self.spec_states[spec_sheet_id]["status"] = "validation_failed"
                self.spec_states[spec_sheet_id]["validation_errors"] = validation_errors

            self.logger.info(f"Spec sheet {spec_sheet_id} validation result: {is_valid}")

        except Exception as e:
            self.logger.error(f"Error handling spec sheet validation: {e}")

    async def _orchestrate_code_generation(self, generation_id: str, spec_sheet: Dict[str, Any], target_language: str):
        """
        Orchestrate the code generation process through multiple phases.

        Args:
            generation_id: The unique identifier for this generation
            spec_sheet: The specification sheet data
            target_language: The target programming language
        """
        try:
            self.logger.info(f"Starting code generation orchestration for {generation_id}")

            # Get generation state
            state = self.generation_states[generation_id]
            spec_id = state["spec_id"]

            # Phase 1: Spec Analysis
            state["current_phase"] = CodeGenPhase.SPEC_ANALYSIS
            spec_analysis_result = await self._execute_spec_analysis(spec_sheet)
            state["spec_analysis"] = spec_analysis_result
            state["phases_completed"].append(CodeGenPhase.SPEC_ANALYSIS)

            # Phase 2: Template Selection
            state["current_phase"] = CodeGenPhase.TEMPLATE_SELECTION
            template = await self._select_template(spec_analysis_result, target_language)
            state["template"] = template
            state["phases_completed"].append(CodeGenPhase.TEMPLATE_SELECTION)

            # Phase 3: AST Generation
            state["current_phase"] = CodeGenPhase.AST_GENERATION
            ast = await self._generate_ast(spec_sheet, template, target_language)
            state["ast"] = ast
            state["phases_completed"].append(CodeGenPhase.AST_GENERATION)

            # Phase 4: Code Generation
            state["current_phase"] = CodeGenPhase.CODE_GENERATION
            result = await self._generate_code(spec_id, ast, template, target_language)

            # If code generation was successful, continue with verification and optimization
            if result.get("success", False):
                # Store generated code
                state["generated_code"] = result.get("code", "")
                state["code_generation_details"] = result
                state["phases_completed"].append(CodeGenPhase.CODE_GENERATION)

                # Phase 5: Code Verification
                state["current_phase"] = CodeGenPhase.CODE_VERIFICATION
                verification_result = await self._verify_code(
                    generation_id, state["generated_code"], spec_sheet, target_language
                )
                state["verification_result"] = verification_result
                state["phases_completed"].append(CodeGenPhase.CODE_VERIFICATION)

                # Phase 6: Code Optimization (if verification passed)
                if verification_result.get("is_valid", False):
                    state["current_phase"] = CodeGenPhase.CODE_OPTIMIZATION
                    optimized_code = await self._optimize_code(state["generated_code"], target_language)
                    state["optimized_code"] = optimized_code
                    state["phases_completed"].append(CodeGenPhase.CODE_OPTIMIZATION)

                    # Use optimized code as the final output
                    state["final_code"] = optimized_code
                else:
                    # Use unoptimized code if verification failed
                    state["final_code"] = state["generated_code"]

                # Phase 7: Documentation
                state["current_phase"] = CodeGenPhase.DOCUMENTATION
                documentation = await self._generate_documentation(state["final_code"], spec_sheet, target_language)
                state["documentation"] = documentation
                state["phases_completed"].append(CodeGenPhase.DOCUMENTATION)

                # Emit completion event
                await self._emit_generation_completed_event(
                    generation_id,
                    spec_id,
                    state["final_code"],
                    state["ast"],
                    result.get("confidence_score", 0.0),
                    result.get("strategy_used", "neural"),
                    time.time() - time.mktime(datetime.fromisoformat(state["start_time"]).timetuple()),
                )

            else:
                # Code generation failed
                await self._emit_generation_failed_event(
                    generation_id,
                    spec_id,
                    result.get("error", "Unknown error during code generation"),
                    "code_generation_error",
                    result.get("partial_result"),
                )

            self.logger.info(f"Completed code generation orchestration for {generation_id}")

        except Exception as e:
            self.logger.error(f"Error during code generation orchestration: {e}")
            # Emit failure event
            await self._emit_generation_failed_event(
                generation_id,
                self.generation_states[generation_id]["spec_id"],
                f"Orchestration error: {str(e)}",
                "orchestration_error",
            )

    async def _execute_spec_analysis(self, spec_sheet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the spec sheet to determine generation approach.

        Args:
            spec_sheet: The specification sheet data

        Returns:
            Analysis results
        """
        # In a production system, this would perform more sophisticated analysis
        # For now, we'll do a simple analysis based on spec type

        spec_type = spec_sheet.get("type", "")

        analysis_result = {
            "complexity": "medium",
            "suggested_approach": "neural",
            "required_components": [],
            "dependencies": [],
        }

        # Determine complexity and approach based on spec type
        if spec_type == "container":
            analysis_result["complexity"] = "high"
            analysis_result["suggested_approach"] = "hybrid_grammar_neural"
            analysis_result["required_components"] = ["container_service", "event_bus"]

        elif spec_type == "api":
            analysis_result["complexity"] = "medium"
            analysis_result["suggested_approach"] = "tree_transformer"
            analysis_result["required_components"] = ["api_service", "router"]

        elif spec_type == "database":
            analysis_result["complexity"] = "medium"
            analysis_result["suggested_approach"] = "syntax_aware"
            analysis_result["required_components"] = ["database_model", "repository"]

        # Simulate analysis time
        await asyncio.sleep(0.1)

        return analysis_result

    async def _select_template(self, analysis_result: Dict[str, Any], target_language: str) -> Dict[str, Any]:
        """
        Select the appropriate template based on analysis results.

        Args:
            analysis_result: Results from spec analysis
            target_language: Target programming language

        Returns:
            Selected template
        """
        # In a production system, this would query a template registry
        # For now, we'll create a simple template based on the analysis

        approach = analysis_result.get("suggested_approach", "neural")
        components = analysis_result.get("required_components", [])

        template = {
            "id": f"template_{approach}_{target_language}",
            "name": f"{approach.capitalize()} {target_language.capitalize()} Template",
            "language": target_language,
            "approach": approach,
            "components": components,
            "structure": {
                "imports": [],
                "class_template": "",
                "function_templates": [],
            },
        }

        # Add language-specific templates
        if target_language == "python":
            template["structure"]["imports"] = [
                "import os",
                "import logging",
                "import json",
                "import asyncio",
            ]
            template["structure"]["class_template"] = "class {name}:\n    def __init__(self):\n        pass"
            template["structure"]["function_templates"] = ['def {name}({params}):\n    """Docstring"""\n    pass']

        elif target_language == "typescript":
            template["structure"]["imports"] = [
                "import * as fs from 'fs'",
                "import * as path from 'path'",
            ]
            template["structure"]["class_template"] = "class {name} {\n  constructor() {}\n}"
            template["structure"]["function_templates"] = ["function {name}({params}): void {\n  // Implementation\n}"]

        # Simulate template selection time
        await asyncio.sleep(0.1)

        return template

    async def _generate_ast(
        self, spec_sheet: Dict[str, Any], template: Dict[str, Any], target_language: str
    ) -> Dict[str, Any]:
        """
        Generate an abstract syntax tree based on the spec and template.

        Args:
            spec_sheet: The specification sheet data
            template: Selected template
            target_language: Target programming language

        Returns:
            Generated AST
        """
        # Extract key information from spec
        spec_type = spec_sheet.get("type", "")
        fields = spec_sheet.get("fields", {})

        # Create a basic AST structure
        ast = {
            "type": "Program",
            "language": target_language,
            "imports": [],
            "declarations": [],
        }

        # Add imports from template
        ast["imports"] = [{"type": "Import", "value": imp} for imp in template["structure"]["imports"]]

        # Generate AST based on spec type
        if spec_type == "container":
            # Container implementation
            container_name = fields.get("container_name", {}).get("value", "Container")
            container_class = {
                "type": "ClassDeclaration",
                "name": container_name,
                "methods": [
                    {
                        "type": "MethodDeclaration",
                        "name": ("__init__" if target_language == "python" else "constructor"),
                        "params": [],
                        "body": [],
                    },
                    {
                        "type": "MethodDeclaration",
                        "name": "start",
                        "params": [],
                        "body": [],
                    },
                    {
                        "type": "MethodDeclaration",
                        "name": "stop",
                        "params": [],
                        "body": [],
                    },
                ],
                "properties": [],
            }

            # Add to AST
            ast["declarations"].append(container_class)

        elif spec_type == "api":
            # API implementation
            api_name = fields.get("api_name", {}).get("value", "ApiService")
            api_class = {
                "type": "ClassDeclaration",
                "name": api_name,
                "methods": [
                    {
                        "type": "MethodDeclaration",
                        "name": ("__init__" if target_language == "python" else "constructor"),
                        "params": [],
                        "body": [],
                    }
                ],
                "properties": [],
            }

            # Add endpoints
            endpoints = fields.get("endpoints", {}).get("value", [])
            if isinstance(endpoints, list):
                for endpoint in endpoints:
                    if isinstance(endpoint, dict):
                        method_name = endpoint.get("name", "endpoint")
                        api_class["methods"].append(
                            {
                                "type": "MethodDeclaration",
                                "name": method_name,
                                "params": [],
                                "body": [],
                            }
                        )

            # Add to AST
            ast["declarations"].append(api_class)

        # Simulate AST generation time
        await asyncio.sleep(0.2)

        return ast

    async def _generate_code(
        self,
        spec_id: str,
        ast: Dict[str, Any],
        template: Dict[str, Any],
        target_language: str,
    ) -> Dict[str, Any]:
        """
        Generate code from the AST using the neural code generator.

        Args:
            spec_id: The spec sheet ID
            ast: Generated abstract syntax tree
            template: Selected template
            target_language: Target programming language

        Returns:
            Generation result
        """
        # In a production system, this would call the actual code generator
        # For now, we'll emit the code generation request event

        generation_id = str(uuid.uuid4())

        # Create the request payload
        spec_sheet = await self._get_spec_sheet(spec_id)
        if not spec_sheet:
            return {"success": False, "error": f"Spec sheet {spec_id} not found"}

        # Create and emit the code generation request event
        event = CodeGenerationRequestedEvent.create(
            source_container=Components.AST_CODE_GENERATOR,
            spec_sheet=spec_sheet,
            target_language=target_language,
            correlation_id=generation_id,
        )

        # Add AST and template to event metadata
        event.metadata.update(
            {
                "ast": ast,
                "template": template,
                "orchestrator_id": self.__class__.__name__,
            }
        )

        # Emit the event
        await self.event_emitter.emit_async(event)

        self.logger.info(f"Emitted code generation request for spec {spec_id}")

        # For demonstration purposes, simulate a successful generation
        # In a real system, this would wait for the completion event
        simulated_code = self._generate_simulated_code(ast, target_language)

        # Return success result with simulated code
        return {
            "success": True,
            "code": simulated_code,
            "confidence_score": 0.85,
            "strategy_used": template["approach"],
        }

    def _generate_simulated_code(self, ast: Dict[str, Any], target_language: str) -> str:
        """
        Generate simulated code for demonstration purposes.

        Args:
            ast: The abstract syntax tree
            target_language: Target programming language

        Returns:
            Generated code
        """
        code_lines = []

        # Add imports
        for imp in ast.get("imports", []):
            code_lines.append(imp["value"])

        code_lines.append("")  # Empty line after imports

        # Add declarations
        for decl in ast.get("declarations", []):
            if decl["type"] == "ClassDeclaration":
                if target_language == "python":
                    # Python class
                    code_lines.append(f"class {decl['name']}:")

                    # Add methods
                    for method in decl.get("methods", []):
                        method_name = method["name"]
                        params = "self"
                        if method_name != "__init__":
                            code_lines.append(f"    def {method_name}({params}):")
                            code_lines.append('        """')
                            code_lines.append(f"        {method_name.capitalize()} method")
                            code_lines.append('        """')
                            code_lines.append(f"        # TODO: Implement {method_name}")
                            code_lines.append("        pass")
                        else:
                            code_lines.append(f"    def {method_name}({params}):")
                            code_lines.append(f"        # Initialize {decl['name']}")
                            code_lines.append("        self.logger = logging.getLogger(self.__class__.__name__)")
                            code_lines.append("        self.initialized = True")

                    code_lines.append("")  # Empty line after class

                elif target_language == "typescript":
                    # TypeScript class
                    code_lines.append(f"class {decl['name']} {{")

                    # Add methods
                    for method in decl.get("methods", []):
                        method_name = method["name"]
                        if method_name == "constructor":
                            code_lines.append("  constructor() {")
                            code_lines.append(f"    // Initialize {decl['name']}")
                            code_lines.append("    this.initialized = true;")
                            code_lines.append("  }")
                        else:
                            code_lines.append(f"  {method_name}(): void {{")
                            code_lines.append(f"    // TODO: Implement {method_name}")
                            code_lines.append("  }")

                    code_lines.append("}")  # Class closing brace
                    code_lines.append("")  # Empty line after class

        # Add main block for Python
        if target_language == "python":
            code_lines.append('if __name__ == "__main__":')
            code_lines.append("    # Set up logging")
            code_lines.append("    logging.basicConfig(level=logging.INFO)")
            code_lines.append("    ")
            if ast.get("declarations") and ast.get("declarations")[0]["type"] == "ClassDeclaration":
                class_name = ast.get("declarations")[0]["name"]
                code_lines.append(f"    # Create {class_name} instance")
                code_lines.append(f"    instance = {class_name}()")

                # Find start method
                has_start = any(method["name"] == "start" for method in ast.get("declarations")[0].get("methods", []))
                if has_start:
                    code_lines.append("    # Start the instance")
                    code_lines.append("    instance.start()")

        # Join code lines
        return "\n".join(code_lines)

    async def _verify_code(
        self,
        generation_id: str,
        code: str,
        spec_sheet: Dict[str, Any],
        target_language: str,
    ) -> Dict[str, Any]:
        """
        Verify the generated code against the spec.

        Args:
            generation_id: Generation ID
            code: Generated code
            spec_sheet: The specification sheet
            target_language: Target programming language

        Returns:
            Verification result
        """
        # In a production system, this would perform code verification
        # For now, we'll do a simple check

        # Check if code is empty
        if not code or len(code.strip()) == 0:
            return {"is_valid": False, "errors": ["Generated code is empty"]}

        # Check if code contains basic elements based on spec type
        spec_type = spec_sheet.get("type", "")

        # Perform language-specific syntax checks
        syntax_valid = self._check_syntax(code, target_language)

        if not syntax_valid:
            return {
                "is_valid": False,
                "errors": ["Generated code contains syntax errors"],
            }

        # Perform spec-specific checks
        if spec_type == "container":
            # Check for container class
            container_name = spec_sheet.get("fields", {}).get("container_name", {}).get("value", "")
            if container_name and container_name not in code:
                return {
                    "is_valid": False,
                    "errors": [f"Container class '{container_name}' not found in code"],
                }

        # Simulate verification time
        await asyncio.sleep(0.2)

        return {"is_valid": True, "errors": []}

    def _check_syntax(self, code: str, language: str) -> bool:
        """
        Check syntax of generated code.

        Args:
            code: Generated code
            language: Target language

        Returns:
            True if syntax is valid, False otherwise
        """
        # In a production system, this would use language-specific parsers
        # For now, we'll do a simple check

        try:
            # For Python, we can use the ast module
            if language == "python":
                import ast

                ast.parse(code)
                return True

            # For other languages, just check for balanced braces
            else:
                # Simple check for balanced braces
                stack = []
                for char in code:
                    if char in "({[":
                        stack.append(char)
                    elif char in ")}]":
                        if not stack:
                            return False

                        opening = stack.pop()
                        if opening == "(" and char != ")":
                            return False
                        if opening == "{" and char != "}":
                            return False
                        if opening == "[" and char != "]":
                            return False

                return len(stack) == 0

        except SyntaxError:
            return False
        except Exception:
            return False

    async def _optimize_code(self, code: str, target_language: str) -> str:
        """
        Optimize the generated code.

        Args:
            code: Generated code
            target_language: Target programming language

        Returns:
            Optimized code
        """
        # In a production system, this would perform code optimization
        # For now, we'll just add an optimization comment

        optimized_code = code

        comment_marker = "# " if target_language == "python" else "// "
        optimization_comment = f"{comment_marker}Optimized by CodeGenOrchestrator\n"

        # Add optimization comment at top
        optimized_code = optimization_comment + optimized_code

        # Simulate optimization time
        await asyncio.sleep(0.1)

        return optimized_code

    async def _generate_documentation(
        self, code: str, spec_sheet: Dict[str, Any], target_language: str
    ) -> Dict[str, Any]:
        """
        Generate documentation for the code.

        Args:
            code: Generated code
            spec_sheet: Specification sheet
            target_language: Target programming language

        Returns:
            Documentation data
        """
        # In a production system, this would generate comprehensive documentation
        # For now, we'll create a simple documentation object

        spec_type = spec_sheet.get("type", "")

        documentation = {
            "overview": f"Generated {spec_type} implementation in {target_language}",
            "usage_examples": [],
            "sections": [],
        }

        # Add usage example
        if target_language == "python":
            documentation["usage_examples"].append(
                f"# Example usage\n"
                f"from {spec_type.lower()} import {spec_type.capitalize()}\n\n"
                f"instance = {spec_type.capitalize()}()\n"
                f"instance.start()"
            )
        elif target_language == "typescript":
            documentation["usage_examples"].append(
                f"// Example usage\n"
                f"import {{ {spec_type.capitalize()} }} from './{spec_type.lower()}';\n\n"
                f"const instance = new {spec_type.capitalize()}();\n"
                f"instance.start();"
            )

        # Add sections based on spec type
        if spec_type == "container":
            documentation["sections"].append(
                {
                    "title": "Container Overview",
                    "content": "This container implements the specified functionality with event handling capabilities.",
                }
            )

            documentation["sections"].append(
                {
                    "title": "Event Handling",
                    "content": "The container handles events through registered event handlers and the event bus.",
                }
            )

        # Simulate documentation generation time
        await asyncio.sleep(0.1)

        return documentation

    async def _update_project_state(self, project_id: str):
        """
        Update project state based on current spec states.

        Args:
            project_id: The project ID
        """
        if project_id not in self.project_states:
            self.logger.warning(f"Unknown project ID: {project_id}")
            return

        project_state = self.project_states[project_id]
        spec_sheet_ids = project_state.get("spec_sheet_ids", [])

        # Count completed specs
        completed_count = 0
        validated_count = 0
        generated_count = 0

        for spec_id in spec_sheet_ids:
            if spec_id in self.spec_states:
                spec_state = self.spec_states[spec_id]

                if spec_state.get("completed", False):
                    completed_count += 1

                if spec_state.get("validated", False):
                    validated_count += 1

                if spec_state.get("generation_completed", False):
                    generated_count += 1

        # Update project counters
        project_state["completed_specs_count"] = completed_count
        project_state["validated_specs_count"] = validated_count
        project_state["generated_specs_count"] = generated_count
        project_state["total_specs_count"] = len(spec_sheet_ids)

        # Determine project status based on counts
        if generated_count == len(spec_sheet_ids) and len(spec_sheet_ids) > 0:
            project_state["status"] = "code_generated"
        elif validated_count == len(spec_sheet_ids) and len(spec_sheet_ids) > 0:
            project_state["status"] = "specs_validated"
        elif completed_count == len(spec_sheet_ids) and len(spec_sheet_ids) > 0:
            project_state["status"] = "specs_completed"
        elif len(spec_sheet_ids) > 0:
            project_state["status"] = "specs_in_progress"
        else:
            project_state["status"] = "created"

        self.logger.info(f"Updated project state: {project_id} - {project_state['status']}")

    async def _request_spec_validation(self, spec_id: str):
        """
        Request validation for a spec sheet.

        Args:
            spec_id: The spec sheet ID
        """
        # In a production system, this would emit a validation request event
        # For now, we'll just log the request

        self.logger.info(f"Requesting validation for spec sheet: {spec_id}")

        # Simulate validation by directly updating the spec state
        if spec_id in self.spec_states:
            self.spec_states[spec_id]["validated"] = True
            self.spec_states[spec_id]["status"] = "validated"

            # Update project state
            project_id = self.spec_states[spec_id].get("project_id")
            if project_id:
                await self._update_project_state(project_id)

    async def _request_code_generation(self, spec_id: str, target_language: str):
        """
        Request code generation for a validated spec sheet.

        Args:
            spec_id: The spec sheet ID
            target_language: Target programming language
        """
        # Get spec sheet data
        spec_sheet = await self._get_spec_sheet(spec_id)
        if not spec_sheet:
            self.logger.error(f"Cannot generate code for unknown spec: {spec_id}")
            return

        # Create generation ID
        generation_id = str(uuid.uuid4())

        # Update spec state
        if spec_id in self.spec_states:
            self.spec_states[spec_id]["generation_id"] = generation_id
            self.spec_states[spec_id]["status"] = "generating"

        # Create generation state
        self.generation_states[generation_id] = {
            "spec_id": spec_id,
            "status": "requested",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "target_language": target_language,
            "current_phase": CodeGenPhase.SPEC_ANALYSIS,
        }

        # Start orchestration
        await self._orchestrate_code_generation(generation_id, spec_sheet, target_language)

    async def _get_spec_sheet(self, spec_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a spec sheet by ID.

        Args:
            spec_id: The spec sheet ID

        Returns:
            Spec sheet data or None if not found
        """
        # In a production system, this would fetch the spec from storage
        # For now, we'll create a simple spec based on the ID

        if spec_id not in self.spec_states:
            return None

        spec_state = self.spec_states[spec_id]

        # Determine spec type from ID
        spec_type = "container"
        if spec_id.startswith("api"):
            spec_type = "api"
        elif spec_id.startswith("db"):
            spec_type = "database"

        # Create a basic spec sheet
        spec_sheet = {"id": spec_id, "type": spec_type, "fields": {}}

        # Add fields based on spec type
        if spec_type == "container":
            spec_sheet["fields"] = {
                "container_name": {
                    "value": f"Container{spec_id.capitalize()}",
                    "type": "string",
                },
                "description": {"value": f"A container for {spec_id}", "type": "text"},
                "dependencies": {"value": ["logging", "asyncio"], "type": "list"},
                "event_handlers": {
                    "value": [
                        {
                            "event_type": "example.event",
                            "handler": "handle_example_event",
                        }
                    ],
                    "type": "list",
                },
                "event_bus_config": {
                    "value": {
                        "broker_url": "pulsar://localhost:6650",
                        "topic": "example-topic",
                    },
                    "type": "json",
                },
                "main_logic": {
                    "value": "async def run():\n    # Main container logic\n    pass",
                    "type": "code",
                },
            }
        elif spec_type == "api":
            spec_sheet["fields"] = {
                "api_name": {"value": f"Api{spec_id.capitalize()}", "type": "string"},
                "endpoints": {
                    "value": [
                        {"path": "/example", "method": "GET", "name": "get_example"},
                        {
                            "path": "/example",
                            "method": "POST",
                            "name": "create_example",
                        },
                    ],
                    "type": "list",
                },
                "auth_method": {"value": "jwt", "type": "string"},
                "documentation": {
                    "value": "API documentation for example endpoints",
                    "type": "text",
                },
            }

        return spec_sheet

    async def _emit_generation_completed_event(
        self,
        generation_id: str,
        spec_id: str,
        code: str,
        ast: Dict[str, Any],
        confidence_score: float,
        strategy_used: str,
        time_taken: float,
    ):
        """
        Emit a generation completed event.

        Args:
            generation_id: Generation ID (used as correlation ID)
            spec_id: Spec sheet ID
            code: Generated code
            ast: Program AST
            confidence_score: Confidence score
            strategy_used: Strategy used for generation
            time_taken: Time taken in seconds
        """
        event = CodeGenerationCompletedEvent.create(
            source_container=Components.AST_CODE_GENERATOR,
            generated_code=code,
            program_ast=ast,
            confidence_score=confidence_score,
            strategy_used=strategy_used,
            time_taken=time_taken,
            correlation_id=generation_id,
        )

        # Add spec ID to metadata
        event.metadata["spec_id"] = spec_id
        event.metadata["orchestrator_id"] = self.__class__.__name__

        # Emit the event
        await self.event_emitter.emit_async(event)

        self.logger.info(f"Emitted code generation completion for spec {spec_id}")

    async def _emit_generation_failed_event(
        self,
        generation_id: str,
        spec_id: str,
        error_message: str,
        error_type: str,
        partial_result: Optional[Dict[str, Any]] = None,
    ):
        """
        Emit a generation failed event.

        Args:
            generation_id: Generation ID (used as correlation ID)
            spec_id: Spec sheet ID
            error_message: Error message
            error_type: Error type
            partial_result: Optional partial result
        """
        event = CodeGenerationFailedEvent.create(
            source_container=Components.AST_CODE_GENERATOR,
            error_message=error_message,
            error_type=error_type,
            partial_result=partial_result,
            correlation_id=generation_id,
        )

        # Add spec ID to metadata
        event.metadata["spec_id"] = spec_id
        event.metadata["orchestrator_id"] = self.__class__.__name__

        # Emit the event
        await self.event_emitter.emit_async(event)

        self.logger.info(f"Emitted code generation failure for spec {spec_id}: {error_message}")

    # Public API methods

    async def create_project(self, project_data: Dict[str, Any]) -> str:
        """
        Create a new project.

        Args:
            project_data: Project data

        Returns:
            Project ID
        """
        project_id = project_data.get("id", str(uuid.uuid4()))

        self.project_states[project_id] = {
            "id": project_id,
            "name": project_data.get("name", f"Project {project_id}"),
            "description": project_data.get("description", ""),
            "status": "created",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "spec_sheet_ids": [],
            "auto_validate_specs": project_data.get("auto_validate_specs", False),
            "auto_generate_code": project_data.get("auto_generate_code", False),
            "target_language": project_data.get("target_language", "python"),
        }

        self.logger.info(f"Created project {project_id}")

        return project_id

    async def start_code_generation(self, spec_id: str, target_language: str = "python") -> str:
        """
        Start code generation for a spec sheet.

        Args:
            spec_id: Spec sheet ID
            target_language: Target programming language

        Returns:
            Generation ID
        """
        # Check if spec exists and is validated
        if spec_id not in self.spec_states:
            raise ValueError(f"Spec sheet {spec_id} not found")

        spec_state = self.spec_states[spec_id]
        if not spec_state.get("validated", False):
            self.logger.warning(f"Spec sheet {spec_id} is not validated, requesting validation")
            await self._request_spec_validation(spec_id)

        # Request code generation
        await self._request_code_generation(spec_id, target_language)

        # Return the generation ID
        return spec_state.get("generation_id", "")

    async def get_generation_status(self, generation_id: str) -> Dict[str, Any]:
        """
        Get status of a code generation process.

        Args:
            generation_id: Generation ID

        Returns:
            Generation status info
        """
        if generation_id not in self.generation_states:
            raise ValueError(f"Generation {generation_id} not found")

        state = self.generation_states[generation_id]

        return {
            "id": generation_id,
            "spec_id": state.get("spec_id"),
            "status": state.get("status"),
            "current_phase": state.get("current_phase"),
            "phases_completed": state.get("phases_completed", []),
            "start_time": state.get("start_time"),
            "completion_time": state.get("completion_time"),
            "generated_code": (state.get("final_code") if state.get("status") == "completed" else None),
        }

    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """
        Get status of a project.

        Args:
            project_id: Project ID

        Returns:
            Project status info
        """
        if project_id not in self.project_states:
            raise ValueError(f"Project {project_id} not found")

        state = self.project_states[project_id]

        return {
            "id": project_id,
            "name": state.get("name"),
            "status": state.get("status"),
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            "spec_sheet_count": len(state.get("spec_sheet_ids", [])),
            "completed_specs_count": state.get("completed_specs_count", 0),
            "validated_specs_count": state.get("validated_specs_count", 0),
            "generated_specs_count": state.get("generated_specs_count", 0),
        }


# Example usage
async def run_orchestrator():
    """Run the code generation orchestrator service."""
    orchestrator = CodeGenOrchestrator(pulsar_service_url="pulsar://localhost:6650", enable_metrics=True)

    try:
        # Initialize and start the orchestrator
        await orchestrator.initialize()

        # Create a test project
        project_id = await orchestrator.create_project(
            {
                "name": "Test Project",
                "description": "A test project for code generation",
                "target_language": "python",
                "auto_validate_specs": True,
                "auto_generate_code": True,
            }
        )

        # Keep the service running
        logger.info("Code Generation Orchestrator running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutting down Code Generation Orchestrator...")
    finally:
        # Shutdown the orchestrator
        await orchestrator.shutdown()


if __name__ == "__main__":
    # Run the orchestrator service
    asyncio.run(run_orchestrator())
