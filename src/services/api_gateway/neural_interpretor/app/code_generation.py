# code_generation_router.py
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pulsar
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CodeGenRequestType(str, Enum):
    """Types of code generation requests"""
    # Workflow phases
    WORKFLOW_SOURCE_OF_TRUTH = "workflow_source_of_truth"
    WORKFLOW_SPEC_SHEETS = "workflow_spec_sheets"
    WORKFLOW_FILLED_SPECS = "workflow_filled_specs"
    WORKFLOW_IMPLEMENTATION = "workflow_implementation"
    WORKFLOW_INTEGRATION = "workflow_integration"
    WORKFLOW_DEBUGGING = "workflow_debugging"

    # One-off requests
    FUNCTION = "function"
    CLASS = "class"
    COMPONENT = "component"
    SINGLE_FILE = "single_file"
    UTILITY = "utility"

    # Conceptual requests
    BRAINSTORMING = "brainstorming"
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"


class ProgrammingLanguage(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    NEXTJS = "nextjs"
    REACT = "react"
    PULSAR = "pulsar"
    HTML_CSS = "html_css"
    OTHER = "other"


class CodeGenRequest(BaseModel):
    """Code generation request object"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    request_type: CodeGenRequestType
    prompt: str
    system_message: Optional[str] = None
    language: Optional[ProgrammingLanguage] = None
    project_id: Optional[str] = None
    workflow_id: Optional[str] = None
    phase: Optional[int] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            Enum: lambda v: v.value
        }


class CodeGenRouter:
    """
    Routes code generation requests to appropriate topics based on
    request type, language, and other factors.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        self.config = config
        self.pulsar_client = None
        self.producers = {}
        self._lock = asyncio.Lock()

        # Keywords for request type classification
        self.request_type_keywords = self.config.get("code_gen_classification", {}).get("request_types", {})
        self.language_keywords = self.config.get("code_gen_classification", {}).get("languages", {})

    async def initialize(self):
        """Initialize Pulsar client and producers"""
        try:
            # Get Pulsar connection settings
            pulsar_url = self.config["pulsar"]["service_url"]
            if self.config["pulsar"]["tls_enabled"]:
                pulsar_url = self.config["pulsar"]["service_url_tls"]

            # Create Pulsar client
            self.pulsar_client = pulsar.Client(
                pulsar_url,
                authentication=pulsar.AuthenticationTls(
                    self.config["pulsar"]["tls_cert_file_path"],
                    self.config["pulsar"]["tls_key_file_path"]
                ) if self.config["pulsar"]["tls_enabled"] else None
            )

            # Create producers for all topics
            topics = self.config.get("topics", {})
            for topic_key, topic_name in topics.items():
                self.producers[topic_key] = self.pulsar_client.create_producer(
                    topic=topic_name,
                    block_if_queue_full=True,
                    batching_enabled=True,
                    batching_max_publish_delay_ms=10
                )

            logger.info(f"Initialized Pulsar client and {len(self.producers)} producers")

        except Exception as e:
            logger.error(f"Failed to initialize Pulsar: {e}")
            if self.pulsar_client:
                self.pulsar_client.close()
            raise

    async def close(self):
        """Close Pulsar connections"""
        if self.pulsar_client:
            for producer_name, producer in self.producers.items():
                try:
                    producer.close()
                except Exception as e:
                    logger.error(f"Error closing producer {producer_name}: {e}")

            self.pulsar_client.close()
            logger.info("Closed Pulsar client and producers")

    def determine_request_type(self, prompt: str) -> CodeGenRequestType:
        """
        Analyze prompt to determine request type
        """
        prompt_lower = prompt.lower()

        # Check for workflow phases
        workflow_keywords = self.request_type_keywords.get("workflow", {})
        if any(kw in prompt_lower for kw in workflow_keywords.get("source_of_truth", [])):
            return CodeGenRequestType.WORKFLOW_SOURCE_OF_TRUTH
        elif any(kw in prompt_lower for kw in workflow_keywords.get("spec_sheets", [])):
            return CodeGenRequestType.WORKFLOW_SPEC_SHEETS
        elif any(kw in prompt_lower for kw in workflow_keywords.get("filled_specs", [])):
            return CodeGenRequestType.WORKFLOW_FILLED_SPECS
        elif any(kw in prompt_lower for kw in workflow_keywords.get("implementation", [])):
            return CodeGenRequestType.WORKFLOW_IMPLEMENTATION
        elif any(kw in prompt_lower for kw in workflow_keywords.get("integration", [])):
            return CodeGenRequestType.WORKFLOW_INTEGRATION
        elif any(kw in prompt_lower for kw in workflow_keywords.get("debugging", [])):
            return CodeGenRequestType.WORKFLOW_DEBUGGING

        # Check for one-off requests
        one_off_keywords = self.request_type_keywords.get("one_off", {})
        if any(kw in prompt_lower for kw in one_off_keywords.get("function", [])):
            return CodeGenRequestType.FUNCTION
        elif any(kw in prompt_lower for kw in one_off_keywords.get("class", [])):
            return CodeGenRequestType.CLASS
        elif any(kw in prompt_lower for kw in one_off_keywords.get("component", [])):
            return CodeGenRequestType.COMPONENT
        elif any(kw in prompt_lower for kw in one_off_keywords.get("file", [])):
            return CodeGenRequestType.SINGLE_FILE
        elif any(kw in prompt_lower for kw in one_off_keywords.get("utility", [])):
            return CodeGenRequestType.UTILITY

        # Check for conceptual requests
        conceptual_keywords = self.request_type_keywords.get("conceptual", {})
        if any(kw in prompt_lower for kw in conceptual_keywords.get("brainstorming", [])):
            return CodeGenRequestType.BRAINSTORMING
        elif any(kw in prompt_lower for kw in conceptual_keywords.get("architecture", [])):
            return CodeGenRequestType.ARCHITECTURE
        elif any(kw in prompt_lower for kw in conceptual_keywords.get("optimization", [])):
            return CodeGenRequestType.OPTIMIZATION

        # Default to single file if no matches
        return CodeGenRequestType.SINGLE_FILE

    def determine_language(self, prompt: str) -> Optional[ProgrammingLanguage]:
        """
        Analyze prompt to determine programming language
        """
        prompt_lower = prompt.lower()

        for lang, keywords in self.language_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                return ProgrammingLanguage(lang)

        return None

    def get_topic_for_request(self, request: CodeGenRequest) -> str:
        """
        Determine the appropriate topic for a request
        """
        # Map request types to topics
        topic_mapping = {
            # Workflow phases
            CodeGenRequestType.WORKFLOW_SOURCE_OF_TRUTH: "workflow_phase0",
            CodeGenRequestType.WORKFLOW_SPEC_SHEETS: "workflow_phase1",
            CodeGenRequestType.WORKFLOW_FILLED_SPECS: "workflow_phase2",
            CodeGenRequestType.WORKFLOW_IMPLEMENTATION: "workflow_phase3",
            CodeGenRequestType.WORKFLOW_INTEGRATION: "workflow_phase4",
            CodeGenRequestType.WORKFLOW_DEBUGGING: "workflow_phase5",

            # One-off requests
            CodeGenRequestType.FUNCTION: "single_requests",
            CodeGenRequestType.CLASS: "single_requests",
            CodeGenRequestType.COMPONENT: "single_requests",
            CodeGenRequestType.SINGLE_FILE: "single_requests",
            CodeGenRequestType.UTILITY: "single_requests",

            # Conceptual requests
            CodeGenRequestType.BRAINSTORMING: "brainstorming",
            CodeGenRequestType.ARCHITECTURE: "conceptual",
            CodeGenRequestType.OPTIMIZATION: "conceptual",
        }

        return topic_mapping.get(request.request_type, "single_requests")

    async def route_request(self, prompt: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a code generation request:
        1. Analyze to determine type and language
        2. Create request object
        3. Route to appropriate topic
        4. Return acknowledgment with metadata
        """
        metadata = metadata or {}
        start_time = time.time()

        try:
            # Determine request type and language
            request_type = self.determine_request_type(prompt)
            language = self.determine_language(prompt)

            # Create request object
            request = CodeGenRequest(
                request_type=request_type,
                prompt=prompt,
                language=language,
                user_id=metadata.get("user_id"),
                project_id=metadata.get("project_id"),
                workflow_id=metadata.get("workflow_id"),
                phase=metadata.get("phase"),
                system_message=metadata.get("system_message"),
                metadata=metadata
            )

            # Determine topic
            topic_key = self.get_topic_for_request(request)

            # Publish to topic
            if topic_key in self.producers:
                producer = self.producers[topic_key]
                message = json.dumps(request.dict())
                producer.send(
                    message.encode('utf-8'),
                    properties={
                        "request_id": request.request_id,
                        "request_type": request.request_type,
                        "language": request.language if request.language else "unknown"
                    }
                )

                logger.info(f"Routed request {request.request_id} to topic {topic_key} as {request.request_type}")

                # Track processing time
                processing_time = time.time() - start_time

                # Return acknowledgment
                return {
                    "request_id": request.request_id,
                    "status": "accepted",
                    "request_type": request.request_type,
                    "language": request.language,
                    "topic": topic_key,
                    "processing_time_ms": int(processing_time * 1000)
                }
            else:
                raise ValueError(f"No producer found for topic key: {topic_key}")

        except Exception as e:
            logger.error(f"Error routing request: {e}")
            raise