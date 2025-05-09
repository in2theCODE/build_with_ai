"""
Neural Interpreter Core - Classifies and broadcasts tasks for agent self-selection

This module provides the core Neural Interpreter that analyzes incoming requests,
assigns rich metadata dimensions, and publishes them to Pulsar topics where
autonomous agents can choose tasks that match their capabilities.
"""

import asyncio
from datetime import datetime
import json
import time
from typing import Any, Dict, Optional
import uuid

import pulsar
from pydantic import BaseModel
from pydantic import Field
from src.services.shared.concurrency.concurrency import AsyncTaskManager
from src.services.shared.logging.logger import get_logger

# Keep the imports exactly as they are - they're correct
from src.services.shared.models import ProcessingMode
from src.services.shared.models import Task
from src.services.shared.models import TaskPriority
from src.services.shared.models import TaskStatus
from src.services.shared.monitoring.metrics import MetricsCollector
from src.services.shared.validation.validator import ValidationResult
from src.services.shared.validation.validator import Validator


# Configure logging
logger = get_logger(__name__)

# Initialize metrics collector
metrics_collector = MetricsCollector("neural_interpreter")


class IntentDimension(BaseModel):
    """Scoring dimensions for the intent classification system"""

    information_retrieval: int = Field(0, ge=0, le=10)
    task_execution: int = Field(0, ge=0, le=10)
    creative_generation: int = Field(0, ge=0, le=10)
    analysis: int = Field(0, ge=0, le=10)
    conversation: int = Field(0, ge=0, le=10)


class DomainDimension(BaseModel):
    """Domain classification scores"""

    technical: int = Field(0, ge=0, le=10)
    business: int = Field(0, ge=0, le=10)
    creative: int = Field(0, ge=0, le=10)
    scientific: int = Field(0, ge=0, le=10)
    general: int = Field(0, ge=0, le=10)


class ComplexityDimension(BaseModel):
    """Task complexity metrics"""

    tokens_required: int = Field(0, ge=0, le=10)
    context_depth: int = Field(0, ge=0, le=10)
    specialized_knowledge: int = Field(0, ge=0, le=10)


class ActionabilityDimension(BaseModel):
    """Scores for what actions the task requires"""

    tool_usage: int = Field(0, ge=0, le=10)
    external_data: int = Field(0, ge=0, le=10)
    computation: int = Field(0, ge=0, le=10)


class UrgencyDimension(BaseModel):
    """Urgency and importance metrics"""

    time_sensitivity: int = Field(0, ge=0, le=10)
    importance: int = Field(0, ge=0, le=10)


class SpecificityDimension(BaseModel):
    """How well-specified the task is"""

    clarity: int = Field(0, ge=0, le=10)
    constraints: int = Field(0, ge=0, le=10)
    examples_provided: int = Field(0, ge=0, le=10)


class MultimodalityDimension(BaseModel):
    """Requirements for different modalities"""

    code_required: int = Field(0, ge=0, le=10)
    visual_required: int = Field(0, ge=0, le=10)
    structured_data: int = Field(0, ge=0, le=10)


class TaskMetadata(BaseModel):
    """Complete metadata for a task, used for agent self-selection"""

    intent: IntentDimension
    domain: DomainDimension
    complexity: ComplexityDimension
    actionability: ActionabilityDimension
    urgency: UrgencyDimension
    specificity: SpecificityDimension
    multimodality: MultimodalityDimension

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to a flat dictionary for easier filtering"""
        result = {}
        for category_name, category in self.dict().items():
            for dimension_name, score in category.items():
                result[f"{category_name}.{dimension_name}"] = score
        return result


class NeuralInterpretor:
    """
    Analyzes tasks to generate rich metadata for agent self-selection
    """

    # Keywords associated with different dimensions
    INTENT_KEYWORDS = {
        "information_retrieval": [
            "find",
            "search",
            "lookup",
            "what is",
            "tell me about",
            "information on",
            "get data",
            "retrieve",
            "look up",
            "where can I find",
        ],
        "task_execution": [
            "create",
            "make",
            "build",
            "generate",
            "execute",
            "perform",
            "run",
            "implement",
            "do",
            "complete",
            "finish",
            "accomplish",
        ],
        "creative_generation": [
            "write",
            "compose",
            "design",
            "draft",
            "create",
            "imagine",
            "story",
            "creative",
            "novel",
            "innovative",
            "artistic",
            "poem",
            "song",
        ],
        "analysis": [
            "analyze",
            "evaluate",
            "assess",
            "examine",
            "study",
            "investigate",
            "research",
            "review",
            "compare",
            "contrast",
            "synthesize",
            "breakdown",
        ],
        "conversation": [
            "chat",
            "talk",
            "discuss",
            "conversation",
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "how are you",
            "what's up",
        ],
    }

    DOMAIN_KEYWORDS = {
        "technical": [
            "code",
            "programming",
            "software",
            "development",
            "algorithm",
            "technical",
            "engineering",
            "system",
            "framework",
            "api",
            "database",
            "infrastructure",
        ],
        "business": [
            "business",
            "finance",
            "marketing",
            "strategy",
            "management",
            "sales",
            "customer",
            "market",
            "revenue",
            "profit",
            "commercial",
            "enterprise",
        ],
        "creative": [
            "creative",
            "art",
            "design",
            "music",
            "writing",
            "story",
            "visual",
            "artistic",
            "aesthetic",
            "composition",
            "style",
            "narrative",
        ],
        "scientific": [
            "science",
            "scientific",
            "research",
            "experiment",
            "theory",
            "hypothesis",
            "analysis",
            "laboratory",
            "investigation",
            "study",
            "observation",
            "data",
        ],
        "general": [
            "general",
            "everyday",
            "common",
            "regular",
            "standard",
            "normal",
            "typical",
            "usual",
            "ordinary",
            "conventional",
            "customary",
        ],
    }

    # Workflow pattern keywords
    WORKFLOW_KEYWORDS = {
        "code_generation": [
            "generate code",
            "write code",
            "implement function",
            "create class",
            "coding task",
            "programming task",
            "code implementation",
        ],
        "data_analysis": [
            "analyze data",
            "data processing",
            "dataset analysis",
            "statistical analysis",
            "data mining",
            "data exploration",
            "data visualization",
        ],
        "content_creation": [
            "create content",
            "generate article",
            "write blog",
            "content creation",
            "draft document",
            "produce content",
            "content generation",
        ],
        "customer_support": [
            "customer inquiry",
            "support request",
            "customer question",
            "help customer",
            "resolve issue",
            "customer problem",
            "support ticket",
        ],
        "knowledge_retrieval": [
            "find information",
            "knowledge lookup",
            "retrieve data",
            "get details",
            "find answer",
            "documentation lookup",
            "information retrieval",
        ],
    }

    def __init__(self):
        """Initialize the metadata analyzer"""
        self.token_counter = self._simple_token_counter
        logger.info("Initialized NeuralInterpretor")
        metrics_collector.set_component_up(True)

    def _simple_token_counter(self, text: str) -> int:
        """
        Simple token counting approximation
        In production, use a proper tokenizer based on your LLM
        """
        return len(text.split())

    def analyze_task(self, prompt: str, system_message: Optional[str] = None) -> TaskMetadata:
        """
        Analyze a task prompt to generate rich metadata

        Args:
            prompt: The task prompt text
            system_message: Optional system context

        Returns:
            TaskMetadata object with scores across all dimensions
        """
        # Track analysis time with metrics collector
        timer = metrics_collector.start_request_timer(strategy="task_analysis")

        # Generate scores for each dimension
        intent_scores = self._analyze_intent(prompt)
        domain_scores = self._analyze_domain(prompt)
        complexity_scores = self._analyze_complexity(prompt, system_message)
        actionability_scores = self._analyze_actionability(prompt)
        urgency_scores = self._analyze_urgency(prompt)
        specificity_scores = self._analyze_specificity(prompt)
        multimodality_scores = self._analyze_multimodality(prompt)

        # Analyze workflow patterns
        self._analyze_workflow_patterns(prompt)

        # Create task metadata
        result = TaskMetadata(
            intent=IntentDimension(**intent_scores),
            domain=DomainDimension(**domain_scores),
            complexity=ComplexityDimension(**complexity_scores),
            actionability=ActionabilityDimension(**actionability_scores),
            urgency=UrgencyDimension(**urgency_scores),
            specificity=SpecificityDimension(**specificity_scores),
            multimodality=MultimodalityDimension(**multimodality_scores),
        )

        # Record token counts for metrics
        if prompt:
            metrics_collector.record_tokens("input", len(prompt.split()))
        if system_message:
            metrics_collector.record_tokens("system", len(system_message.split()))

        return result

    def _analyze_intent(self, text: str) -> Dict[str, int]:
        """
        Analyze the intent dimensions of a task

        Args:
            text: The task text

        Returns:
            Dictionary of intent dimension scores
        """
        text_lower = text.lower()
        scores = {}

        # Calculate scores based on keyword matches
        for dimension, keywords in self.INTENT_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            # Normalize to 0-10 scale, max score is 3
            scores[dimension] = min(10, int(score * 3.33))

        # Adjust based on question marks (indicates information retrieval)
        if "?" in text:
            scores["information_retrieval"] = min(10, scores.get("information_retrieval", 0) + 2)

        # Ensure we have values for all dimensions
        for dimension in IntentDimension.__annotations__.keys():
            if dimension not in scores:
                scores[dimension] = 0

        return scores

    def _analyze_domain(self, text: str) -> Dict[str, int]:
        """
        Analyze the domain dimensions of a task

        Args:
            text: The task text

        Returns:
            Dictionary of domain dimension scores
        """
        text_lower = text.lower()
        scores = {}

        # Calculate scores based on keyword matches
        for dimension, keywords in self.DOMAIN_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            # Normalize to 0-10 scale, max score is 3
            scores[dimension] = min(10, int(score * 3.33))

        # Always have at least some general score
        scores["general"] = max(3, scores.get("general", 0))

        # Ensure we have values for all dimensions
        for dimension in DomainDimension.__annotations__.keys():
            if dimension not in scores:
                scores[dimension] = 0

        return scores

    def _analyze_complexity(
        self, prompt: str, system_message: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Analyze the complexity dimensions of a task

        Args:
            prompt: The task prompt
            system_message: Optional system context

        Returns:
            Dictionary of complexity dimension scores
        """
        token_count = self.token_counter(prompt)
        if system_message:
            token_count += self.token_counter(system_message)

        # Token requirement score
        if token_count < 50:
            tokens_required = 1
        elif token_count < 200:
            tokens_required = 3
        elif token_count < 500:
            tokens_required = 5
        elif token_count < 1000:
            tokens_required = 7
        else:
            tokens_required = 10

        # Context depth - check for complex reasoning indicators
        context_indicators = [
            "context",
            "background",
            "history",
            "previously",
            "before",
            "after",
            "remember",
            "recall",
            "given",
            "assuming",
            "based on",
            "according to",
        ]

        context_score = 0
        prompt_lower = prompt.lower()
        for indicator in context_indicators:
            if indicator in prompt_lower:
                context_score += 1

        context_depth = min(10, context_score * 2)

        # Specialized knowledge - check for domain-specific terminology
        specialized_indicators = [
            "specifically",
            "specialized",
            "expert",
            "technical",
            "advanced",
            "professional",
            "field",
            "domain",
            "discipline",
            "expertise",
            "jargon",
            "terminology",
            "framework",
            "theory",
            "concept",
        ]

        specialized_score = 0
        for indicator in specialized_indicators:
            if indicator in prompt_lower:
                specialized_score += 1

        specialized_knowledge = min(10, specialized_score * 2)

        return {
            "tokens_required": tokens_required,
            "context_depth": context_depth,
            "specialized_knowledge": specialized_knowledge,
        }

    def _analyze_actionability(self, text: str) -> Dict[str, int]:
        """
        Analyze the actionability dimensions of a task

        Args:
            text: The task text

        Returns:
            Dictionary of actionability dimension scores
        """
        text_lower = text.lower()

        # Tool usage indicators
        tool_indicators = [
            "use",
            "tool",
            "api",
            "search",
            "query",
            "lookup",
            "find",
            "calculate",
            "compute",
            "call",
            "fetch",
            "retrieve",
        ]

        tool_score = 0
        for indicator in tool_indicators:
            if indicator in text_lower:
                tool_score += 1

        tool_usage = min(10, tool_score * 2)

        # External data indicators
        data_indicators = [
            "data",
            "database",
            "information",
            "source",
            "reference",
            "website",
            "document",
            "file",
            "record",
            "report",
            "article",
            "paper",
        ]

        data_score = 0
        for indicator in data_indicators:
            if indicator in text_lower:
                data_score += 1

        external_data = min(10, data_score * 2)

        # Computation indicators
        computation_indicators = [
            "calculate",
            "compute",
            "solve",
            "math",
            "equation",
            "formula",
            "algorithm",
            "function",
            "process",
            "analyze",
            "evaluate",
            "assess",
        ]

        computation_score = 0
        for indicator in computation_indicators:
            if indicator in text_lower:
                computation_score += 1

        computation = min(10, computation_score * 2)

        return {
            "tool_usage": tool_usage,
            "external_data": external_data,
            "computation": computation,
        }

    def _analyze_urgency(self, text: str) -> Dict[str, int]:
        """
        Analyze the urgency dimensions of a task

        Args:
            text: The task text

        Returns:
            Dictionary of urgency dimension scores
        """
        text_lower = text.lower()

        # Time sensitivity indicators
        time_indicators = [
            "urgent",
            "quickly",
            "asap",
            "immediately",
            "as soon as possible",
            "by today",
            "by tomorrow",
            "deadline",
            "due",
            "fast",
            "rapid",
            "soon",
        ]

        time_score = 0
        for indicator in time_indicators:
            if indicator in text_lower:
                time_score += 2  # Higher weight for urgency

        time_sensitivity = min(10, time_score * 2)

        # Importance indicators
        importance_indicators = [
            "important",
            "critical",
            "crucial",
            "essential",
            "vital",
            "key",
            "significant",
            "major",
            "primary",
            "core",
            "fundamental",
            "necessary",
        ]

        importance_score = 0
        for indicator in importance_indicators:
            if indicator in text_lower:
                importance_score += 1

        importance = min(10, importance_score * 2)

        return {"time_sensitivity": time_sensitivity, "importance": importance}

    def _analyze_specificity(self, text: str) -> Dict[str, int]:
        """
        Analyze the specificity dimensions of a task

        Args:
            text: The task text

        Returns:
            Dictionary of specificity dimension scores
        """
        text_lower = text.lower()

        # Clarity indicators
        clarity_indicators = [
            "specific",
            "clear",
            "well-defined",
            "detailed",
            "precise",
            "exact",
            "explicit",
            "definite",
            "particular",
            "distinct",
            "focused",
        ]

        clarity_score = 0
        for indicator in clarity_indicators:
            if indicator in text_lower:
                clarity_score += 1

        # Increase clarity score based on detailed instructions
        if len(text) > 200:
            clarity_score += 1
        if len(text.split("\n")) > 3:  # Structured with newlines
            clarity_score += 1

        clarity = min(10, clarity_score * 2)

        # Constraints indicators
        constraint_indicators = [
            "constraint",
            "limit",
            "parameter",
            "boundary",
            "scope",
            "restriction",
            "guideline",
            "requirement",
            "must",
            "should",
            "only",
            "ensure",
            "make sure",
        ]

        constraint_score = 0
        for indicator in constraint_indicators:
            if indicator in text_lower:
                constraint_score += 1

        constraints = min(10, constraint_score * 2)

        # Examples provided indicators
        example_indicators = [
            "example",
            "instance",
            "case",
            "illustration",
            "demonstration",
            "sample",
            "like this",
            "such as",
            "for instance",
            "e.g.",
        ]

        example_score = 0
        for indicator in example_indicators:
            if indicator in text_lower:
                example_score += 1

        examples_provided = min(10, example_score * 3)

        return {
            "clarity": clarity,
            "constraints": constraints,
            "examples_provided": examples_provided,
        }

    def _analyze_multimodality(self, text: str) -> Dict[str, int]:
        """
        Analyze the multimodality dimensions of a task

        Args:
            text: The task text

        Returns:
            Dictionary of multimodality dimension scores
        """
        text_lower = text.lower()

        # Code required indicators
        code_indicators = [
            "code",
            "program",
            "script",
            "function",
            "class",
            "method",
            "api",
            "programming",
            "develop",
            "implement",
            "software",
            "algorithm",
            "python",
            "javascript",
            "java",
            "typescript",
            "c#",
            "c++",
            "go",
        ]

        code_score = 0
        for indicator in code_indicators:
            if indicator in text_lower:
                code_score += 1

        code_required = min(10, code_score * 2)

        # Visual required indicators
        visual_indicators = [
            "image",
            "picture",
            "graphic",
            "chart",
            "graph",
            "diagram",
            "visual",
            "illustration",
            "draw",
            "plot",
            "map",
            "sketch",
            "figure",
            "visualization",
        ]

        visual_score = 0
        for indicator in visual_indicators:
            if indicator in text_lower:
                visual_score += 1

        visual_required = min(10, visual_score * 2)

        # Structured data indicators
        data_indicators = [
            "data",
            "table",
            "csv",
            "excel",
            "spreadsheet",
            "database",
            "json",
            "xml",
            "structured",
            "format",
            "schema",
            "field",
            "record",
            "entry",
            "row",
            "column",
        ]

        data_score = 0
        for indicator in data_indicators:
            if indicator in text_lower:
                data_score += 1

        structured_data = min(10, data_score * 2)

        return {
            "code_required": code_required,
            "visual_required": visual_required,
            "structured_data": structured_data,
        }

    def _analyze_workflow_patterns(self, text: str) -> Dict[str, float]:
        """
        Analyze the text for common workflow patterns

        Args:
            text: The task text

        Returns:
            Dictionary mapping workflow types to confidence scores (0-1)
        """
        text_lower = text.lower()
        scores = {}

        # Calculate scores based on workflow pattern matches
        for workflow_type, patterns in self.WORKFLOW_KEYWORDS.items():
            matches = 0
            for pattern in patterns:
                if pattern in text_lower:
                    matches += 1

            if matches > 0:
                score = min(1.0, matches / len(patterns) * 2)
                scores[workflow_type] = score

                # Log workflow identification
                logger.debug(
                    f"Identified potential workflow: {workflow_type} (confidence: {score:.2f})"
                )

        return scores

    def determine_service_route(
        self,
        metadata: TaskMetadata,
        workflow_id: Optional[str] = None,
        workflow_phase: Optional[str] = None,
    ) -> str:
        """
        Determine the appropriate service to route this task to

        Args:
            metadata: Task metadata
            workflow_id: Optional workflow identifier
            workflow_phase: Optional phase in the workflow

        Returns:
            Service routing identifier
        """
        # If we have explicit workflow information, use it for routing
        if workflow_id:
            if workflow_id.startswith("code_gen"):
                return "code_generator"
            elif workflow_id.startswith("data_proc"):
                return "data_processor"
            elif workflow_id.startswith("content"):
                return "content_creator"

            # If we have a phase, add it to the route
            if workflow_phase:
                return f"{workflow_id}.{workflow_phase}"

            return workflow_id

        # Otherwise, infer routing from metadata
        # Code generation route
        if metadata.multimodality.code_required >= 7:
            return "code_generator"

        # Data processing route
        if metadata.multimodality.structured_data >= 7:
            return "data_processor"

        # Creative content route
        if metadata.intent.creative_generation >= 7:
            return "content_creator"

        # Information retrieval route
        if metadata.intent.information_retrieval >= 7:
            return "knowledge_retriever"

        # Conversation route
        if metadata.intent.conversation >= 7:
            return "conversation_agent"

        # Default route based on highest intent score
        highest_intent = max(
            ("information_retrieval", metadata.intent.information_retrieval),
            ("task_execution", metadata.intent.task_execution),
            ("creative_generation", metadata.intent.creative_generation),
            ("analysis", metadata.intent.analysis),
            ("conversation", metadata.intent.conversation),
            key=lambda x: x[1],
        )

        intent_map = {
            "information_retrieval": "knowledge_retriever",
            "task_execution": "task_executor",
            "creative_generation": "content_creator",
            "analysis": "data_analyzer",
            "conversation": "conversation_agent",
        }

        return intent_map.get(highest_intent[0], "general_agent")


class NeuralInterpreter:
    """
    Core Neural Interpreter that analyzes tasks and publishes them to Pulsar
    topics for autonomous agent selection.
    """

    def __init__(
        self,
        pulsar_url: str = "pulsar://localhost:6650",
        tasks_topic: str = "persistent://public/default/tasks",
        results_topic: str = "persistent://public/default/results",
        metrics_topic: str = "persistent://public/default/metrics",
        config: Dict[str, Any] = None,
    ):
        """
        Initialize the Neural Interpreter

        Args:
            pulsar_url: Pulsar service URL
            tasks_topic: Topic for publishing tasks
            results_topic: Topic for receiving task results
            metrics_topic: Topic for publishing system metrics
            config: Additional configuration parameters
        """
        self.pulsar_url = pulsar_url
        self.tasks_topic = tasks_topic
        self.results_topic = results_topic
        self.metrics_topic = metrics_topic
        self.config = config or {}

        # Initialize services
        self.metadata_analyzer = NeuralInterpretor()
        self.client = None
        self.producers = {}
        self.consumers = {}
        self._lock = asyncio.Lock()

        # Initialize task pool for concurrent processing
        self.task_pool = AsyncTaskManager()

        # Initialize validators
        self.prompt_validator = Validator()

        # Set up metrics
        metrics_collector.set_component_up(True)

        logger.info("Neural Interpreter initialized")

    async def initialize(self):
        """Initialize Pulsar connections"""
        try:
            # Create Pulsar client
            timer = metrics_collector.start_request_timer(strategy="pulsar_connection")
            self.client = pulsar.Client(self.pulsar_url)

            # Create producers
            self.producers["tasks"] = self.client.create_producer(
                topic=self.tasks_topic, block_if_queue_full=True, batching_enabled=True
            )

            self.producers["metrics"] = self.client.create_producer(
                topic=self.metrics_topic,
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=100,
            )

            # Create consumers
            self.consumers["results"] = self.client.subscribe(
                topic=self.results_topic,
                subscription_name="neural-interpreter-results",
                consumer_type=pulsar.ConsumerType.Shared,
            )

            logger.info(f"Initialized Pulsar connections to {self.pulsar_url}")
            metrics_collector.record_request(strategy="pulsar_connection")

            # Publish system startup metrics
            await self.publish_metrics(
                {
                    "event": "system_startup",
                    "component": "neural_interpreter",
                    "status": "initialized",
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize Pulsar client: {e}")
            metrics_collector.record_request(status="error", strategy="pulsar_connection")
            metrics_collector.record_error("pulsar_connection_error")

            # Clean up if initialization fails
            await self.close()
            raise

    async def close(self):
        """Close Pulsar connections"""
        try:
            # Publish shutdown metrics
            if "metrics" in self.producers:
                await self.publish_metrics(
                    {
                        "event": "system_shutdown",
                        "component": "neural_interpreter",
                        "status": "shutting_down",
                    }
                )

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

            # Shutdown async task manager
            await self.task_pool.shutdown()

            # Clear references
            self.producers = {}
            self.consumers = {}
            self.client = None

            # Update metrics
            metrics_collector.set_component_up(False)

            logger.info("Closed all Pulsar connections")
        except Exception as e:
            logger.error(f"Error closing Pulsar client: {e}")
            metrics_collector.record_error("pulsar_close_error")

    def determine_processing_mode(self, metadata: TaskMetadata) -> ProcessingMode:
        """
        Determine processing mode based on task metadata

        Args:
            metadata: Task metadata from analyzer

        Returns:
            Processing mode for the task
        """
        # Default processing mode thresholds
        complexity_threshold = self.config.get("complexity_threshold", 7)
        collaboration_threshold = self.config.get("collaboration_threshold", 8)

        # Get max complexity score
        max_complexity = max(
            metadata.complexity.tokens_required,
            metadata.complexity.context_depth,
            metadata.complexity.specialized_knowledge,
        )

        # Check if task requires collaboration
        if (
            max_complexity >= collaboration_threshold
            or metadata.actionability.tool_usage >= collaboration_threshold
            or metadata.actionability.external_data >= collaboration_threshold
        ):
            return ProcessingMode.COLLABORATIVE

        # Check if task is complex
        elif (
            max_complexity >= complexity_threshold
            or metadata.actionability.computation >= complexity_threshold
        ):
            return ProcessingMode.DELIBERATIVE

        # Default to reactive for simpler tasks
        else:
            return ProcessingMode.REACTIVE

    def estimate_processing_time(
        self, prompt: str, metadata: TaskMetadata, mode: ProcessingMode
    ) -> float:
        """
        Estimate task processing time in seconds

        Args:
            prompt: Task prompt
            metadata: Task metadata
            mode: Processing mode

        Returns:
            Estimated processing time in seconds
        """
        token_count = len(prompt.split())

        # Base processing times by mode
        if mode == ProcessingMode.REACTIVE:
            base_time = 1.0  # 1 second for simple tasks
        elif mode == ProcessingMode.DELIBERATIVE:
            base_time = 5.0  # 5 seconds for complex tasks
        else:  # COLLABORATIVE
            base_time = 15.0  # 15 seconds for collaborative tasks

        # Adjust for complexity
        complexity_factor = (
            1.0
            + (
                metadata.complexity.tokens_required
                + metadata.complexity.context_depth
                + metadata.complexity.specialized_knowledge
            )
            / 10.0
        )

        # Adjust for actionability
        action_factor = (
            1.0
            + (
                metadata.actionability.tool_usage
                + metadata.actionability.external_data
                + metadata.actionability.computation
            )
            / 15.0
        )

        # Calculate final estimate
        estimated_time = base_time * complexity_factor * action_factor

        # Add token-based processing time (assuming 10 tokens/second processing)
        token_time = token_count / 10.0

        return estimated_time + token_time

    def determine_priority(self, metadata: TaskMetadata) -> TaskPriority:
        """
        Determine task priority based on metadata

        Args:
            metadata: Task metadata

        Returns:
            Task priority level
        """
        # Check for critical priority indicators
        if metadata.urgency.time_sensitivity >= 9 and metadata.urgency.importance >= 8:
            return TaskPriority.CRITICAL

        # Check for high priority
        elif metadata.urgency.time_sensitivity >= 7 or metadata.urgency.importance >= 7:
            return TaskPriority.HIGH

        # Check for medium priority
        elif metadata.urgency.time_sensitivity >= 4 or metadata.urgency.importance >= 4:
            return TaskPriority.MEDIUM

        # Check for background tasks
        elif metadata.urgency.time_sensitivity <= 2 and metadata.urgency.importance <= 2:
            return TaskPriority.BACKGROUND

        # Default to low priority
        else:
            return TaskPriority.LOW

    def validate_request(
        self, prompt: str, system_message: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate request parameters

        Args:
            prompt: User prompt
            system_message: Optional system message

        Returns:
            Validation result
        """
        # Create validation result
        result = ValidationResult()

        # Validate prompt
        if not prompt or not isinstance(prompt, str):
            result.add_error("Prompt must be a non-empty string")

        # Validate system message if provided
        if system_message is not None and not isinstance(system_message, str):
            result.add_error("System message must be a string if provided")

        return result

    async def process_query(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Dict[str, Any] = None,
        require_collaboration: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a query, analyze it, and publish to appropriate topic

        Args:
            prompt: The query/task prompt
            system_message: Optional system context message
            session_id: Optional session identifier
            user_id: Optional user identifier
            context: Optional context data
            require_collaboration: Force collaborative processing

        Returns:
            Response with task ID and other tracking info
        """
        start_time = time.time()
        task_id = str(uuid.uuid4())
        context = context or {}

        # Start request timer for metrics
        timer = metrics_collector.start_request_timer()

        try:
            # 1. Validate input
            validation_result = self.validate_request(prompt, system_message)
            if not validation_result.valid:
                logger.error(f"Input validation failed: {validation_result.error_message}")
                metrics_collector.record_request(status="error", strategy="validation")
                metrics_collector.record_error("input_validation_error")
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": f"Input validation failed: {validation_result.error_message}",
                }

            # 2. Analyze task to generate metadata
            metadata = self.metadata_analyzer.analyze_task(prompt, system_message)

            # 3. Determine processing mode
            processing_mode = (
                ProcessingMode.COLLABORATIVE
                if require_collaboration
                else self.determine_processing_mode(metadata)
            )

            # 4. Estimate processing time
            estimated_time = self.estimate_processing_time(prompt, metadata, processing_mode)

            # 5. Determine priority
            priority = self.determine_priority(metadata)

            # 6. Determine service route
            workflow_id = context.get("workflow_id")
            workflow_phase = context.get("workflow_phase")
            service_route = self.metadata_analyzer.determine_service_route(
                metadata, workflow_id, workflow_phase
            )

            # For workflow tasks, add project ID
            project_id = context.get("project_id")

            # 7. Create task object
            # Use Task constructor directly - avoid issues with default Enum values
            task = Task(
                task_id=task_id,
                prompt=prompt,
                system_message=system_message,
                created_at=datetime.utcnow(),
                metadata=metadata.to_dict(),
                status=TaskStatus.PENDING,
                priority=priority,
                estimated_processing_time=estimated_time,
                session_id=session_id,
                context=context,
                user_id=user_id,
                service_route=service_route,
                workflow_id=workflow_id,
                phase=workflow_phase,
                project_id=project_id,
                processing_mode=processing_mode,  # Set this explicitly rather than relying on default
            )

            # 8. Publish task to Pulsar
            await self.publish_task(task)

            # 9. Publish metrics
            analysis_time = time.time() - start_time
            await self.publish_metrics(
                {
                    "event": "task_analyzed",
                    "task_id": task_id,
                    "analysis_time_ms": int(analysis_time * 1000),
                    "processing_mode": processing_mode.value,
                    "priority": priority.value,
                    "service_route": service_route,
                }
            )

            # Record completion
            metrics_collector.record_request(status="success", strategy="task_processing")

            # 10. Return response
            return {
                "task_id": task_id,
                "processing_mode": processing_mode.value,
                "priority": priority.value,
                "estimated_processing_time": estimated_time,
                "status": "submitted",
                "analysis_time_ms": int(analysis_time * 1000),
                "service_route": service_route,
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            # Record error
            metrics_collector.record_request(status="error", strategy="task_processing")
            metrics_collector.record_error("task_processing_error")

            # Publish error metrics
            await self.publish_metrics(
                {"event": "task_analysis_error", "task_id": task_id, "error": str(e)}
            )

            # Re-raise for proper error handling
            raise

    async def publish_task(self, task: Task):
        """
        Publish a task to Pulsar

        Args:
            task: Task to publish
        """
        if "tasks" not in self.producers:
            raise RuntimeError("Task producer not initialized")

        # Track publishing time
        timer = metrics_collector.start_request_timer(strategy="publish_task")

        try:
            # Convert task to JSON
            task_json = json.dumps(task.dict())

            # Publish to Pulsar
            self.producers["tasks"].send(
                task_json.encode("utf-8"),
                properties={
                    "task_id": task.task_id,
                    "priority": str(task.priority.value),
                    "processing_mode": task.processing_mode.value,
                    "service_route": task.service_route,
                },
            )

            logger.info(
                f"Published task {task.task_id} to {self.tasks_topic} [mode={task.processing_mode.value}, priority={task.priority.value}, route={task.service_route}]"
            )

            # Record publishing success
            metrics_collector.record_request(status="success", strategy="publish_task")

        except Exception as e:
            logger.error(f"Error publishing task: {e}", exc_info=True)
            metrics_collector.record_request(status="error", strategy="publish_task")
            metrics_collector.record_error("task_publish_error")
            raise

    async def publish_metrics(self, metrics: Dict[str, Any]):
        """
        Publish metrics to Pulsar metrics topic

        Args:
            metrics: Metrics data to publish
        """
        if "metrics" not in self.producers:
            logger.warning("Metrics producer not initialized")
            return

        try:
            # Add timestamp
            metrics["timestamp"] = datetime.utcnow().isoformat()
            metrics["component"] = "neural-interpreter"

            # Convert to JSON
            metrics_json = json.dumps(metrics)

            # Publish to Pulsar
            self.producers["metrics"].send(metrics_json.encode("utf-8"))

            # Record event emission
            metrics_collector.record_event_emitted(metrics.get("event", "metrics"))

        except Exception as e:
            logger.error(f"Error publishing metrics: {e}")
            metrics_collector.record_error("metrics_publish_error")

    async def start_result_consumer(self):
        """Start consuming task results"""
        if "results" not in self.consumers:
            raise RuntimeError("Results consumer not initialized")

        logger.info(f"Starting to consume results from {self.results_topic}")

        while True:
            try:
                # Receive message with timeout
                msg = self.consumers["results"].receive(timeout_millis=10000)
                try:
                    # Process message
                    timer = metrics_collector.start_request_timer(strategy="process_result")
                    payload = json.loads(msg.data().decode("utf-8"))
                    task_id = payload.get("task_id")

                    if task_id:
                        logger.info(f"Received result for task {task_id}")
                        metrics_collector.record_event_received("task_result")

                        # Process result (e.g., update system state)
                        await self.process_result(payload)

                        # Acknowledge message
                        self.consumers["results"].acknowledge(msg)
                        metrics_collector.record_request(
                            status="success", strategy="process_result"
                        )
                    else:
                        logger.warning("Received result without task_id")
                        metrics_collector.record_error("invalid_result_message")
                        self.consumers["results"].negative_acknowledge(msg)

                except Exception as e:
                    logger.error(f"Error processing result: {e}", exc_info=True)
                    metrics_collector.record_request(status="error", strategy="process_result")
                    metrics_collector.record_error("result_processing_error")
                    self.consumers["results"].negative_acknowledge(msg)

            except Exception as e:
                # Handle timeout or other errors
                if "timeout" not in str(e).lower():
                    logger.error(f"Error receiving results: {e}", exc_info=True)
                    metrics_collector.record_error("result_consumer_error")
                    await asyncio.sleep(1)  # Avoid tight loop if there's an error

    async def process_result(self, payload: Dict[str, Any]):
        """
        Process a task result received from an agent

        Args:
            payload: Result message payload
        """
        task_id = payload.get("task_id")
        status = payload.get("status", "unknown")
        result = payload.get("result")

        # Publish metrics for the result
        await self.publish_metrics(
            {
                "event": "task_result_received",
                "task_id": task_id,
                "status": status,
                "processing_time": payload.get("processing_time"),
            }
        )

        # For workflow tasks, trigger next phase if configured
        workflow_id = payload.get("workflow_id")
        if workflow_id and status == "completed":
            current_phase = payload.get("phase")
            next_phase = payload.get("next_phase")

            if next_phase:
                logger.info(f"Triggering next workflow phase: {workflow_id} -> {next_phase}")

                # Create a new task for the next phase
                context = payload.get("context", {})
                context.update(
                    {
                        "workflow_id": workflow_id,
                        "workflow_phase": next_phase,
                        "previous_phase": current_phase,
                        "previous_result": result,
                    }
                )

                # Process next task in workflow
                await self.process_query(
                    prompt=payload.get(
                        "next_prompt", f"Process workflow {workflow_id} phase {next_phase}"
                    ),
                    system_message=payload.get("next_system_message"),
                    session_id=payload.get("session_id"),
                    user_id=payload.get("user_id"),
                    context=context,
                )


async def run_interpreter():
    """Run the Neural Interpreter service"""
    interpreter = NeuralInterpreter()

    try:
        # Initialize interpreter
        await interpreter.initialize()

        # Start result consumer
        asyncio.create_task(interpreter.start_result_consumer())

        # Keep interpreter running
        while True:
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutting down Neural Interpreter...")
    finally:
        # Clean up
        await interpreter.close()


if __name__ == "__main__":
    # Run the interpreter
    asyncio.run(run_interpreter())
