"""
Enumeration types for system-wide constants and identifiers.

This module defines enumeration types that are used throughout the system
to ensure consistent naming and values for various concepts. These enums
help maintain type safety and improve code readability.

Classes:
    Components: Constants for component identification
    Events: Constants for event types and topics
    EventPriority: Event priority levels
    EventType: Standardized event types
    Database: Constants for database configuration
    DatabaseConfig: Database configuration constants
    Paths: Constants for file and directory paths
    Techniques: Constants for neural code generation techniques
    ModelConfig: Constants for model configuration
    Metrics: Constants for metrics collection
    MetricsConfig: Configuration for metrics collection
    DeploymentConfig: Constants for deployment configuration
    ErrorCodes: Constants for error and status codes
    Constants: Container for all system constants
    ProcessingMode: Processing modes for query handling
    TaskStatus: Status of a task in the system
    TaskPriority: Priority levels for tasks
    ProjectType: Project types
    ProjectStatus: Project status
    SynthesisStrategy: Types of synthesis strategies
    DisclosureLevel: Progressive disclosure levels for code synthesis
    HealthStatus: Health status constants
    VerificationResult: Result of verification process
"""

from enum import auto
from enum import Enum

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class Components(str, Enum):
    """Constants for component identification"""

    # Core services

    SPEC_REGISTRY = None
    NEURAL_INTERPRETOR = "neural_interpretor"
    PROJECT_MANAGER = "project_manager"
    NEURAL_CODE_GENERATOR = "neural_code_generator"
    ENHANCED_NEURAL_CODE_GENERATOR = "enhanced_neural_code_generator"
    AST_CODE_GENERATOR = "ast_code_generator"
    KNOWLEDGE_BASE = "knowledge_base"
    VECTOR_KNOWLEDGE_BASE = "vector_knowledge_base"

    # code generation helpers
    CONSTRAINT_RELAXER = "constraint_relaxer"

    # Events and messaging
    EVENT_EMITTER = "event_emitter"
    EVENT_LISTENER = "event_listener"

    # Database adapters
    DATABASE_ADAPTER = "database_adapter"
    VECTOR_SEARCH = "vector_search"
    CACHE = "cache"

    # Service services
    SERVICE_MODULE = "service_module"
    STANDALONE_MODULE = "standalone_module"
    HEALTH_CHECK = "health_check"
    METRICS_COLLECTOR = "metrics_collector"

    # Integration services
    SYNTHESIS_INTEGRATOR = "synthesis_integrator"
    RESPONSE_AGGREGATOR = "response_aggregator"

    # Verification services
    STATISTICAL_VERIFIER = "statistical_verifier"
    SYMBOLIC_EXECUTOR = "symbolic_executor"

    FEEDBACK_COLLECTOR = "feedback_collector"
    INCREMENTAL_SYNTHESIS = "incremental_synthesis"
    LANGUAGE_INTEROP = "language_interop"
    META_LEARNER = "meta_learner"
    SPEC_INFERENCE = "spec_inference"
    SYNTHESIS_ENGINE = "synthesis_engine"
    VERSION_MANAGER = "version_manager"


class Events(str, Enum):
    """Constants for event types and topics"""

    # Event types
    CODE_GENERATION_REQUESTED = "ast_code_generator.generation_requested"
    CODE_GENERATION_COMPLETED = "ast_code_generator.generation_completed"
    CODE_GENERATION_FAILED = "ast_code_generator.generation_failed"

    KNOWLEDGE_QUERY_REQUESTED = "knowledge.query_requested"
    KNOWLEDGE_QUERY_COMPLETED = "knowledge.query_completed"
    KNOWLEDGE_UPDATED = "knowledge.knowledge_updated"

    VERIFICATION_REQUESTED = "verification.verification_requested"
    VERIFICATION_COMPLETED = "verification.verification_completed"
    VERIFICATION_FAILED = "verification.verification_failed"

    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_ERROR = "system.error"
    SYSTEM_SHUTDOWN = "system.shutdown"

    # Topics
    TOPIC_CODE_GENERATOR = "code-generator"
    TOPIC_KNOWLEDGE = "knowledge"
    TOPIC_VERIFICATION = "verification"
    TOPIC_SYSTEM = "system"

    # Default topics
    DEFAULT_INPUT_TOPIC = "code-generation-requests"
    DEFAULT_OUTPUT_TOPIC = "code-generation-results"
    DEFAULT_SUBSCRIPTION_NAME = "code-generator-worker"


# Consolidated from base.py and events.py
class EventPriority(int, Enum):
    """Event priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventType(str, Enum):
    """Standardized event types for the system."""

    # Neural code generator events
    CODE_GENERATION_REQUESTED = "ast_code_generator.generation_requested"
    CODE_GENERATION_COMPLETED = "ast_code_generator.generation_completed"
    CODE_GENERATION_FAILED = "ast_code_generator.generation_failed"

    # Spec sheet events
    SPEC_SHEET_CREATED = "spec_sheet.created"
    SPEC_SHEET_UPDATED = "spec_sheet.updated"
    SPEC_SHEET_DELETED = "spec_sheet.deleted"
    SPEC_SHEET_PUBLISHED = "spec_sheet.published"
    SPEC_SHEET_DEPRECATED = "spec_sheet.deprecated"
    SPEC_SHEET_ARCHIVED = "spec_sheet.archived"

    # Spec sheet instance events
    SPEC_INSTANCE_CREATED = "spec_instance.created"
    SPEC_INSTANCE_UPDATED = "spec_instance.updated"
    SPEC_INSTANCE_COMPLETED = "spec_instance.completed"
    SPEC_INSTANCE_VALIDATED = "spec_instance.validated"
    SPEC_INSTANCE_DELETED = "spec_instance.deleted"

    # Spec sheet evolution events
    SPEC_SHEET_ANALYSIS_REQUESTED = "spec_sheet.analysis.requested"
    SPEC_SHEET_ANALYSIS_COMPLETED = "spec_sheet.analysis.completed"
    SPEC_SHEET_EVOLUTION_SUGGESTED = "spec_sheet.evolution.suggested"
    SPEC_SHEET_EVOLUTION_APPLIED = "spec_sheet.evolution.applied"

    # Knowledge base events
    KNOWLEDGE_QUERY_REQUESTED = "knowledge.query_requested"
    KNOWLEDGE_QUERY_COMPLETED = "knowledge.query_completed"
    KNOWLEDGE_UPDATED = "knowledge.knowledge_updated"

    # System events
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_ERROR = "system.error"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_INFO = "system.info"

    # Verification events
    VERIFICATION_REQUESTED = "verification.verification_requested"
    VERIFICATION_COMPLETED = "verification.verification_completed"
    VERIFICATION_FAILED = "verification.verification_failed"

    # Authentication events
    AUTH_TOKEN_ISSUED = "auth.token_issued"
    AUTH_TOKEN_REVOKED = "auth.token_revoked"
    AUTH_FAILED = "auth.auth_failed"


class Database(str, Enum):
    """Constants for database configuration"""

    # Vector database
    VECTOR_DB_TYPE_MILVUS = "milvus"
    VECTOR_DB_TYPE_QDRANT = "qdrant"
    VECTOR_DB_TYPE_FILE = "file"

    RELATIONAL_DB_TYPE_POSTGRESQL = "postgresql"
    RELATIONAL_DB_TYPE_SQLITE = "sqlite"

    # Cache
    CACHE_TYPE_REDIS = "redis"
    CACHE_TYPE_MEMORY = "memory"


class DatabaseConfig(BaseModel):
    """Database configuration constants"""

    DEFAULT_VECTOR_DIMENSION: int = Field(
        1536, description="Default vector dimension for embeddings"
    )
    DEFAULT_COLLECTION_NAME: str = Field(
        "neural_code_embeddings", description="Default collection name for vector database"
    )
    DEFAULT_VECTOR_INDEX_TYPE: str = Field("HNSW", description="Default vector index type")
    DEFAULT_VECTOR_METRIC_TYPE: str = Field("COSINE", description="Default vector metric type")
    DEFAULT_DATABASE_NAME: str = Field("neural_code_generator", description="Default database name")
    DEFAULT_CACHE_TTL: int = Field(3600, description="Default cache TTL in seconds")

    model_config = ConfigDict(frozen=True)


class Paths(BaseModel):
    """Constants for file and directory paths"""

    BASE_PATH: str = Field("program_synthesis_system", description="Base path for the system")
    COMPONENTS_PATH: str = Field(
        "program_synthesis_system/components", description="Path for system components"
    )
    SHARED_PATH: str = Field(
        "program_synthesis_system/shared", description="Path for shared modules"
    )
    CONFIGS_PATH: str = Field(
        "program_synthesis_system/configs", description="Path for configuration files"
    )

    NEURAL_CODE_GENERATOR_PATH: str = Field(
        "build_with_ai/src/neural_code_generator", description="Path for neural code generator"
    )
    KNOWLEDGE_BASE_PATH: str = Field(
        "build_with_ai/src/knowledge_base", description="Path for knowledge base"
    )
    AST_CODE_GENERATOR_PATH: str = Field(
        "build_with_ai/components/ast_code_generator", description="Path for AST code generator"
    )

    DEFAULT_MODEL_PATH: str = Field(
        "/app/models/deepseek-coder-6.7b-instruct", description="Default path for models"
    )
    DEFAULT_KNOWLEDGE_BASE_PATH: str = Field(
        "/app/knowledge_base", description="Default path for knowledge base"
    )

    model_config = ConfigDict(frozen=True)


class Techniques(str, Enum):
    """Constants for neural code generation techniques"""

    MULTI_HEAD_ATTENTION = "multi_head_attention"
    RETRIEVAL_AUGMENTATION = "retrieval_augmentation"
    TREE_TRANSFORMERS = "tree_transformers"
    HIERARCHICAL_GENERATION = "hierarchical_generation"
    SYNTAX_AWARE_SEARCH = "syntax_aware_search"
    HYBRID_GRAMMAR_NEURAL = "hybrid_grammar_neural"

    # Strategy names for logging and metrics
    STRATEGY_NEURAL = "neural"
    STRATEGY_STATISTICAL = "statistical"
    STRATEGY_HYBRID = "hybrid"
    STRATEGY_TREE_TRANSFORMER = "tree_transformer"
    STRATEGY_HIERARCHICAL = "hierarchical"
    STRATEGY_HYBRID_GRAMMAR_NEURAL = "hybrid_grammar_neural"
    STRATEGY_ATTENTION = "attention"


class ModelConfig(BaseModel):
    """Constants for model configuration"""

    DEFAULT_MODEL_NAME: str = Field("DeepSeek-Coder 8B", description="Default model name")
    DEFAULT_MODEL_PATH: str = Field(
        "/app/models/deepseek-coder-8b-instruct", description="Default model path"
    )
    DEFAULT_CONTEXT_LENGTH: int = Field(8192, description="Default context length")
    DEFAULT_TEMPERATURE: float = Field(0.2, description="Default temperature")
    DEFAULT_TOP_P: float = Field(0.95, description="Default top-p value")
    DEFAULT_TOP_K: int = Field(50, description="Default top-k value")
    DEFAULT_QUANTIZATION: str = Field("int8", description="Default quantization")
    DEFAULT_EMBEDDING_MODEL: str = Field("all-mpnet-base-v2", description="Default embedding model")

    model_config = ConfigDict(frozen=True)

    class SynthesisStrategy(str, Enum):
        """Types of synthesis strategies."""

        BOTTOM_UP = "bottom_up"
        TOP_DOWN = "top_down"
        ENUMERATIVE = "enumerative"
        DEDUCTIVE = "deductive"
        INDUCTIVE = "inductive"
        CONSTRAINT_BASED = "constraint_based"
        EXAMPLE_GUIDED = "example_guided"
        NEURAL_GUIDED = "neural_guided"

    class DisclosureLevel(Enum):
        """Progressive disclosure levels for code synthesis."""

        HIGH_LEVEL = auto()  # Only signatures and high-level descriptions
        MID_LEVEL = auto()  # Implementation with simplified details
        DETAILED = auto()  # Complete implementation with all details


class Metrics(str, Enum):
    """Constants for metrics collection"""

    # Metric names
    METRIC_REQUESTS_TOTAL = "neural_code_generator_requests_total"
    METRIC_REQUEST_DURATION = "neural_code_generator_request_duration_seconds"
    METRIC_REQUEST_TOKENS = "neural_code_generator_request_tokens"
    METRIC_RESULT_CONFIDENCE = "neural_code_generator_result_confidence"
    METRIC_GENERATED_CODE_LENGTH = "neural_code_generator_generated_code_length"
    METRIC_CACHE_HITS = "neural_code_generator_cache_hits_total"
    METRIC_CACHE_MISSES = "neural_code_generator_cache_misses_total"
    METRIC_GPU_MEMORY = "neural_code_generator_gpu_memory_bytes"
    METRIC_MODEL_LOADING_TIME = "neural_code_generator_model_loading_time_seconds"
    METRIC_EVENTS_EMITTED = "neural_code_generator_events_emitted_total"
    METRIC_EVENTS_RECEIVED = "neural_code_generator_events_received_total"
    METRIC_EVENT_PROCESSING_TIME = "neural_code_generator_event_processing_time_seconds"
    METRIC_COMPONENT_UP = "neural_code_generator_component_up"
    METRIC_ERRORS_TOTAL = "neural_code_generator_errors_total"
    METRIC_VECTOR_DB_QUERY_TIME = "neural_code_generator_vector_db_query_time_seconds"
    METRIC_VECTOR_DB_OPERATIONS = "neural_code_generator_vector_db_operations_total"


class MetricsConfig(BaseModel):
    """Configuration for metrics collection"""

    DEFAULT_METRICS_PORT: int = Field(8081, description="Default metrics port")
    DEFAULT_METRICS_ENDPOINT: str = Field("/metrics", description="Default metrics endpoint")

    model_config = ConfigDict(frozen=True)


class DeploymentConfig(BaseModel):
    """Constants for deployment configuration"""

    DEFAULT_IMAGE: str = Field(
        "program-synthesis/neural-code-generator:latest", description="Default container image"
    )
    DEFAULT_PORT: int = Field(8000, description="Default port")
    DEFAULT_CPU_REQUEST: str = Field("2", description="Default CPU request")
    DEFAULT_MEMORY_REQUEST: str = Field("16Gi", description="Default memory request")
    DEFAULT_GPU_REQUEST: str = Field("1", description="Default GPU request")
    DEFAULT_CPU_LIMIT: str = Field("4", description="Default CPU limit")
    DEFAULT_MEMORY_LIMIT: str = Field("24Gi", description="Default memory limit")
    DEFAULT_GPU_LIMIT: str = Field("1", description="Default GPU limit")
    DEFAULT_NAMESPACE: str = Field("program-synthesis", description="Default Kubernetes namespace")
    DEFAULT_REPLICAS: int = Field(1, description="Default number of replicas")
    READINESS_PATH: str = Field("/readiness", description="Path for readiness probe")
    LIVENESS_PATH: str = Field("/liveness", description="Path for liveness probe")
    STARTUP_PATH: str = Field("/startup", description="Path for startup probe")
    DEFAULT_UID: int = Field(1000, description="Default UID")
    DEFAULT_GID: int = Field(1000, description="Default GID")

    model_config = ConfigDict(frozen=True)


class ErrorCodes(int, Enum):
    """Constants for error and status codes"""

    SUCCESS = 0
    ERROR_GENERAL = 1
    ERROR_INVALID_INPUT = 2
    ERROR_MODEL_LOAD_FAILED = 3
    ERROR_GENERATION_FAILED = 4
    ERROR_VERIFICATION_FAILED = 5
    ERROR_DATABASE_CONNECTION = 6
    ERROR_EVENT_PROCESSING = 7
    ERROR_TIMEOUT = 8


class Constants(BaseModel):
    """Container for all system constants"""

    VERSION: str = Field("1.0.0", description="System version")

    model_config = ConfigDict(frozen=True)


class ProcessingMode(str, Enum):
    """Processing modes for query handling"""

    REACTIVE = "reactive"  # Fast path for pattern matching
    DELIBERATIVE = "deliberative"  # Thoughtful path for complex queries
    COLLABORATIVE = "collaborative"  # Requires multiple agents


class TaskStatus(str, Enum):
    """Status of a task in the system"""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(int, Enum):
    """Priority levels for tasks"""

    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    BACKGROUND = 0


class ProjectType(str, Enum):
    """Project types."""

    WEB_APP = "WEB_APP"
    MOBILE_APP = "MOBILE_APP"
    API_SERVICE = "API_SERVICE"
    LIBRARY = "LIBRARY"
    CLI_TOOL = "CLI_TOOL"


class ProjectStatus(str, Enum):
    """Project status."""

    INITIALIZING = "INITIALIZING"
    ANALYZING = "ANALYZING"
    SPEC_SHEETS_GENERATED = "SPEC_SHEETS_GENERATED"
    SPEC_SHEETS_COMPLETED = "SPEC_SHEETS_COMPLETED"
    GENERATING_CODE = "GENERATING_CODE"
    CODE_GENERATED = "CODE_GENERATED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SynthesisStrategy(str, Enum):
    """Types of synthesis strategies."""

    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    ENUMERATIVE = "enumerative"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    CONSTRAINT_BASED = "constraint_based"
    EXAMPLE_GUIDED = "example_guided"
    NEURAL_GUIDED = "neural_guided"


class DisclosureLevel(Enum):
    """Progressive disclosure levels for code synthesis."""

    HIGH_LEVEL = auto()  # Only signatures and high-level descriptions
    MID_LEVEL = auto()  # Implementation with simplified details
    DETAILED = auto()  # Complete implementation with all details


class HealthStatus(str, Enum):
    """Health status constants."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


class VerificationResult(str, Enum):
    """Result of verification process."""

    COUNTEREXAMPLE_FOUND = "counterexample_found"
    VERIFIED = "verified"
    FALSIFIED = "falsified"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"
