# src/services/shared/models/constants.py
from pydantic import BaseModel, Field, ConfigDict


class DatabaseConfig(BaseModel):
    """Database configuration constants"""
    DEFAULT_VECTOR_DIMENSION: int = Field(1536, description="Default vector dimension for embeddings")
    DEFAULT_COLLECTION_NAME: str = Field("neural_code_embeddings",
                                         description="Default collection name for vector database")
    DEFAULT_VECTOR_INDEX_TYPE: str = Field("HNSW", description="Default vector index type")
    DEFAULT_VECTOR_METRIC_TYPE: str = Field("COSINE", description="Default vector metric type")
    DEFAULT_DATABASE_NAME: str = Field("neural_code_generator", description="Default database name")
    DEFAULT_CACHE_TTL: int = Field(3600, description="Default cache TTL in seconds")

    model_config = ConfigDict(frozen=True)


class Paths(BaseModel):
    """Constants for file and directory paths"""
    BASE_PATH: str = Field("build_with_ai", description="Base path for the system")
    SRC_PATH: str = Field("build_with_ai/src", description="Source code path")
    PULSAR_PATH: str = Field("build_with_ai/src/pulsar", description="Pulsar path")
    SERVICES_PATH: str = Field("build_with_ai/src/services", description="Services path")

    # Core services paths
    AGGREGATORS_PATH: str = Field("build_with_ai/src/services/aggregators", description="Aggregators path")
    API_GATEWAY_PATH: str = Field("build_with_ai/src/services/api_gateway", description="API Gateway path")
    AST_CODE_GENERATOR_PATH: str = Field("build_with_ai/src/services/ast_code_generator",
                                         description="AST code generator path")
    CONSTRAINT_RELAXER_PATH: str = Field("build_with_ai/src/services/constraint_relaxer",
                                         description="Constraint relaxer path")
    FEEDBACK_COLLECTOR_PATH: str = Field("build_with_ai/src/services/feedback_collector",
                                         description="Feedback collector path")
    GRAFANA_PATH: str = Field("build_with_ai/src/services/grafana", description="Grafana path")
    INCREMENTAL_SYNTHESIS_PATH: str = Field("build_with_ai/src/services/incremental_synthesis",
                                            description="Incremental synthesis path")
    KNOWLEDGE_BASE_PATH: str = Field("build_with_ai/src/services/knowledge_base", description="Knowledge base path")
    LANGUAGE_INTEROP_PATH: str = Field("build_with_ai/src/services/language_interop",
                                       description="Language interoperability path")
    LLM_VOLUME_PATH: str = Field("build_with_ai/src/services/llm_volume", description="LLM volume path")
    META_LEARNER_PATH: str = Field("build_with_ai/src/services/meta_learner", description="Meta learner path")
    NEURAL_CODE_GENERATOR_PATH: str = Field("build_with_ai/src/services/neural_code_generator",
                                            description="Neural code generator path")
    PROJECT_MANAGER_PATH: str = Field("build_with_ai/src/services/project_manager", description="Project manager path")
    PROMETHEUS_PATH: str = Field("build_with_ai/src/services/prometheus", description="Prometheus path")
    SHARED_PATH: str = Field("build_with_ai/src/services/shared", description="Shared modules path")
    SPEC_INFERENCE_PATH: str = Field("build_with_ai/src/services/spec_inference", description="Spec inference path")
    SPEC_REGISTRY_PATH: str = Field("build_with_ai/src/services/spec_registry", description="Spec registry path")
    SPECIFICATION_PARSER_PATH: str = Field("build_with_ai/src/services/specification_parser",
                                           description="Specification parser path")
    SYNTHESIS_ENGINE_PATH: str = Field("build_with_ai/src/services/synthesis_engine",
                                       description="Synthesis engine path")
    TEMPLATE_DISCOVERY_PATH: str = Field("build_with_ai/src/services/template_discovery",
                                         description="Template discovery path")
    TEMPLATE_LIB_VOLUME_PATH: str = Field("build_with_ai/src/services/template_lib_volume",
                                          description="Template library volume path")
    VERIFIER_PATH: str = Field("build_with_ai/src/services/verifier", description="Verifier path")
    VERSION_MANAGER_PATH: str = Field("build_with_ai/src/services/version_manager", description="Version manager path")
    WORKFLOW_ORCHESTRATOR_PATH: str = Field("build_with_ai/src/services/workflow_orchestrator",
                                            description="Workflow orchestrator path")
    WORKFLOW_REGISTRY_PATH: str = Field("build_with_ai/src/services/workflow_registry",
                                        description="Workflow registry path")

    # Configuration paths
    CONFIGS_PATH: str = Field("build_with_ai/src/services/configs", description="Configuration files path")

    # Model paths (absolute)
    DEFAULT_MODEL_PATH: str = Field("/Users/justinrussell/.models/deepseek-coder-6.7b-base",
                                    description="Default path for code generation model")
    DEFAULT_INFERENCE_MODEL_PATH: str = Field("/Users/justinrussell/.models/mistral-mlx",
                                              description="Path for neural interpreter model")

    model_config = ConfigDict(frozen=True)


class ModelConfig(BaseModel):
    """Constants for model configuration"""
    DEFAULT_MODEL_NAME: str = Field("DeepSeek-Coder 6.7B", description="Default model name")
    DEFAULT_MODEL_PATH: str = Field("/Users/justinrussell/.models/deepseek-coder-6.7b-base",
                                    description="Default model path")
    DEFAULT_INFERENCE_MODEL_NAME: str = Field("Mistral MLX", description="Default inference model name")
    DEFAULT_INFERENCE_MODEL_PATH: str = Field("/Users/justinrussell/.models/mistral-mlx",
                                              description="Path for neural interpreter model")
    DEFAULT_CONTEXT_LENGTH: int = Field(8192, description="Default context length")
    DEFAULT_TEMPERATURE: float = Field(0.2, description="Default temperature")
    DEFAULT_TOP_P: float = Field(0.95, description="Default top-p value")
    DEFAULT_TOP_K: int = Field(50, description="Default top-k value")
    DEFAULT_QUANTIZATION: str = Field("int8", description="Default quantization")
    DEFAULT_EMBEDDING_MODEL: str = Field("all-mpnet-base-v2", description="Default embedding model")

    model_config = ConfigDict(frozen=True)


class MetricsConfig(BaseModel):
    """Configuration for metrics collection"""
    DEFAULT_METRICS_PORT: int = Field(8081, description="Default metrics port")
    DEFAULT_METRICS_ENDPOINT: str = Field("/metrics", description="Default metrics endpoint")

    model_config = ConfigDict(frozen=True)


class DeploymentConfig(BaseModel):
    """Constants for deployment configuration"""
    DEFAULT_IMAGE: str = Field("program-synthesis/neural-code-generator:latest", description="Default container image")
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


class Constants(BaseModel):
    """Container for all system constants"""
    VERSION: str = Field("1.0.0", description="System version")

    model_config = ConfigDict(frozen=True)