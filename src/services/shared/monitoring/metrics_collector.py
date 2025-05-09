"""Prometheus metrics collector for the neural code generator system.

This module provides metrics collection for the neural code generator system,
with Prometheus integration for monitoring in a containerized environment.
"""

import logging

from prometheus_client.context_managers import Timer


try:
    import prometheus_client
    from prometheus_client import Counter
    from prometheus_client import Gauge
    from prometheus_client import Histogram
    from prometheus_client import Summary

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install with: pip install prometheus-client")

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Prometheus metrics collector for the neural code generator system.

    This class provides metrics collection and exposure via a Prometheus endpoint,
    suitable for containerized deployment with Kubernetes.
    """

    def __init__(
        self,
        component_name: str,
        metrics_port: int = 8081,
        metrics_endpoint: str = "/metrics",
        collect_process_metrics: bool = True,
        enabled: bool = True,
    ):
        """
        Initialize the metrics collector.

        Args:
            component_name: Name of the component (used as label)
            metrics_port: Port to expose metrics on
            metrics_endpoint: Endpoint for metrics
            collect_process_metrics: Whether to collect process metrics
            enabled: Whether metrics collection is enabled
        """
        self.component_name = component_name
        self.metrics_port = metrics_port
        self.metrics_endpoint = metrics_endpoint
        self.collect_process_metrics = collect_process_metrics
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        if not self.enabled:
            if not PROMETHEUS_AVAILABLE:
                logger.warning("Prometheus client not available. Metrics collection disabled.")
            else:
                logger.info("Metrics collection disabled.")
            return

        # Initialize Prometheus metrics
        self._init_metrics()

        # Start Prometheus HTTP server
        try:
            prometheus_client.start_http_server(self.metrics_port)
            logger.info(
                f"Prometheus metrics available at :{self.metrics_port}{self.metrics_endpoint}"
            )
        except Exception as e:
            logger.error(f"Failed to start Prometheus HTTP server: {e}")
            self.enabled = False

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Request metrics
        self.requests_total = Counter(
            "neural_code_generator_requests_total",
            "Total number of code generation requests",
            ["component", "status", "strategy"],
        )

        self.request_duration = Histogram(
            "neural_code_generator_request_duration_seconds",
            "Duration of code generation requests",
            ["component", "strategy"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
        )

        self.request_tokens = Histogram(
            "neural_code_generator_request_tokens",
            "Number of tokens in code generation requests",
            ["component", "token_type"],
            buckets=(100, 500, 1000, 2000, 5000, 10000, 15000),
        )

        # Result metrics
        self.result_confidence = Histogram(
            "neural_code_generator_result_confidence",
            "Confidence scores of code generation results",
            ["component", "strategy"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
        )

        self.generated_code_length = Histogram(
            "neural_code_generator_generated_code_length",
            "Length of generated code in characters",
            ["component", "language"],
            buckets=(100, 500, 1000, 2500, 5000, 10000, 25000, 50000),
        )

        # Cache metrics
        self.cache_hits = Counter(
            "neural_code_generator_cache_hits_total",
            "Total number of cache hits",
            ["component", "cache_type"],
        )

        self.cache_misses = Counter(
            "neural_code_generator_cache_misses_total",
            "Total number of cache misses",
            ["component", "cache_type"],
        )

        # Resource metrics
        self.gpu_memory_usage = Gauge(
            "neural_code_generator_gpu_memory_bytes",
            "GPU memory usage in bytes",
            ["component", "gpu_id"],
        )

        self.model_loading_time = Histogram(
            "neural_code_generator_model_loading_time_seconds",
            "Time taken to load models",
            ["component", "model_type"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
        )

        # Event system metrics
        self.events_emitted = Counter(
            "neural_code_generator_events_emitted_total",
            "Total number of events emitted",
            ["component", "event_type"],
        )

        self.events_received = Counter(
            "neural_code_generator_events_received_total",
            "Total number of events received",
            ["component", "event_type"],
        )

        self.event_processing_time = Histogram(
            "neural_code_generator_event_processing_time_seconds",
            "Time taken to process events",
            ["component", "event_type"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0),
        )

        # Health metrics
        self.component_up = Gauge(
            "neural_code_generator_component_up",
            "Whether the component is running properly",
            ["component"],
        )

        self.errors_total = Counter(
            "neural_code_generator_errors_total",
            "Total number of errors encountered",
            ["component", "error_type"],
        )

        # Vector database metrics
        self.vector_db_query_time = Histogram(
            "neural_code_generator_vector_db_query_time_seconds",
            "Time taken for vector database queries",
            ["component", "operation"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0),
        )

        self.vector_db_operations = Counter(
            "neural_code_generator_vector_db_operations_total",
            "Total number of vector database operations",
            ["component", "operation", "status"],
        )

        # Set component as up
        self.component_up.labels(component=self.component_name).set(1)

        # If enabled, collect process metrics
        if self.collect_process_metrics:
            prometheus_client.process_collector.ProcessCollector()

    # Request metrics methods
    def record_request(self, status: str = "success", strategy: str = "default"):
        """
        Record a code generation request.

        Args:
            status: Status of the request (success, failure)
            strategy: Strategy used for generation
        """
        if not self.enabled:
            return

        self.requests_total.labels(
            component=self.component_name, status=status, strategy=strategy
        ).inc()

    def start_request_timer(self, strategy: str = "default") -> Timer | None:
        """
        Start a timer for a code generation request.

        Args:
            strategy: Strategy used for generation

        Returns:
            Start time or None if metrics are disabled
        """
        if not self.enabled:
            return None

        return self.request_duration.labels(component=self.component_name, strategy=strategy).time()

    def record_tokens(self, token_type: str, count: int):
        """
        Record the number of tokens in a request or response.

        Args:
            token_type: Type of tokens (input, output)
            count: Number of tokens
        """
        if not self.enabled:
            return

        self.request_tokens.labels(component=self.component_name, token_type=token_type).observe(
            count
        )

    # Result metrics methods
    def record_confidence(self, confidence: float, strategy: str = "default"):
        """
        Record the confidence score of a generation result.

        Args:
            confidence: Confidence score (0-1)
            strategy: Strategy used for generation
        """
        if not self.enabled:
            return

        self.result_confidence.labels(component=self.component_name, strategy=strategy).observe(
            confidence
        )

    def record_code_length(self, length: int, language: str = "python"):
        """
        Record the length of generated code.

        Args:
            length: Length in characters
            language: Programming language
        """
        if not self.enabled:
            return

        self.generated_code_length.labels(component=self.component_name, language=language).observe(
            length
        )

    # Cache metrics methods
    def record_cache_hit(self, cache_type: str = "knowledge_base"):
        """
        Record a cache hit.

        Args:
            cache_type: Type of cache
        """
        if not self.enabled:
            return

        self.cache_hits.labels(component=self.component_name, cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str = "knowledge_base"):
        """
        Record a cache miss.

        Args:
            cache_type: Type of cache
        """
        if not self.enabled:
            return

        self.cache_misses.labels(component=self.component_name, cache_type=cache_type).inc()

    def record_component_status(self, status: str, value: int = 1):
        """
        Record a component status (e.g., started, stopped, initializing).

        This method creates a gauge metric to track the current state of a component,
        which is useful for monitoring service lifecycle events.

        Args:
            status: Status name (e.g., "started", "stopped", "initializing")
            value: Value to set (typically 1 for active, 0 for inactive)
        """
        if not self.enabled:
            return

        # Lazily create status gauges as needed
        status_gauge_name = f"component_status_{status}"

        # Check if we already have this gauge
        if not hasattr(self, status_gauge_name):
            # Create the gauge if it doesn't exist
            setattr(
                self,
                status_gauge_name,
                Gauge(
                    f"neural_code_generator_{status}", f"Component {status} status", ["component"]
                ),
            )

        # Get the gauge and set its value
        gauge = getattr(self, status_gauge_name)
        gauge.labels(component=self.component_name).set(value)

        # Log the status change
        logger.debug(f"Component {self.component_name} status '{status}' set to {value}")

    # Resource metrics methods
    def update_gpu_memory_usage(self, gpu_id: str, memory_bytes: float):
        """
        Update GPU memory usage.

        Args:
            gpu_id: GPU identifier
            memory_bytes: Memory usage in bytes
        """
        if not self.enabled:
            return

        self.gpu_memory_usage.labels(component=self.component_name, gpu_id=gpu_id).set(memory_bytes)

    def start_model_loading_timer(self, model_type: str) -> Timer | None:
        """
        Start a timer for model loading.

        Args:
            model_type: Type of model

        Returns:
            Start time or None if metrics are disabled
        """
        if not self.enabled:
            return None

        return self.model_loading_time.labels(
            component=self.component_name, model_type=model_type
        ).time()

    # Event system metrics methods
    def record_event_emitted(self, event_type: str):
        """
        Record an emitted event.

        Args:
            event_type: Type of event
        """
        if not self.enabled:
            return

        self.events_emitted.labels(component=self.component_name, event_type=event_type).inc()

    def record_event_received(self, event_type: str):
        """
        Record a received event.

        Args:
            event_type: Type of event
        """
        if not self.enabled:
            return

        self.events_received.labels(component=self.component_name, event_type=event_type).inc()

    def start_event_processing_timer(self, event_type: str) -> Timer | None:
        """
        Start a timer for event processing.

        Args:
            event_type: Type of event

        Returns:
            Start time or None if metrics are disabled
        """
        if not self.enabled:
            return None

        return self.event_processing_time.labels(
            component=self.component_name, event_type=event_type
        ).time()

    # Health metrics methods
    def set_component_up(self, up: bool = True):
        """
        Set whether the component is up.

        Args:
            up: Whether the component is up
        """
        if not self.enabled:
            return

        self.component_up.labels(component=self.component_name).set(1 if up else 0)

    def record_error(self, error_type: str):
        """
        Record an error.

        Args:
            error_type: Type of error
        """
        if not self.enabled:
            return

        self.errors_total.labels(component=self.component_name, error_type=error_type).inc()

    # Vector database metrics methods
    def start_vector_db_timer(self, operation: str) -> Timer | None:
        """
        Start a timer for a vector database operation.

        Args:
            operation: Type of operation (search, insert, delete)

        Returns:
            Start time or None if metrics are disabled
        """
        if not self.enabled:
            return None

        return self.vector_db_query_time.labels(
            component=self.component_name, operation=operation
        ).time()

    def record_vector_db_operation(self, operation: str, status: str = "success"):
        """
        Record a vector database operation.

        Args:
            operation: Type of operation (search, insert, delete)
            status: Status of the operation (success, failure)
        """
        if not self.enabled:
            return

        self.vector_db_operations.labels(
            component=self.component_name, operation=operation, status=status
        ).inc()
