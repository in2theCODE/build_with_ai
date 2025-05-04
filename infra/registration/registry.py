from threading import Lock

from src.services.shared.logging.logger import get_logger
from src.services.shared.monitoring.circuit_breaker import CircuitBreaker
from src.services.shared.monitoring.health_monitor import HealthMonitor
from src.services.shared.monitoring.metrics_collector import MetricsCollector


class SingletonMeta(type):
    """
    Thread-safe implementation of the Singleton pattern using a metaclass.
    """
    _instances = {}
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class ServiceRegistry(metaclass=SingletonMeta):
    """
    Registry for microservices implementing the Singleton pattern.
    Provides centralized service discovery and management.
    """

    def __init__(self):
        """Initialize the service registry."""
        self.services = {}
        self.event_emitter = None
        self.metrics_collector = None
        self.logger = logger.get_logger("service_registry")
        self.health_monitors = {}
        self.circuit_breakers = {}
        self.pulsar_url = None

    def initialize(self, pulsar_url, metrics_port=8081):
        """Initialize the service registry with required dependencies."""
        self.pulsar_url = pulsar_url

        # Set up metrics collector
        self.metrics_collector = MetricsCollector(
            component_name="service_registry",
            metrics_port=metrics_port
        )

        # Set up event emitter
        self.event_emitter = SecureEventEmitter(
            service_url=pulsar_url,
            tenant="public",
            namespace="program-synthesis"
        )

        self.logger.info("Service registry initialized")
        return self

    async def register_service(self, service_name, service_instance, service_type, endpoints=None):
        """Register a service with the registry."""
        if service_name in self.services:
            self.logger.warning(f"Service {service_name} already registered. Updating...")

        service_info = {
            "instance": service_instance,
            "type": service_type,
            "endpoints": endpoints or [],
            "registered_at": time.time(),
            "health_status": "starting"
        }

        self.services[service_name] = service_info

        # Set up health monitoring for this service
        health_monitor = HealthMonitor(service_name, self.metrics_collector)
        self.health_monitors[service_name] = health_monitor

        # Set up circuit breaker for this service
        circuit_breaker = CircuitBreaker(
            name=service_name,
            failure_threshold=5,
            reset_timeout=30,
            metrics_collector=self.metrics_collector
        )
        self.circuit_breakers[service_name] = circuit_breaker

        # Emit event for service registration
        if self.event_emitter:
            await self.event_emitter.emit_async(
                BaseEvent(
                    event_type=EventType.SYSTEM_INFO,
                    source_container="service_registry",
                    payload={
                        "action": "service_registered",
                        "service_name": service_name,
                        "service_type": service_type.value
                    }
                )
            )

        self.logger.info(f"Service {service_name} registered successfully")
        return True

    def get_service(self, service_name):
        """Get a service by name with circuit breaker pattern applied."""
        if service_name not in self.services:
            self.logger.warning(f"Service {service_name} not found in registry")
            return None

        # Apply circuit breaker pattern
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker and circuit_breaker.is_open():
            self.logger.warning(f"Circuit breaker open for {service_name}, service unavailable")
            return None

        return self.services[service_name]["instance"]

    def get_services_by_type(self, service_type):
        """Get all services of a specific type."""
        return {
            name: info["instance"]
            for name, info in self.services.items()
            if info["type"] == service_type
        }