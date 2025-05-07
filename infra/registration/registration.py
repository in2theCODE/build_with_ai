#!/usr/bin/env python3
"""
Service Registry for the Program Synthesis System.

This module implements a thread-safe ServiceRegistry component that provides
centralized service discovery and management for the microservice's architecture.
It incorporates fault tolerance patterns like Circuit Breaker to prevent
cascading failures across the system.

Design Patterns:
- Singleton Pattern: Thread-safe singleton implementation for service registry
- Service Registry Pattern: Central service discovery mechanism
- Circuit Breaker Pattern: Fault tolerance for service calls
- Event-Driven Architecture: Asynchronous event communication
"""

import asyncio
import logging
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Type, TypeVar, cast, Coroutine, Tuple

# Import shared modules
from src.services.shared.logging.logger import configure_logging, get_logger, log_execution_time
from src.services.shared.models.base import BaseEvent
from src.services.shared.models.enums import Components as ComponentType
from src.services.shared.models.enums import EventType
from src.services.shared.monitoring.circuit_breaker import CircuitBreaker, CircuitState
from src.services.shared.monitoring.health_monitor import HealthMonitor
from src.services.shared.monitoring.metrics_collector import MetricsCollector
from src.services.shared.pulsar.client_factory import create_pulsar_client
from src.services.shared.pulsar.event_emitter import SecureEventEmitter

# Type variable for components
T = TypeVar("T")


class SingletonMeta(type):
    """
    Thread-safe implementation of the Singleton pattern using a metaclass.

    This ensures only one instance of ServiceRegistry exists across all threads.
    """

    _instances: Dict[type, Any] = {}
    _lock: Lock = Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class ServiceRegistry(metaclass=SingletonMeta):
    """
    Registry for microservices implementing the thread-safe Singleton pattern.

    This class provides a centralized repository of all running services,
    with support for service discovery, health monitoring, and fault tolerance.

    Note: This class uses asyncio for several operations. When using these methods,
    you must call them from an asyncio event loop.
    """

    def __init__(self) -> None:
        """Initialize the service registry."""
        # Set up logging first for proper traceability
        self._setup_logging()

        self.services: Dict[str, Dict[str, Any]] = {}
        self.event_emitter: Optional[SecureEventEmitter] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.health_monitors: Dict[str, HealthMonitor] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.pulsar_client: Any = None
        self.pulsar_url: Optional[str] = None
        self.initialized: bool = False

        # Asyncio loop related attributes
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.logger.info("Service registry instance created")

    def _setup_logging(self) -> None:
        """
        Set up logging for the service registry.

        This method configures comprehensive logging with proper formatting,
        handlers, and fallback mechanisms.
        """
        try:
            # Configure logging with default settings if not already configured
            log_config = {
                "level": logging.INFO,
                "console": True,
                "file": True,
                "use_colors": True,
                "directory": "logs",
                "collect_performance": True,
                "log_format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                "file_rotation": True,
                "max_size_mb": 10,
                "backup_count": 5,
                "multi_tenant": True,
            }

            # Use the configure_logging from shared module if available
            try:
                configure_logging(log_config)
            except Exception as e:
                # Fallback to basic logging if the shared module fails
                logging.basicConfig(
                    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
                )
                logging.warning(
                    f"Failed to configure logging using shared module: {e}. Using basic logging."
                )

            # Get the logger for this class
            self.logger = get_logger("service_registry")
            self.logger.info("Logging configured for service registry")
        except Exception as e:
            # Last resort fallback
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("service_registry")
            self.logger.error(f"Failed to set up logging properly: {e}")

    async def initialize(self, pulsar_url: str, metrics_port: int = 8081) -> "ServiceRegistry":
        """
        Initialize the service registry with required dependencies.

        This is an asynchronous method and must be awaited.

        Args:
            pulsar_url: URL for the Pulsar service
            metrics_port: Port for the metrics HTTP server

        Returns:
            The initialized service registry instance

        Raises:
            RuntimeError: If initialization fails
        """
        if self.initialized:
            self.logger.info("Service registry already initialized")
            return self

        self.pulsar_url = pulsar_url
        self.logger.info(f"Initializing service registry with Pulsar URL: {pulsar_url}")

        # Store the current event loop
        self._loop = asyncio.get_running_loop()

        try:
            # Set up metrics collector
            self.logger.debug("Setting up metrics collector")
            self.metrics_collector = MetricsCollector(
                component_name="service_registry", metrics_port=metrics_port
            )

            # Create Pulsar client
            self.logger.debug("Creating Pulsar client")
            self.pulsar_client = create_pulsar_client(service_url=pulsar_url)

            # Set up event emitter
            self.logger.debug("Setting up event emitter")
            self.event_emitter = SecureEventEmitter(
                service_url=pulsar_url, namespace="program-synthesis"
            )

            self.initialized = True
            self.logger.info("Service registry initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize service registry: {e}", exc_info=True)
            raise RuntimeError(f"Service registry initialization failed: {e}") from e

        return self

    @log_execution_time
    async def register_service(
        self,
        service_name: str,
        service_instance: Any,
        service_type: ComponentType,
        endpoints: Optional[List[str]] = None,
    ) -> bool:
        """
        Register a service with the registry.

        This is an asynchronous method and must be awaited.

        Args:
            service_name: Unique name of the service
            service_instance: Instance of the service
            service_type: Type of the service component
            endpoints: Optional list of service endpoints

        Returns:
            True if registration succeeded, False otherwise
        """
        if not self.initialized:
            self.logger.error("Cannot register service: registry not initialized")
            return False

        try:
            self.logger.info(f"Registering service: {service_name} of type {service_type}")

            if service_name in self.services:
                self.logger.warning(f"Service {service_name} already registered. Updating...")

            service_info = {
                "instance": service_instance,
                "type": service_type,
                "endpoints": endpoints or [],
                "registered_at": time.time(),
                "health_status": "starting",
            }

            self.services[service_name] = service_info

            # Set up health monitoring for this service
            try:
                self.logger.debug(f"Setting up health monitor for {service_name}")
                health_monitor = HealthMonitor(
                    service_name=service_name, metrics_collector=self.metrics_collector
                )
                self.health_monitors[service_name] = health_monitor
            except Exception as e:
                self.logger.error(
                    f"Failed to create health monitor for {service_name}: {e}", exc_info=True
                )

            # Set up circuit breaker for this service
            try:
                self.logger.debug(f"Setting up circuit breaker for {service_name}")
                circuit_breaker = CircuitBreaker(
                    name=service_name, metrics_collector=self.metrics_collector
                )
                self.circuit_breakers[service_name] = circuit_breaker
            except Exception as e:
                self.logger.error(
                    f"Failed to create circuit breaker for {service_name}: {e}", exc_info=True
                )

            # Emit event for service registration
            if self.event_emitter:
                try:
                    self.logger.debug(f"Emitting registration event for {service_name}")
                    event = BaseEvent(
                        event_type=EventType.SYSTEM_INFO,
                        source_container="service_registry",
                        payload={
                            "action": "service_registered",
                            "service_name": service_name,
                            "service_type": service_type.value,
                        },
                    )

                    # Use the async version
                    await self.event_emitter.emit_async(event)
                except Exception as e:
                    self.logger.error(
                        f"Failed to emit service registration event: {e}", exc_info=True
                    )

            self.logger.info(f"Service {service_name} registered successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register service {service_name}: {e}", exc_info=True)
            return False

    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get a service by name with circuit breaker pattern applied.

        This is a synchronous method that can be called from any context.

        Args:
            service_name: Name of the service to retrieve

        Returns:
            Service instance if available, None otherwise
        """
        if not self.initialized:
            self.logger.error("Cannot get service: registry not initialized")
            return None

        if service_name not in self.services:
            self.logger.warning(f"Service {service_name} not found in registry")
            return None

        # Apply circuit breaker pattern - synchronous check
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker and circuit_breaker.is_open():
            self.logger.warning(f"Circuit breaker open for {service_name}, service unavailable")
            return None

        self.logger.debug(f"Retrieved service: {service_name}")
        return self.services[service_name]["instance"]

    def get_service_with_type(self, service_name: str, expected_type: Type[T]) -> Optional[T]:
        """
        Get a service by name with expected type.

        This is a synchronous method that can be called from any context.

        Args:
            service_name: Name of the service to retrieve
            expected_type: Expected type of the service

        Returns:
            Service instance cast to expected type if available, None otherwise
        """
        service = self.get_service(service_name)
        if service is None:
            return None

        if not isinstance(service, expected_type):
            self.logger.warning(
                f"Service {service_name} is not of expected type {expected_type.__name__}"
            )
            return None

        return cast(T, service)

    def get_services_by_type(self, service_type: ComponentType) -> Dict[str, Any]:
        """
        Get all services of a specific type.

        This is a synchronous method that can be called from any context.

        Args:
            service_type: Type of services to retrieve

        Returns:
            Dictionary mapping service names to instances
        """
        if not self.initialized:
            self.logger.error("Cannot get services: registry not initialized")
            return {}

        result = {
            name: info["instance"]
            for name, info in self.services.items()
            if info["type"] == service_type
        }

        self.logger.debug(f"Found {len(result)} services of type {service_type}")
        return result

    async def health_check(self) -> Dict[str, str]:
        """
        Perform health check on all registered services.

        This is an asynchronous method and must be awaited.

        Returns:
            Dictionary mapping service names to health status
        """
        if not self.initialized:
            self.logger.error("Cannot perform health check: registry not initialized")
            return {}

        self.logger.info(f"Performing health check on {len(self.health_monitors)} services")
        results = {}

        # Create tasks for all health checks to run in parallel
        health_check_tasks = []
        for name, monitor in self.health_monitors.items():
            task = self._check_service_health(name, monitor)
            health_check_tasks.append(task)

        # Wait for all health checks to complete
        if health_check_tasks:
            completed_results = await asyncio.gather(*health_check_tasks, return_exceptions=True)

            # Process results
            for result in completed_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Health check task failed: {result}", exc_info=True)
                    continue

                name, status = result
                results[name] = status
                if name in self.services:
                    self.services[name]["health_status"] = status

        return results

    async def _check_service_health(
        self, service_name: str, monitor: HealthMonitor
    ) -> Tuple[str, str]:
        """
        Check health for a single service.

        Args:
            service_name: Name of the service
            monitor: Health monitor for the service

        Returns:
            Tuple of (service_name, status)
        """
        try:
            self.logger.debug(f"Checking health for service: {service_name}")
            status = await monitor.check_health()
            self.logger.debug(f"Health status for {service_name}: {status}")
            return service_name, status
        except Exception as e:
            self.logger.error(f"Health check failed for {service_name}: {e}", exc_info=True)
            return service_name, "error"

    def get_service_health(self, service_name: str) -> Optional[str]:
        """
        Get the current health status of a specific service.

        This is a synchronous method that returns the last known health status.
        For a real-time health check, use health_check() instead.

        Args:
            service_name: Name of the service

        Returns:
            Health status string if available, None otherwise
        """
        if not self.initialized or service_name not in self.services:
            return None

        status = self.services[service_name].get("health_status")
        self.logger.debug(f"Retrieved health status for {service_name}: {status}")
        return status

    def get_service_metadata(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a registered service.

        This is a synchronous method that can be called from any context.

        Args:
            service_name: Name of the service

        Returns:
            Dictionary with service metadata if available, None otherwise
        """
        if not self.initialized or service_name not in self.services:
            return None

        service_info = self.services[service_name].copy()
        # Don't expose the actual instance
        service_info.pop("instance", None)

        # Add circuit breaker status if available
        if service_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[service_name]
            service_info["circuit_breaker"] = circuit_breaker.get_metrics()

        self.logger.debug(f"Retrieved metadata for service: {service_name}")
        return service_info

    async def reset_circuit_breaker(self, service_name: str) -> bool:
        """
        Manually reset a service's circuit breaker.

        This is an asynchronous method and must be awaited.

        Args:
            service_name: Name of the service

        Returns:
            True if reset was successful, False otherwise
        """
        if not self.initialized or service_name not in self.circuit_breakers:
            return False

        try:
            self.logger.info(f"Manually resetting circuit breaker for {service_name}")
            breaker = self.circuit_breakers[service_name]

            if breaker.get_state() == CircuitState.OPEN:
                # Access the state transition method properly
                # We need to call the breaker's internal method for this
                if hasattr(breaker, "_state_transition"):
                    # The proper way to access a protected method when necessary
                    await getattr(breaker, "_state_transition")(CircuitState.HALF_OPEN)
                    # Reset the counter for half-open requests
                    if hasattr(breaker, "half_open_requests"):
                        setattr(breaker, "half_open_requests", 0)

                    self.logger.info(f"Successfully reset circuit breaker for {service_name}")
                    return True
        except Exception as e:
            self.logger.error(
                f"Failed to reset circuit breaker for {service_name}: {e}", exc_info=True
            )

        return False

    def list_services(self) -> List[Dict[str, Any]]:
        """
        Get a list of all registered services with their status.

        This is a synchronous method that can be called from any context.

        Returns:
            List of service information dictionaries
        """
        if not self.initialized:
            return []

        self.logger.debug(f"Listing {len(self.services)} registered services")
        services_list = []
        for name, info in self.services.items():
            service_info = {
                "name": name,
                "type": (
                    str(info["type"].value) if hasattr(info["type"], "value") else str(info["type"])
                ),
                "health_status": info.get("health_status", "unknown"),
                "registered_at": info.get("registered_at", 0),
                "circuit_state": "unknown",
            }

            # Add circuit breaker status if available
            if name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers.get(name)
                if circuit_breaker:
                    service_info["circuit_state"] = circuit_breaker.get_state().value

            services_list.append(service_info)

        return services_list

    async def cleanup(self) -> None:
        """
        Clean up resources associated with the service registry.

        This is an asynchronous method and should be awaited.
        """
        if not self.initialized:
            return

        self.logger.info("Cleaning up service registry resources")

        # Close Pulsar client
        if self.pulsar_client:
            try:
                self.logger.debug("Closing Pulsar client")
                self.pulsar_client.close()
            except Exception as e:
                self.logger.error(f"Error closing Pulsar client: {e}", exc_info=True)

        # Close event emitter
        if self.event_emitter:
            try:
                self.logger.debug("Closing event emitter")
                self.event_emitter.close()
            except Exception as e:
                self.logger.error(f"Error closing event emitter: {e}", exc_info=True)

        # Clean up metrics collector
        if self.metrics_collector and hasattr(self.metrics_collector, "shutdown"):
            try:
                self.logger.debug("Shutting down metrics collector")
                self.metrics_collector.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down metrics collector: {e}", exc_info=True)

        # Mark as uninitialized
        self.initialized = False
        self.logger.info("Service registry cleaned up successfully")

    @classmethod
    def run_async(cls, coro: Coroutine) -> Any:
        """
        Run an asynchronous coroutine in a synchronous context.

        This is a helper method to bridge between sync and async code.

        Args:
            coro: Coroutine to run

        Returns:
            Result of the coroutine

        Raises:
            Exception: If the coroutine raises an exception
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coro)
