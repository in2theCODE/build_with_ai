#!/usr/bin/env python3
"""
Registration module for the Program Synthesis System.

This script registers components from the project structure into the component factory
and sets up the unified shared folder for communication between services.
It implements a robust service discovery mechanism using Apache Pulsar for event-driven
architecture following microservices best practices.

Design Patterns:
- Factory Pattern: Dynamic component registration and instantiation
- Service Registry Pattern: Central service discovery
- Event-Driven Architecture: Async communication via Apache Pulsar
- Circuit Breaker & Bulkhead Patterns: Fault tolerance and failure isolation
"""

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Type, Any, Optional

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.services.shared.models.enums import Components as ComponentType
from src.services.shared.models.events.events import EventType, BaseEvent
from infra.component_factory import ComponentFactory
from src.services.shared.logging.logger import (
    get_logger, configure_logging, log_execution_time
)
from src.services.shared.pulsar.event_emitter import SecureEventEmitter
from src.services.shared.monitoring.metrics import MetricsCollector
from src.services.shared.monitoring.circuit_breaker import CircuitBreaker
from src.services.shared.monitoring.health_monitor import HealthMonitor
from src.services.api_gateway.neural_interpretor.app.nueral_interpretor import NeuralInterpretor

# Import all component classes for registration
# Core components based on the directory structure
from src.services.neural_code_generator.app.enhanced_neural_code_generator import EnhancedNeuralCodeGenerator
from src.services.constraint_relaxer.app.constraint_relaxer import ModelBasedConstraintRelaxer
from src.services.feedback_collector.feedback_collector import FeedbackCollector
from src.services.incremental_synthesis.incremental_synthesis import IncrementalSynthesis
from src.services.language_interop.language_interop import LanguageInterop
from src.services.meta_learner.meta_learner import MetaLearner
from src.services.spec_inference.spec_inference import SpecInference
from src.services.synthesis_engine.synthesis_engine import SpecBasedSynthesisEngine
from src.services.version_manager.version_manager import VersionManager
from src.services.knowledge_base.vector_knowledge_base import VectorKnowledgeBase
from src.services.ast_code_generator.app.ast_code_generator import ASTCodeGenerator
from src.services.project_manager.app.project_manager import ProjectManager

# Define component mapping to simplify registration
COMPONENT_MAPPING: Dict[ComponentType, Dict[str, Type[Any]]] = {
    ComponentType.ENHANCED_NEURAL_CODE_GENERATOR: {
        "enhanced_neural_code_generator": EnhancedNeuralCodeGenerator
    },
    ComponentType.CONSTRAINT_RELAXER: {
        "constraint_relaxer": ModelBasedConstraintRelaxer
    },
    ComponentType.FEEDBACK_COLLECTOR: {
        "feedback_collector": FeedbackCollector
    },
    ComponentType.INCREMENTAL_SYNTHESIS: {
        "incremental_synthesis": IncrementalSynthesis
    },
    ComponentType.LANGUAGE_INTEROP: {
        "language_interop": LanguageInterop
    },
    ComponentType.META_LEARNER: {
        "meta_learner": MetaLearner
    },
    ComponentType.SPEC_INFERENCE: {
        "spec_inference": SpecInference
    },
    ComponentType.SYNTHESIS_ENGINE: {
        "synthesis_engine": SpecBasedSynthesisEngine
    },
    ComponentType.VERSION_MANAGER: {
        "version_manager": VersionManager
    },
    ComponentType.KNOWLEDGE_BASE: {
        "vector_knowledge_base": VectorKnowledgeBase
    },
    ComponentType.AST_CODE_GENERATOR: {
        "ast_code_generator": ASTCodeGenerator
    },
    ComponentType.PROJECT_MANAGER: {
        "project_manager": ProjectManager
    }
    ComponentType.NEURAL_INTERPRETOR: {
        "neural_interpretor": NeuralInterpretor
    }
}


class SharedFolderManager:
    """
    Manages the unified shared folder for component communication.

    This class handles the creation, mounting, and permissions management
    for a unified shared folder accessible to all microservices in the system.
    It enables efficient data sharing while maintaining isolation boundaries.
    """

    def __init__(self, base_path: str = "shared"):
        """Initialize the shared folder manager.

        Args:
            base_path: The base path for the shared folder, relative to project root
        """
        self.logger = get_logger("shared_folder_manager")
        self.base_path = Path(project_root) / base_path
        self.shared_dirs = [
            "models",
            "pulsar",
            "monitoring",
            "logging",
            "concurrency",
            "validation"
        ]
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in self.shared_dirs:
            dir_path = self.base_path / directory
            if not dir_path.exists():
                self.logger.info(f"Creating shared directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)

    def create_symlinks(self, services_dir: Path) -> None:
        """
        Create symbolic links to the shared folder from all service directories.

        Args:
            services_dir: Path to the services directory
        """
        # Get all service directories (exclude shared)
        service_paths = [
            p for p in services_dir.iterdir()
            if p.is_dir() and p.name != "shared"
        ]

        for service_path in service_paths:
            target_shared = service_path / "shared"

            # Skip if it's already a symlink to our shared folder
            if target_shared.is_symlink() and os.path.realpath(target_shared) == str(self.base_path):
                self.logger.debug(f"Symlink already exists for {service_path.name}")
                continue

            # Remove existing directory or symlink if it exists
            if target_shared.exists():
                if target_shared.is_symlink():
                    target_shared.unlink()
                else:
                    import shutil
                    shutil.rmtree(target_shared)

            # Create the symlink
            self.logger.info(f"Creating symlink for {service_path.name} to shared folder")
            os.symlink(str(self.base_path), str(target_shared), target_is_directory=True)

    def collect_shared_imports(self) -> None:
        """
        Collect and standardize imports across shared modules.
        Creates __init__.py files with proper imports for shared modules.
        """
        # For each shared directory, create an __init__.py that imports all modules
        for dir_name in self.shared_dirs:
            dir_path = self.base_path / dir_name
            init_file = dir_path / "__init__.py"

            # Collect Python modules in this directory
            modules = [
                p.stem for p in dir_path.glob("*.py")
                if p.is_file() and p.name != "__init__.py"
            ]

            # Create or update __init__.py
            with open(init_file, "w") as f:
                f.write(f'"""\n{dir_name.capitalize()} module shared across all services.\n"""\n\n')

                # Import all modules with proper relative imports
                for module in sorted(modules):
                    f.write(f"from .{module} import *\n")

            self.logger.info(f"Created unified imports for {dir_name}")

    def setup(self, services_dir: Path) -> None:
        """
        Complete setup for the shared folder system.

        Args:
            services_dir: Path to the services directory
        """
        self._ensure_directories()
        self.collect_shared_imports()
        self.create_symlinks(services_dir)
        self.logger.info("Shared folder system setup complete")


class ServiceRegistry:
    """
    Registry for microservices implementing the Singleton pattern.
    Tracks all active services and facilitates service discovery.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
            cls._instance.services = {}
            cls._instance.event_emitter = "secure_event_emmiter"
            cls._instance.metrics_collector = None
            cls._instance.logger = get_logger("service_registry")
            cls._instance.health_monitors: Dict[str, HealthMonitor] = {}
            cls._instance.circuit_breakers: Dict[str, CircuitBreaker] = {}
        return cls._instance

    def initialize(self, pulsar_url: str, metrics_port: int = 8081):
        """Initialize the service registry with required dependencies."""
        self.pulsar_url = pulsar_url

        # Set up metrics collector (Metrics Pattern)
        self.metrics_collector = MetricsCollector(
            component_name="service_registry",
            metrics_port=metrics_port
        )

        # Set up event emitter (Observer Pattern)
        self.event_emitter = SecureEventEmitter(
            service_url=pulsar_url,
            tenant="public",
            namespace="program-synthesis"
        )

        self.logger.info("Service registry initialized")
        return self

    async def register_service(self, service_name: str, service_instance: Any,
                               service_type: ComponentType, endpoints: List[str] = None):
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

        # Set up health monitoring for this service (Health Check Pattern)
        health_monitor = HealthMonitor(service_name, self.metrics_collector)
        self.health_monitors[service_name] = health_monitor

        # Set up circuit breaker for this service (Circuit Breaker Pattern)
        circuit_breaker = CircuitBreaker(
            name=service_name,
            failure_threshold=5,
            reset_timeout=30,
            metrics_collector=self.metrics_collector
        )
        self.circuit_breakers[service_name] = circuit_breaker

        # Emit event for service registration (Observer Pattern)
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

    def get_service(self, service_name: str) -> Optional[Any]:
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

    def get_services_by_type(self, service_type: ComponentType) -> Dict[str, Any]:
        """Get all services of a specific type."""
        return {
            name: info["instance"]
            for name, info in self.services.items()
            if info["type"] == service_type
        }

    async def health_check(self) -> Dict[str, str]:
        """Perform health check on all registered services."""
        results = {}
        for name, monitor in self.health_monitors.items():
            status = await monitor.check_health()
            results[name] = status

            # Update service status
            if name in self.services:
                self.services[name]["health_status"] = status

        return results


@log_execution_time
async def register_services(
        pulsar_url: str = "pulsar://localhost:6650",
        setup_shared_folder: bool = True
) -> ServiceRegistry:
    """
    Register all services based on the project folder structure.

    Args:
        pulsar_url: URL for the Pulsar service
        setup_shared_folder: Whether to set up the unified shared folder

    Returns:
        The initialized service registry
    """
    logger = get_logger("service_registration")
    logger.info("Starting service registration")

    # Initialize service registry
    registry = ServiceRegistry().initialize(pulsar_url=pulsar_url)

    # Set up the shared folder if requested
    services_dir = Path(project_root) / "src" / "services"
    if setup_shared_folder:
        shared_manager = SharedFolderManager(base_path="src/services/shared")
        shared_manager.setup(services_dir)
        logger.info("Shared folder setup complete")

    # Initialize metrics for the registration process
    metrics = MetricsCollector(
        component_name="service_registration",
        metrics_port=8082
    )

    # Create component factory
    factory = ComponentFactory()

    # Register services from component mapping
    registered_count = 0
    for component_type, implementations in COMPONENT_MAPPING.items():
        for name, implementation_class in implementations.items():
            # Register with component factory
            factory.register_component(component_type, name, implementation_class)

            try:
                # Apply bulkhead pattern by isolating component initialization
                with metrics.start_request_timer(f"instantiate_{name}"):
                    component_instance = implementation_class(
                        component_name=name,
                        metrics_collector=metrics,  # Removed create_child method call
                        event_emitter=registry.event_emitter
                    )

                    # Register with service registry
                    await registry.register_service(
                        service_name=name,
                        service_instance=component_instance,
                        service_type=component_type
                    )

                    registered_count += 1  # Fixed 'a' to '1'
                    logger.info(f"Registered and initialized {name} as {component_type.value}")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {str(e)}")
                metrics.record_error(f"{name}_initialization_error")

    # Set up Template Registry Adapter
    try:
        spec_registry_adapter = SpecRegistryEventAdapter(
            pulsar_service_url=pulsar_url,
            enable_events=True
        )
        await spec_registry_adapter.start()
        await registry.register_service(
            "spec_registry_adapter",
            spec_registry_adapter,
            ComponentType.SYNTHESIS_ENGINE
        )
        logger.info("Template Registry Adapter initialized and registered")
    except Exception as e:
        logger.error(f"Failed to initialize Template Registry Adapter: {str(e)}")

    # Register the additional API gateway services visible in your screenshots
    try:
        # These would be properly imported and instantiated in a real implementation
        from src.services.api_gateway.neural_interpretor.app.nueral_interpretor import NeuralInterpreter
        neural_interpreter = NeuralInterpreter(
        )
        await registry.register_service(
            "neural_interpreter",
            neural_interpreter,
            ComponentType.NEURAL_INTERPRETOR
        )
        logger.info("Neural Interpreter gateway registered")
    except Exception as e:
        logger.error(f"Failed to initialize Neural Interpreter: {str(e)}")

    logger.info(f"Successfully registered {registered_count} services")
    return registry


@log_execution_time
def register_components(pulsar_url: str = "pulsar://localhost:6650"):
    """Register all components and set up the shared folder for the program synthesis system."""
    # Configure logging
    configure_logging({
        "level": logging.INFO,
        "console": True,
        "file": True,
        "use_colors": True,
        "directory": "logs",
        "collect_performance": True,
        "multi_tenant": True
    })

    logger = get_logger("component_registration")
    logger.info("Starting component registration")

    # Create event loop
    event_loop = asyncio.get_event_loop()

    # Set up signal handlers for graceful shutdown
    registry = None

    def shutdown_handler():
        nonlocal registry
        if registry:
            asyncio.create_task(shutdown(registry, event_loop))

    for sig in (signal.SIGTERM, signal.SIGINT):
        event_loop.add_signal_handler(sig, shutdown_handler)

    # Register services
    registry = event_loop.run_until_complete(register_services(pulsar_url=pulsar_url))

    # Start health check HTTP server
    start_health_check_server(registry)

    return registry


def start_health_check_server(registry: ServiceRegistry):
    """Start the health check HTTP server for Kubernetes probes."""
    from src.services.neural_code_generator.app.healthcheck import start_server
    # Start in a separate thread to not block the main thread
    import threading
    health_thread = threading.Thread(
        target=start_server,
        args=(registry,),
        daemon=True
    )
    health_thread.start()


async def shutdown(registry: ServiceRegistry, loop: asyncio.AbstractEventLoop):
    """Gracefully shut down all components."""
    logger = get_logger("component_registration")
    logger.info("Shutting down all components")

    # Emit shutdown event
    if registry.event_emitter:
        try:
            await registry.event_emitter.emit_async(
                BaseEvent(
                    event_type=EventType.SYSTEM_SHUTDOWN,
                    source_container="service_registry",
                    payload={
                        "reason": "graceful_shutdown"
                    }
                )
            )
        except Exception as e:
            logger.error(f"Error emitting shutdown event: {str(e)}")

    # Shut down all services
    for name, service_info in registry.services.items():
        try:
            if hasattr(service_info["instance"], "stop"):
                logger.info(f"Stopping service {name}")
                if asyncio.iscoroutinefunction(service_info["instance"].stop):
                    await service_info["instance"].stop()
                else:
                    service_info["instance"].stop()
        except Exception as e:
            logger.error(f"Error stopping service {name}: {str(e)}")

    # Close event emitter
    if registry.event_emitter:
        try:
            registry.event_emitter.close()
        except Exception as e:
            logger.error(f"Error closing event emitter: {str(e)}")

    # Stop the event loop
    loop.stop()


if __name__ == "__main__":
    registry = register_components()

    # Keep the script running to maintain services
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure we shut down gracefully
        asyncio.get_event_loop().run_until_complete(
            shutdown(registry, asyncio.get_event_loop())
        )