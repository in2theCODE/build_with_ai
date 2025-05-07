"""
Health monitoring system for microservices.

This module provides health monitoring capabilities for microservices,
working with both the healthcheck API and circuit breaker pattern.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional


# Try to import the metrics collector
try:
    from src.shared.metrics.metrics_collector import MetricsCollector
except ImportError:
    MetricsCollector = None

logger = logging.getLogger(__name__)


class HealthStatus:
    """Health status constants."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


class HealthMonitor:
    """
    Health monitor for microservices.

    This class provides health monitoring capabilities for microservices,
    integrating with the healthcheck API and circuit breaker pattern.
    """

    def __init__(
        self,
        service_name: str,
        metrics_collector: Optional[Any] = None,
        initial_status: str = HealthStatus.STARTING,
    ):
        """
        Initialize the health monitor.

        Args:
            service_name: Name of the service to monitor
            metrics_collector: Optional metrics collector
            initial_status: Initial health status
        """
        self.service_name = service_name
        self.metrics_collector = metrics_collector
        self.status = initial_status
        self.last_check_time = time.time()
        self.check_count = 0
        self.error_count = 0
        self.dependencies = {}
        self.subsystems = {}

        # Register with metrics if available
        if self.metrics_collector and hasattr(self.metrics_collector, "component_up"):
            self.metrics_collector.component_up.labels(component=self.service_name).set(
                1 if initial_status == HealthStatus.HEALTHY else 0
            )

        logger.info(
            f"Health monitor initialized for service '{service_name}' with status '{initial_status}'"
        )

    def register_dependency(self, name: str, check_function: Callable[[], bool]) -> None:
        """
        Register a dependency to check.

        Args:
            name: Dependency name
            check_function: Function to check dependency health
        """
        self.dependencies[name] = {
            "check_function": check_function,
            "status": HealthStatus.STARTING,
            "last_check_time": 0,
            "error_count": 0,
        }
        logger.debug(f"Registered dependency '{name}' for service '{self.service_name}'")

    def register_subsystem(self, name: str, check_function: Callable[[], bool]) -> None:
        """
        Register a subsystem to check.

        Args:
            name: Subsystem name
            check_function: Function to check subsystem health
        """
        self.subsystems[name] = {
            "check_function": check_function,
            "status": HealthStatus.STARTING,
            "last_check_time": 0,
            "error_count": 0,
        }
        logger.debug(f"Registered subsystem '{name}' for service '{self.service_name}'")

    async def check_health(self) -> str:
        """
        Check health of service and all dependencies/subsystems.

        Returns:
            Health status
        """
        self.check_count += 1
        self.last_check_time = time.time()
        all_healthy = True
        critical_failure = False

        # Check dependencies
        for name, info in self.dependencies.items():
            try:
                healthy = info["check_function"]()
                info["status"] = HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY
                info["last_check_time"] = time.time()

                if not healthy:
                    all_healthy = False
                    info["error_count"] += 1
                    logger.warning(
                        f"Dependency '{name}' for service '{self.service_name}' is unhealthy"
                    )

                    # Dependencies are critical
                    critical_failure = True
                else:
                    info["error_count"] = 0

            except Exception as e:
                all_healthy = False
                critical_failure = True
                info["status"] = HealthStatus.UNHEALTHY
                info["error_count"] += 1
                info["last_check_time"] = time.time()
                logger.error(
                    f"Error checking dependency '{name}' for service '{self.service_name}': {e}"
                )

        # Check subsystems
        for name, info in self.subsystems.items():
            try:
                healthy = info["check_function"]()
                info["status"] = HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY
                info["last_check_time"] = time.time()

                if not healthy:
                    all_healthy = False
                    info["error_count"] += 1
                    logger.warning(
                        f"Subsystem '{name}' for service '{self.service_name}' is unhealthy"
                    )
                else:
                    info["error_count"] = 0

            except Exception as e:
                all_healthy = False
                info["status"] = HealthStatus.UNHEALTHY
                info["error_count"] += 1
                info["last_check_time"] = time.time()
                logger.error(
                    f"Error checking subsystem '{name}' for service '{self.service_name}': {e}"
                )

        # Update service status
        old_status = self.status
        if critical_failure:
            self.status = HealthStatus.UNHEALTHY
            self.error_count += 1
        elif all_healthy:
            self.status = HealthStatus.HEALTHY
            self.error_count = 0
        else:
            self.status = HealthStatus.DEGRADED
            self.error_count += 1

        # Log status change
        if old_status != self.status:
            logger.info(
                f"Service '{self.service_name}' health status changed: {old_status} -> {self.status}"
            )

            # Update metrics if available
            if self.metrics_collector and hasattr(self.metrics_collector, "component_up"):
                self.metrics_collector.component_up.labels(component=self.service_name).set(
                    1 if self.status == HealthStatus.HEALTHY else 0
                )

        return self.status

    def get_detailed_status(self) -> Dict[str, Any]:
        """
        Get detailed health status for all components.

        Returns:
            Dictionary with detailed health information
        """
        return {
            "service": {
                "name": self.service_name,
                "status": self.status,
                "last_check_time": self.last_check_time,
                "check_count": self.check_count,
                "error_count": self.error_count,
            },
            "dependencies": {
                name: {
                    "status": info["status"],
                    "last_check_time": info["last_check_time"],
                    "error_count": info["error_count"],
                }
                for name, info in self.dependencies.items()
            },
            "subsystems": {
                name: {
                    "status": info["status"],
                    "last_check_time": info["last_check_time"],
                    "error_count": info["error_count"],
                }
                for name, info in self.subsystems.items()
            },
        }

    def is_healthy(self) -> bool:
        """
        Check if service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        return self.status == HealthStatus.HEALTHY

    def is_available(self) -> bool:
        """
        Check if service is available (healthy or degraded).

        Returns:
            True if service is available, False otherwise
        """
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


def start_server(registry=None):
    """
    Start the health check server with service registry integration.

    Args:
        registry: Optional service registry for additional health checks
    """
    from src.services.shared.health.healthcheck import (
        start_server as start_healthcheck_server,
    )

    # If registry is provided, integrate health checks
    if registry and hasattr(registry, "health_check"):
        # Modify the health check endpoint to include registry health
        from fastapi import Response
        from src.services.shared.health.healthcheck import app

        @app.get("/service-health")
        async def service_health():
            """Get health status of all registered services."""
            try:
                health_status = await registry.health_check()
                return health_status
            except Exception as e:
                logger.error(f"Error in service health check: {e}")
                return Response(content=str(e), status_code=500)

    # Start the server
    start_healthcheck_server()
