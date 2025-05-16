"""
Health monitor for the Knowledge Node service.

This module provides health monitoring capabilities to ensure
the Knowledge Node service is functioning properly.
"""

import asyncio
import logging
from typing import Dict, Any, Callable, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Health monitoring for the Knowledge Node service.

    Provides health check mechanisms, component monitoring,
    and self-healing capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the health monitor."""
        self.config = config

        # Monitoring state
        self.component_statuses: Dict[str, Dict[str, Any]] = {}
        self.last_global_health_check = datetime.now()
        self.health_check_interval = config.get("health_check_interval", 60)  # seconds
        self.critical_components = config.get("critical_components", [])

        # Recovery actions
        self.recovery_actions: Dict[str, Callable] = {}

        # Health metrics
        self.metrics = {
            "component_failures": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "last_incident": None,
            "uptime_start": datetime.now(),
        }

        logger.info("Health Monitor initialized")

    async def start(self):
        """Start the health monitor."""
        logger.info("Starting Health Monitor")

        # Start health check task
        asyncio.create_task(self._run_periodic_health_checks())

        logger.info("Health Monitor started")

    async def stop(self):
        """Stop the health monitor."""
        logger.info("Stopping Health Monitor")
        # Cleanup any resources
        logger.info("Health Monitor stopped")

    def register_component(
        self,
        component_name: str,
        health_check_func: Callable[[], bool],
        recovery_func: Optional[Callable] = None,
        is_critical: bool = False,
    ):
        """
        Register a component for health monitoring.

        Args:
            component_name: Name of the component
            health_check_func: Function to check component health
            recovery_func: Function to recover component (optional)
            is_critical: Whether this is a critical component
        """
        self.component_statuses[component_name] = {
            "healthy": True,
            "last_check": datetime.now(),
            "last_failure": None,
            "failure_count": 0,
            "health_check_func": health_check_func,
        }

        if recovery_func:
            self.recovery_actions[component_name] = recovery_func

        if is_critical and component_name not in self.critical_components:
            self.critical_components.append(component_name)

        logger.info(f"Registered component for health monitoring: {component_name}")

    async def check_component_health(self, component_name: str) -> bool:
        """
        Check the health of a specific component.

        Args:
            component_name: Name of the component

        Returns:
            True if healthy, False otherwise
        """
        if component_name not in self.component_statuses:
            logger.warning(f"Unknown component for health check: {component_name}")
            return False

        component = self.component_statuses[component_name]

        try:
            # Call health check function
            health_check_func = component["health_check_func"]
            is_healthy = health_check_func()

            # Update status
            component["last_check"] = datetime.now()

            if not is_healthy:
                component["healthy"] = False
                component["last_failure"] = datetime.now()
                component["failure_count"] += 1

                logger.warning(f"Component unhealthy: {component_name}")

                # Attempt recovery if available
                if component_name in self.recovery_actions:
                    await self._recover_component(component_name)
            else:
                component["healthy"] = True

            return is_healthy

        except Exception as e:
            logger.error(f"Error checking component health: {component_name}: {e}")
            component["healthy"] = False
            component["last_failure"] = datetime.now()
            component["failure_count"] += 1

            return False

    async def check_global_health(self) -> Dict[str, Any]:
        """
        Check the overall health of the system.

        Returns:
            Dictionary with health status
        """
        self.last_global_health_check = datetime.now()

        # Check all components
        component_results = {}
        for component_name in self.component_statuses:
            is_healthy = await self.check_component_health(component_name)
            component_results[component_name] = is_healthy

        # Calculate global health
        critical_healthy = all(
            self.component_statuses[c]["healthy"] for c in self.critical_components if c in self.component_statuses
        )

        non_critical_healthy = all(
            self.component_statuses[c]["healthy"] for c in self.component_statuses if c not in self.critical_components
        )

        # Determine status
        if critical_healthy and non_critical_healthy:
            status = "healthy"
        elif critical_healthy:
            status = "degraded"
        else:
            status = "unhealthy"

        # Calculate uptime
        uptime_seconds = (datetime.now() - self.metrics["uptime_start"]).total_seconds()

        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "components": component_results,
            "uptime_seconds": uptime_seconds,
            "metrics": self.metrics,
        }

    async def _recover_component(self, component_name: str) -> bool:
        """
        Attempt to recover a component.

        Args:
            component_name: Name of the component

        Returns:
            True if recovery successful, False otherwise
        """
        if component_name not in self.recovery_actions:
            return False

        logger.info(f"Attempting to recover component: {component_name}")
        self.metrics["recovery_attempts"] += 1

        try:
            # Call recovery function
            recovery_func = self.recovery_actions[component_name]
            success = recovery_func()

            if success:
                logger.info(f"Successfully recovered component: {component_name}")
                self.metrics["successful_recoveries"] += 1
            else:
                logger.warning(f"Failed to recover component: {component_name}")

            return success

        except Exception as e:
            logger.error(f"Error recovering component: {component_name}: {e}")
            return False

    async def _run_periodic_health_checks(self):
        """Run periodic health checks."""
        while True:
            try:
                # Run global health check
                health_status = await self.check_global_health()

                # Log status
                if health_status["status"] != "healthy":
                    logger.warning(f"System health degraded: {health_status}")
                    self.metrics["last_incident"] = datetime.now().isoformat()

            except Exception as e:
                logger.error(f"Error in health check: {e}")

            # Wait for next check
            await asyncio.sleep(self.health_check_interval)
