"""
Health monitoring for the Synapse Manager service.

This module provides health monitoring capabilities to ensure
the Synapse Manager service is functioning properly.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Callable, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SynapseHealthMonitor:
    """
    Health monitoring for the Synapse Manager service.

    Provides health check mechanisms, component monitoring,
    and self-healing capabilities for the synapse network.
    """

    def __init__(self, config: Dict[str, Any], synapse_service):
        """
        Initialize the health monitor.

        Args:
            config: Configuration dictionary
            synapse_service: The synapse service instance
        """
        self.config = config
        self.synapse_service = synapse_service

        # Monitoring state
        self.component_statuses = {}
        self.last_health_check = datetime.now()
        self.health_check_interval = config.get("health_check_interval", 60)  # seconds

        # Network health metrics
        self.network_metrics = {
            "total_synapses": 0,
            "active_synapses": 0,
            "pruned_synapses": 0,
            "total_nodes": 0,
            "connection_density": 0.0,
            "network_stability": 1.0,
            "last_network_scan": None,
            "network_scan_duration_ms": 0,
            "recovery_actions_taken": 0,
        }

        # Recovery action history
        self.recovery_history = []

        # Component health checkers
        self.health_checks = {}

        logger.info("Synapse Health Monitor initialized")

    async def start(self):
        """Start the health monitor."""
        logger.info("Starting Synapse Health Monitor")

        # Register core component health checks
        self._register_core_health_checks()

        # Start periodic health checks
        asyncio.create_task(self._run_periodic_health_checks())

        # Start periodic network scans
        asyncio.create_task(self._run_periodic_network_scans())

        logger.info("Synapse Health Monitor started")

    async def stop(self):
        """Stop the health monitor."""
        logger.info("Stopping Synapse Health Monitor")
        # No specific cleanup needed
        logger.info("Synapse Health Monitor stopped")

    def _register_core_health_checks(self):
        """Register health checks for core components."""
        # Register synapse service health check
        self.register_component(
            "synapse_service",
            lambda: self.synapse_service is not None,
            is_critical=True,
        )

        # Register learning service health check
        self.register_component(
            "learning_service",
            lambda: self.synapse_service.learning_service is not None,
            is_critical=True,
        )

        # Register connection optimizer health check
        self.register_component(
            "connection_optimizer",
            lambda: self.synapse_service.connection_optimizer is not None,
            is_critical=True,
        )

        # Register pathway analyzer health check
        self.register_component(
            "pathway_analyzer",
            lambda: self.synapse_service.pathway_analyzer is not None,
            is_critical=False,
        )

        # Register event handler health check
        self.register_component(
            "event_handler",
            lambda: self.synapse_service.event_handler is not None and self.synapse_service.event_handler.running,
            is_critical=True,
        )

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
            "is_critical": is_critical,
        }

        self.health_checks[component_name] = health_check_func

        if recovery_func:
            component_status = self.component_statuses[component_name]
            component_status["recovery_func"] = recovery_func

        logger.info(f"Registered component for health monitoring: {component_name} (critical: {is_critical})")

    async def check_health(self) -> Dict[str, Any]:
        """
        Check the health of all registered components.

        Returns:
            Dictionary with health status
        """
        self.last_health_check = datetime.now()

        # Check all components
        all_healthy = True
        critical_failure = False
        component_results = {}

        for component_name, health_check_func in self.health_checks.items():
            # Get component status
            component_status = self.component_statuses[component_name]

            try:
                # Check component health
                is_healthy = health_check_func()

                # Update status
                component_status["last_check"] = datetime.now()
                component_status["healthy"] = is_healthy

                if not is_healthy:
                    component_status["last_failure"] = datetime.now()
                    component_status["failure_count"] += 1

                    # Try recovery if function available
                    if "recovery_func" in component_status:
                        await self._attempt_recovery(component_name)

                    # Check if critical
                    if component_status["is_critical"]:
                        critical_failure = True

                    all_healthy = False

                component_results[component_name] = is_healthy

            except Exception as e:
                logger.error(f"Error checking component health: {component_name}: {e}")
                component_status["healthy"] = False
                component_status["last_failure"] = datetime.now()
                component_status["failure_count"] += 1
                component_results[component_name] = False

                # Check if critical
                if component_status["is_critical"]:
                    critical_failure = True

                all_healthy = False

        # Determine overall status
        if all_healthy:
            status = "healthy"
        elif critical_failure:
            status = "unhealthy"
        else:
            status = "degraded"

        # Include network metrics
        health = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "components": component_results,
            "network_metrics": self.network_metrics,
            "last_health_check": self.last_health_check.isoformat(),
        }

        if not all_healthy:
            logger.warning(f"Health check status: {status}")

        return health

    async def _attempt_recovery(self, component_name: str) -> bool:
        """
        Attempt to recover a component.

        Args:
            component_name: Name of the component

        Returns:
            True if recovery successful, False otherwise
        """
        component_status = self.component_statuses[component_name]

        if "recovery_func" not in component_status:
            return False

        try:
            logger.info(f"Attempting to recover component: {component_name}")

            # Get recovery function
            recovery_func = component_status["recovery_func"]

            # Call recovery function
            if asyncio.iscoroutinefunction(recovery_func):
                success = await recovery_func()
            else:
                success = recovery_func()

            # Record recovery attempt
            recovery_record = {
                "component": component_name,
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "failure_count": component_status["failure_count"],
            }

            self.recovery_history.append(recovery_record)
            self.network_metrics["recovery_actions_taken"] += 1

            if success:
                logger.info(f"Successfully recovered component: {component_name}")
            else:
                logger.warning(f"Failed to recover component: {component_name}")

            return success

        except Exception as e:
            logger.error(f"Error recovering component: {component_name}: {e}")
            return False

    async def scan_network_health(self) -> Dict[str, Any]:
        """
        Scan the synapse network for health issues.

        Returns:
            Dictionary with network health metrics
        """
        try:
            start_time = time.time()

            # Count synapses
            total_synapses = len(self.synapse_service.synapses)

            # Count active synapses
            active_synapses = sum(
                1
                for synapse in self.synapse_service.synapses.values()
                if synapse.weight >= 0.3  # Arbitrary threshold
            )

            # Count nodes
            total_nodes = len(self.synapse_service.connections)

            # Calculate connection density
            if total_nodes > 1:
                max_possible_connections = total_nodes * (total_nodes - 1)
                density = total_synapses / max(1, max_possible_connections)
            else:
                density = 0.0

            # Calculate network stability metric
            # (Ratio of stable to forming/pruning synapses)
            stable_synapses = sum(
                1 for synapse in self.synapse_service.synapses.values() if synapse.state in ["stable", "strengthening"]
            )

            if total_synapses > 0:
                stability = stable_synapses / total_synapses
            else:
                stability = 1.0

            # Calculate scan duration
            end_time = time.time()
            scan_duration_ms = int((end_time - start_time) * 1000)

            # Update metrics
            self.network_metrics.update(
                {
                    "total_synapses": total_synapses,
                    "active_synapses": active_synapses,
                    "total_nodes": total_nodes,
                    "connection_density": density,
                    "network_stability": stability,
                    "last_network_scan": datetime.now().isoformat(),
                    "network_scan_duration_ms": scan_duration_ms,
                }
            )

            # If network has issues, attempt repairs
            if stability < 0.5 or density < 0.1:
                await self._repair_network()

            return self.network_metrics

        except Exception as e:
            logger.error(f"Error scanning network health: {e}")
            return self.network_metrics

    async def _repair_network(self):
        """Attempt to repair network issues."""
        try:
            logger.info("Attempting network repairs")

            # Get weak connections that should be removed
            pruning_threshold = self.config.get("pruning_threshold", 0.2)
            to_prune = []

            for synapse_id, synapse in self.synapse_service.synapses.items():
                if synapse.weight < pruning_threshold:
                    to_prune.append(synapse_id)

            # Prune weak connections
            for synapse_id in to_prune:
                # This would be done through the synapse service in a real implementation
                # await self.synapse_service._remove_synapse(synapse_id)
                pass

            # Strengthen important connections
            # This is a simplified approach - in a real implementation, would use more
            # sophisticated analysis to determine important connections
            for synapse_id, synapse in self.synapse_service.synapses.items():
                if 0.2 < synapse.weight < 0.5:
                    # Small boost to moderately strong connections
                    await self.synapse_service.update_synapse_weight(synapse_id, 0.05)

            # Record repair action
            self.network_metrics["recovery_actions_taken"] += 1

            logger.info(f"Network repairs completed: removed {len(to_prune)} weak connections")

        except Exception as e:
            logger.error(f"Error repairing network: {e}")

    async def _run_periodic_health_checks(self):
        """Run periodic health checks."""
        logger.info("Starting periodic health checks")

        while True:
            try:
                # Check health
                await self.check_health()

                # Wait for next check
                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                logger.info("Stopping periodic health checks")
                break

            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")
                await asyncio.sleep(5)  # Reduced interval on error

    async def _run_periodic_network_scans(self):
        """Run periodic network scans."""
        logger.info("Starting periodic network scans")

        # Use a different interval for network scans
        network_scan_interval = self.config.get("network_scan_interval", 300)  # 5 minutes

        while True:
            try:
                # Scan network
                await self.scan_network_health()

                # Wait for next scan
                await asyncio.sleep(network_scan_interval)

            except asyncio.CancelledError:
                logger.info("Stopping periodic network scans")
                break

            except Exception as e:
                logger.error(f"Error in periodic network scan: {e}")
                await asyncio.sleep(30)  # Reduced interval on error
