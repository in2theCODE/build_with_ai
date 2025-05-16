from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime, timedelta
import numpy as np


from .metrics_collector import MetricsCollector
from .metrics_analyzer import MetricsAnalyzer
from .event_handlers import MetricsNodeEventHandler

logger = logging.getLogger(__name__)


class MetricsService:
    """
    Service collecting and analyzing metrics for the neural mesh.

    Tracks usage patterns, performance metrics, and other statistics
    to provide insights for optimization and evolution.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the metrics service."""
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.metrics_analyzer = MetricsAnalyzer(config)
        self.event_handler = MetricsNodeEventHandler(self)

        # Metrics storage
        self.activation_metrics: Dict[str, Dict[str, Any]] = {}
        self.synapse_metrics: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}

        # Aggregation intervals
        self.aggregation_intervals = [
            60,  # 1 minute
            300,  # 5 minutes
            3600,  # 1 hour
            86400,  # 1 day
        ]

        # Historical storage
        self.historical_metrics: Dict[int, List[Dict[str, Any]]] = {
            interval: [] for interval in self.aggregation_intervals
        }

        # Retention periods (in seconds)
        self.retention_periods = {
            60: 3600,  # Keep 1-minute metrics for 1 hour
            300: 86400,  # Keep 5-minute metrics for 1 day
            3600: 604800,  # Keep 1-hour metrics for 1 week
            86400: 2592000,  # Keep 1-day metrics for 30 days
        }

        logger.info("Metrics Service initialized")

    async def start(self):
        """Start the metrics service."""
        logger.info("Starting Metrics Service")

        # Initialize components
        await self.metrics_collector.initialize()
        await self.metrics_analyzer.initialize()

        # Start event handler
        await self.event_handler.start()

        # Start background tasks
        asyncio.create_task(self._run_metrics_aggregation())
        asyncio.create_task(self._run_metrics_cleanup())

        logger.info("Metrics Service started")

    async def stop(self):
        """Stop the metrics service."""
        logger.info("Stopping Metrics Service")

        # Stop event handler
        await self.event_handler.stop()

        logger.info("Metrics Service stopped")

    async def record_activation(self, node_id: str, context_type: str, activation_value: float):
        """
        Record a node activation event.

        Args:
            node_id: Node ID
            context_type: Context type
            activation_value: Activation value
        """
        try:
            # Initialize if not yet tracked
            if node_id not in self.activation_metrics:
                self.activation_metrics[node_id] = {
                    "count": 0,
                    "sum_value": 0.0,
                    "context_type": context_type,
                    "first_seen": datetime.now(),
                    "last_seen": datetime.now(),
                    "history": [],
                }

            # Update metrics
            now = datetime.now()
            metrics = self.activation_metrics[node_id]
            metrics["count"] += 1
            metrics["sum_value"] += activation_value
            metrics["last_seen"] = now

            # Add to recent history
            metrics["history"].append({"timestamp": now, "value": activation_value})

            # Limit history size
            if len(metrics["history"]) > 100:
                metrics["history"] = metrics["history"][-100:]

            logger.debug(f"Recorded activation: {node_id} = {activation_value}")

        except Exception as e:
            logger.error(f"Error recording activation: {e}")

    async def record_synapse_change(
        self,
        synapse_id: str,
        from_node_id: str,
        to_node_id: str,
        weight_change: float,
        new_state: str,
    ):
        """
        Record a synapse state change event.

        Args:
            synapse_id: Synapse ID
            from_node_id: Source node ID
            to_node_id: Target node ID
            weight_change: Weight change amount
            new_state: New synapse state
        """
        try:
            # Initialize if not yet tracked
            if synapse_id not in self.synapse_metrics:
                self.synapse_metrics[synapse_id] = {
                    "from_node_id": from_node_id,
                    "to_node_id": to_node_id,
                    "weight_changes": [],
                    "state_changes": [],
                    "first_seen": datetime.now(),
                    "last_updated": datetime.now(),
                }

            # Update metrics
            now = datetime.now()
            metrics = self.synapse_metrics[synapse_id]
            metrics["last_updated"] = now

            # Add weight change
            metrics["weight_changes"].append({"timestamp": now, "value": weight_change})

            # Add state change
            metrics["state_changes"].append({"timestamp": now, "state": new_state})

            # Limit history size
            if len(metrics["weight_changes"]) > 100:
                metrics["weight_changes"] = metrics["weight_changes"][-100:]

            if len(metrics["state_changes"]) > 100:
                metrics["state_changes"] = metrics["state_changes"][-100:]

            logger.debug(f"Recorded synapse change: {synapse_id} = {weight_change}")

        except Exception as e:
            logger.error(f"Error recording synapse change: {e}")

    async def record_performance_metric(self, metric_name: str, value: float, context: Optional[Dict[str, Any]] = None):
        """
        Record a performance metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            context: Optional context information
        """
        try:
            # Initialize if not yet tracked
            if metric_name not in self.performance_metrics:
                self.performance_metrics[metric_name] = {
                    "count": 0,
                    "sum": 0.0,
                    "min": float("inf"),
                    "max": float("-inf"),
                    "sum_squared": 0.0,
                    "history": [],
                }

            # Update metrics
            now = datetime.now()
            metrics = self.performance_metrics[metric_name]
            metrics["count"] += 1
            metrics["sum"] += value
            metrics["min"] = min(metrics["min"], value)
            metrics["max"] = max(metrics["max"], value)
            metrics["sum_squared"] += value * value

            # Add to history
            metrics["history"].append({"timestamp": now, "value": value, "context": context})

            # Limit history size
            if len(metrics["history"]) > 1000:
                metrics["history"] = metrics["history"][-1000:]

            logger.debug(f"Recorded performance metric: {metric_name} = {value}")

        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")

    async def get_node_metrics(
        self,
        node_id: Optional[str] = None,
        context_type: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Get metrics for nodes.

        Args:
            node_id: Optional specific node ID
            context_type: Optional context type filter
            time_range: Optional time range

        Returns:
            Dict of node metrics
        """
        try:
            # If specific node requested
            if node_id:
                if node_id in self.activation_metrics:
                    return {node_id: self.activation_metrics[node_id]}
                else:
                    return {}

            # Filter by context type if specified
            if context_type:
                return {
                    node_id: metrics
                    for node_id, metrics in self.activation_metrics.items()
                    if metrics.get("context_type") == context_type
                }

            # Filter by time range if specified
            if time_range:
                start_time, end_time = time_range
                return {
                    node_id: metrics
                    for node_id, metrics in self.activation_metrics.items()
                    if metrics.get("last_seen") >= start_time and metrics.get("first_seen") <= end_time
                }

            # Return all metrics
            return self.activation_metrics

        except Exception as e:
            logger.error(f"Error getting node metrics: {e}")
            return {}

    async def get_synapse_metrics(
        self,
        synapse_id: Optional[str] = None,
        from_node_id: Optional[str] = None,
        to_node_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Get metrics for synapses.

        Args:
            synapse_id: Optional specific synapse ID
            from_node_id: Optional source node ID filter
            to_node_id: Optional target node ID filter
            time_range: Optional time range

        Returns:
            Dict of synapse metrics
        """
        try:
            # If specific synapse requested
            if synapse_id:
                if synapse_id in self.synapse_metrics:
                    return {synapse_id: self.synapse_metrics[synapse_id]}
                else:
                    return {}

            # Apply filters
            filtered_metrics = {}

            for s_id, metrics in self.synapse_metrics.items():
                # Check from_node_id filter
                if from_node_id and metrics.get("from_node_id") != from_node_id:
                    continue

                # Check to_node_id filter
                if to_node_id and metrics.get("to_node_id") != to_node_id:
                    continue

                # Check time range filter
                if time_range:
                    start_time, end_time = time_range
                    if metrics.get("last_updated") < start_time or metrics.get("first_seen") > end_time:
                        continue

                # Add to filtered results
                filtered_metrics[s_id] = metrics

            return filtered_metrics

        except Exception as e:
            logger.error(f"Error getting synapse metrics: {e}")
            return {}

    async def get_performance_metrics(
        self,
        metric_name: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Get performance metrics.

        Args:
            metric_name: Optional specific metric name
            time_range: Optional time range

        Returns:
            Dict of performance metrics
        """
        try:
            # If specific metric requested
            if metric_name:
                if metric_name in self.performance_metrics:
                    return {metric_name: self.performance_metrics[metric_name]}
                else:
                    return {}

            # Return all metrics (time filtering would be applied here)
            return self.performance_metrics

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    async def get_historical_metrics(self, interval: int, metric_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical metrics at a specific aggregation interval.

        Args:
            interval: Aggregation interval in seconds
            metric_type: Type of metric
            limit: Maximum number of records to return

        Returns:
            List of historical metrics
        """
        try:
            if interval not in self.historical_metrics:
                return []

            # Filter by metric type
            filtered_metrics = [
                metric for metric in self.historical_metrics[interval] if metric.get("metric_type") == metric_type
            ]

            # Return most recent metrics up to limit
            return filtered_metrics[-limit:]

        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return []

    async def _run_metrics_aggregation(self):
        """Run periodic metrics aggregation."""
        while True:
            try:
                now = datetime.now()

                # Aggregate metrics for each interval
                for interval in self.aggregation_intervals:
                    # Check if it's time to aggregate
                    if now.timestamp() % interval < 10:  # Within 10 seconds of interval boundary
                        await self._aggregate_metrics(interval)

            except Exception as e:
                logger.error(f"Error in metrics aggregation: {e}")

            # Run every 10 seconds
            await asyncio.sleep(10)

    async def _aggregate_metrics(self, interval: int):
        """
        Aggregate metrics for a specific interval.

        Args:
            interval: Aggregation interval in seconds
        """
        logger.debug(f"Aggregating metrics for {interval}s interval")

        # Current time
        now = datetime.now()

        # Interval start and end
        interval_end = now
        interval_start = now - timedelta(seconds=interval)

        # Aggregate activation metrics
        activation_aggregate = await self._aggregate_activation_metrics(interval_start, interval_end)

        # Aggregate synapse metrics
        synapse_aggregate = await self._aggregate_synapse_metrics(interval_start, interval_end)

        # Aggregate performance metrics
        performance_aggregate = await self._aggregate_performance_metrics(interval_start, interval_end)

        # Store the aggregated metrics
        self.historical_metrics[interval].append(
            {
                "timestamp": now,
                "interval_start": interval_start,
                "interval_end": interval_end,
                "activation": activation_aggregate,
                "synapse": synapse_aggregate,
                "performance": performance_aggregate,
            }
        )

    async def _aggregate_activation_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Aggregate activation metrics for a time range.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            Aggregated metrics
        """
        # Initialize aggregates
        total_activations = 0
        active_nodes = 0
        activations_by_type = {}

        # Process each node's metrics
        for node_id, metrics in self.activation_metrics.items():
            # Filter history by time range
            history_in_range = [h for h in metrics.get("history", []) if start_time <= h.get("timestamp") <= end_time]

            # Count activations in range
            activations_in_range = len(history_in_range)

            if activations_in_range > 0:
                # Count active node
                active_nodes += 1

                # Add to total
                total_activations += activations_in_range

                # Aggregate by context type
                context_type = metrics.get("context_type", "unknown")
                if context_type not in activations_by_type:
                    activations_by_type[context_type] = 0
                activations_by_type[context_type] += activations_in_range

        return {
            "total_activations": total_activations,
            "active_nodes": active_nodes,
            "activations_by_type": activations_by_type,
        }

    async def _aggregate_synapse_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Aggregate synapse metrics for a time range.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            Aggregated metrics
        """
        # Initialize aggregates
        total_changes = 0
        active_synapses = 0
        weight_increases = 0
        weight_decreases = 0
        state_changes = {}

        # Process each synapse's metrics
        for synapse_id, metrics in self.synapse_metrics.items():
            # Filter weight changes by time range
            weight_changes_in_range = [
                c for c in metrics.get("weight_changes", []) if start_time <= c.get("timestamp") <= end_time
            ]

            # Filter state changes by time range
            state_changes_in_range = [
                c for c in metrics.get("state_changes", []) if start_time <= c.get("timestamp") <= end_time
            ]

            # Count changes in range
            changes_in_range = len(weight_changes_in_range) + len(state_changes_in_range)

            if changes_in_range > 0:
                # Count active synapse
                active_synapses += 1

                # Add to total
                total_changes += changes_in_range

                # Count weight increases and decreases
                for change in weight_changes_in_range:
                    if change.get("value", 0) > 0:
                        weight_increases += 1
                    elif change.get("value", 0) < 0:
                        weight_decreases += 1

                # Aggregate state changes
                for change in state_changes_in_range:
                    state = change.get("state", "unknown")
                    if state not in state_changes:
                        state_changes[state] = 0
                    state_changes[state] += 1

        return {
            "total_changes": total_changes,
            "active_synapses": active_synapses,
            "weight_increases": weight_increases,
            "weight_decreases": weight_decreases,
            "state_changes": state_changes,
        }

    async def _aggregate_performance_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Aggregate performance metrics for a time range.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            Aggregated metrics
        """
        # Initialize aggregates
        metric_aggregates = {}

        # Process each performance metric
        for metric_name, metrics in self.performance_metrics.items():
            # Filter history by time range
            history_in_range = [h for h in metrics.get("history", []) if start_time <= h.get("timestamp") <= end_time]

            # Skip if no data in range
            if not history_in_range:
                continue

            # Calculate aggregate statistics
            values = [h.get("value", 0) for h in history_in_range]

            metric_aggregates[metric_name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": np.mean(values),
                "median": np.median(values),
                "std_dev": np.std(values),
            }

        return metric_aggregates

    async def _run_metrics_cleanup(self):
        """Run periodic metrics cleanup."""
        while True:
            try:
                # Cleanup historical metrics
                await self._cleanup_historical_metrics()

            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")

            # Run every hour
            await asyncio.sleep(3600)

    async def _cleanup_historical_metrics(self):
        """Clean up old historical metrics."""
        logger.debug("Cleaning up historical metrics")

        now = datetime.now()

        # Clean up for each interval
        for interval, retention_period in self.retention_periods.items():
            # Calculate cutoff time
            cutoff_time = now - timedelta(seconds=retention_period)

            # Remove old metrics
            self.historical_metrics[interval] = [
                metric for metric in self.historical_metrics[interval] if metric.get("timestamp", now) >= cutoff_time
            ]

            logger.debug(f"Cleaned up {interval}s interval metrics: {len(self.historical_metrics[interval])} remaining")
