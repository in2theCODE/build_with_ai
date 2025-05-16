"""
Metrics collector for the Neural Context Mesh.

This module collects and tracks metrics from various mesh components,
providing a centralized view of system performance and behavior.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects metrics from neural mesh components.

    Provides a centralized collection point for metrics from
    different mesh components, supporting aggregation and tracking.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the metrics collector."""
        self.config = config

        # Metrics storage
        self.metrics_data = {}

        # Collection intervals
        self.collection_intervals = {
            "realtime": 1,  # 1 second
            "short": 60,  # 1 minute
            "medium": 300,  # 5 minutes
            "long": 3600,  # 1 hour
            "daily": 86400,  # 24 hours
        }

        # Metric types
        self.metric_types = {
            "counter": self._aggregate_counter,
            "gauge": self._aggregate_gauge,
            "histogram": self._aggregate_histogram,
            "summary": self._aggregate_summary,
        }

        # Latest metrics cache
        self.latest_metrics = {}

        # Historical metrics
        self.historical_metrics = {}
        for interval in self.collection_intervals:
            self.historical_metrics[interval] = {}

        logger.info("Metrics Collector initialized")

    async def initialize(self):
        """Initialize the metrics collector."""
        logger.info("Initializing Metrics Collector")

        # Start collection tasks
        for interval_name, seconds in self.collection_intervals.items():
            asyncio.create_task(self._collect_interval_metrics(interval_name, seconds))

        logger.info("Metrics Collector initialized")

    async def record_metric(
        self,
        metric_name: str,
        value: Any,
        metric_type: str = "gauge",
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            metric_type: Type of metric (counter, gauge, histogram, summary)
            labels: Optional metric labels
            timestamp: Optional timestamp (default: now)
        """
        if metric_type not in self.metric_types:
            logger.warning(f"Unknown metric type: {metric_type}")
            return

        # Use current time if not provided
        if timestamp is None:
            timestamp = datetime.now()

        # Convert timestamp to ISO format
        timestamp_str = timestamp.isoformat()

        # Format labels
        labels_key = self._format_labels(labels)

        # Create metric key
        metric_key = f"{metric_name}{labels_key}"

        # Initialize metric if needed
        if metric_key not in self.metrics_data:
            self.metrics_data[metric_key] = {
                "name": metric_name,
                "type": metric_type,
                "labels": labels or {},
                "values": [],
            }

        # Add value
        self.metrics_data[metric_key]["values"].append(
            {
                "timestamp": timestamp_str,
                "value": value,
            }
        )

        # Update latest metrics
        self.latest_metrics[metric_key] = {
            "name": metric_name,
            "type": metric_type,
            "labels": labels or {},
            "value": value,
            "timestamp": timestamp_str,
        }

        # Limit values list size
        max_values = self.config.get("max_values_per_metric", 1000)
        if len(self.metrics_data[metric_key]["values"]) > max_values:
            self.metrics_data[metric_key]["values"] = self.metrics_data[metric_key]["values"][-max_values:]

    async def get_latest_metrics(
        self, metric_name: Optional[str] = None, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Get latest metrics values.

        Args:
            metric_name: Optional metric name filter
            labels: Optional labels filter

        Returns:
            Dict of latest metrics
        """
        if metric_name is None:
            # Return all metrics
            return self.latest_metrics

        # Filter by name
        filtered_metrics = {}
        for key, metric in self.latest_metrics.items():
            if metric["name"] == metric_name:
                # Filter by labels if specified
                if labels is not None:
                    match = True
                    for label_key, label_value in labels.items():
                        if label_key not in metric["labels"] or metric["labels"][label_key] != label_value:
                            match = False
                            break

                    if match:
                        filtered_metrics[key] = metric
                else:
                    filtered_metrics[key] = metric

        return filtered_metrics

    async def get_historical_metrics(
        self,
        interval: str,
        metric_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get historical metrics for an interval.

        Args:
            interval: Interval name
            metric_name: Optional metric name filter
            labels: Optional labels filter
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            Dict of historical metrics
        """
        if interval not in self.historical_metrics:
            return {}

        # Get metrics for interval
        interval_metrics = self.historical_metrics[interval]

        # Filter by name
        if metric_name is not None:
            filtered_metrics = {}
            for key, metric in interval_metrics.items():
                if metric["name"] == metric_name:
                    filtered_metrics[key] = metric
            interval_metrics = filtered_metrics

        # Filter by labels
        if labels is not None:
            filtered_metrics = {}
            for key, metric in interval_metrics.items():
                match = True
                for label_key, label_value in labels.items():
                    if label_key not in metric["labels"] or metric["labels"][label_key] != label_value:
                        match = False
                        break

                if match:
                    filtered_metrics[key] = metric
            interval_metrics = filtered_metrics

        # Filter by time range
        if start_time is not None or end_time is not None:
            filtered_metrics = {}
            for key, metric in interval_metrics.items():
                filtered_values = []

                for value in metric["values"]:
                    timestamp = datetime.fromisoformat(value["timestamp"])

                    if start_time is not None and timestamp < start_time:
                        continue

                    if end_time is not None and timestamp > end_time:
                        continue

                    filtered_values.append(value)

                if filtered_values:
                    filtered_metric = metric.copy()
                    filtered_metric["values"] = filtered_values
                    filtered_metrics[key] = filtered_metric

            interval_metrics = filtered_metrics

        return interval_metrics

    async def _collect_interval_metrics(self, interval_name: str, seconds: int):
        """
        Collect metrics at regular intervals.

        Args:
            interval_name: Name of the interval
            seconds: Collection interval in seconds
        """
        logger.info(f"Starting metrics collection for interval: {interval_name}")

        # Initial delay to stagger collections
        initial_delay = seconds * 0.1 * hash(interval_name) % 1.0
        await asyncio.sleep(initial_delay)

        while True:
            try:
                # Collect metrics for this interval
                await self._aggregate_interval_metrics(interval_name)

                # Wait until next interval
                await asyncio.sleep(seconds)

            except Exception as e:
                logger.error(f"Error collecting metrics for interval {interval_name}: {e}")
                await asyncio.sleep(1)  # Short delay before retry

    async def _aggregate_interval_metrics(self, interval_name: str):
        """
        Aggregate metrics for an interval.

        Args:
            interval_name: Name of the interval
        """
        now = datetime.now()
        interval_metrics = {}

        # Process each metric
        for metric_key, metric in self.metrics_data.items():
            # Get aggregation function
            aggregate_func = self.metric_types.get(metric["type"])

            if aggregate_func:
                # Aggregate values for this interval
                aggregated = aggregate_func(metric, interval_name, now)

                if aggregated:
                    interval_metrics[metric_key] = aggregated

        # Store interval metrics
        self.historical_metrics[interval_name] = interval_metrics

    def _aggregate_counter(self, metric: Dict[str, Any], interval_name: str, now: datetime) -> Optional[Dict[str, Any]]:
        """
        Aggregate counter metric.

        Args:
            metric: Metric data
            interval_name: Interval name
            now: Current time

        Returns:
            Aggregated metric or None
        """
        interval_seconds = self.collection_intervals[interval_name]
        start_time = now.timestamp() - interval_seconds

        # Filter values in interval
        values = [v for v in metric["values"] if datetime.fromisoformat(v["timestamp"]).timestamp() >= start_time]

        if not values:
            return None

        # For counter, we take the sum of increments
        total = sum(v["value"] for v in values)
        rate = total / interval_seconds

        return {
            "name": metric["name"],
            "type": metric["type"],
            "labels": metric["labels"],
            "interval": interval_name,
            "total": total,
            "rate": rate,
            "timestamp": now.isoformat(),
            "values": values,
        }

    def _aggregate_gauge(self, metric: Dict[str, Any], interval_name: str, now: datetime) -> Optional[Dict[str, Any]]:
        """
        Aggregate gauge metric.

        Args:
            metric: Metric data
            interval_name: Interval name
            now: Current time

        Returns:
            Aggregated metric or None
        """
        interval_seconds = self.collection_intervals[interval_name]
        start_time = now.timestamp() - interval_seconds

        # Filter values in interval
        values = [v for v in metric["values"] if datetime.fromisoformat(v["timestamp"]).timestamp() >= start_time]

        if not values:
            return None

            # For gauge, we take min, max, and average
            # For gauge, we take min, max, and average
            min_value = min(v["value"] for v in values)
            max_value = max(v["value"] for v in values)
            avg_value = sum(v["value"] for v in values) / len(values)

            return {
                "name": metric["name"],
                "type": metric["type"],
                "labels": metric["labels"],
                "interval": interval_name,
                "min": min_value,
                "max": max_value,
                "avg": avg_value,
                "last": values[-1]["value"],
                "timestamp": now.isoformat(),
                "values": values,
            }

        def _aggregate_histogram(
            self, metric: Dict[str, Any], interval_name: str, now: datetime
        ) -> Optional[Dict[str, Any]]:
            """
            Aggregate histogram metric.

            Args:
                metric: Metric data
                interval_name: Interval name
                now: Current time

            Returns:
                Aggregated metric or None
            """
            interval_seconds = self.collection_intervals[interval_name]
            start_time = now.timestamp() - interval_seconds

            # Filter values in interval
            values = [v for v in metric["values"] if datetime.fromisoformat(v["timestamp"]).timestamp() >= start_time]

            if not values:
                return None

            # Extract numeric values
            numeric_values = [float(v["value"]) for v in values]

            # Sort for percentiles
            sorted_values = sorted(numeric_values)

            # Calculate histogram buckets
            buckets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            bucket_counts = [0] * len(buckets)

            for value in numeric_values:
                for i, bucket in enumerate(buckets):
                    if value <= bucket:
                        bucket_counts[i] += 1

            # Calculate percentiles
            count = len(sorted_values)
            percentiles = {
                "50": sorted_values[int(count * 0.5)] if count > 0 else 0,
                "90": sorted_values[int(count * 0.9)] if count > 0 else 0,
                "95": sorted_values[int(count * 0.95)] if count > 0 else 0,
                "99": sorted_values[int(count * 0.99)] if count > 0 else 0,
            }

            return {
                "name": metric["name"],
                "type": metric["type"],
                "labels": metric["labels"],
                "interval": interval_name,
                "count": count,
                "min": min(numeric_values) if numeric_values else 0,
                "max": max(numeric_values) if numeric_values else 0,
                "avg": sum(numeric_values) / count if count > 0 else 0,
                "percentiles": percentiles,
                "buckets": dict(zip([str(b) for b in buckets], bucket_counts)),
                "timestamp": now.isoformat(),
                "values": values,
            }

        def _aggregate_summary(
            self, metric: Dict[str, Any], interval_name: str, now: datetime
        ) -> Optional[Dict[str, Any]]:
            """
            Aggregate summary metric.

            Args:
                metric: Metric data
                interval_name: Interval name
                now: Current time

            Returns:
                Aggregated metric or None
            """
            interval_seconds = self.collection_intervals[interval_name]
            start_time = now.timestamp() - interval_seconds

            # Filter values in interval
            values = [v for v in metric["values"] if datetime.fromisoformat(v["timestamp"]).timestamp() >= start_time]

            if not values:
                return None

            # Extract numeric values
            numeric_values = [float(v["value"]) for v in values]

            # Calculate summary statistics
            count = len(numeric_values)
            min_value = min(numeric_values) if numeric_values else 0
            max_value = max(numeric_values) if numeric_values else 0
            avg_value = sum(numeric_values) / count if count > 0 else 0

            # Calculate standard deviation
            variance = sum((x - avg_value) ** 2 for x in numeric_values) / count if count > 0 else 0
            std_dev = variance**0.5

            return {
                "name": metric["name"],
                "type": metric["type"],
                "labels": metric["labels"],
                "interval": interval_name,
                "count": count,
                "min": min_value,
                "max": max_value,
                "avg": avg_value,
                "std_dev": std_dev,
                "timestamp": now.isoformat(),
                "values": values,
            }

        def _format_labels(self, labels: Optional[Dict[str, str]]) -> str:
            """
            Format labels into a string key.

            Args:
                labels: Labels dict or None

            Returns:
                Formatted labels string
            """
            if not labels:
                return ""

            # Sort labels for consistent keys
            sorted_labels = sorted(labels.items())
            labels_str = ",".join(f"{k}={v}" for k, v in sorted_labels)

            return f"{{{labels_str}}}"
