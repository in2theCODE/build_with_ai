#!/usr/bin/env python3
"""
Telemetry manager for collecting metrics across the system.
"""

import logging
from typing import Dict, Any, List, Optional
import threading
from datetime import datetime, timezone


class TelemetryManager:
    """
    Singleton class for managing telemetry across the system.
    Records metrics and provides reporting capabilities.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the telemetry manager."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = {}
        self.start_time = datetime.now(timezone.utc)

    def record_metric(self, name: str, value: Any):
        """
        Record a metric with the given name and value.

        Args:
            name: The name of the metric
            value: The value of the metric
        """
        timestamp = datetime.now().isoformat()

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            "value": value,
            "timestamp": timestamp
        })

    def get_metrics(self, name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get metrics by name or all metrics if no name is provided.

        Args:
            name: Optional metric name to filter by

        Returns:
            Dictionary of metrics
        """
        if name:
            return {name: self.metrics.get(name, [])}
        else:
            return self.metrics

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.start_time = datetime.now(timezone.utc)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.

        Returns:
            Summary dictionary
        """
        summary = {
            "start_time": self.start_time,
            "elapsed_time": datetime.now(timezone.utc) - self.start_time,
            "metrics_count": len(self.metrics)
        }

        # Add summaries for numeric metrics
        for name, values in self.metrics.items():
            numeric_values = [v["value"] for v in values if isinstance(v["value"], (int, float))]

            if numeric_values:
                summary[f"{name}_min"] = min(numeric_values)
                summary[f"{name}_max"] = max(numeric_values)
                summary[f"{name}_avg"] = sum(numeric_values) / len(numeric_values)
                summary[f"{name}_count"] = len(numeric_values)

        return summary