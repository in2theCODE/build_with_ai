"""
Metrics analyzer for the Neural Context Mesh.

This module analyzes metrics collected from mesh components,
identifying patterns, anomalies, and optimization opportunities.
"""

import logging
import asyncio
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class MetricsAnalyzer:
    """
    Analyzes metrics from neural mesh components.

    Identifies patterns, detects anomalies, and suggests
    optimization opportunities based on collected metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the metrics analyzer."""
        self.config = config

        # Analysis parameters
        self.analysis_interval = config.get("analysis_interval", 300)  # 5 minutes
        self.anomaly_threshold = config.get("anomaly_threshold", 3.0)  # 3 standard deviations
        self.trend_min_periods = config.get("trend_min_periods", 3)
        self.correlation_threshold = config.get("correlation_threshold", 0.7)

        # Analysis results
        self.anomalies = []
        self.trends = {}
        self.correlations = {}
        self.reports = []

        # Last analysis timestamps
        self.last_anomaly_detection = datetime.now()
        self.last_trend_analysis = datetime.now()
        self.last_correlation_analysis = datetime.now()

        logger.info("Metrics Analyzer initialized")

    async def initialize(self):
        """Initialize the metrics analyzer."""
        logger.info("Initializing Metrics Analyzer")

        # Start analysis tasks
        asyncio.create_task(self._run_periodic_analysis())

        logger.info("Metrics Analyzer initialized")

    async def detect_anomalies(
        self, metrics: Dict[str, Dict[str, Any]], interval: str = "medium"
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metrics.

        Args:
            metrics: Metrics data
            interval: Analysis interval

        Returns:
            List of detected anomalies
        """
        self.last_anomaly_detection = datetime.now()
        detected_anomalies = []

        for metric_key, metric in metrics.items():
            # Skip metrics with insufficient data
            if "values" not in metric or len(metric["values"]) < 5:
                continue

            # Extract values
            values = [float(v["value"]) for v in metric["values"]]

            # Calculate statistics
            mean = np.mean(values)
            std_dev = np.std(values)

            # Check last value for anomaly
            last_value = values[-1]
            z_score = abs(last_value - mean) / max(std_dev, 0.0001)  # Avoid division by zero

            if z_score > self.anomaly_threshold:
                # Create anomaly record
                anomaly = {
                    "metric_key": metric_key,
                    "metric_name": metric["name"],
                    "labels": metric["labels"],
                    "timestamp": datetime.now().isoformat(),
                    "value": last_value,
                    "expected": mean,
                    "z_score": z_score,
                    "std_dev": std_dev,
                    "is_high": last_value > mean,
                    "is_low": last_value < mean,
                    "interval": interval,
                }

                detected_anomalies.append(anomaly)
                self.anomalies.append(anomaly)

                logger.info(f"Detected anomaly in {metric['name']}: {anomaly}")

        # Limit anomalies history
        max_anomalies = self.config.get("max_anomalies_history", 100)
        if len(self.anomalies) > max_anomalies:
            self.anomalies = self.anomalies[-max_anomalies:]

        return detected_anomalies

    async def analyze_trends(
        self, metrics: Dict[str, Dict[str, Any]], interval: str = "long"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze trends in metrics.

        Args:
            metrics: Metrics data
            interval: Analysis interval

        Returns:
            Dict of metric trends
        """
        self.last_trend_analysis = datetime.now()
        detected_trends = {}

        for metric_key, metric in metrics.items():
            # Skip metrics with insufficient data
            if "values" not in metric or len(metric["values"]) < self.trend_min_periods:
                continue

            # Extract values and timestamps
            values = [(datetime.fromisoformat(v["timestamp"]).timestamp(), float(v["value"])) for v in metric["values"]]

            # Sort by timestamp
            values.sort(key=lambda x: x[0])

            # Extract x and y
            x = np.array([v[0] for v in values])
            y = np.array([v[1] for v in values])

            # Normalize x to [0, 1] range for numerical stability
            x_min = x.min()
            x_range = x.max() - x_min
            if x_range > 0:
                x_norm = (x - x_min) / x_range
            else:
                x_norm = np.zeros_like(x)

            # Simple linear regression
            slope, intercept = self._linear_regression(x_norm, y)

            # Calculate fit quality (R^2)
            y_pred = slope * x_norm + intercept
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

            # Determine trend direction and significance
            trend_direction = "stable"
            if abs(slope) > 0.1:  # Arbitrary threshold
                trend_direction = "increasing" if slope > 0 else "decreasing"

            trend_strength = "weak"
            if r_squared >= 0.7:
                trend_strength = "strong"
            elif r_squared >= 0.3:
                trend_strength = "moderate"

            # Create trend record
            trend = {
                "metric_key": metric_key,
                "metric_name": metric["name"],
                "labels": metric["labels"],
                "timestamp": datetime.now().isoformat(),
                "slope": slope,
                "r_squared": r_squared,
                "direction": trend_direction,
                "strength": trend_strength,
                "interval": interval,
                "data_points": len(values),
            }

            detected_trends[metric_key] = trend
            self.trends[metric_key] = trend

            if trend_strength != "weak":
                logger.info(f"Detected {trend_strength} {trend_direction} trend in {metric['name']}")

        return detected_trends

    async def analyze_correlations(
        self, metrics: Dict[str, Dict[str, Any]], interval: str = "long"
    ) -> List[Dict[str, Any]]:
        """
        Analyze correlations between metrics.

        Args:
            metrics: Metrics data
            interval: Analysis interval

        Returns:
            List of metric correlations
        """
        self.last_correlation_analysis = datetime.now()
        detected_correlations = []

        # Need at least 2 metrics for correlation
        if len(metrics) < 2:
            return []

        # Get all metric pairs
        metric_keys = list(metrics.keys())

        for i in range(len(metric_keys)):
            for j in range(i + 1, len(metric_keys)):
                key1 = metric_keys[i]
                key2 = metric_keys[j]

                metric1 = metrics[key1]
                metric2 = metrics[key2]

                # Skip if insufficient data points
                if (
                    "values" not in metric1
                    or len(metric1["values"]) < 5
                    or "values" not in metric2
                    or len(metric2["values"]) < 5
                ):
                    continue

                # Get timestamps and values
                values1 = {
                    datetime.fromisoformat(v["timestamp"]).timestamp(): float(v["value"]) for v in metric1["values"]
                }
                values2 = {
                    datetime.fromisoformat(v["timestamp"]).timestamp(): float(v["value"]) for v in metric2["values"]
                }

                # Find common timestamps
                common_timestamps = set(values1.keys()) & set(values2.keys())

                # Skip if insufficient common points
                if len(common_timestamps) < 5:
                    continue

                # Extract values at common timestamps
                x = np.array([values1[ts] for ts in common_timestamps])
                y = np.array([values2[ts] for ts in common_timestamps])

                # Calculate correlation coefficient
                correlation = self._correlation(x, y)

                # Only report significant correlations
                if abs(correlation) >= self.correlation_threshold:
                    correlation_type = "positive" if correlation > 0 else "negative"

                    # Create correlation record
                    corr_record = {
                        "metric1_key": key1,
                        "metric1_name": metric1["name"],
                        "metric1_labels": metric1["labels"],
                        "metric2_key": key2,
                        "metric2_name": metric2["name"],
                        "metric2_labels": metric2["labels"],
                        "correlation": correlation,
                        "type": correlation_type,
                        "strength": abs(correlation),
                        "timestamp": datetime.now().isoformat(),
                        "data_points": len(common_timestamps),
                        "interval": interval,
                    }

                    detected_correlations.append(corr_record)

                    # Store in correlation dict
                    corr_key = f"{key1}:{key2}"
                    self.correlations[corr_key] = corr_record

                    logger.info(
                        f"Detected {correlation_type} correlation ({correlation:.2f}) between {metric1['name']} and {metric2['name']}"
                    )

        return detected_correlations

    async def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.

        Returns:
            Report dictionary
        """
        now = datetime.now()

        # Create report
        report = {
            "timestamp": now.isoformat(),
            "anomalies": self.anomalies[-10:],  # Last 10 anomalies
            "trends": list(self.trends.values()),
            "correlations": list(self.correlations.values()),
            "summary": {
                "anomaly_count": len(self.anomalies),
                "trend_count": len(self.trends),
                "correlation_count": len(self.correlations),
            },
            "insights": [],
        }

        # Generate insights
        if self.anomalies:
            # Group anomalies by metric
            anomalies_by_metric = {}
            for anomaly in self.anomalies:
                metric_name = anomaly["metric_name"]
                if metric_name not in anomalies_by_metric:
                    anomalies_by_metric[metric_name] = []
                anomalies_by_metric[metric_name].append(anomaly)

            # Report frequent anomalies
            for metric_name, anomalies in anomalies_by_metric.items():
                if len(anomalies) >= 3:
                    report["insights"].append(
                        {
                            "type": "frequent_anomalies",
                            "metric_name": metric_name,
                            "count": len(anomalies),
                            "description": f"Frequent anomalies detected in {metric_name} ({len(anomalies)} instances)",
                        }
                    )

        # Add report to history
        self.reports.append(report)

        # Limit report history
        max_reports = self.config.get("max_reports_history", 10)
        if len(self.reports) > max_reports:
            self.reports = self.reports[-max_reports:]

        return report

    async def _run_periodic_analysis(self):
        """Run periodic analysis tasks."""
        logger.info("Starting periodic metrics analysis")

        while True:
            try:
                # Wait for next analysis time
                await asyncio.sleep(self.analysis_interval)

                # In a real implementation, this would get metrics from collector
                # metrics = await metrics_collector.get_historical_metrics("medium")

                # For this example, use empty metrics
                metrics = {}

                # Run analysis
                await self.detect_anomalies(metrics)
                await self.analyze_trends(metrics)
                await self.analyze_correlations(metrics)

                # Generate report
                await self.generate_report()

            except Exception as e:
                logger.error(f"Error in periodic metrics analysis: {e}")

    def _linear_regression(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """
        Perform linear regression.

        Args:
            x: X values
            y: Y values

        Returns:
            Tuple of (slope, intercept)
        """
        # Linear regression: y = slope * x + intercept
        n = len(x)
        if n == 0:
            return 0, 0

        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x_squared = np.sum(x * x)
        sum_xy = np.sum(x * y)

        # Calculate slope
        denominator = n * sum_x_squared - sum_x**2
        if denominator == 0:
            return 0, np.mean(y)

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Calculate intercept
        intercept = (sum_y - slope * sum_x) / n

        return slope, intercept

    def _correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Pearson correlation coefficient.

        Args:
            x: X values
            y: Y values

        Returns:
            Correlation coefficient
        """
        # Calculate means
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        # Calculate deviations
        dev_x = x - mean_x
        dev_y = y - mean_y

        # Calculate covariance and variances
        cov_xy = np.sum(dev_x * dev_y)
        var_x = np.sum(dev_x**2)
        var_y = np.sum(dev_y**2)

        # Calculate correlation
        denominator = np.sqrt(var_x * var_y)
        if denominator == 0:
            return 0

        correlation = cov_xy / denominator

        return correlation
