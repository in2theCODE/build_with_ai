"""
Configuration for the Metrics Node service.

This module provides configuration management for
the Metrics Node service.
"""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables and files.

    Returns:
        Dictionary of configuration values
    """
    # Default configuration
    config = {
        # Service configuration
        "service_name": "metrics_node",
        "service_version": "1.0.0",
        # Metrics configuration
        "max_values_per_metric": 1000,
        "max_anomalies_history": 100,
        "max_reports_history": 10,
        # Analysis configuration
        "analysis_interval": 300,  # 5 minutes
        "anomaly_threshold": 3.0,  # 3 standard deviations
        "trend_min_periods": 3,
        "correlation_threshold": 0.7,
        # API configuration
        "api_host": "0.0.0.0",
        "api_port": 8084,
        # Logging configuration
        "log_level": "INFO",
        # Pulsar configuration
        "pulsar_service_url": os.environ.get("PULSAR_SERVICE_URL", "pulsar://pulsar:6650"),
        "pulsar_admin_url": os.environ.get("PULSAR_ADMIN_URL", "http://pulsar:8080"),
        "pulsar_subscription_name": "metrics-node",
        # Health monitoring
        "health_check_interval": 60,  # 1 minute
        "critical_components": ["metrics_collector", "metrics_analyzer"],
    }

    # Load from environment
    for key in config:
        env_key = key.upper()
        if env_key in os.environ:
            # Handle type conversion
            if isinstance(config[key], int):
                config[key] = int(os.environ[env_key])
            elif isinstance(config[key], float):
                config[key] = float(os.environ[env_key])
            elif isinstance(config[key], bool):
                config[key] = os.environ[env_key].lower() in ("true", "yes", "1")
            else:
                config[key] = os.environ[env_key]

    # Load from config file if specified
    config_file = os.environ.get("CONFIG_FILE")
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                file_config = json.load(f)
                config.update(file_config)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")

    # Log configuration (excluding sensitive values)
    safe_config = config.copy()
    for key in safe_config:
        if "password" in key.lower() or "secret" in key.lower() or "key" in key.lower():
            safe_config[key] = "******"

    logger.info(f"Configuration: {safe_config}")

    return config
