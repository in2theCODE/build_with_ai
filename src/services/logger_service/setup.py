"""
Utility for setting up application-wide logging.
Includes console logging and Pulsar logging.
"""

import logging
import os
import sys
from .pulsar_handler import PulsarLogHandler  # Relative import

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(service_name)s - %(message)s"


def setup_application_logging(service_name: str, log_level: str = None) -> logging.Logger:
    """
    Configures logging for an application.

    Sets up:
    1. A console handler (StreamHandler) for local viewing.
    2. A PulsarLogHandler to send logs to a central Pulsar topic.

    The root logger's level is set, and handlers are added to it.
    Named loggers will inherit this configuration.

    Args:
        service_name (str): The name of the service, used for identifying logs.
        log_level (str, optional): The logging level (e.g., "INFO", "DEBUG").
            Defaults to LOG_LEVEL env var or "INFO".

    Returns:
        logging.Logger: The configured root logger.
    """
    # Determine log level
    effective_log_level_str = (log_level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric_log_level = getattr(logging, effective_log_level_str, logging.INFO)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_log_level)

    # Clear any existing handlers on the root logger to avoid duplication
    # if this function is called multiple times (though it ideally shouldn't be).
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 1. Console Handler (for local development/stdout)
    console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    # Add a filter to inject service_name into records for the console formatter
    class ServiceNameFilter(logging.Filter):
        def __init__(self, service_name_to_inject):
            super().__init__()
            self.service_name_to_inject = service_name_to_inject

        def filter(self, record):
            record.service_name = self.service_name_to_inject
            return True

    console_handler.addFilter(ServiceNameFilter(service_name))
    root_logger.addHandler(console_handler)

    # 2. Pulsar Handler (for centralized logging)
    # PulsarLogHandler gets its Pulsar connection details (URL, topic)
    # from environment variables by default, or they can be passed to its constructor.
    # It also takes the service_name for tagging logs.
    try:
        pulsar_log_handler = PulsarLogHandler(service_name=service_name)
        # PulsarLogHandler does its own formatting into a dictionary,
        # so a logging.Formatter is not applied to it here for the Pulsar message content.
        root_logger.addHandler(pulsar_log_handler)
        logging.info(f"Pulsar logging configured for service: {service_name}. Log level: {effective_log_level_str}")
    except ImportError:
        logging.warning(f"Pulsar client library not found for service: {service_name}. Pulsar logging disabled.")
    except Exception as e:
        logging.error(
            f"Failed to initialize PulsarLogHandler for service {service_name}: {e}. Pulsar logging disabled."
        )

    # Note on Sentry:
    # Sentry integration is typically done by initializing the Sentry SDK,
    # often with its own DSN from environment variables.
    # e.g., import sentry_sdk; sentry_sdk.init(dsn=os.environ.get("SENTRY_DSN"), traces_sample_rate=1.0)
    # Sentry will automatically capture unhandled exceptions and can be configured
    # to capture logging messages as breadcrumbs or events.

    logging.info(
        f"Application logging setup complete for service: {service_name}. Log level: {effective_log_level_str}"
    )
    return root_logger
