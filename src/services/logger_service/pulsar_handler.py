"""
PulsarLogHandler for sending log records to Apache Pulsar.
"""

import json
import logging
import os
import sys
from datetime import datetime

try:
    import pulsar
except ImportError:
    # This allows the module to be imported even if pulsar is not installed,
    # for example, in environments where this handler isn't used.
    # An error will be raised if the handler is instantiated without pulsar.
    pulsar = None


class PulsarLogHandler(logging.Handler):
    """
    A logging handler that publishes log messages to an Apache Pulsar topic.
    """

    def __init__(self, service_url=None, topic=None, service_name=None):
        """
        Initialize the handler.

        Args:
            service_url (str, optional): The Pulsar service URL.
                Defaults to PULSAR_SERVICE_URL env var or "pulsar://pulsar:6650".
            topic (str, optional): The Pulsar topic to publish logs to.
                Defaults to PULSAR_LOG_TOPIC env var or "log.entry".
            service_name (str, optional): Name of the service generating the logs.
                Defaults to SERVICE_NAME env var or "unknown_service".
        """
        super().__init__()

        if pulsar is None:
            raise ImportError(
                "The 'pulsar' library is not installed. "
                "Please install it to use PulsarLogHandler: pip install pulsar-client"
            )

        self.service_url = service_url or os.environ.get("PULSAR_SERVICE_URL", "pulsar://pulsar:6650")
        self.topic = topic or os.environ.get("PULSAR_LOG_TOPIC", "log.entry")
        self.service_name = service_name or os.environ.get(
            "SERVICE_NAME", "unknown_service"
        )  # Changed from CONTAINER_NAME

        self.client = None
        self.producer = None
        self._connect()

    def _connect(self):
        """
        Connect to Pulsar.
        Retries connection if it fails initially.
        """
        if self.client and self.producer:
            return True

        try:
            self.client = pulsar.Client(self.service_url)
            self.producer = self.client.create_producer(self.topic)
            # Using sys.stdout for this initial connection message as logger might not be fully set up
            sys.stdout.write(
                f"PulsarLogHandler: Connected to Pulsar at {self.service_url}, topic: {self.topic} for service: {self.service_name}\n"
            )
            return True
        except Exception as e:
            sys.stderr.write(f"PulsarLogHandler: Error connecting to Pulsar for service {self.service_name}: {e}\n")
            self.client = None
            self.producer = None
            return False

    def emit(self, record: logging.LogRecord):
        """
        Emit a log record to Pulsar.
        Attempts to reconnect if the producer is not available.
        """
        if not self.producer:
            if not self._connect():
                self.handleError(record)  # Could not connect, report error
                return

        try:
            log_data = self.format(record)  # This calls our custom format method
            json_data = json.dumps(log_data)
            self.producer.send(json_data.encode("utf-8"))
        except Exception:
            self.handleError(record)  # Report error during send

    def format(self, record: logging.LogRecord) -> dict:
        """
        Format the log record as a dictionary.
        This method is called by emit().
        """
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "service_name": self.service_name,
            "level": record.levelname,
            "level_number": record.levelno,
            "logger_name": record.name,
            "message": record.getMessage(),  # Ensures proper formatting of %-style messages
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        }

        if record.exc_info:
            log_entry["exception"] = logging.Formatter.formatException(self, record.exc_info)

        if hasattr(record, "stack_info") and record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)

        # Add extra attributes passed in logging calls
        # e.g., logger.info("message", extra={"key": "value"})
        standard_attrs = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in standard_attrs and key not in log_entry:
                    log_entry[key] = str(value)  # Ensure serializable

        return log_entry

    def close(self):
        """
        Close the handler and the Pulsar client/producer.
        """
        try:
            if self.producer:
                self.producer.close()
            if self.client:
                self.client.close()
        except Exception as e:
            sys.stderr.write(f"PulsarLogHandler: Error closing Pulsar connections: {e}\n")
        finally:
            self.producer = None
            self.client = None
            super().close()
