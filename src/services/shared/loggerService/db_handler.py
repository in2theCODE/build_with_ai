import asyncio
import json
import logging
import os
import pulsar
from datetime import datetime


class PulsarLogHandler(logging.Handler):
    """Logging handler that publishes log messages to Pulsar."""

    def __init__(self, service_url=None, topic=None):
        """Initialize the handler."""
        super().__init__()

        # Set default values from environment variables if not provided
        self.service_url = service_url or os.environ.get(
            "PULSAR_SERVICE_URL", "pulsar://pulsar:6650"
        )
        self.topic = topic or os.environ.get("PULSAR_TOPIC", "log.entry")
        self.container_name = os.environ.get("CONTAINER_NAME", "unknown")

        # Initialize Pulsar client
        self.client = None
        self.producer = None
        self._connect()

    def _connect(self):
        """Connect to Pulsar."""
        try:
            self.client = pulsar.Client(self.service_url)
            self.producer = self.client.create_producer(self.topic)
            print(f"Connected to Pulsar at {self.service_url}, topic: {self.topic}")
        except Exception as e:
            print(f"Error connecting to Pulsar: {e}")
            self.client = None
            self.producer = None

    def emit(self, record):
        """Emit a log record to Pulsar."""
        if not self.client or not self.producer:
            try:
                self._connect()
            except Exception:
                # Can't log if we can't connect
                return

        try:
            # Format the record
            log_data = self.format(record)

            # Add container name
            log_data["container"] = self.container_name

            # Convert to JSON
            json_data = json.dumps(log_data)

            # Send to Pulsar
            self.producer.send(json_data.encode("utf-8"))
        except Exception as e:
            print(f"Error sending log to Pulsar: {e}")

    def format(self, record):
        """Format the log record as a dictionary."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "level_number": record.levelno,
            "logger": record.name,
            "message": self.format_message(record),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        }

        # Add exception info if available
        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            log_data["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra attributes
        for key, value in record.__dict__.items():
            if key not in (
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "id",
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
            ):
                log_data[key] = value

        return log_data

    def format_message(self, record):
        """Format the log message."""
        if isinstance(record.msg, str):
            return record.getMessage()
        else:
            # Handle non-string messages
            return str(record.msg)

    def close(self):
        """Close the handler."""
        if self.producer:
            self.producer.close()
        if self.client:
            self.client.close()
        super().close()
