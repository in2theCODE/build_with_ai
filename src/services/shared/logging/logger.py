#!/usr/bin/env python3
"""
Advanced logging utility for the Program Synthesis System.

This module provides a sophisticated logging system with structured logging,
multi-tenant support, log rotation, and performance monitoring capabilities.
"""
import contextlib
from contextvars import ContextVar
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import datetime
from enum import Enum
import functools
import inspect
import json
import logging
import logging.config
import logging.handlers
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TextIO,
    TypeVar,
    Union,
)
import uuid


try:
    import colorama
    from colorama import Back
    from colorama import Fore
    from colorama import Style

    COLORS_AVAILABLE = True
    colorama.init()
except ImportError:
    COLORS_AVAILABLE = False


# Type variables for function decorators
F = TypeVar("F", bound=Callable[..., Any])


# Context variables for storing request context
current_tenant_id: ContextVar[str] = ContextVar("current_tenant_id", default="")
current_request_id: ContextVar[str] = ContextVar("current_request_id", default="")
current_user_id: ContextVar[str] = ContextVar("current_user_id", default="")
current_correlation_id: ContextVar[str] = ContextVar("current_correlation_id", default="")
current_session_id: ContextVar[str] = ContextVar("current_session_id", default="")
current_operation: ContextVar[str] = ContextVar("current_operation", default="")


class LogLevel(Enum):
    """Custom log levels with additional context."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    SUCCESS = 25  # Between INFO and WARNING
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    SECURITY = 35  # Between WARNING and ERROR
    PERFORMANCE = 15  # Between DEBUG and INFO
    TRACE = 5  # Below DEBUG


@dataclass
class LogContext:
    """Context information for structured logging."""

    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    tenant_id: str = field(default_factory=lambda: current_tenant_id.get())
    request_id: str = field(default_factory=lambda: current_request_id.get())
    user_id: str = field(default_factory=lambda: current_user_id.get())
    correlation_id: str = field(default_factory=lambda: current_correlation_id.get())
    session_id: str = field(default_factory=lambda: current_session_id.get())
    operation: str = field(default_factory=lambda: current_operation.get())
    component: str = ""
    file: str = ""
    line: int = 0
    function: str = ""
    thread_id: int = field(default_factory=lambda: threading.get_ident())
    process_id: int = field(default_factory=lambda: os.getpid())
    extra: Dict[str, Any] = field(default_factory=dict)


class StructuredLogRecord(logging.LogRecord):
    """Extended LogRecord class for structured logging."""

    def __init__(self, *args, **kwargs):
        """Initialize with standard LogRecord arguments."""
        super().__init__(*args, **kwargs)

        # Add context information
        self.tenant_id = current_tenant_id.get()
        self.request_id = current_request_id.get()
        self.user_id = current_user_id.get()
        self.correlation_id = current_correlation_id.get()
        self.session_id = current_session_id.get()
        self.operation = current_operation.get()

        # Additional structured fields will be stored here
        self.structured_data = {}


class StructuredLogger(logging.Logger):
    """Logger that supports structured logging with context."""

    def makeRecord(
        self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None
    ):
        """Create a LogRecord with structured data support."""
        # Create record with the extended class
        record = StructuredLogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)

        # Add extra fields
        if extra is not None:
            for key, value in extra.items():
                if key == "structured_data" and isinstance(value, dict):
                    # Special handling for structured data
                    record.structured_data = value
                elif key != "message":
                    setattr(record, key, value)

        return record

    def struct(self, level: int, msg: str, structured_data: Dict[str, Any], *args, **kwargs):
        """Log a message with structured data."""
        if not self.isEnabledFor(level):
            return

        # Ensure we have an extra dict
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        # Add structured data
        kwargs["extra"]["structured_data"] = structured_data

        # Log the message
        self._log(level, msg, args, **kwargs)

    def success(self, msg: str, *args, **kwargs):
        """Log a success message."""
        self.log(LogLevel.SUCCESS.value, msg, *args, **kwargs)

    def security(self, msg: str, *args, **kwargs):
        """Log a security message."""
        self.log(LogLevel.SECURITY.value, msg, *args, **kwargs)

    def performance(self, msg: str, *args, **kwargs):
        """Log a performance message."""
        self.log(LogLevel.PERFORMANCE.value, msg, *args, **kwargs)

    def trace(self, msg: str, *args, **kwargs):
        """Log a trace message."""
        self.log(LogLevel.TRACE.value, msg, *args, **kwargs)


class MultiTenantFilter(logging.Filter):
    """Filter that allows controlling log output by tenant."""

    def __init__(
        self,
        tenant_whitelist: Optional[Set[str]] = None,
        tenant_blacklist: Optional[Set[str]] = None,
    ):
        """
        Initialize the filter.

        Args:
            tenant_whitelist: Set of tenant IDs to allow (None = allow all)
            tenant_blacklist: Set of tenant IDs to block (None = block none)
        """
        super().__init__()
        self.tenant_whitelist = tenant_whitelist
        self.tenant_blacklist = tenant_blacklist

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on tenant ID."""
        # Get tenant ID from record
        tenant_id = getattr(record, "tenant_id", "")

        # If no tenant ID, allow the record
        if not tenant_id:
            return True

        # Check blacklist first
        if self.tenant_blacklist and tenant_id in self.tenant_blacklist:
            return False

        # Then check whitelist
        if self.tenant_whitelist and tenant_id not in self.tenant_whitelist:
            return False

        # If passed all checks, allow the record
        return True


class ColorFormatter(logging.Formatter):
    """Log formatter that adds ANSI colors based on log level."""

    COLORS = {
        LogLevel.DEBUG.value: Style.DIM + Fore.CYAN,
        LogLevel.INFO.value: Fore.WHITE,
        LogLevel.SUCCESS.value: Fore.GREEN,
        LogLevel.WARNING.value: Fore.YELLOW,
        LogLevel.ERROR.value: Fore.RED,
        LogLevel.CRITICAL.value: Fore.RED + Style.BRIGHT,
        LogLevel.SECURITY.value: Fore.MAGENTA,
        LogLevel.PERFORMANCE.value: Fore.CYAN,
        LogLevel.TRACE.value: Style.DIM + Fore.BLUE,
    }

    RESET = Style.RESET_ALL

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        validate: bool = True,
        use_colors: bool = True,
    ):
        """
        Initialize the formatter.

        Args:
            fmt: Format string
            datefmt: Date format string
            style: Format style
            validate: Whether to validate the format string
            use_colors: Whether to use colors (will be disabled if colorama is not available)
        """
        super().__init__(fmt, datefmt, style, validate)
        self.use_colors = use_colors and COLORS_AVAILABLE

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Format the record first
        formatted = super().format(record)

        # Add colors if enabled
        if self.use_colors:
            # Get color for this level
            color = self.COLORS.get(record.levelno, Fore.WHITE)

            # Add color to formatted string
            formatted = color + formatted + self.RESET

        return formatted


class JsonFormatter(logging.Formatter):
    """Log formatter that outputs structured JSON."""

    def __init__(self, indent: Optional[int] = None, include_stack_trace: bool = True):
        """
        Initialize the formatter.

        Args:
            indent: JSON indentation level (None for compact output)
            include_stack_trace: Whether to include stack traces for exceptions
        """
        super().__init__()
        self.indent = indent
        self.include_stack_trace = include_stack_trace

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Create a dictionary with standard fields
        log_data = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "level_number": record.levelno,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        }

        # Add context fields
        for field in [
            "tenant_id",
            "request_id",
            "user_id",
            "correlation_id",
            "session_id",
            "operation",
        ]:
            if hasattr(record, field):
                value = getattr(record, field)
                if value:  # Only add non-empty values
                    log_data[field] = value

        # Add structured data if available
        if hasattr(record, "structured_data") and record.structured_data:
            log_data["data"] = record.structured_data

        # Add exception info if available
        if record.exc_info and self.include_stack_trace:
            exc_type, exc_value, exc_traceback = record.exc_info
            log_data["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": traceback.format_exception(exc_type, exc_value, exc_traceback),
            }

        # Convert to JSON
        return json.dumps(log_data, indent=self.indent)


class PerformanceHandler(logging.Handler):
    """Handler that collects performance metrics from logs."""

    def __init__(self, level=LogLevel.PERFORMANCE.value):
        """Initialize the handler with the performance log level."""
        super().__init__(level)
        self.metrics = {}
        self.lock = threading.RLock()

    def emit(self, record: logging.LogRecord) -> None:
        """Process the log record and extract performance metrics."""
        # Only process performance log records
        if record.levelno != LogLevel.PERFORMANCE.value:
            return

        # Extract structured data
        data = getattr(record, "structured_data", {})

        # Check for required fields
        if "metric" not in data or "value" not in data:
            return

        metric = data["metric"]
        value = data["value"]

        # Additional metadata
        tags = data.get("tags", {})
        tenant_id = getattr(record, "tenant_id", "")

        # Create a unique key for this metric
        key_parts = [metric]
        if tenant_id:
            key_parts.append(f"tenant:{tenant_id}")
        for tag_name, tag_value in sorted(tags.items()):
            key_parts.append(f"{tag_name}:{tag_value}")

        key = "|".join(key_parts)

        # Update metrics
        with self.lock:
            if key not in self.metrics:
                self.metrics[key] = {
                    "metric": metric,
                    "tenant_id": tenant_id,
                    "tags": tags,
                    "count": 0,
                    "sum": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                }

            self.metrics[key]["count"] += 1
            self.metrics[key]["sum"] += value
            self.metrics[key]["min"] = min(self.metrics[key]["min"], value)
            self.metrics[key]["max"] = max(self.metrics[key]["max"], value)

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all collected metrics."""
        with self.lock:
            # Make a deep copy to avoid external modification
            return {k: v.copy() for k, v in self.metrics.items()}

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self.metrics = {}


class LoggerFactory:
    """Factory for creating loggers with consistent configuration."""

    def __init__(self):
        """Initialize the factory with default settings."""
        # Register custom log levels
        logging.addLevelName(LogLevel.SUCCESS.value, "SUCCESS")
        logging.addLevelName(LogLevel.SECURITY.value, "SECURITY")
        logging.addLevelName(LogLevel.PERFORMANCE.value, "PERFORMANCE")
        logging.addLevelName(LogLevel.TRACE.value, "TRACE")

        # Register custom logger class
        logging.setLoggerClass(StructuredLogger)

        # Default configuration
        self.default_level = logging.INFO
        self.log_directory = "logs"
        self.log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        self.use_json = False
        self.use_colors = COLORS_AVAILABLE
        self.multi_tenant = False
        self.collect_performance = False

        # Performance handler
        self.performance_handler = None

        # Flag to track if configured
        self.is_configured = False

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the logging system.

        Args:
            config: Configuration dictionary
        """
        # Extract configuration
        self.default_level = config.get("level", logging.INFO)
        self.log_directory = config.get("directory", "logs")
        self.log_format = config.get("format", self.log_format)
        self.use_json = config.get("use_json", False)
        self.use_colors = config.get("use_colors", COLORS_AVAILABLE) and COLORS_AVAILABLE
        self.multi_tenant = config.get("multi_tenant", False)
        self.collect_performance = config.get("collect_performance", False)

        # Create log directory if it doesn't exist
        os.makedirs(self.log_directory, exist_ok=True)

        # Create handlers
        handlers = {}

        # Console handler
        if config.get("console", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_level = config.get("console_level", self.default_level)
            console_handler.setLevel(console_level)

            # Add formatter
            if self.use_json:
                console_handler.setFormatter(JsonFormatter())
            else:
                console_format = config.get("console_format", self.log_format)
                console_formatter = (
                    ColorFormatter(console_format)
                    if self.use_colors
                    else logging.Formatter(console_format)
                )
                console_handler.setFormatter(console_formatter)

            handlers["console"] = console_handler

        # File handler
        if config.get("file", True):
            # Determine log file path
            log_file = config.get("log_file")
            if not log_file:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(self.log_directory, f"app_{timestamp}.log")

            # Create handler with rotation
            max_bytes = config.get("max_file_size", 10 * 1024 * 1024)  # 10 MB
            backup_count = config.get("backup_count", 5)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )

            file_level = config.get("file_level", self.default_level)
            file_handler.setLevel(file_level)

            # Add formatter
            if self.use_json:
                file_handler.setFormatter(JsonFormatter())
            else:
                file_format = config.get("file_format", self.log_format)
                file_formatter = logging.Formatter(file_format)
                file_handler.setFormatter(file_formatter)

            handlers["file"] = file_handler

        # Performance handler
        if self.collect_performance:
            self.performance_handler = PerformanceHandler()
            handlers["performance"] = self.performance_handler

        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.default_level)

        # Remove existing handlers
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

        # Add the new handlers
        for handler in handlers.values():
            root_logger.addHandler(handler)

        # Add multi-tenant filter if enabled
        if self.multi_tenant:
            tenant_filter = MultiTenantFilter()
            for handler in handlers.values():
                handler.addFilter(tenant_filter)

        self.is_configured = True

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the given name.

        Args:
            name: Logger name

        Returns:
            Configured logger
        """
        # Configure with defaults if not already configured
        if not self.is_configured:
            self.configure({})

        return logging.getLogger(name)

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics collected by the performance handler.

        Returns:
            Dictionary of metrics
        """
        if self.performance_handler:
            return self.performance_handler.get_metrics()
        return {}

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        if self.performance_handler:
            self.performance_handler.reset_metrics()


# Global logger factory
logger_factory = LoggerFactory()


def configure_logging(config: Dict[str, Any]) -> None:
    """
    Configure the logging system.

    Args:
        config: Configuration dictionary
    """
    logger_factory.configure(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    return logger_factory.get_logger(name)


def get_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Get performance metrics.

    Returns:
        Dictionary of metrics
    """
    return logger_factory.get_metrics()


def reset_metrics() -> None:
    """Reset performance metrics."""
    logger_factory.reset_metrics()


@contextlib.contextmanager
def log_context(**kwargs):
    """
    Context manager for setting log context variables.

    Args:
        **kwargs: Context variables to set
    """
    # Store current values
    old_values = {}
    tokens = {}

    # Set new values
    for key, value in kwargs.items():
        if key == "tenant_id":
            old_values[key] = current_tenant_id.get()
            tokens[key] = current_tenant_id.set(value)
        elif key == "request_id":
            old_values[key] = current_request_id.get()
            tokens[key] = current_request_id.set(value)
        elif key == "user_id":
            old_values[key] = current_user_id.get()
            tokens[key] = current_user_id.set(value)
        elif key == "correlation_id":
            old_values[key] = current_correlation_id.get()
            tokens[key] = current_correlation_id.set(value)
        elif key == "session_id":
            old_values[key] = current_session_id.get()
            tokens[key] = current_session_id.set(value)
        elif key == "operation":
            old_values[key] = current_operation.get()
            tokens[key] = current_operation.set(value)

    try:
        yield
    finally:
        # Restore old values
        for key in kwargs:
            if key in tokens:
                if key == "tenant_id":
                    current_tenant_id.reset(tokens[key])
                elif key == "request_id":
                    current_request_id.reset(tokens[key])
                elif key == "user_id":
                    current_user_id.reset(tokens[key])
                elif key == "correlation_id":
                    current_correlation_id.reset(tokens[key])
                elif key == "session_id":
                    current_session_id.reset(tokens[key])
                elif key == "operation":
                    current_operation.reset(tokens[key])


def log_execution_time(logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """
    Decorator to log the execution time of a function.

    Args:
        logger: Logger to use (if None, a logger will be created based on the function's module)
        level: Log level
    """

    def decorator(func: F) -> F:
        # Get function metadata
        module_name = func.__module__
        function_name = func.__qualname__

        # Create logger if not provided
        nonlocal logger
        if logger is None:
            logger = get_logger(module_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Log start
            logger.log(level, f"Starting {function_name}")

            try:
                # Call function
                result = func(*args, **kwargs)

                # Log end and execution time
                end_time = time.time()
                execution_time = end_time - start_time

                # Use performance level if it's a StructuredLogger
                if isinstance(logger, StructuredLogger):
                    logger.performance(
                        f"Completed {function_name}",
                        extra={
                            "structured_data": {
                                "metric": "execution_time",
                                "value": execution_time,
                                "tags": {"function": function_name, "module": module_name},
                            }
                        },
                    )
                else:
                    logger.log(level, f"Completed {function_name} in {execution_time:.3f} seconds")

                return result

            except Exception as e:
                # Log exception
                end_time = time.time()
                execution_time = end_time - start_time

                logger.exception(
                    f"Error in {function_name} after {execution_time:.3f} seconds: {e}"
                )

                # Re-raise
                raise

        return wrapper  # type: ignore

    return decorator


def log_method_calls(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """
    Decorator to log all method calls in a class.

    Args:
        logger: Logger to use (if None, a logger will be created for each method)
        level: Log level
    """

    def decorator(cls):
        # Get class metadata
        module_name = cls.__module__
        class_name = cls.__qualname__

        # Create class logger if not provided
        class_logger = logger
        if class_logger is None:
            class_logger = get_logger(f"{module_name}.{class_name}")

        # For each method in the class
        for attr_name, attr_value in cls.__dict__.items():
            # Skip special methods and non-callables
            if attr_name.startswith("__") or not callable(attr_value):
                continue

            # Get method
            method = attr_value

            # Define wrapper
            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                # Format arguments for logging
                args_str = ", ".join([repr(a) for a in args])
                kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                params_str = ", ".join(filter(None, [args_str, kwargs_str]))

                # Log method call
                class_logger.log(level, f"Calling {class_name}.{attr_name}({params_str})")

                # Call method
                result = method(self, *args, **kwargs)

                # Log method return
                class_logger.log(level, f"Returned from {class_name}.{attr_name}")

                return result

            # Replace method with wrapper
            setattr(cls, attr_name, wrapper)

        return cls

    return decorator


class RequestLogger:
    """Logger for tracking requests end-to-end."""

    def __init__(self, logger_name: str = "request"):
        """
        Initialize the request logger.

        Args:
            logger_name: Base logger name
        """
        self.logger = get_logger(logger_name)
        self.tenant_id = ""
        self.request_id = ""
        self.user_id = ""
        self.correlation_id = ""
        self.session_id = ""
        self.operation = ""

    def start_request(self, operation: str, **context):
        """
        Start tracking a request.

        Args:
            operation: Operation name
            **context: Additional context variables
        """
        # Generate a request ID if not provided
        self.request_id = context.get("request_id", str(uuid.uuid4()))

        # Store context
        self.tenant_id = context.get("tenant_id", "")
        self.user_id = context.get("user_id", "")
        self.correlation_id = context.get("correlation_id", "")
        self.session_id = context.get("session_id", "")
        self.operation = operation

        # Set context variables
        if self.tenant_id:
            current_tenant_id.set(self.tenant_id)
        if self.request_id:
            current_request_id.set(self.request_id)
        if self.user_id:
            current_user_id.set(self.user_id)
        if self.correlation_id:
            current_correlation_id.set(self.correlation_id)
        if self.session_id:
            current_session_id.set(self.session_id)
        if self.operation:
            current_operation.set(self.operation)

        # Log request start
        extra = {"structured_data": {**context, "event": "request_start"}}

        self.logger.info(f"Request started: {operation}", extra=extra)

    def end_request(self, status: str = "success", **context):
        """
        End tracking a request.

        Args:
            status: Request status
            **context: Additional context variables
        """
        # Log request end
        extra = {"structured_data": {**context, "event": "request_end", "status": status}}

        self.logger.info(f"Request completed: {self.operation} - {status}", extra=extra)

    def log(self, level: Union[int, str], msg: str, **context):
        """
        Log a message within the request context.

        Args:
            level: Log level
            msg: Message
            **context: Additional context variables
        """
        # Convert string levels to integers
        if isinstance(level, str):
            level_name = level.upper()
            level = (
                LogLevel[level_name].value if level_name in LogLevel.__members__ else logging.INFO
            )

        # Log message
        extra = {"structured_data": context} if context else None
        self.logger.log(level, msg, extra=extra)

    @contextlib.contextmanager
    def request_context(self, operation: str, **context):
        """
        Context manager for tracking a request.

        Args:
            operation: Operation name
            **context: Additional context variables
        """
        try:
            self.start_request(operation, **context)
            yield self
        except Exception as e:
            # Log exception
            self.logger.exception(f"Error during request: {e}")
            self.end_request(status="error", error=str(e))
            raise
        else:
            self.end_request(status="success")
