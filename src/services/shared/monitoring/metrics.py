# src/shared/metrics.py

import functools
import time
import asyncio
import logging
from typing import Callable, Any, Optional

# Import your existing metrics collector
from metrics_collector import MetricsCollector

# Global metrics collector instance that can be set by the application
_metrics_collector: Optional[MetricsCollector] = None


def set_metrics_collector(collector: MetricsCollector):
    """Set the global metrics collector instance."""
    global _metrics_collector
    _metrics_collector = collector


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the global metrics collector instance."""
    return _metrics_collector


def track_inference_time(func):
    """
    Decorator to track the execution time of inference functions.
    Works with both synchronous and asynchronous functions.

    Usage:

    @track_inference_time
    def my_function():
        ...

    @track_inference_time
    async def my_async_function():
        ...
    """

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        component_name = func.__module__.split('.')[-1]
        function_name = func.__name__

        # Start timer
        start_time = time.time()

        try:
            # Execute the function
            result = func(*args, **kwargs)
            _record_success(component_name, function_name, start_time)
            return result
        except Exception as e:
            _record_error(component_name, function_name, start_time, e)
            raise

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        component_name = func.__module__.split('.')[-1]
        function_name = func.__name__

        # Start timer
        start_time = time.time()

        try:
            # Execute the async function
            result = await func(*args, **kwargs)
            _record_success(component_name, function_name, start_time)
            return result
        except Exception as e:
            _record_error(component_name, function_name, start_time, e)
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def _record_success(component_name: str, function_name: str, start_time: float):
    """Record successful execution metrics."""
    duration = time.time() - start_time

    # Log the execution time
    logging.debug(f"{component_name}.{function_name} completed in {duration:.4f}s")

    # Record in Prometheus metrics if available
    global _metrics_collector
    if _metrics_collector:
        # Use your existing metrics collector to record the time
        # This assumes your metrics collector has appropriate methods
        try:
            _metrics_collector.record_request(status="success", strategy=function_name)

            # If you have a histogram for function duration
            timer = _metrics_collector.request_duration.labels(
                component=component_name,
                strategy=function_name
            )
            if hasattr(timer, 'observe'):
                timer.observe(duration)
        except Exception as e:
            logging.error(f"Error recording metrics: {e}")


def _record_error(component_name: str, function_name: str, start_time: float, error: Exception):
    """Record error metrics."""
    duration = time.time() - start_time

    # Log the error
    logging.error(f"{component_name}.{function_name} failed after {duration:.4f}s: {error}")

    # Record in Prometheus metrics if available
    global _metrics_collector
    if _metrics_collector:
        try:
            _metrics_collector.record_request(status="error", strategy=function_name)
            _metrics_collector.record_error(error_type=error.__class__.__name__)
        except Exception as e:
            logging.error(f"Error recording metrics: {e}")