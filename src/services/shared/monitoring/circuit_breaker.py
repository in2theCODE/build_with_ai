#!/usr/bin/env python3
"""
Circuit Breaker implementation for the Program Synthesis System.

This module implements the Circuit Breaker pattern to prevent cascading failures
in a distributed microservices architecture. It provides a mechanism to detect
failures and prevent operation when the system is not functioning correctly.
"""

import asyncio
import logging
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

# Import metrics collector if available
try:
    from src.shared.metrics.metrics_collector import MetricsCollector
except ImportError:
    MetricsCollector = None

# Type variables for function decorators
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

# Configure logging
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failure detected, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Exception raised when the circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit Breaker implementation to prevent cascading failures.

    This class implements the Circuit Breaker pattern, which prevents a service
    from repeatedly trying to execute an operation that's likely to fail, allowing
    it to continue operating without waiting for the failing service to recover.
    """

    def __init__(
            self,
            name: str,
            failure_threshold: int = 5,
            reset_timeout: float = 30.0,
            half_open_max_requests: int = 2,
            metrics_collector: Optional[Any] = None
    ):
        """
        Initialize the circuit breaker.

        Args:
            name: Name of the circuit breaker (used for logging and metrics)
            failure_threshold: Number of consecutive failures before opening circuit
            reset_timeout: Time in seconds before trying to reset to half-open
            half_open_max_requests: Max requests to allow in half-open state
            metrics_collector: Optional metrics collector for instrumentation
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_requests = half_open_max_requests
        self.metrics_collector = metrics_collector

        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = time.time()
        self.half_open_requests = 0

        # Concurrency control
        self._lock = asyncio.Lock()

        logger.info(f"Circuit breaker '{name}' initialized with threshold {failure_threshold}")

        # Register with metrics if available
        if self.metrics_collector and hasattr(self.metrics_collector, 'component_up'):
            self.metrics_collector.component_up.labels(
                component=self.name).set(1)

    async def _state_transition(self, new_state: CircuitState) -> None:
        """
        Transition to a new state with proper logging and metrics.

        Args:
            new_state: The new circuit state
        """
        old_state = self.state
        self.state = new_state

        logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")

        # Record state change in metrics if available
        if self.metrics_collector:
            try:
                # Record event if the metrics collector has that method
                if hasattr(self.metrics_collector, 'record_event_emitted'):
                    self.metrics_collector.record_event_emitted(
                        f"circuit_breaker_{new_state.value}")

                # Update component status based on circuit state
                if hasattr(self.metrics_collector, 'set_component_up'):
                    self.metrics_collector.set_component_up(new_state != CircuitState.OPEN)

            except Exception as e:
                logger.error(f"Error recording circuit breaker metrics: {e}")

    async def success(self) -> None:
        """Record a successful operation."""
        async with self._lock:
            self.last_success_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.half_open_requests += 1

                # If we've had enough successful requests in half-open state,
                # transition back to closed
                if self.half_open_requests >= self.half_open_max_requests:
                    await self._state_transition(CircuitState.CLOSED)
                    self.failure_count = 0
                    self.half_open_requests = 0

            # In closed state, reset failure count on success
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    async def failure(self) -> None:
        """Record a failed operation."""
        async with self._lock:
            self.last_failure_time = time.time()

            if self.state == CircuitState.CLOSED:
                self.failure_count += 1

                # If we've reached the threshold, open the circuit
                if self.failure_count >= self.failure_threshold:
                    await self._state_transition(CircuitState.OPEN)

            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit again
                await self._state_transition(CircuitState.OPEN)
                self.half_open_requests = 0

    async def check_state(self) -> None:
        """
        Check and potentially update the circuit state.

        Raises:
            CircuitBreakerError: If the circuit is open
        """
        async with self._lock:
            now = time.time()

            # Check if we should try to recover from open state
            if self.state == CircuitState.OPEN:
                if now - self.last_failure_time >= self.reset_timeout:
                    # Transition to half-open to test the service
                    await self._state_transition(CircuitState.HALF_OPEN)
                    self.half_open_requests = 0
                else:
                    # Still in timeout period
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' is open. Service unavailable."
                    )

            # In half-open state, only allow limited requests
            elif self.state == CircuitState.HALF_OPEN:
                if self.half_open_requests >= self.half_open_max_requests:
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' is half-open and at capacity."
                    )

    def is_open(self) -> bool:
        """
        Check if the circuit is currently open.

        Returns:
            True if the circuit is in OPEN state, False otherwise
        """
        return self.state == CircuitState.OPEN

    def get_state(self) -> CircuitState:
        """
        Get the current circuit state.

        Returns:
            Current CircuitState
        """
        return self.state

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get circuit breaker metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "half_open_requests": self.half_open_requests
        }

    def __str__(self) -> str:
        """String representation of the circuit breaker."""
        return f"CircuitBreaker({self.name}, state={self.state.value}, failures={self.failure_count})"


def circuit_breaker(breaker: CircuitBreaker) -> Callable[[F], F]:
    """
    Decorator to apply circuit breaker pattern to synchronous functions.

    Args:
        breaker: The circuit breaker instance to use

    Returns:
        Decorated function with circuit breaker protection
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Check circuit state
            try:
                loop.run_until_complete(breaker.check_state())
            except CircuitBreakerError as e:
                logger.warning(f"Circuit breaker prevented call to {func.__name__}: {e}")
                raise

            # Execute function
            try:
                result = func(*args, **kwargs)
                loop.run_until_complete(breaker.success())
                return result
            except Exception as e:
                loop.run_until_complete(breaker.failure())
                logger.error(f"Circuit breaker recorded failure in {func.__name__}: {e}")
                raise

        return cast(F, wrapper)

    return decorator


def async_circuit_breaker(breaker: CircuitBreaker) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to apply circuit breaker pattern to asynchronous functions.

    Args:
        breaker: The circuit breaker instance to use

    Returns:
        Decorated async function with circuit breaker protection
    """

    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check circuit state
            await breaker.check_state()

            # Execute function
            try:
                result = await func(*args, **kwargs)
                await breaker.success()
                return result
            except Exception as e:
                await breaker.failure()
                logger.error(f"Circuit breaker recorded failure in {func.__name__}: {e}")
                raise

        return cast(AsyncF, wrapper)

    return decorator


# Convenience function to create circuit breaker from config
def create_circuit_breaker_from_config(
        config: Dict[str, Any],
        name: str,
        metrics_collector: Optional[Any] = None
) -> CircuitBreaker:
    """
    Create a circuit breaker instance from configuration.

    Args:
        config: Configuration dictionary with circuit breaker settings
        name: Name for the circuit breaker
        metrics_collector: Optional metrics collector

    Returns:
        Configured CircuitBreaker instance
    """
    cb_config = config.get("circuit_breaker", {})
    return CircuitBreaker(
        name=name,
        failure_threshold=cb_config.get("failure_threshold", 5),
        reset_timeout=cb_config.get("reset_timeout_seconds", 30.0),
        half_open_max_requests=cb_config.get("half_open_max_requests", 2),
        metrics_collector=metrics_collector
    )