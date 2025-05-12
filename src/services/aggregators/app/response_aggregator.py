import asyncio
import xyz
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union
from datetime import datetime, timedelta
from redis.asyncio import Redis
from motor.motor_asyncio import AsyncIOMotorClient

from src.services.shared.models import BaseComponent
from src.services.shared.models import BaseEvent, EventType
from src.services.shared.logging.logger import logger
from src.services.shared.monitoring.metrics_collector import MetricsCollector


class ResponseAggregator(BaseComponent):
    """
    Modern event-driven ResponseAggregator that aggregates responses from multiple services.

    This component:
    - Collects and aggregates responses from multiple services via events
    - Supports timeouts and partial responses
    - Provides configurable retry strategies
    - Implements the Circuit Breaker pattern for failing services
    - Includes comprehensive metrics and monitoring
    - Uses MongoDB for persistent storage and Redis for ephemeral state
    """

    get_logger = logger("ResponseAggregator")

    def __init__(self, **params):
        super().__init__(**params)

        # Configuration
        self.default_timeout = params.get("default_timeout", 60)  # Default timeout in seconds
        self.cleanup_interval = params.get("cleanup_interval", 30)  # Cleanup every 30 seconds
        self.max_retries = params.get("max_retries", 3)  # Maximum retry attempts
        self.retry_delay = params.get("retry_delay", 5)  # Seconds between retries
        self.circuit_breaker_threshold = params.get(
            "circuit_breaker_threshold", 5
        )  # Failures before circuit breaks
        self.circuit_reset_timeout = params.get(
            "circuit_reset_timeout", 60
        )  # Seconds before circuit resets

        # State
        self.running = False
        self.pending_requests = {}  # Tracks in-progress requests
        self.response_processors = {}  # Custom processors for event types
        self.circuit_breakers = {}  # Tracks service health

        # Initialize Redis for ephemeral state
        self.redis = Redis(
            host=params.get("redis_host", "response-aggregator-redis"),
            port=params.get("redis_port", 6379),
            db=params.get("redis_db", 0),
            decode_responses=True,
            password=params.get("redis_password", None),
        )

        # Initialize MongoDB for persistent storage
        mongodb_uri = params.get(
            "mongodb_uri",
            f"mongodb://admin:{params.get('mongodb_password', '')}@mongodb:27017/event_system?authSource=admin",
        )
        self.mongodb_client = AsyncIOMotorClient(mongodb_uri)
        self.db = self.mongodb_client[params.get("mongodb_database", "event_system")]
        self.responses_collection = self.db["aggregated_responses"]
        self.events_collection = self.db["events"]

        # Metrics collector
        self.metrics = MetricsCollector("response_aggregator")

        self.logger.info(f"ResponseAggregator initialized with timeout={self.default_timeout}s")

    def register_response_processor(self, event_type: Union[str, EventType], processor: Callable):
        """
        Register a custom processor for specific event types.

        Args:
            event_type: The event type to process
            processor: Function to process the event
        """
        if isinstance(event_type, EventType):
            event_type_str = event_type.value
        else:
            event_type_str = event_type

        self.response_processors[event_type_str] = processor
        self.logger.info(f"Registered custom processor for event type: {event_type_str}")

    async def start(self):
        """Start the response aggregator service."""
        if self.running:
            self.logger.info("ResponseAggregator is already running")
            return

        self.running = True
        self.logger.info("Starting ResponseAggregator")

        # Test database connections
        try:
            # Test Redis connection
            await self.redis.ping()
            self.logger.info("Connected to Redis")

            # Test MongoDB connection
            await self.mongodb_client.admin.command("ping")
            self.logger.info("Connected to MongoDB")
        except Exception as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            raise

        # Initialize event system connections
        await self._initialize_event_system()

        # Start background tasks
        asyncio.create_task(self._process_events())
        asyncio.create_task(self._cleanup_stale_requests())
        asyncio.create_task(self._reset_circuit_breakers())
        asyncio.create_task(self._collect_metrics())

        self.metrics.record_component_status("started", 1)
        self.logger.info("ResponseAggregator started successfully")

    async def stop(self):
        """Stop the response aggregator service."""
        self.logger.info("Stopping ResponseAggregator")
        self.running = False

        # Stop event system connections
        await self._shutdown_event_system()

        # Close database connections
        await self.redis.close()
        self.mongodb_client.close()

        # Wait for cleanup to finish
        await asyncio.sleep(1)
        self.metrics.record_component_status("started", 0)
        self.logger.info("ResponseAggregator stopped successfully")

    async def _initialize_event_system(self):
        """Initialize connections to the event system."""
        try:
            # Connect to the event bus
            self.event_bus = self.get_component("event_bus")

            # Subscribe to response events
            event_types = self._get_response_event_types()
            await self.event_bus.subscribe(
                event_types=event_types,
                handler=self._handle_event,
                subscription_name="response_aggregator",
            )

            self.logger.info(f"Subscribed to event types: {event_types}")
        except Exception as e:
            self.logger.error(f"Failed to initialize event system: {str(e)}", exc_info=True)
            raise

    async def _shutdown_event_system(self):
        """Shutdown connections to the event system."""
        try:
            await self.event_bus.unsubscribe("response_aggregator")
            self.logger.info("Unsubscribed from event system")
        except Exception as e:
            self.logger.error(f"Error unsubscribing from event system: {str(e)}", exc_info=True)

    def _get_response_event_types(self) -> List[str]:
        """Get the list of event types to subscribe to."""
        # Base set of events to listen for
        event_types = [
            EventType.CODE_GENERATION_COMPLETED.value,
            EventType.CODE_GENERATION_FAILED.value,
            EventType.VERIFICATION_COMPLETED.value,
            EventType.VERIFICATION_FAILED.value,
        ]

        # Add any custom registered event types
        event_types.extend([et for et in self.response_processors.keys() if et not in event_types])

        return event_types

    async def _process_events(self):
        """Process incoming events from the event listener."""
        while self.running:
            try:
                # Process events in batches for efficiency
                events = await self.event_bus.receive_events(max_count=10, timeout=0.5)

                for event in events:
                    await self._handle_event(event)

                # Small sleep to prevent CPU spinning
                if not events:
                    await asyncio.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Error processing events: {str(e)}", exc_info=True)
                self.metrics.record_error("event_processing_error")
                await asyncio.sleep(1)  # Backoff on error

    async def _handle_event(self, event: BaseEvent):
        """
        Handle an incoming event.

        Args:
            event: The event to handle
        """
        start_time = time.time()
        correlation_id = event.correlation_id

        # Check for duplicate events using Redis
        processed_key = f"processed:{correlation_id}"
        if await self.redis.get(processed_key):
            self.logger.debug(f"Duplicate event ignored: {correlation_id}")
            return

        # Mark this event as processed
        await self.redis.setex(processed_key, 3600, "1")  # 1 hour TTL

        # Store event in MongoDB for persistence and analysis
        try:
            await self.events_collection.insert_one(
                {
                    "correlation_id": correlation_id,
                    "event_type": event.event_type,
                    "timestamp": datetime.utcnow(),
                    "data": event.data,
                    "metadata": event.metadata,
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to store event in MongoDB: {str(e)}")

        self.logger.debug(
            f"Handling event: {event.event_type} with correlation_id: {correlation_id}"
        )

        # Check if circuit breaker is open for this service
        service_name = event.metadata.get("source_service")
        if service_name and self._is_circuit_open(service_name):
            self.logger.warning(f"Circuit breaker open for {service_name}, ignoring event")
            self.metrics.record_event("circuit_breaker_block", {"service": service_name})
            return

        # Initialize request context if this is first event
        if correlation_id not in self.pending_requests:
            self._initialize_request(correlation_id, event)

        # Add response to the pending request
        request_data = self.pending_requests[correlation_id]

        # Record event metrics
        self.metrics.record_event(
            "event_received",
            {
                "event_type": event.event_type,
                "correlation_id": correlation_id,
                "expected_responses": request_data["expected_responses"],
                "current_responses": len(request_data["responses"]),
            },
        )

        # Handle success or failure events
        if EventType.is_failure_event(event.event_type):
            await self._handle_failure_event(correlation_id, event, service_name)
        else:
            await self._handle_success_event(correlation_id, event)

        # Process with custom handler if registered
        if event.event_type in self.response_processors:
            try:
                await self.response_processors[event.event_type](
                    event, self.pending_requests[correlation_id]
                )
            except Exception as e:
                self.logger.error(
                    f"Error in custom processor for {event.event_type}: {str(e)}", exc_info=True
                )
                self.metrics.record_error("custom_processor_error")

        # Check if we have all expected responses or if we should complete with partial results
        await self._check_request_completion(correlation_id)

        # Record processing time
        processing_time = time.time() - start_time
        self.metrics.record_latency("event_processing_time", processing_time)

    def _initialize_request(self, correlation_id: str, event: BaseEvent):
        """Initialize a new request entry."""
        # Extract metadata from the event
        request_data = event.metadata.get("request_data", {})
        expected_responses = event.metadata.get("expected_responses", 1)
        timeout = event.metadata.get("timeout", self.default_timeout)
        service_timeouts = event.metadata.get("service_timeouts", {})
        min_responses = event.metadata.get("min_responses", expected_responses)

        self.pending_requests[correlation_id] = {
            "request_data": request_data,
            "responses": [],
            "failures": [],
            "timestamp": time.time(),
            "timeout": timeout,
            "expected_responses": expected_responses,
            "min_responses": min_responses,
            "service_timeouts": service_timeouts,
            "retry_attempts": {},
            "completion_checks": 0,
        }

        self.logger.info(
            f"New request initialized: correlation_id={correlation_id}, "
            f"expected_responses={expected_responses}, timeout={timeout}s"
        )

    async def _handle_success_event(self, correlation_id: str, event: BaseEvent):
        """Handle a success event."""
        # Add this response to the pending request
        request_data = self.pending_requests[correlation_id]

        response_data = {
            "event_type": event.event_type,
            "data": event.data,
            "timestamp": time.time(),
            "service": event.metadata.get("source_service"),
            "duration": event.metadata.get("processing_duration", 0),
        }

        request_data["responses"].append(response_data)

        self.logger.debug(
            f"Added success response for correlation_id={correlation_id}, "
            f"event_type={event.event_type}, "
            f"total_responses={len(request_data['responses'])}"
        )

    async def _handle_failure_event(
        self, correlation_id: str, event: BaseEvent, service_name: Optional[str] = None
    ):
        """Handle a failure event with potential retry logic."""
        request_data = self.pending_requests[correlation_id]

        # Extract error information
        error_data = {
            "event_type": event.event_type,
            "error": event.data.get("error", "Unknown error"),
            "timestamp": time.time(),
            "service": service_name or event.metadata.get("source_service"),
            "retryable": event.metadata.get("retryable", False),
        }

        request_data["failures"].append(error_data)

        # Record failure in circuit breaker if service is identified
        if service_name:
            self._record_service_failure(service_name)

        # Check if we should retry
        if error_data["retryable"]:
            service = error_data["service"]

            # Check Redis for retry count
            retry_key = f"retry:{correlation_id}:{service}"
            retry_count = int(await self.redis.get(retry_key) or 0)

            if retry_count < self.max_retries:
                # Increment retry counter in Redis
                await self.redis.incr(retry_key)
                await self.redis.expire(retry_key, 3600)  # 1 hour TTL

                # Update local state too
                request_data["retry_attempts"][service] = retry_count + 1

                # Calculate delay with exponential backoff
                delay = self.retry_delay * (2**retry_count)

                self.logger.info(
                    f"Scheduling retry for service={service}, "
                    f"correlation_id={correlation_id}, "
                    f"retry_count={retry_count + 1}, "
                    f"delay={delay}s"
                )

                # Persist retry state to Redis
                try:
                    await self.event_bus.publish_event(
                        event_type="state.write",
                        correlation_id=correlation_id,
                        data={
                            "key": f"retry:{correlation_id}:{service}",
                            "value": retry_count + 1,
                            "ttl": 3600,
                        },
                        metadata={"target": "redis"},
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to publish Redis state event: {str(e)}", exc_info=True
                    )

                # Schedule retry
                asyncio.create_task(self._retry_request(correlation_id, service, delay, event))
                return

        self.logger.warning(
            f"Added failure for correlation_id={correlation_id}, "
            f"event_type={event.event_type}, "
            f"error={error_data['error']}, "
            f"total_failures={len(request_data['failures'])}"
        )

    async def _retry_request(
        self, correlation_id: str, service: str, delay: float, original_event: BaseEvent
    ):
        """Retry a request after delay."""
        await asyncio.sleep(delay)

        # Check if request still exists and we're still running
        if not self.running or correlation_id not in self.pending_requests:
            return

        # Create retry event
        retry_event = EventType.get_retry_event_type(original_event.event_type)

        if not retry_event:
            self.logger.error(f"No retry event type defined for {original_event.event_type}")
            return

        # Get retry count from Redis
        retry_key = f"retry:{correlation_id}:{service}"
        retry_count = int(await self.redis.get(retry_key) or 0)

        retry_metadata = {
            "correlation_id": correlation_id,
            "original_event_type": original_event.event_type,
            "retry_count": retry_count,
            "request_data": self.pending_requests[correlation_id]["request_data"],
        }

        self.logger.info(f"Executing retry for service={service}, correlation_id={correlation_id}")

        # Emit retry event
        try:
            await self.event_bus.publish_event(
                event_type=retry_event,
                correlation_id=correlation_id,
                data=original_event.data.get("retry_data", {}),
                metadata=retry_metadata,
            )

            self.metrics.record_event(
                "retry_attempt", {"service": service, "correlation_id": correlation_id}
            )
        except Exception as e:
            self.logger.error(f"Failed to publish retry event: {str(e)}", exc_info=True)
            self.metrics.record_error("retry_publish_error")

    async def _check_request_completion(self, correlation_id: str):
        """
        Check if a request is complete and process it if so.

        Args:
            correlation_id: The correlation ID of the request
        """
        request_data = self.pending_requests.get(correlation_id)
        if not request_data:
            return

        # Increment completion check counter
        request_data["completion_checks"] += 1

        responses = request_data["responses"]
        expected = request_data["expected_responses"]
        min_responses = request_data["min_responses"]
        elapsed_time = time.time() - request_data["timestamp"]
        timeout = request_data["timeout"]

        # Check if we have all expected responses
        if len(responses) >= expected:
            self.logger.info(
                f"Request complete with all responses: correlation_id={correlation_id}, "
                f"responses={len(responses)}/{expected}"
            )
            await self._process_completed_request(correlation_id)
            return

        # Check if we've reached timeout but have minimum responses
        if elapsed_time >= timeout and len(responses) >= min_responses:
            self.logger.warning(
                f"Request timed out but has minimum responses: correlation_id={correlation_id}, "
                f"responses={len(responses)}/{expected}, min_required={min_responses}"
            )
            await self._process_completed_request(correlation_id, timed_out=True)
            return

        # Log progress periodically for long-running requests
        if request_data["completion_checks"] % 5 == 0:
            self.logger.debug(
                f"Request in progress: correlation_id={correlation_id}, "
                f"responses={len(responses)}/{expected}, "
                f"elapsed={elapsed_time:.1f}s/{timeout}s"
            )

    async def _process_completed_request(self, correlation_id: str, timed_out: bool = False):
        """
        Process a completed request.

        Args:
            correlation_id: The correlation ID of the completed request
            timed_out: Whether the request timed out
        """
        request_data = self.pending_requests.pop(correlation_id, None)
        if not request_data:
            return

        start_time = time.time()

        # Aggregate the responses
        aggregated_result = self._aggregate_responses(
            request_data["responses"], request_data["failures"], timed_out
        )

        # Add request metadata
        total_time = time.time() - request_data["timestamp"]
        aggregated_result["request"] = {
            "correlation_id": correlation_id,
            "total_time": total_time,
            "expected_responses": request_data["expected_responses"],
            "actual_responses": len(request_data["responses"]),
            "failures": len(request_data["failures"]),
            "timed_out": timed_out,
        }

        # Store the completed aggregation in MongoDB
        try:
            # Convert to a format suitable for MongoDB
            mongo_doc = {
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow(),
                "status": aggregated_result["status"],
                "responses": aggregated_result["responses"],
                "failures": aggregated_result.get("failures", []),
                "total_time": total_time,
                "expected_responses": request_data["expected_responses"],
                "actual_responses": len(request_data["responses"]),
                "timed_out": timed_out,
            }

            await self.responses_collection.insert_one(mongo_doc)
            self.logger.info(f"Stored aggregated result in MongoDB: {correlation_id}")
        except Exception as e:
            self.logger.error(f"Failed to store aggregated result in MongoDB: {str(e)}")

        # Record metrics
        self.metrics.record_latency("total_request_time", total_time)
        self.metrics.record_completion(
            "request_completed",
            {
                "correlation_id": correlation_id,
                "responses": len(request_data["responses"]),
                "expected": request_data["expected_responses"],
                "timed_out": timed_out,
            },
        )

        # Emit a completion event
        event_type = (
            EventType.AGGREGATION_COMPLETED_PARTIAL
            if timed_out
            else EventType.AGGREGATION_COMPLETED
        )

        try:
            await self.event_bus.publish_event(
                event_type=event_type,
                correlation_id=correlation_id,
                data=aggregated_result,
                metadata={
                    "original_request": request_data["request_data"],
                    "processing_time": time.time() - start_time,
                },
            )

            self.logger.info(
                f"Published completion event: correlation_id={correlation_id}, "
                f"event_type={event_type}, "
                f"responses={len(request_data['responses'])}/{request_data['expected_responses']}"
            )

            # Clean up Redis keys for this correlation_id
            await self.redis.delete(f"processed:{correlation_id}")
            # Get all retry keys for this correlation_id
            retry_keys = await self.redis.keys(f"retry:{correlation_id}:*")
            if retry_keys:
                await self.redis.delete(*retry_keys)

        except Exception as e:
            self.logger.error(f"Failed to publish completion event: {str(e)}", exc_info=True)
            self.metrics.record_error("completion_event_error")

    def _aggregate_responses(
        self, responses: List[Dict[str, Any]], failures: List[Dict[str, Any]], timed_out: bool
    ) -> Dict[str, Any]:
        """
        Aggregate multiple responses into a single result.

        Args:
            responses: List of response data
            failures: List of failure data
            timed_out: Whether the request timed out

        Returns:
            Aggregated result dictionary
        """
        # Sort responses by timestamp
        sorted_responses = sorted(responses, key=lambda x: x["timestamp"])

        # Status determination
        status = "success"
        if failures:
            status = "partial_failure" if responses else "failure"
        if timed_out:
            status = "partial_success_timeout"

        # Get services that responded
        responding_services = {r.get("service") for r in responses if r.get("service")}
        failing_services = {f.get("service") for f in failures if f.get("service")}

        # Basic aggregation
        result = {
            "status": status,
            "timestamp": time.time(),
            "response_count": len(responses),
            "failure_count": len(failures),
            "timed_out": timed_out,
            "responding_services": list(responding_services),
            "failing_services": list(failing_services),
            "responses": sorted_responses,
        }

        # Include failures if any
        if failures:
            result["failures"] = failures

        return result

    async def _cleanup_stale_requests(self):
        """Periodically clean up stale requests that have timed out."""
        while self.running:
            try:
                stale_count = 0
                current_time = time.time()
                stale_ids = []

                for correlation_id, request_data in self.pending_requests.items():
                    timeout = request_data.get("timeout", self.default_timeout)

                    # Allow for service-specific timeouts
                    service_timeouts = request_data.get("service_timeouts", {})
                    max_service_timeout = max(service_timeouts.values()) if service_timeouts else 0

                    # Use the larger of the global timeout or any service-specific timeout
                    effective_timeout = max(timeout, max_service_timeout)

                    # Check if request has timed out
                    if current_time - request_data["timestamp"] > effective_timeout:
                        stale_ids.append(correlation_id)

                # Process timeouts
                for correlation_id in stale_ids:
                    self.logger.warning(f"Request timed out: {correlation_id}")
                    await self._process_completed_request(correlation_id, timed_out=True)
                    stale_count += 1

                # Record cleanup metrics
                if stale_count > 0:
                    self.metrics.record_event("stale_requests_cleaned", {"count": stale_count})
                    self.logger.info(f"Cleaned up {stale_count} stale requests")

            except Exception as e:
                self.logger.error(f"Error in cleanup task: {str(e)}", exc_info=True)
                self.metrics.record_error("cleanup_error")

            await asyncio.sleep(self.cleanup_interval)

    def _record_service_failure(self, service_name: str):
        """Record a service failure for circuit breaker."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = {
                "failure_count": 0,
                "last_failure": time.time(),
                "status": "closed",  # closed, open, half-open
            }

        circuit = self.circuit_breakers[service_name]
        circuit["failure_count"] += 1
        circuit["last_failure"] = time.time()

        # Check if we should open the circuit
        if (
            circuit["status"] == "closed"
            and circuit["failure_count"] >= self.circuit_breaker_threshold
        ):
            circuit["status"] = "open"
            circuit["opened_at"] = time.time()
            self.logger.warning(
                f"Circuit breaker opened for service {service_name} "
                f"after {circuit['failure_count']} failures"
            )
            self.metrics.record_event("circuit_breaker_opened", {"service": service_name})

    def _is_circuit_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for a service."""
        if service_name not in self.circuit_breakers:
            return False

        circuit = self.circuit_breakers[service_name]
        return circuit["status"] == "open"

    async def _reset_circuit_breakers(self):
        """Periodically check and reset circuit breakers."""
        while self.running:
            try:
                current_time = time.time()

                for service_name, circuit in self.circuit_breakers.items():
                    if circuit["status"] == "open":
                        # Check if it's time to try half-open
                        if current_time - circuit.get("opened_at", 0) > self.circuit_reset_timeout:
                            circuit["status"] = "half-open"
                            circuit["failure_count"] = 0
                            self.logger.info(
                                f"Circuit breaker for {service_name} changed to half-open state"
                            )
                            self.metrics.record_event(
                                "circuit_breaker_half_open", {"service": service_name}
                            )
                    elif circuit["status"] == "half-open":
                        # If no failures in half-open state for a while, close it
                        if current_time - circuit["last_failure"] > self.circuit_reset_timeout:
                            circuit["status"] = "closed"
                            circuit["failure_count"] = 0
                            self.logger.info(
                                f"Circuit breaker for {service_name} reset to closed state"
                            )
                            self.metrics.record_event(
                                "circuit_breaker_closed", {"service": service_name}
                            )
            except Exception as e:
                self.logger.error(f"Error in circuit breaker task: {str(e)}", exc_info=True)
                self.metrics.record_error("circuit_breaker_error")

            await asyncio.sleep(self.circuit_reset_timeout / 2)

    async def _collect_metrics(self):
        """Periodically collect and report metrics."""
        while self.running:
            try:
                # Record current state
                self.metrics.record_gauge("pending_requests", len(self.pending_requests))

                # Record circuit breaker states
                open_circuits = sum(
                    1 for c in self.circuit_breakers.values() if c["status"] == "open"
                )
                half_open_circuits = sum(
                    1 for c in self.circuit_breakers.values() if c["status"] == "half-open"
                )

                self.metrics.record_gauge("open_circuits", open_circuits)
                self.metrics.record_gauge("half_open_circuits", half_open_circuits)

            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}", exc_info=True)

            await asyncio.sleep(15)  # Collect metrics every 15 seconds
