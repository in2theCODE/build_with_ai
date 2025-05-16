"""
Pulsar Event Bus Implementation Module.

This module implements the EventBus abstract interface using Apache Pulsar as the
underlying message broker technology. It provides a concrete implementation
that enables the template registry system to use Pulsar for event-driven communication.
"""

import abc
import asyncio
from datetime import datetime
import json
import logging
from typing import Any, Callable, Dict, List, Optional
import uuid

import pulsar


logger = logging.getLogger(__name__)

"""
EventBus Abstract Interface Module.

This module defines the abstract interface for an event bus to enable
event-driven communication between different components of the system.
"""


class EventBus(abc):
    """
    Abstract interface for an Event Bus.

    This class defines the common interface that all event bus implementations
    should follow, regardless of the underlying message broker technology
    (like Pulsar, RabbitMQ, Kafka, etc).
    """

    @abc.abstractmethod
    async def start(self) -> bool:
        """
        Initialize and start the event bus.

        Returns:
            bool: True if successfully started, False otherwise
        """
        pass

    @abc.abstractmethod
    async def stop(self) -> None:
        """
        Shut down the event bus and clean up resources.
        """
        pass

    @abc.abstractmethod
    async def publish_event(self, event_type: str, payload: Dict[str, Any], **kwargs) -> bool:
        """
        Publish an event to the event bus.

        Args:
            event_type: The type/name of the event
            payload: The event data
            **kwargs: Additional metadata for the event

        Returns:
            bool: True if successfully published, False otherwise
        """
        pass

    @abc.abstractmethod
    def subscribe(
        self,
        event_types: List[str],
        handler: Callable,
        subscription_name: str,
        subscription_type: str = "exclusive",
        **kwargs,
    ) -> bool:
        """
        Subscribe to events from the bus.

        Args:
            event_types: List of event types to subscribe to
            handler: Callback function to process events
            subscription_name: Unique name for this subscription
            subscription_type: Type of subscription (exclusive, shared, etc.)
            **kwargs: Additional subscription options

        Returns:
            bool: True if successfully subscribed, False otherwise
        """
        pass


class PulsarEventBus(EventBus):
    """
    Pulsar implementation of the EventBus interface.

    This implementation uses Apache Pulsar as the underlying message broker
    for publishing and subscribing to events.
    """

    def __init__(
        self,
        service_url: str,
        topic_prefix: str = "persistent://public/default/template-",
        consumer_timeout_ms: int = 1000,  # Added the missing comma here
        schema_registry_url: Optional[str] = None,
        schema_registry_auth_token: Optional[str] = None,
    ):
        """
        Initialize the Pulsar Event Bus.

        Args:
            service_url: Pulsar service URL
            topic_prefix: Prefix for Pulsar topics
            consumer_timeout_ms: Consumer timeout in milliseconds
            schema_registry_url: URL for the schema registry
            schema_registry_auth_token: Auth token for the schema registry
        """
        self.service_url = service_url
        self.topic_prefix = topic_prefix
        self.consumer_timeout_ms = consumer_timeout_ms
        self.client = None
        self.producers = {}
        self.consumers = {}
        self.handlers = {}
        self.running = False
        self.consumer_tasks = []
        self._lock = asyncio.Lock()

        self.schema_registry = None
        if schema_registry_url:
            from src.services.shared.models.schema_registry import SchemaRegistryClient

            self.schema_registry = SchemaRegistryClient(url=schema_registry_url, auth_token=schema_registry_auth_token)

    async def start(self) -> bool:
        """
        Initialize the Pulsar client, producers, and consumers.

        Returns:
            bool: True if successfully started, False otherwise
        """
        try:
            # Create Pulsar client
            self.client = pulsar.Client(self.service_url)
            self.running = True

            # Start consumer tasks
            for subscription_name, handler_info in self.handlers.items():
                await self._create_consumer(
                    handler_info["event_types"],
                    handler_info["handler"],
                    subscription_name,
                    handler_info["subscription_type"],
                    **handler_info["options"],
                )

            logger.info(f"Started Pulsar Event Bus with service URL: {self.service_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to start Pulsar Event Bus: {e}")
            await self.stop()
            return False

    async def stop(self) -> None:
        """
        Shut down the Pulsar client, producers, and consumers.
        """
        self.running = False

        # Cancel consumer tasks
        for task in self.consumer_tasks:
            task.cancel()

        try:
            # Wait for tasks to complete
            if self.consumer_tasks:
                await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
            self.consumer_tasks = []
        except Exception as e:
            logger.error(f"Error waiting for consumer tasks to complete: {e}")

        try:
            # Close consumers
            for consumer_name, consumer in self.consumers.items():
                try:
                    consumer.close()
                except Exception as e:
                    logger.error(f"Error closing consumer {consumer_name}: {e}")

            # Close producers
            for producer_name, producer in self.producers.items():
                try:
                    producer.close()
                except Exception as e:
                    logger.error(f"Error closing producer {producer_name}: {e}")

            # Close client
            if self.client:
                self.client.close()

            # Clear references
            self.consumers = {}
            self.producers = {}
            self.client = None

            logger.info("Stopped Pulsar Event Bus")
        except Exception as e:
            logger.error(f"Error stopping Pulsar Event Bus: {e}")

    async def publish_event(self, event_type: str, payload: Dict[str, Any], **kwargs) -> bool:
        """
        Publish an event to the appropriate Pulsar topic.

        Args:
            event_type: The type/name of the event
            payload: The event data
            **kwargs: Additional metadata for the event

        Returns:
            bool: True if successfully published, False otherwise
        """
        try:
            # Create event
            event = self._create_event(event_type, payload, **kwargs)

            if self.schema_registry:
                subject = f"{event_type}-value"
                schema_id = self.schema_registry.get_schema_id(subject)
                if schema_id:
                    if not self.schema_registry.validate_event_against_schema(event, schema_id):
                        logger.error(f"Schema validation failed for event {event_type}: {schema_id}")
                        return False

            # Get or create producer
            producer = await self._get_producer(event_type)

            # Serialize event
            event_json = json.dumps(event)

            # Publish event
            producer.send(event_json.encode("utf-8"))

            logger.debug(f"Published event {event['event_id']} of type {event_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish event of type {event_type}: {e}")
            return False

    def subscribe(
        self,
        event_types: List[str],
        handler: Callable,
        subscription_name: str,
        subscription_type: str = "exclusive",
        **kwargs,
    ) -> bool:
        """
        Subscribe to events from the bus.

        Args:
            event_types: List of event types to subscribe to
            handler: Callback function to process events
            subscription_name: Unique name for this subscription
            subscription_type: Type of subscription (exclusive, shared, etc.)
            **kwargs: Additional subscription options

        Returns:
            bool: True if successfully subscribed, False otherwise
        """
        try:
            # Store handler information
            self.handlers[subscription_name] = {
                "event_types": event_types,
                "handler": handler,
                "subscription_type": subscription_type,
                "options": kwargs,
            }

            # If already running, create consumer
            if self.running and self.client:
                asyncio.create_task(
                    self._create_consumer(
                        event_types,
                        handler,
                        subscription_name,
                        subscription_type,
                        **kwargs,
                    )
                )

            logger.info(f"Registered subscription '{subscription_name}' for event types: {', '.join(event_types)}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to event types {event_types}: {e}")
            return False

    async def _get_producer(self, event_type: str):
        """
        Get or create a producer for the specified event type.

        Args:
            event_type: Event type for the producer

        Returns:
            Pulsar producer
        """
        async with self._lock:
            if event_type not in self.producers:
                if not self.client:
                    raise ValueError("Pulsar client not initialized")

                topic = f"{self.topic_prefix}{event_type}"
                self.producers[event_type] = self.client.create_producer(
                    topic=topic,
                    block_if_queue_full=True,
                    batching_enabled=True,
                    batching_max_publish_delay_ms=10,
                )
                logger.debug(f"Created producer for event type {event_type} on topic {topic}")

            return self.producers[event_type]

    async def _create_consumer(
        self,
        event_types: List[str],
        handler: Callable,
        subscription_name: str,
        subscription_type: str,
        **kwargs,
    ):
        """
        Create consumers for the specified event types.

        Args:
            event_types: List of event types to subscribe to
            handler: Callback function to process events
            subscription_name: Unique name for this subscription
            subscription_type: Type of subscription (exclusive, shared, etc.)
            **kwargs: Additional subscription options
        """
        if not self.client:
            raise ValueError("Pulsar client not initialized")

        # Map subscription type to Pulsar consumer type
        consumer_type_map = {
            "exclusive": pulsar.ConsumerType.Exclusive,
            "shared": pulsar.ConsumerType.Shared,
            "failover": pulsar.ConsumerType.Failover,
            "key_shared": pulsar.ConsumerType.KeyShared,
        }

        consumer_type = consumer_type_map.get(subscription_type.lower(), pulsar.ConsumerType.Exclusive)

        # Create a consumer for each event type
        for event_type in event_types:
            topic = f"{self.topic_prefix}{event_type}"
            consumer_key = f"{subscription_name}_{event_type}"

            # Create consumer
            consumer = self.client.subscribe(
                topic=topic,
                subscription_name=subscription_name,
                consumer_type=consumer_type,
                **kwargs,
            )

            self.consumers[consumer_key] = consumer

            # Start consumer task
            task = asyncio.create_task(self._consume_events(consumer, event_type, handler))
            self.consumer_tasks.append(task)

            logger.debug(
                f"Created consumer for event type {event_type} on topic {topic} with subscription {subscription_name}"
            )

    async def _consume_events(self, consumer, event_type: str, handler: Callable):
        """
        Consume events from a Pulsar topic and pass to handler.

        Args:
            consumer: Pulsar consumer
            event_type: Type of events being consumed
            handler: Callback function to process events
        """
        while self.running:
            try:
                # Receive message with timeout
                msg = consumer.receive(timeout_millis=self.consumer_timeout_ms)

                # Parse event
                event_json = msg.data().decode("utf-8")
                event = json.loads(event_json)

                # Process event
                try:
                    # Call handler
                    await handler(event)

                    # Acknowledge message
                    consumer.acknowledge(msg)
                    logger.debug(f"Processed event {event.get('event_id', 'unknown')} of type {event_type}")

                except Exception as e:
                    # Negative acknowledge on handler error
                    consumer.negative_acknowledge(msg)
                    logger.error(f"Error processing event of type {event_type}: {e}")

            except pulsar.Timeout:
                # Timeout is normal, just continue
                pass

            except Exception as e:
                logger.error(f"Error consuming events of type {event_type}: {e}")
                # Sleep to avoid tight loop in case of persistent errors
                await asyncio.sleep(1)

    def _create_event(self, event_type: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Create an event dictionary with standard fields.

        Args:
            event_type: Type of event
            payload: Event payload
            **kwargs: Additional metadata

        Returns:
            Event dictionary
        """
        event_id = kwargs.get("event_id", str(uuid.uuid4()))
        timestamp = kwargs.get("timestamp", datetime.utcnow().isoformat())
        source = kwargs.get("source", "template-registry")
        version = kwargs.get("version", "1.0")

        # Extract metadata
        metadata = {}
        for key, value in kwargs.items():
            if key not in ["event_id", "timestamp", "source", "version"]:
                metadata[key] = value

        return {
            "event_id": event_id,
            "timestamp": timestamp,
            "source": source,
            "event_type": event_type,
            "payload": payload,
            "metadata": metadata,
            "version": version,
        }
