"""
Pulsar Event Bus Implementation.

This module provides the Pulsar-specific implementation of the EventBus interface,
handling all the details of connecting to and interacting with an Apache Pulsar cluster.

It encapsulates the complexities of Pulsar client management, topic naming, message
serialization/deserialization, and error handling to provide a robust and reliable
event bus for production environments.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Callable, Optional
import pulsar

from src.services.shared.pulsar.event_bus import EventBus

logger = logging.getLogger(__name__)

class PulsarEventBus(EventBus):
    """
    Apache Pulsar implementation of the EventBus interface.

    This class provides a production-ready event bus implementation using Apache Pulsar
    as the underlying message broker. It handles connection management, topic creation,
    message publishing, and subscription management.
    """

    def __init__(
        self,
        client_url: str = "pulsar://localhost:6650",
        tenant: str = "template-system",
        namespace: str = "events",
        auth_token: Optional[str] = None,
        operation_timeout: int = 30,
        connection_timeout: int = 30
    ):
        """
        Initialize the Pulsar event bus.

        Args:
            client_url: Pulsar broker URL
            tenant: Pulsar tenant
            namespace: Pulsar namespace
            auth_token: Authentication token (if required)
            operation_timeout: Operation timeout in seconds
            connection_timeout: Connection timeout in seconds
        """
        self.client_url = client_url
        self.tenant = tenant
        self.namespace = namespace
        self.auth_token = auth_token
        self.operation_timeout = operation_timeout
        self.connection_timeout = connection_timeout

        # Will be initialized in start()
        self.client = None
        self.producers = {}
        self.consumers = {}
        self.running = False

    async def start(self) -> bool:
        """
        Initialize the Pulsar client and test connection.

        Returns:
            bool: True if successfully connected, False otherwise
        """
        try:
            # Create the client
            client_args = {
                "service_url": self.client_url,
                "operation_timeout_seconds": self.operation_timeout,
                "connection_timeout_seconds": self.connection_timeout
            }

            # Add authentication if provided
            if self.auth_token:
                client_args["authentication"] = pulsar.AuthenticationToken(self.auth_token)

            self.client = pulsar.Client(**client_args)

            # Test connection
            client_id = self.client.get_client_id()
            logger.info(f"Connected to Pulsar (Client ID: {client_id})")

            self.running = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Pulsar: {str(e)}")
            logger.error(f"Broker URL: {self.client_url}")
            logger.error("Make sure your Pulsar instance is running")
            await self._cleanup()
            return False

    async def stop(self) -> None:
        """
        Close all producers, consumers, and the client connection.
        """
        await self._cleanup()

    async def _cleanup(self) -> None:
        """
        Internal cleanup method to release all resources.
        """
        # Close all producers
        for producer in self.producers.values():
            try:
                producer.close()
            except Exception as e:
                logger.warning(f"Error closing producer: {str(e)}")

        # Close all consumers
        for consumer in self.consumers.values():
            try:
                consumer.close()
            except Exception as e:
                logger.warning(f"Error closing consumer: {str(e)}")

        # Close client
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error closing Pulsar client: {str(e)}")

        # Reset state
        self.client = None
        self.producers = {}
        self.consumers = {}
        self.running = False

    def _get_topic_name(self, event_type: str) -> str:
        """
        Convert event type to a fully qualified Pulsar topic name.

        Args:
            event_type: The event type

        Returns:
            Fully qualified topic name
        """
        return f"persistent://{self.tenant}/{self.namespace}/{event_type}"

    async def publish_event(self, event_type: str, payload: Dict[str, Any], **kwargs) -> bool:
        """
        Publish an event to the specified topic.

        Args:
            event_type: Event type (used as topic name)
            payload: Event data
            **kwargs: Additional metadata

        Returns:
            bool: True if published successfully, False otherwise
        """
        if not self.running or not self.client:
            logger.error("Cannot publish event: Pulsar client not connected")
            return False

        try:
            # Create producer if it doesn't exist for this topic
            topic = self._get_topic_name(event_type)
            if topic not in self.producers:
                self.producers[topic] = self.client.create_producer(
                    topic=topic,
                    send_timeout_millis=self.operation_timeout * 1000,
                    block_if_queue_full=True,
                    batching_enabled=True
                )

            # Prepare event data
            event_data = {
                "event_type": event_type,
                "payload": payload,
                "metadata": kwargs
            }

            # Serialize to JSON
            serialized_data = json.dumps(event_data).encode('utf-8')

            # Send message
            self.producers[topic].send(serialized_data)
            logger.debug(f"Published event to {topic}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish event to {event_type}: {str(e)}")
            return False

    def subscribe(
        self,
        event_types: List[str],
        handler: Callable,
        subscription_name: str,
        subscription_type: str = "exclusive",
        **kwargs
    ) -> bool:
        """
        Subscribe to events from specified topics.

        Args:
            event_types: List of event types to subscribe to
            handler: Callback function to process events
            subscription_name: Unique name for this subscription
            subscription_type: Subscription type (exclusive, shared, etc.)
            **kwargs: Additional subscription options

        Returns:
            bool: True if successfully subscribed, False otherwise
        """
        if not self.running or not self.client:
            logger.error("Cannot subscribe: Pulsar client not connected")
            return False

        try:
            # Convert subscription type to enum
            sub_type = getattr(pulsar.SubscriptionType, subscription_type.upper())

            # Create consumer for each topic
            for event_type in event_types:
                topic = self._get_topic_name(event_type)

                # Create a unique consumer key for this topic + subscription
                consumer_key = f"{topic}_{subscription_name}"

                # If we already have a consumer for this, skip
                if consumer_key in self.consumers:
                    continue

                # Create the consumer
                consumer = self.client.subscribe(
                    topic=topic,
                    subscription_name=subscription_name,
                    subscription_type=sub_type,
                    consumer_type=pulsar.ConsumerType.Shared if subscription_type.lower() == "shared" else pulsar.ConsumerType.Exclusive,
                    unacked_messages_timeout_ms=kwargs.get("ack_timeout_seconds", 60) * 1000
                )

                self.consumers[consumer_key] = consumer

                # Start message listener
                self._start_consumer_loop(consumer, handler)

            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to events: {str(e)}")
            return False

    def _start_consumer_loop(self, consumer, handler: Callable) -> None:
        """
        Start a background task to continuously receive messages.

        Args:
            consumer: Pulsar consumer
            handler: Callback function to process messages
        """
        loop = asyncio.get_event_loop()
        loop.create_task(self._consumer_loop(consumer, handler))

    async def _consumer_loop(self, consumer, handler: Callable) -> None:
        """
        Continuously receive messages from a consumer and process them.

        Args:
            consumer: Pulsar consumer
            handler: Callback function to process messages
        """
        while self.running:
            try:
                # Receive message with timeout
                message = consumer.receive(timeout_millis=1000)

                # Parse message data
                try:
                    data = json.loads(message.data().decode('utf-8'))

                    # Create event object
                    event = self._create_event_object(
                        data.get("event_type", "unknown"),
                        data.get("payload", {}),
                        data.get("metadata", {})
                    )

                    # Process the event
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)

                    # Acknowledge message
                    consumer.acknowledge(message)

                except json.JSONDecodeError:
                    logger.error(f"Failed to decode message data: {message.data()}")
                    consumer.negative_acknowledge(message)

                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    consumer.negative_acknowledge(message)

            except pulsar.Timeout:
                # Just a timeout, continue the loop
                pass

            except Exception as e:
                logger.error(f"Error in consumer loop: {str(e)}")
                await asyncio.sleep(1)  # Prevent tight loop in case of persistent errors

        # If we're here, we're shutting down
        logger.debug("Consumer loop exiting")

    def _create_event_object(self, event_type, payload, metadata):
        """
        Create an event object with the expected interface.

        Args:
            event_type: Event type
            payload: Event data
            metadata: Event metadata

        Returns:
            Event object
        """
        # Create a simple object with the expected interface
        return type('Event', (), {
            'event_type': event_type,
            'payload': payload,
            'metadata': type('Metadata', (), metadata)
        })
