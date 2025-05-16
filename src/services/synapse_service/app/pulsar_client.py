"""
Pulsar client initialization for event messaging.

This module provides functions to create event listeners and emitters
for Apache Pulsar, handling connection setup and configuration.
"""

import logging
import asyncio
from typing import Dict, Any, List, Callable
from datetime import datetime
import json

# Pulsar client imports
import pulsar

logger = logging.getLogger(__name__)


async def create_event_listener(config: Dict[str, Any], subscription_name: str, topics: List[str]) -> "EventListener":
    """
    Create an event listener for Pulsar topics.

    Args:
        config: Configuration dictionary
        subscription_name: Name for the subscription
        topics: List of topics to subscribe to

    Returns:
        EventListener instance
    """
    try:
        # Create listener
        listener = EventListener(
            service_url=config.get("pulsar_service_url", "pulsar://pulsar:6650"),
            subscription_name=subscription_name,
            topics=topics,
            consumer_timeout_ms=config.get("consumer_timeout_ms", 1000),
            subscription_type=config.get("subscription_type", "Shared"),
        )

        # Initialize client
        await listener.initialize()

        return listener

    except Exception as e:
        logger.error(f"Error creating event listener: {e}")
        raise


async def create_event_emitter(config: Dict[str, Any]) -> "EventEmitter":
    """
    Create an event emitter for Pulsar topics.

    Args:
        config: Configuration dictionary

    Returns:
        EventEmitter instance
    """
    try:
        # Create emitter
        emitter = EventEmitter(
            service_url=config.get("pulsar_service_url", "pulsar://pulsar:6650"),
            topic_prefix=config.get("topic_prefix", "neural-context-"),
        )

        # Initialize client
        await emitter.initialize()

        return emitter

    except Exception as e:
        logger.error(f"Error creating event emitter: {e}")
        raise


class EventListener:
    """Event listener for Pulsar topics."""

    def __init__(
        self,
        service_url: str,
        subscription_name: str,
        topics: List[str],
        consumer_timeout_ms: int = 1000,
        subscription_type: str = "Shared",
    ):
        """
        Initialize the event listener.

        Args:
            service_url: Pulsar service URL
            subscription_name: Name for the subscription
            topics: List of topics to subscribe to
            consumer_timeout_ms: Consumer timeout in milliseconds
            subscription_type: Subscription type (Exclusive, Shared, Failover)
        """
        self.service_url = service_url
        self.subscription_name = subscription_name
        self.topics = topics
        self.consumer_timeout_ms = consumer_timeout_ms

        # Map subscription type string to Pulsar enum
        self.subscription_type_map = {
            "Exclusive": pulsar.ConsumerType.Exclusive,
            "Shared": pulsar.ConsumerType.Shared,
            "Failover": pulsar.ConsumerType.Failover,
            "KeyShared": pulsar.ConsumerType.KeyShared,
        }
        self.subscription_type = self.subscription_type_map.get(subscription_type, pulsar.ConsumerType.Shared)

        # Pulsar client
        self.client = None
        self.consumers = {}

        # Event handlers
        self.handlers = {}

        # Running state
        self.running = False
        self.consumer_tasks = []

        logger.info(f"Created event listener for topics: {', '.join(topics)}")

    async def initialize(self):
        """Initialize the event listener."""
        try:
            # Create Pulsar client
            self.client = pulsar.Client(self.service_url)

            logger.info(f"Connected to Pulsar service: {self.service_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Pulsar service: {e}")
            raise

    def register_handler(self, event_type: str, handler: Callable):
        """
        Register an event handler.

        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        self.handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")

    async def start(self):
        """Start listening for events."""
        if self.running:
            logger.warning("Event listener already running")
            return

        self.running = True

        try:
            # Create consumers for each topic
            for topic in self.topics:
                consumer = self.client.subscribe(
                    topic=f"persistent://public/default/{topic}",
                    subscription_name=self.subscription_name,
                    consumer_type=self.subscription_type,
                )

                self.consumers[topic] = consumer

                # Start consumer task
                task = asyncio.create_task(self._consume_events(consumer, topic))
                self.consumer_tasks.append(task)

                logger.info(f"Started consumer for topic: {topic}")

        except Exception as e:
            logger.error(f"Error starting event listener: {e}")
            self.running = False
            raise

    async def stop(self):
        """Stop listening for events."""
        if not self.running:
            return

        self.running = False

        # Cancel consumer tasks
        for task in self.consumer_tasks:
            task.cancel()

        # Wait for tasks to complete with exception handling
        if self.consumer_tasks:
            await asyncio.gather(*self.consumer_tasks, return_exceptions=True)

        # Close consumers
        for topic, consumer in self.consumers.items():
            consumer.close()
            logger.info(f"Closed consumer for topic: {topic}")

        # Close client
        if self.client:
            self.client.close()
            logger.info("Closed Pulsar client")

    async def _consume_events(self, consumer: pulsar.Consumer, topic: str):
        """
        Consume events from a topic.

        Args:
            consumer: Pulsar consumer
            topic: Topic name
        """
        logger.info(f"Starting event consumption for topic: {topic}")

        while self.running:
            try:
                # Receive message with timeout
                msg = consumer.receive(timeout_millis=self.consumer_timeout_ms)

                # Process message
                await self._process_message(msg, consumer, topic)

            except pulsar.Timeout:
                # No message received within timeout
                pass

            except asyncio.CancelledError:
                # Task cancelled
                logger.info(f"Event consumption cancelled for topic: {topic}")
                break

            except Exception as e:
                logger.error(f"Error consuming events from topic {topic}: {e}")
                # Sleep briefly to avoid busy waiting in case of persistent errors
                await asyncio.sleep(0.1)

    async def _process_message(self, msg: pulsar.Message, consumer: pulsar.Consumer, topic: str):
        """
        Process a received message.

        Args:
            msg: Pulsar message
            consumer: Pulsar consumer
            topic: Topic name
        """
        try:
            # Decode message data
            message_data = msg.data().decode("utf-8")
            event = json.loads(message_data)

            # Get event type
            event_type = event.get("event_type", topic)

            # Check if we have a handler for this event type
            if event_type in self.handlers:
                handler = self.handlers[event_type]

                # Call handler
                if asyncio.iscoroutinefunction(handler):
                    # Async handler
                    await handler(event)
                else:
                    # Sync handler
                    handler(event)

            # Acknowledge message
            consumer.acknowledge(msg)

        except Exception as e:
            logger.error(f"Error processing message from topic {topic}: {e}")
            # Negative acknowledge on error
            consumer.negative_acknowledge(msg)


class EventEmitter:
    """Event emitter for Pulsar topics."""

    def __init__(self, service_url: str, topic_prefix: str = "neural-context-"):
        """
        Initialize the event emitter.

        Args:
            service_url: Pulsar service URL
            topic_prefix: Prefix for topic names
        """
        self.service_url = service_url
        self.topic_prefix = topic_prefix

        # Pulsar client
        self.client = None

        # Producer cache
        self.producers = {}

        logger.info(f"Created event emitter with topic prefix: {topic_prefix}")

    async def initialize(self):
        """Initialize the event emitter."""
        try:
            # Create Pulsar client
            self.client = pulsar.Client(self.service_url)

            logger.info(f"Connected to Pulsar service: {self.service_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Pulsar service: {e}")
            raise

    async def emit_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """
        Emit an event to a topic.

        Args:
            event_type: Type of event
            payload: Event payload

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get or create producer
            producer = await self._get_producer(event_type)

            # Create event
            event = {
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "payload": payload,
            }

            # Serialize to JSON
            event_json = json.dumps(event)

            # Send message
            producer.send(event_json.encode("utf-8"))

            logger.debug(f"Emitted event: {event_type}")
            return True

        except Exception as e:
            logger.error(f"Error emitting event {event_type}: {e}")
            return False

    async def _get_producer(self, event_type: str) -> pulsar.Producer:
        """
        Get or create a producer for an event type.

        Args:
            event_type: Event type

        Returns:
            Pulsar producer
        """
        if event_type not in self.producers:
            # Get topic name
            topic = f"persistent://public/default/{event_type}"

            # Create producer
            self.producers[event_type] = self.client.create_producer(
                topic=topic,
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=10,
            )

            logger.debug(f"Created producer for event type: {event_type}")

        return self.producers[event_type]

    async def close(self):
        """Close the event emitter."""
        # Close producers
        for event_type, producer in self.producers.items():
            producer.close()
            logger.info(f"Closed producer for event type: {event_type}")

        # Close client
        if self.client:
            self.client.close()
            logger.info("Closed Pulsar client")
