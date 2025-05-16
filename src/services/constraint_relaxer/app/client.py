#!/usr/bin/env python3
"""
Event bus client for connecting to Apache Pulsar.
"""

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, Type, TypeVar

import pulsar
from src.services.shared.models.base import BaseMessage


logger = logging.getLogger(__name__)

# Define a type variable for BaseMessage subclasses
T = TypeVar("T", bound=BaseMessage)


class EventBusClient:
    """Client for interacting with the Apache Pulsar event bus."""

    def __init__(self, config):
        """
        Initialize the event bus client.

        Args:
            config: Application configuration
        """
        self.config = config
        self.client = None
        self.producers = {}
        self.consumers = {}
        self.logger = logger

    async def connect(self):
        """Connect to the Pulsar event bus."""
        try:
            service_url = f"pulsar://{self.config.EVENT_BUS_HOST}:{self.config.EVENT_BUS_PORT}"
            self.logger.info(f"Connecting to Pulsar at {service_url}")

            # Create client
            self.client = pulsar.Client(service_url, operation_timeout_seconds=30)

            self.logger.info("Connected to Pulsar event bus")

        except Exception as e:
            self.logger.error(f"Failed to connect to Pulsar: {str(e)}")
            raise

    async def close(self):
        """Close the connection to the event bus."""
        try:
            # Close producers
            for producer in self.producers.values():
                producer.close()

            # Close consumers
            for consumer in self.consumers.values():
                consumer.close()

            # Close client
            if self.client:
                self.client.close()

            self.logger.info("Closed Pulsar event bus connection")

        except Exception as e:
            self.logger.error(f"Error closing Pulsar connection: {str(e)}")

    async def publish_message(self, topic: str, message: BaseMessage):
        """
        Publish a BaseMessage object to a topic.

        Args:
            topic: The topic to publish to
            message: The BaseMessage to publish
        """
        # Convert BaseMessage to dictionary using model_dump instead of dict()
        message_dict = message.model_dump()

        # Publish the dictionary
        await self.publish(topic, message_dict)

    async def publish(self, topic: str, message: Dict[str, Any]):
        """
        Publish a message to a topic.

        Args:
            topic: The topic to publish to
            message: The message to publish
        """
        try:
            # Get or create producer
            if topic not in self.producers:
                full_topic = f"persistent://{self.config.EVENT_BUS_TENANT}/{self.config.EVENT_BUS_NAMESPACE}/{topic}"
                self.producers[topic] = self.client.create_producer(full_topic)

            producer = self.producers[topic]

            # Serialize message
            message_data = json.dumps(message).encode("utf-8")

            # Send message
            producer.send(message_data)
            self.logger.debug(f"Published message to {topic}")

        except Exception as e:
            self.logger.error(f"Failed to publish message to {topic}: {str(e)}")
            raise

    async def subscribe_with_type(self, topic: str, message_type: Type[T], handler: Callable[[T], Awaitable[None]]):
        """
        Subscribe to a topic with a specific message type.

        Args:
            topic: The topic to subscribe to
            message_type: The BaseMessage subclass to parse messages as
            handler: Async callback function to handle typed messages
        """

        # Create a wrapper handler that parses raw dictionaries into the specified type
        async def type_handler(message_dict: Dict[str, Any]):
            # Parse the message as the specified type
            typed_message = message_type(**message_dict)
            # Call the original handler with the typed message
            await handler(typed_message)

        # Subscribe with the wrapper handler
        await self.subscribe(topic, type_handler)

    async def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Subscribe to a topic.

        Args:
            topic: The topic to subscribe to
            handler: Async callback function to handle messages
        """
        try:
            full_topic = f"persistent://{self.config.EVENT_BUS_TENANT}/{self.config.EVENT_BUS_NAMESPACE}/{topic}"
            subscription_name = "constraint-relaxer-service"

            # Create consumer
            consumer = self.client.subscribe(full_topic, subscription_name, consumer_type=pulsar.ConsumerType.Shared)

            self.consumers[topic] = consumer
            self.logger.info(f"Subscribed to topic {topic}")

            # Start message listener
            asyncio.create_task(self._listen_for_messages(topic, consumer, handler))

        except Exception as e:
            self.logger.error(f"Failed to subscribe to {topic}: {str(e)}")
            raise

    async def _listen_for_messages(self, topic: str, consumer, handler: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Listen for messages on a topic.

        Args:
            topic: Topic name
            consumer: Pulsar consumer
            handler: Message handler function
        """
        self.logger.info(f"Started listening for messages on {topic}")

        while topic in self.consumers:
            try:
                # Receive message with timeout
                msg = consumer.receive(timeout_millis=1000)

                if msg:
                    # Parse message
                    message_data = msg.data().decode("utf-8")
                    message = json.loads(message_data)

                    # Process message
                    try:
                        await handler(message)
                        # Acknowledge successful processing
                        consumer.acknowledge(msg)
                    except Exception as e:
                        self.logger.error(f"Error processing message: {str(e)}")
                        # Negative acknowledge to have the message redelivered
                        consumer.negative_acknowledge(msg)

            except pulsar.Timeout:
                # Timeout is expected, just continue
                pass
            except Exception as e:
                self.logger.error(f"Error receiving message from {topic}: {str(e)}")
                await asyncio.sleep(1)  # Prevent tight loop in case of errors
