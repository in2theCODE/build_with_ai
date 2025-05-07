"""Secure event listener for the event-driven neural code generator system.

This module provides a secure event listener that verifies and processes
events from Apache Pulsar, implementing the zero-trust security model.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import pulsar
from pulsar.schema import *

from ..models import BaseEvent
from ..models import EventType


logger = logging.getLogger(__name__)

# Type for event handlers - can be either synchronous or asynchronous
EventHandlerType = Callable[[BaseEvent], Union[None, Awaitable[None]]]


class EventVerificationError(Exception):
    """Exception raised when event verification fails."""

    pass


class SecureEventListener:
    """
    Secure event listener for Apache Pulsar.

    This class provides a secure way to listen for events from Apache Pulsar topics,
    with support for HMAC-based message verification for the zero-trust security model.
    """

    def __init__(
        self,
        service_url: str,
        subscription_name: str,
        event_types: List[EventType],
        secret_key: Optional[str] = None,
        tenant: str = "public",
        namespace: str = "code-generator",
        consumer_type: str = "Shared",
        ssl_cert_path: Optional[str] = None,
        max_retry_attempts: int = 3,
    ):
        """
        Initialize the secure event listener.

        Args:
            service_url: Pulsar service URL
            subscription_name: Name of the subscription
            event_types: List of event types to listen for
            secret_key: Secret key for HMAC verification. If None, will use PULSAR_SECRET_KEY env var
            tenant: Pulsar tenant
            namespace: Pulsar namespace
            consumer_type: Pulsar consumer type (Exclusive, Shared, Failover)
            ssl_cert_path: Path to SSL certificate for secure connection
            max_retry_attempts: Maximum number of retry attempts for failed message processing
        """
        self.service_url = service_url
        self.subscription_name = subscription_name
        self.event_types = event_types
        self.tenant = tenant
        self.namespace = namespace
        self.ssl_cert_path = ssl_cert_path
        self.max_retry_attempts = max_retry_attempts

        # Map consumer type string to Pulsar enum
        consumer_type_map = {
            "Exclusive": pulsar.ConsumerType.Exclusive,
            "Shared": pulsar.ConsumerType.Shared,
            "Failover": pulsar.ConsumerType.Failover,
        }
        self.consumer_type = consumer_type_map.get(consumer_type, pulsar.ConsumerType.Shared)

        # Get secret key for HMAC verification
        self.secret_key = secret_key or os.environ.get("PULSAR_SECRET_KEY", "default-secret-key")
        if self.secret_key == "default-secret-key":
            logger.warning("Using default secret key for HMAC verification. This is insecure!")

        # Create Pulsar client
        client_config = {"service_url": service_url}

        # Add SSL configuration if specified
        if self.ssl_cert_path:
            client_config["authentication"] = pulsar.AuthenticationTLS(self.ssl_cert_path)
            client_config["tls_validate_hostname"] = True
            client_config["tls_trust_certs_file_path"] = self.ssl_cert_path

        self.client = pulsar.Client(**client_config)

        # Dictionary to store event handlers
        self.handlers: Dict[str, List[EventHandlerType]] = {}

        # Dictionary to store consumers
        self.consumers = {}

        # Flag to indicate if the listener is running
        self.running = False

        logger.info(f"Initialized SecureEventListener for {service_url}")

    def _get_topic_names(self) -> List[str]:
        """
        Get the full topic names for all event types.

        Returns:
            List of full topic names
        """
        # Group event types by category (first part of the event type)
        categories = set()
        for event_type in self.event_types:
            if isinstance(event_type, EventType):
                category = event_type.value.split(".")[0]
            else:
                category = event_type.split(".")[0]
            categories.add(category)

        # Create topic names for each category
        return [
            f"persistent://{self.tenant}/{self.namespace}/{category}-events"
            for category in categories
        ]

    def _verify_signature(self, data: Dict[str, Any]) -> bool:
        """
        Verify the HMAC signature of event data.

        Args:
            data: The event data with signature

        Returns:
            True if signature is valid, False otherwise
        """
        # Extract signature
        signature = data.pop("_signature", None)
        if not signature:
            logger.warning("No signature found in event data")
            return False

        # Create canonical representation for verification
        canonical = json.dumps(data, sort_keys=True)

        # Generate expected signature
        expected_signature = hmac.new(
            key=self.secret_key.encode(), msg=canonical.encode(), digestmod=hashlib.sha256
        ).digest()
        expected_signature_b64 = base64.b64encode(expected_signature).decode()

        # Verify signature
        result = hmac.compare_digest(signature, expected_signature_b64)

        # Add signature back to data
        data["_signature"] = signature

        return result

    def register_handler(self, event_type: Union[str, EventType], handler: EventHandlerType):
        """
        Register a handler for a specific event type.

        Args:
            event_type: The event type to handle
            handler: The handler function
        """
        if isinstance(event_type, EventType):
            event_type_str = event_type.value
        else:
            event_type_str = event_type

        if event_type_str not in self.handlers:
            self.handlers[event_type_str] = []

        self.handlers[event_type_str].append(handler)
        logger.debug(f"Registered handler for event type {event_type_str}")

    async def start(self):
        """Start listening for events."""
        if self.running:
            logger.warning("Event listener already running")
            return

        self.running = True

        # Get topic names
        topic_names = self._get_topic_names()

        # Create consumers for each topic
        for topic in topic_names:
            try:
                consumer = self.client.subscribe(
                    topic=topic,
                    subscription_name=self.subscription_name,
                    consumer_type=self.consumer_type,
                    max_pending_messages=1000,
                    receiver_queue_size=1000,
                )
                self.consumers[topic] = consumer
                logger.info(f"Subscribed to topic {topic}")

                # Start a task to receive messages from this consumer
                asyncio.create_task(self._receive_loop(consumer))
            except Exception as e:
                logger.error(f"Failed to subscribe to topic {topic}: {e}")

    async def _receive_loop(self, consumer: pulsar.Consumer):
        """
        Continuous loop to receive messages from a consumer.

        Args:
            consumer: The Pulsar consumer
        """
        while self.running:
            try:
                # Receive message with timeout
                message = consumer.receive(timeout_millis=1000)

                # Process the message
                await self._process_message(message, consumer)
            except pulsar.Timeout:
                # No message received within timeout
                continue
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                # Sleep briefly to avoid busy waiting in case of persistent errors
                await asyncio.sleep(0.1)

    async def _process_message(self, message: pulsar.Message, consumer: pulsar.Consumer):
        """
        Process a received message.

        Args:
            message: The Pulsar message
            consumer: The Pulsar consumer
        """
        try:
            # Decode message data
            message_data = message.data().decode("utf-8")
            event_data = json.loads(message_data)

            # Verify signature
            if not self._verify_signature(event_data):
                logger.warning(f"Signature verification failed for message {message.message_id()}")
                # Acknowledge message even if verification fails to avoid clogging the queue
                consumer.acknowledge(message)
                return

            # Create event object
            try:
                event = BaseEvent.from_dict(event_data)
            except Exception as e:
                logger.error(f"Failed to parse event: {e}")
                consumer.acknowledge(message)
                return

            # Find and call handlers
            event_type_str = event_data.get("event_type")
            if event_type_str in self.handlers:
                handler_tasks = []

                for handler in self.handlers[event_type_str]:
                    try:
                        # Check if handler is async or sync
                        if asyncio.iscoroutinefunction(handler):
                            # Async handler
                            task = asyncio.create_task(handler(event))
                        else:
                            # Sync handler
                            task = asyncio.create_task(asyncio.to_thread(handler, event))
                        handler_tasks.append(task)
                    except Exception as e:
                        logger.error(f"Error calling handler for event {event.event_id}: {e}")

                # Wait for all handlers to complete
                if handler_tasks:
                    await asyncio.gather(*handler_tasks, return_exceptions=True)

            # Acknowledge message
            consumer.acknowledge(message)
            logger.debug(f"Processed event {event.event_id}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Negative acknowledge to retry
            consumer.negative_acknowledge(message)

    async def stop(self):
        """Stop listening for events."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping event listener")

        # Close all consumers
        for topic, consumer in self.consumers.items():
            try:
                consumer.close()
                logger.debug(f"Closed consumer for topic {topic}")
            except Exception as e:
                logger.error(f"Error closing consumer for topic {topic}: {e}")

        # Clear consumers dictionary
        self.consumers = {}

        # Close Pulsar client
        try:
            self.client.close()
            logger.debug("Closed Pulsar client")
        except Exception as e:
            logger.error(f"Error closing Pulsar client: {e}")
