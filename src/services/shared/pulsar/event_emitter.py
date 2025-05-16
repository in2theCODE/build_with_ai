"""Secure event emitter for the event-driven neural code generator system.

This module provides a secure event emitter that signs and emits events
to Apache Pulsar, implementing the zero-trust security model.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
from typing import Any, Dict, Optional

import pulsar
from pulsar.schema import *

from ..models import BaseEvent


logger = logging.getLogger(__name__)


class SecureEventEmitter:
    """
    Secure event emitter for Apache Pulsar.

    This class provides a secure way to emit events to Apache Pulsar topics,
    with support for HMAC-based message signing for the zero-trust security model.
    """

    def __init__(
        self,
        service_url: str,
        secret_key: Optional[str] = None,
        tenant: str = "public",
        namespace: str = "code-generator",
        ssl_cert_path: Optional[str] = None,
    ):
        """
        Initialize the secure event emitter.

        Args:
            service_url: Pulsar service URL
            secret_key: Secret key for HMAC signing. If None, will use PULSAR_SECRET_KEY env var
            tenant: Pulsar tenant
            namespace: Pulsar namespace
            ssl_cert_path: Path to SSL certificate for secure connection
        """
        self.service_url = service_url
        self.tenant = tenant
        self.namespace = namespace
        self.ssl_cert_path = ssl_cert_path

        # Get secret key for HMAC signing
        self.secret_key = secret_key or os.environ.get("PULSAR_SECRET_KEY", "default-secret-key")
        if self.secret_key == "default-secret-key":
            logger.warning("Using default secret key for HMAC signing. This is insecure!")

        # Create Pulsar client
        client_config = {"service_url": service_url}

        # Add SSL configuration if specified
        if self.ssl_cert_path:
            client_config["authentication"] = pulsar.AuthenticationTLS(self.ssl_cert_path)
            client_config["tls_validate_hostname"] = True
            client_config["tls_trust_certs_file_path"] = self.ssl_cert_path

        self.client = pulsar.Client(**client_config)

        # Dictionary to store producers for different topics
        self.producers = {}

        logger.info(f"Initialized SecureEventEmitter for {service_url}")

    def _get_topic_name(self, event_type: str) -> str:
        """
        Get the full topic name for an event type.

        Args:
            event_type: The event type

        Returns:
            The full topic name
        """
        event_category = event_type.split(".")[0]
        return f"persistent://{self.tenant}/{self.namespace}/{event_category}-events"

    def _get_producer(self, topic: str) -> pulsar.Producer:
        """
        Get or create a producer for a topic.

        Args:
            topic: The topic name

        Returns:
            A Pulsar producer for the topic
        """
        if topic not in self.producers:
            self.producers[topic] = self.client.create_producer(
                topic=topic,
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=10,
                send_timeout_ms=30000,  # 30 seconds
            )
        return self.producers[topic]

    def _sign_message(self, message: str) -> str:
        """
        Sign a message using HMAC-SHA256.

        Args:
            message: The message to sign

        Returns:
            Base64-encoded signature
        """
        signature = hmac.new(key=self.secret_key.encode(), msg=message.encode(), digestmod=hashlib.sha256).digest()
        return base64.b64encode(signature).decode()

    def _add_signature(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a signature to event data for zero-trust verification.

        Args:
            event_data: The event data dictionary

        Returns:
            Event data with added signature
        """
        # Create a copy of the data to avoid modifying the original
        signed_data = event_data.copy()

        # Remove any existing signature
        signed_data.pop("_signature", None)

        # Create a canonical representation for signing
        canonical = json.dumps(signed_data, sort_keys=True)

        # Generate signature
        signature = self._sign_message(canonical)

        # Add signature to data
        signed_data["_signature"] = signature

        return signed_data

    def emit(self, event: BaseEvent) -> str:
        """
        Emit an event to the appropriate topic.

        Args:
            event: The event to emit

        Returns:
            Message ID of the emitted event
        """
        # Convert event to dictionary
        event_data = event.to_dict()

        # Add signature for zero-trust verification
        signed_data = self._add_signature(event_data)

        # Serialize to JSON
        serialized_data = json.dumps(signed_data)

        # Get topic name
        topic = self._get_topic_name(event_data["event_type"])

        # Get or create producer
        producer = self._get_producer(topic)

        # Send the message
        try:
            message_id = producer.send(serialized_data.encode("utf-8"))
            logger.debug(f"Emitted event {event.event_id} to topic {topic}")
            return message_id
        except Exception as e:
            logger.error(f"Failed to emit event {event.event_id}: {e}")
            raise

    async def emit_async(self, event: BaseEvent) -> str:
        """
        Emit an event asynchronously.

        Args:
            event: The event to emit

        Returns:
            Message ID of the emitted event
        """
        # This is a simple wrapper that runs the synchronous emit method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.emit, event)

    def close(self):
        """Close the Pulsar client and all producers."""
        logger.info("Closing SecureEventEmitter")
        for topic, producer in self.producers.items():
            producer.close()
        self.client.close()
