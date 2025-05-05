"""Apache Pulsar event client for the program synthesis system."""

from typing import Optional, Callable
import logging

from src.services.shared.models import BaseEvent, load_avro_schema
from src.services.shared.models.events.events import EventType
from infra.registration.schema_registry import SchemaRegistryClient

logger = logging.getLogger(__name__)


class PulsarEventClient:
    """Client for producing and consuming events using Apache Pulsar."""

    def __init__(
        self,
        service_url: str,
        tenant: str = "public",
        namespace: str = "program-synthesis",
        schema_registry_url: Optional[str] = None,
        schema_registry_token: Optional[str] = None,
    ):
        """Initialize the Pulsar event client."""
        try:
            import pulsar
            from pulsar.schema import AvroSchema
        except ImportError:
            raise ImportError(
                "Pulsar client not installed. Install with: pip install pulsar-client"
            )

        self.client = pulsar.Client(service_url)
        self.tenant = tenant
        self.namespace = namespace
        self.producers = {}
        self.consumers = {}
        self.event_handlers = {}

        # Load Event schema for Pulsar
        self.event_schema = AvroSchema(load_avro_schema("Event"))

        # Configure schema registry if provided
        self.schema_registry = None
        if schema_registry_url:
            self.schema_registry = SchemaRegistryClient(
                url=schema_registry_url, auth_token=schema_registry_token
            )

        logger.info(f"Initialized Pulsar client with service URL: {service_url}")

    def get_topic_name(self, event_type: EventType) -> str:
        """Get the appropriate topic name for an event type."""
        # Map event types to topics - customize as needed
        if event_type == EventType.CODE_GENERATION_REQUESTED:
            return "code-generator"
        elif event_type == EventType.CODE_GENERATION_COMPLETED:
            return "code-generator-results"
        elif event_type in (
            EventType.KNOWLEDGE_QUERY_REQUESTED,
            EventType.KNOWLEDGE_QUERY_COMPLETED,
        ):
            return "knowledge"
        elif event_type in (
            EventType.SYSTEM_ERROR,
            EventType.SYSTEM_HEALTH_CHECK,
            EventType.SYSTEM_SHUTDOWN,
        ):
            return "system"
        else:
            # Default to event type value
            return event_type.value.replace(".", "-")

    def get_producer(self, topic: str):
        """Get or create a producer for the given topic."""
        if topic not in self.producers:
            full_topic = f"persistent://{self.tenant}/{self.namespace}/{topic}"
            self.producers[topic] = self.client.create_producer(
                full_topic, schema=self.event_schema
            )
            logger.debug(f"Created producer for topic: {full_topic}")
        return self.producers[topic]

    def send_event(self, event: BaseEvent, topic: Optional[str] = None):
        """Send an event to a Pulsar topic."""
        # Determine topic from event type if not specified
        if topic is None:
            topic = self.get_topic_name(event.event_type)

        # Convert event to Avro-compatible format
        event_data = event.to_dict()

        # Send message
        producer = self.get_producer(topic)
        producer.send(event_data)
        logger.debug(f"Sent event {event.event_id} to topic {topic}")

    def subscribe(
        self,
        topic: str,
        subscription_name: str,
        event_handler: Callable[[BaseEvent], None],
        subscription_type: str = "Shared",
    ):
        """Subscribe to a topic."""
        try:
            import pulsar
        except ImportError:
            raise ImportError("Pulsar client not installed")

        if (topic, subscription_name) not in self.consumers:
            full_topic = f"persistent://{self.tenant}/{self.namespace}/{topic}"

            # Determine subscription type
            sub_type = pulsar.SubscriptionType.Shared
            if subscription_type == "Exclusive":
                sub_type = pulsar.SubscriptionType.Exclusive
            elif subscription_type == "Failover":
                sub_type = pulsar.SubscriptionType.Failover

            # Create consumer
            consumer = self.client.subscribe(
                full_topic,
                subscription_name,
                subscription_type=sub_type,
                schema=self.event_schema,
            )

            self.consumers[(topic, subscription_name)] = consumer
            self.event_handlers[(topic, subscription_name)] = event_handler

            # Start receive loop in background thread
            import threading

            thread = threading.Thread(
                target=self._receive_loop,
                args=(consumer, topic, subscription_name),
                daemon=True,
            )
            thread.start()

            logger.info(
                f"Subscribed to topic {full_topic} with subscription {subscription_name}"
            )

    def _receive_loop(self, consumer, topic, subscription_name):
        """Continuous loop to receive messages."""
        handler = self.event_handlers.get((topic, subscription_name))
        if not handler:
            logger.error(f"No handler registered for {topic}/{subscription_name}")
            return

        while True:
            try:
                msg = consumer.receive(timeout_millis=10000)  # 10 second timeout
                if msg:
                    try:
                        # Parse event data
                        event_data = msg.value()

                        # Create event object
                        event = BaseEvent.from_dict(event_data)

                        # Call handler
                        handler(event)

                        # Acknowledge message
                        consumer.acknowledge(msg)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        consumer.negative_acknowledge(msg)
            except Exception as e:
                if "timeout" not in str(e).lower():  # Ignore timeout exceptions
                    logger.error(f"Error receiving message: {e}")

    def close(self):
        """Close all producers, consumers and the client."""
        logger.info("Closing Pulsar client...")

        for producer in self.producers.values():
            try:
                producer.close()
            except Exception as e:
                logger.error(f"Error closing producer: {e}")

        for consumer in self.consumers.values():
            try:
                consumer.close()
            except Exception as e:
                logger.error(f"Error closing consumer: {e}")

        try:
            self.client.close()
        except Exception as e:
            logger.error(f"Error closing Pulsar client: {e}")
