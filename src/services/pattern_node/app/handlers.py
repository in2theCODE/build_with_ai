"""
Event handlers for the Pattern Node service.

This module provides event handling for the Pattern Node
service, including processing incoming events and emitting
outgoing events.
"""

import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PatternNodeEventHandler:
    """
    Event handler for the Pattern Node service.

    Processes incoming events and emits outgoing events
    for pattern-related operations.
    """

    def __init__(self, pattern_service):
        """
        Initialize the event handler.

        Args:
            pattern_service: The pattern service instance
        """
        self.pattern_service = pattern_service
        self.config = pattern_service.config
        self.event_emitter = None
        self.event_listener = None
        self.running = False

        logger.info("Pattern Node Event Handler initialized")

    async def start(self):
        """Start the event handler."""
        logger.info("Starting Pattern Node Event Handler")

        # Initialize Pulsar client and event listener
        self.event_listener, self.event_emitter = await self._create_pulsar_clients()

        # Register message handlers
        self.event_listener.register_handler("context.node.activated", self._handle_node_activation)
        self.event_listener.register_handler("pattern.store", self._handle_store_pattern)
        self.event_listener.register_handler("pattern.search", self._handle_search_patterns)

        # Start listening for events
        await self.event_listener.start()
        self.running = True

        logger.info("Pattern Node Event Handler started")

    async def stop(self):
        """Stop the event handler."""
        logger.info("Stopping Pattern Node Event Handler")

        self.running = False

        # Stop the event listener
        if self.event_listener:
            await self.event_listener.stop()

        # Close the event emitter
        if self.event_emitter:
            await self.event_emitter.close()

        logger.info("Pattern Node Event Handler stopped")

    async def _create_pulsar_clients(self):
        """
        Create Pulsar clients for event handling.

        Returns:
            Tuple of (event_listener, event_emitter)
        """
        try:
            # Import necessary modules from your existing infrastructure
            from ...shared.pulsar.event_listener import SecureEventListener
            from ...shared.pulsar.event_emitter import SecureEventEmitter
            from src.services.shared.models import EventType, BaseEvent

            # Get Pulsar service URL from config
            pulsar_service_url = self.config.get("pulsar_service_url", "pulsar://pulsar:6650")

            # Create subscription name with timestamp to ensure uniqueness
            subscription_name = f"pattern-node-{datetime.now().timestamp()}"

            # Define event types to listen for
            event_types = [
                EventType("context.node.activated"),
                EventType("pattern.store"),
                EventType("pattern.search"),
            ]

            # Create event listener
            listener = SecureEventListener(
                service_url=pulsar_service_url,
                subscription_name=subscription_name,
                event_types=event_types,
            )

            # Create event emitter
            emitter = SecureEventEmitter(service_url=pulsar_service_url)

            # Create a wrapper for the emitter to maintain the same interface
            class EmitterWrapper:
                """
                emmiter wrapperm
                """

                def __init__(self, secure_emitter):
                    self.secure_emitter = secure_emitter

                async def emit_event(self, event_type, payload):
                    # Create a BaseEvent from the event_type and payload
                    event = BaseEvent(event_type=event_type, payload=payload)
                    # Emit the event using the secure emitter
                    await self.secure_emitter.emit_async(event)

                async def close(self):
                    self.secure_emitter.close()

            # Return the wrapped emitter to maintain the expected interface
            return listener, EmitterWrapper(emitter)

        except Exception as e:
            logger.error(f"Error creating Pulsar clients: {e}")

            # If we can't create the real clients, use placeholders as a fallback
            class PlaceholderEmitter:
                async def emit_event(self, event_type, payload):
                    logger.info(f"Would emit event: {event_type} with payload: {payload}")

                async def close(self):
                    pass

            class PlaceholderListener:
                def register_handler(self, event_type, handler):
                    logger.info(f"Registered handler for event type: {event_type}")

                async def start(self):
                    pass

                async def stop(self):
                    pass

            listener = PlaceholderListener()
            emitter = PlaceholderEmitter()

            return listener, emitter

    async def _handle_node_activation(self, event):
        """
        Handle node activation events.

        Args:
            event: The node activation event
        """
        try:
            payload = event["payload"]
            node_id = payload.get("node_id")
            activation_value = payload.get("activation_value", 0.0)
            query_vector = payload.get("query_vector")

            logger.debug(f"Handling node activation: {node_id}")

            # Process through pattern service
            await self.pattern_service.handle_activation(node_id, activation_value, query_vector)

        except Exception as e:
            logger.error(f"Error handling node activation: {e}")

    async def _handle_store_pattern(self, event):
        """
        Handle pattern store events.

        Args:
            event: The pattern store event
        """
        try:
            payload = event["payload"]
            pattern_code = payload.get("pattern_code")
            metadata = payload.get("metadata", {})

            if not pattern_code:
                logger.warning("Missing pattern code in store pattern event")
                return

            # Store pattern through pattern service
            pattern_id = await self.pattern_service.store_pattern(pattern_code, metadata)

            # Emit response event if requested
            response_topic = payload.get("response_topic")
            if response_topic and self.event_emitter:
                response_payload = {
                    "pattern_id": pattern_id,
                    "success": bool(pattern_id),
                    "request_id": payload.get("request_id"),
                }

                await self.event_emitter.emit_event(event_type=response_topic, payload=response_payload)

        except Exception as e:
            logger.error(f"Error handling store pattern: {e}")

    async def _handle_search_patterns(self, event):
        """
        Handle pattern search events.

        Args:
            event: The pattern search event
        """
        try:
            payload = event["payload"]
            query = payload.get("query")
            limit = payload.get("limit", 10)
            min_similarity = payload.get("min_similarity", 0.7)

            if not query:
                logger.warning("Missing query in search patterns event")
                return

            # Check if query is code or text
            if self.pattern_service.pattern_embedder.is_code(query):
                # If code, extract tree structure
                tree_structure = await self.pattern_service.pattern_embedder.extract_tree_structure(query)
            else:
                tree_structure = None

            # Generate embedding
            embedding = await self.pattern_service.pattern_embedder.embed_pattern(query)

            # Search patterns
            results = await self.pattern_service.pattern_storage.find_similar_patterns(
                embedding=embedding,
                tree_structure=tree_structure,
                limit=limit,
                min_similarity=min_similarity,
            )

            # Emit response event if requested
            response_topic = payload.get("response_topic")
            if response_topic and self.event_emitter:
                response_payload = {
                    "results": results,
                    "query": query,
                    "request_id": payload.get("request_id"),
                }

                await self.event_emitter.emit_event(event_type=response_topic, payload=response_payload)

        except Exception as e:
            logger.error(f"Error handling search patterns: {e}")

    async def emit_pattern_activation(
        self,
        pattern_id: str,
        activation_value: float,
        source_node_id: Optional[str] = None,
    ):
        """
        Emit a pattern activation event.

        Args:
            pattern_id: Pattern ID
            activation_value: Activation value
            source_node_id: Optional source node ID
        """
        if not self.event_emitter:
            logger.warning("No event emitter available")
            return

        try:
            # Create payload
            payload = {
                "node_id": pattern_id,
                "context_type": "pattern",
                "activation_value": activation_value,
                "timestamp": datetime.now().isoformat(),
            }

            if source_node_id:
                payload["source_node_id"] = source_node_id
                payload["related_contexts"] = [source_node_id]

            # Emit event
            await self.event_emitter.emit_event(event_type="context.node.activated", payload=payload)

            logger.debug(f"Emitted pattern activation: {pattern_id}")

        except Exception as e:
            logger.error(f"Error emitting pattern activation: {e}")
