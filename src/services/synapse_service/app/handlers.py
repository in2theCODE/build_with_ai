"""
Event handlers for the Synapse Manager service.

This module provides event handling for the Synapse Manager
service, including processing incoming events and emitting
outgoing events.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SynapseManagerEventHandler:
    """
    Event handler for the Synapse Manager service.

    Processes incoming events and emits outgoing events
    for synapse-related operations.
    """

    def __init__(self, synapse_service):
        """
        Initialize the event handler.

        Args:
            synapse_service: The synapse service instance
        """
        self.synapse_service = synapse_service
        self.config = synapse_service.config
        self.event_emitter = None
        self.event_listener = None
        self.running = False

        logger.info("Synapse Manager Event Handler initialized")

    async def start(self):
        """Start the event handler."""
        logger.info("Starting Synapse Manager Event Handler")

        # Initialize Pulsar client and event listener
        self.event_listener, self.event_emitter = await self._create_pulsar_clients()

        # Register message handlers
        self.event_listener.register_handler("context.node.activated", self._handle_node_activation)
        self.event_listener.register_handler("synapse.create", self._handle_create_synapse)
        self.event_listener.register_handler("synapse.update", self._handle_update_synapse)
        self.event_listener.register_handler("pathway.find", self._handle_find_pathway)

        # Start listening for events
        await self.event_listener.start()
        self.running = True

        logger.info("Synapse Manager Event Handler started")

    async def stop(self):
        """Stop the event handler."""
        logger.info("Stopping Synapse Manager Event Handler")

        self.running = False

        # Stop the event listener
        if self.event_listener:
            await self.event_listener.stop()

        # Close the event emitter
        if self.event_emitter:
            await self.event_emitter.close()

        logger.info("Synapse Manager Event Handler stopped")

    async def _create_pulsar_clients(self):
        """
        Create Pulsar clients for event handling.

        Returns:
            Tuple of (event_listener, event_emitter)
        """
        try:
            from pulsar_client_init import create_event_listener, create_event_emitter

            # Create listener
            subscription_name = f"synapse-manager-{datetime.now().timestamp()}"
            listener = await create_event_listener(
                self.config,
                subscription_name=subscription_name,
                topics=[
                    "context.node.activated",
                    "synapse.create",
                    "synapse.update",
                    "pathway.find",
                ],
            )

            # Create emitter
            emitter = await create_event_emitter(self.config)

            return listener, emitter

        except Exception as e:
            logger.error(f"Error creating Pulsar clients: {e}")
            raise

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
            source_node_id = payload.get("source_node_id")
            related_contexts = payload.get("related_contexts", [])

            logger.debug(f"Handling node activation: {node_id}")

            # If there's a source node, update synapse connection
            if source_node_id:
                # Handle co-activation (this node was activated by another node)
                await self.synapse_service.handle_co_activation(
                    source_node_id,
                    node_id,
                    (
                        activation_value,
                        activation_value,
                    ),  # Both activations same for now
                )

            # Process related contexts (if any)
            for related_id in related_contexts:
                if related_id != node_id and related_id != source_node_id:
                    # These are indirectly related, so use a lower activation correlation
                    await self.synapse_service.handle_co_activation(
                        related_id,
                        node_id,
                        (
                            activation_value * 0.5,
                            activation_value,
                        ),  # Lower source activation
                    )

        except Exception as e:
            logger.error(f"Error handling node activation: {e}")

    async def _handle_create_synapse(self, event):
        """
        Handle synapse creation requests.

        Args:
            event: The synapse creation event
        """
        try:
            payload = event["payload"]
            from_node_id = payload.get("from_node_id")
            to_node_id = payload.get("to_node_id")
            initial_weight = payload.get("initial_weight", 0.5)
            metadata = payload.get("metadata", {})

            if not from_node_id or not to_node_id:
                logger.warning("Missing from_node_id or to_node_id in create synapse event")
                return

            # Create synapse
            synapse = await self.synapse_service.create_synapse(
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                initial_weight=initial_weight,
                metadata=metadata,
            )

            # Emit response event if requested
            response_topic = payload.get("response_topic")
            if response_topic and self.event_emitter:
                response_payload = {
                    "synapse_id": synapse.id,
                    "success": True,
                    "request_id": payload.get("request_id"),
                }

                await self.event_emitter.emit_event(event_type=response_topic, payload=response_payload)

        except Exception as e:
            logger.error(f"Error handling create synapse: {e}")

    async def _handle_update_synapse(self, event):
        """
        Handle synapse update requests.

        Args:
            event: The synapse update event
        """
        try:
            payload = event["payload"]
            synapse_id = payload.get("synapse_id")
            weight_change = payload.get("weight_change", 0.0)

            if not synapse_id:
                logger.warning("Missing synapse_id in update synapse event")
                return

            # Update synapse
            synapse = await self.synapse_service.update_synapse_weight(
                synapse_id=synapse_id, weight_change=weight_change
            )

            # Emit response event if requested
            response_topic = payload.get("response_topic")
            if response_topic and self.event_emitter and synapse:
                response_payload = {
                    "synapse_id": synapse_id,
                    "new_weight": synapse.weight,
                    "new_state": synapse.state,
                    "success": synapse is not None,
                    "request_id": payload.get("request_id"),
                }

                await self.event_emitter.emit_event(event_type=response_topic, payload=response_payload)

        except Exception as e:
            logger.error(f"Error handling update synapse: {e}")

    async def _handle_find_pathway(self, event):
        """
        Handle pathway finding requests.

        Args:
            event: The pathway finding event
        """
        try:
            payload = event["payload"]
            from_node_ids = payload.get("from_node_ids", [])
            to_node_ids = payload.get("to_node_ids", [])

            if not from_node_ids or not to_node_ids:
                logger.warning("Missing from_node_ids or to_node_ids in find pathway event")
                return

            # Find pathway
            pathway = await self.synapse_service.find_optimal_pathway(
                from_node_ids=from_node_ids, to_node_ids=to_node_ids
            )

            # Emit response event if requested
            response_topic = payload.get("response_topic")
            if response_topic and self.event_emitter:
                response_payload = {
                    "pathway": pathway,
                    "success": bool(pathway),
                    "request_id": payload.get("request_id"),
                }

                await self.event_emitter.emit_event(event_type=response_topic, payload=response_payload)

        except Exception as e:
            logger.error(f"Error handling find pathway: {e}")

    async def emit_synapse_state_changed(self, payload):
        """
        Emit a synapse state changed event.

        Args:
            payload: Event payload
        """
        if not self.event_emitter:
            logger.warning("No event emitter available")
            return

        try:
            # Emit event
            await self.event_emitter.emit_event(event_type="context.synapse.changed", payload=payload)

            logger.debug(f"Emitted synapse state changed: {payload.get('synapse_id')}")

        except Exception as e:
            logger.error(f"Error emitting synapse state changed: {e}")
