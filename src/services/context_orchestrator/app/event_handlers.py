import logging


logger = logging.getLogger(__name__)


class OrchestratorEventHandler:
    """Handles events for the context orchestrator."""

    def __init__(self, orchestrator):
        """Initialize the event handler."""
        self.orchestrator = orchestrator
        self.config = orchestrator.config
        self.event_listener = None
        self.running = False

        logger.info("Orchestrator Event Handler initialized")

    async def start(self):
        """Start the event handler."""
        logger.info("Starting Orchestrator Event Handler")

        # Initialize Pulsar client and event listener
        from src.services.shared.pulsar.event_listener import create_event_listener

        self.event_listener = await create_event_listener(
            self.config,
            subscription_name="orchestrator-events",
            topics=[
                "context.query",
                "context.node.activated",
                "context.synapse.changed",
                "context.evolution",
            ],
        )

        # Register message handlers
        self.event_listener.register_handler("context.query", self._handle_context_query)
        self.event_listener.register_handler("context.node.activated", self._handle_node_activated)
        self.event_listener.register_handler("context.synapse.changed", self._handle_synapse_changed)
        self.event_listener.register_handler("context.evolution", self._handle_evolution_event)

        # Start listening for events
        await self.event_listener.start()
        self.running = True

        logger.info("Orchestrator Event Handler started")

    async def stop(self):
        """Stop the event handler."""
        logger.info("Stopping Orchestrator Event Handler")

        self.running = False

        # Stop the event listener
        if self.event_listener:
            await self.event_listener.stop()

        logger.info("Orchestrator Event Handler stopped")

    async def _handle_context_query(self, event):
        """
        Handle context query events.

        Args:
            event: The context.query event
        """
        try:
            payload = event["payload"]
            query = payload.get("query")
            context_type = payload.get("context_type")
            min_activation = payload.get("min_activation", 0.5)

            # Process query through orchestrator
            result = await self.orchestrator.activate_context(query, context_type, min_activation)

            # Send response event
            # Implementation depends on your event system

            logger.debug(f"Handled context query: {query}")
        except Exception as e:
            logger.error(f"Error handling context query: {e}")

    async def _handle_node_activated(self, event):
        """
        Handle node activation events.

        Args:
            event: The context.node.activated event
        """
        try:
            payload = event["payload"]
            node_id = payload.get("node_id")
            activation_value = payload.get("activation_value")

            # Update orchestrator's active nodes
            self.orchestrator.active_nodes[node_id] = activation_value

            logger.debug(f"Handled node activation: {node_id} = {activation_value}")
        except Exception as e:
            logger.error(f"Error handling node activation: {e}")

    async def _handle_synapse_changed(self, event):
        """
        Handle synapse state changed events.

        Args:
            event: The context.synapse.changed event
        """
        try:
            payload = event["payload"]
            synapse_id = payload.get("synapse_id")
            new_state = payload.get("new_state")

            # Update internal tracking as needed
            logger.debug(f"Handled synapse change: {synapse_id} = {new_state}")
        except Exception as e:
            logger.error(f"Error handling synapse change: {e}")

    async def _handle_evolution_event(self, event):
        """
        Handle evolution events.

        Args:
            event: The context.evolution event
        """
        try:
            payload = event["payload"]
            evolution_id = payload.get("evolution_id")
            mechanism = payload.get("mechanism")

            # Update internal tracking as needed
            logger.debug(f"Handled evolution event: {evolution_id} using {mechanism}")
        except Exception as e:
            logger.error(f"Error handling evolution event: {e}")
