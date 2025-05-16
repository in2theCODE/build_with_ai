from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from src.services.shared.models.enums import SynapseState
from .synapses import Synapse
from src.services.shared.models.events import (
    ContextNodeActivatedPayload,
    SynapseStateChangedPayload,
)

logger = logging.getLogger(__name__)


class ContextRouter:
    """
    Routes context activations through the neural mesh.

    Manages the connections (synapses) between context nodes and
    handles propagation of activations through the network.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the context router."""
        self.config = config

        # Pulsar client for sending events
        self.event_emitter = None  # Will be initialized on start

        # Synapse tracking
        self.synapses: Dict[str, Dict[str, Synapse]] = {}  # from_id -> to_id -> Synapse

        # Learning parameters
        self.learning_rate = config.get("learning_rate", 0.1)
        self.pruning_threshold = config.get("pruning_threshold", 0.2)

        logger.info("Context Router initialized")

    async def start(self):
        """Start the context router."""
        logger.info("Starting Context Router")

        # Initialize event emitter

        self.event_emitter = await create_event_emitter(self.config)

        # Load existing synapses from storage
        await self._load_synapses()

        logger.info("Context Router started")

    async def stop(self):
        """Stop the context router."""
        logger.info("Stopping Context Router")

        # Persist synapses to storage
        await self._persist_synapses()

        # Close event emitter
        if self.event_emitter:
            await self.event_emitter.close()

        logger.info("Context Router stopped")

    async def route_activations(self, activations: Dict[str, float], query_vector: Optional[List[float]] = None):
        """
        Route activations to appropriate nodes via events.

        Args:
            activations: Dict mapping node IDs to activation values
            query_vector: Optional original query vector
        """
        for node_id, activation_value in activations.items():
            # Get node type (would come from your actual node service)
            context_type = "unknown"  # Placeholder

            # Create payload
            payload = ContextNodeActivatedPayload(
                node_id=node_id,
                context_type=context_type,
                activation_value=activation_value,
                query_vector=query_vector,
            )

            # Emit activation event
            await self.event_emitter.emit_event(event_type="context.node.activated", payload=payload.dict())

            # Propagate activation through synapses
            await self._propagate_activation(node_id, activation_value)

    async def _propagate_activation(self, node_id: str, activation_value: float):
        """
        Propagate activation through connected synapses.

        Args:
            node_id: ID of the activated node
            activation_value: Activation value of the node
        """
        if node_id not in self.synapses:
            return

        # For each connected node
        for to_node_id, synapse in self.synapses[node_id].items():
            # Calculate propagated activation value
            propagated_value = activation_value * synapse.weight

            # Only propagate if above threshold
            if propagated_value >= self.config.get("propagation_threshold", 0.3):
                # Create payload with source information
                payload = ContextNodeActivatedPayload(
                    node_id=to_node_id,
                    context_type="propagated",
                    activation_value=propagated_value,
                    related_contexts=[node_id],
                )

                # Emit propagated activation event
                await self.event_emitter.emit_event(event_type="context.node.activated", payload=payload.dict())

    async def update_synapse_weights(self, co_activation_matrix: Dict[str, Dict[str, float]]):
        """
        Update synapse weights based on co-activation patterns.

        Args:
            co_activation_matrix: Matrix of co-activation strengths
        """
        # Track changed synapses for events
        changed_synapses = []

        # For each source node
        for from_node_id, connections in co_activation_matrix.items():
            if from_node_id not in self.synapses:
                self.synapses[from_node_id] = {}

            # For each target node
            for to_node_id, co_activation in connections.items():
                if from_node_id == to_node_id:
                    continue  # Skip self-connections

                # Get or create synapse
                if to_node_id not in self.synapses[from_node_id]:
                    synapse = Synapse(
                        id=f"{from_node_id}_{to_node_id}",
                        from_node_id=from_node_id,
                        to_node_id=to_node_id,
                        weight=0.1,  # Initial weight
                        state=SynapseState.FORMING,
                        last_updated=datetime.now(),
                    )
                    self.synapses[from_node_id][to_node_id] = synapse
                else:
                    synapse = self.synapses[from_node_id][to_node_id]

                # Update weight based on co-activation (Hebbian learning)
                old_weight = synapse.weight
                old_state = synapse.state

                # Apply learning rule - strengthen if co-activation is high
                synapse.weight += self.learning_rate * co_activation

                # Ensure weight stays in [0, 1]
                synapse.weight = max(0, min(1, synapse.weight))

                # Update state based on weight
                if synapse.weight < self.pruning_threshold:
                    synapse.state = SynapseState.PRUNING
                elif synapse.weight > 0.8:
                    synapse.state = SynapseState.STABLE
                elif synapse.weight > old_weight:
                    synapse.state = SynapseState.STRENGTHENING
                elif synapse.weight < old_weight:
                    synapse.state = SynapseState.WEAKENING

                # Record significant changes
                if abs(synapse.weight - old_weight) > 0.1 or synapse.state != old_state:
                    synapse.last_updated = datetime.now()
                    changed_synapses.append((synapse, old_weight, old_state))

        # Emit events for changed synapses
        for synapse, old_weight, old_state in changed_synapses:
            payload = SynapseStateChangedPayload(
                synapse_id=synapse.id,
                from_node_id=synapse.from_node_id,
                to_node_id=synapse.to_node_id,
                previous_state=old_state,
                new_state=synapse.state,
                weight_change=synapse.weight - old_weight,
            )

            await self.event_emitter.emit_event(event_type="context.synapse.changed", payload=payload.dict())

    async def prune_weak_synapses(self) -> int:
        """
        Remove weak synapses from the network.

        Returns:
            Number of pruned synapses
        """
        pruned_count = 0

        # For each source node
        for from_node_id in list(self.synapses.keys()):
            to_remove = []

            # Find weak connections to remove
            for to_node_id, synapse in self.synapses[from_node_id].items():
                if synapse.weight < self.pruning_threshold:
                    to_remove.append(to_node_id)

            # Remove weak connections
            for to_node_id in to_remove:
                del self.synapses[from_node_id][to_node_id]
                pruned_count += 1

            # Remove empty source nodes
            if not self.synapses[from_node_id]:
                del self.synapses[from_node_id]

        return pruned_count

    async def _load_synapses(self):
        """Load synapses from persistent storage."""
        # In production, this would load from your database
        logger.info("Loading synapses from storage")
        # Placeholder - actual implementation would load from DB

    async def _persist_synapses(self):
        """Save synapses to persistent storage."""
        # In production, this would save to your database
        logger.info("Persisting synapses to storage")
        # Placeholder - actual implementation would save to DB
