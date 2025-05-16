from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import asyncio
from datetime import datetime

from src.services.shared.models.enums import SynapseState, LearningStrategy
from src.services.shared.models.synapses import Synapse
from src.services.shared.models.events import SynapseStateChangedPayload

from .learning_service import LearningService
from .connection_optimizer import ConnectionOptimizer
from .pathway_analyzer import PathwayAnalyzer
from .event_handlers import SynapseManagerEventHandler

logger = logging.getLogger(__name__)


class SynapseService:
    """
    Service managing synapses (connections) in the neural mesh.

    Handles connection creation, adaptation, learning, and optimization
    of the mesh network structure.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the synapse service."""
        self.config = config
        self.learning_service = LearningService(config)
        self.connection_optimizer = ConnectionOptimizer(config)
        self.pathway_analyzer = PathwayAnalyzer(config)
        self.event_handler = SynapseManagerEventHandler(self)

        # Synapse storage
        self.synapses: Dict[str, Synapse] = {}

        # Connection graph by node
        self.connections: Dict[str, Set[str]] = {}

        # Learning configuration
        self.learning_strategy = config.get("learning_strategy", LearningStrategy.HEBBIAN)
        self.learning_rate = config.get("learning_rate", 0.1)
        self.decay_rate = config.get("decay_rate", 0.01)

        # Optimization interval
        self.optimization_interval = config.get("optimization_interval", 3600)  # 1 hour
        self.last_optimization = datetime.now()

        logger.info("Synapse Service initialized with strategy: " + str(self.learning_strategy))

    async def start(self):
        """Start the synapse service."""
        logger.info("Starting Synapse Service")

        # Initialize components
        await self.learning_service.initialize()
        await self.connection_optimizer.initialize()
        await self.pathway_analyzer.initialize()

        # Start event handler
        await self.event_handler.start()

        # Load existing synapses from storage
        await self._load_synapses()

        # Start background tasks
        asyncio.create_task(self._run_periodic_maintenance())

        logger.info("Synapse Service started")

    async def stop(self):
        """Stop the synapse service."""
        logger.info("Stopping Synapse Service")

        # Save synapses to storage
        await self._save_synapses()

        # Stop event handler
        await self.event_handler.stop()

        logger.info("Synapse Service stopped")

    async def create_synapse(
        self,
        from_node_id: str,
        to_node_id: str,
        initial_weight: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Synapse:
        """
        Create a new synapse between nodes.

        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            initial_weight: Initial connection weight
            metadata: Optional metadata

        Returns:
            The created synapse
        """
        # Generate ID
        synapse_id = f"{from_node_id}_to_{to_node_id}"

        # Check if already exists
        if synapse_id in self.synapses:
            return self.synapses[synapse_id]

        # Create synapse
        synapse = Synapse(
            id=synapse_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            weight=initial_weight,
            state=SynapseState.FORMING,
            metadata=metadata or {},
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )

        # Store synapse
        self.synapses[synapse_id] = synapse

        # Update connection graph
        if from_node_id not in self.connections:
            self.connections[from_node_id] = set()
        self.connections[from_node_id].add(to_node_id)

        # Emit event
        await self._emit_synapse_state_changed(synapse, previous_state=None, previous_weight=0)

        logger.info(f"Created synapse: {synapse_id} with weight {initial_weight}")
        return synapse

    async def update_synapse_weight(self, synapse_id: str, weight_change: float) -> Optional[Synapse]:
        """
        Update the weight of a synapse.

        Args:
            synapse_id: Synapse ID
            weight_change: Change in weight (positive or negative)

        Returns:
            Updated synapse or None if not found
        """
        # Check if synapse exists
        if synapse_id not in self.synapses:
            logger.warning(f"Synapse not found: {synapse_id}")
            return None

        synapse = self.synapses[synapse_id]
        previous_weight = synapse.weight
        previous_state = synapse.state

        # Update weight with bounds checking
        synapse.weight = max(0.0, min(1.0, synapse.weight + weight_change))

        # Update state based on new weight
        if synapse.weight < 0.2:
            synapse.state = SynapseState.PRUNING
        elif synapse.weight < 0.4:
            synapse.state = SynapseState.WEAKENING
        elif synapse.weight > 0.8:
            synapse.state = SynapseState.STABLE
        elif synapse.weight > previous_weight:
            synapse.state = SynapseState.STRENGTHENING

        # Update timestamp
        synapse.last_updated = datetime.now()

        # Emit event if significant change or state change
        if abs(synapse.weight - previous_weight) > 0.05 or synapse.state != previous_state:
            await self._emit_synapse_state_changed(synapse, previous_state, previous_weight)

        logger.debug(f"Updated synapse: {synapse_id} weight from {previous_weight} to {synapse.weight}")
        return synapse

    async def get_synapse(self, synapse_id: str) -> Optional[Synapse]:
        """
        Get a synapse by ID.

        Args:
            synapse_id: Synapse ID

        Returns:
            The synapse or None if not found
        """
        return self.synapses.get(synapse_id)

    async def get_connections(self, node_id: str) -> List[Synapse]:
        """
        Get all connections from a node.

        Args:
            node_id: Node ID

        Returns:
            List of synapses connecting from the node
        """
        # Get target node IDs
        target_ids = self.connections.get(node_id, set())

        # Get synapses
        return [
            self.synapses[f"{node_id}_to_{target_id}"]
            for target_id in target_ids
            if f"{node_id}_to_{target_id}" in self.synapses
        ]

    async def handle_co_activation(self, from_node_id: str, to_node_id: str, activation_values: Tuple[float, float]):
        """
        Handle co-activation of two nodes for learning.

        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            activation_values: (source_activation, target_activation)
        """
        try:
            # Ignore self-connections
            if from_node_id == to_node_id:
                return

            # Get or create synapse
            synapse_id = f"{from_node_id}_to_{to_node_id}"
            if synapse_id not in self.synapses:
                await self.create_synapse(from_node_id, to_node_id, 0.1)

            synapse = self.synapses[synapse_id]

            # Calculate weight change based on learning strategy
            weight_change = await self.learning_service.compute_weight_change(
                synapse, activation_values, self.learning_strategy
            )

            # Apply weight change
            await self.update_synapse_weight(synapse_id, weight_change)

        except Exception as e:
            logger.error(f"Error handling co-activation: {e}")

    async def find_optimal_pathway(self, from_node_ids: List[str], to_node_ids: List[str]) -> List[str]:
        """
        Find optimal pathway between sets of nodes.

        Args:
            from_node_ids: Source node IDs
            to_node_ids: Target node IDs

        Returns:
            List of node IDs representing optimal pathway
        """
        try:
            pathway = await self.pathway_analyzer.find_optimal_pathway(
                from_node_ids, to_node_ids, self.synapses, self.connections
            )

            return pathway
        except Exception as e:
            logger.error(f"Error finding optimal pathway: {e}")
            return []

    async def _run_periodic_maintenance(self):
        """Run periodic maintenance on synapses."""
        while True:
            try:
                now = datetime.now()

                # Run weight decay
                await self._apply_weight_decay()

                # Run optimization if interval elapsed
                if (now - self.last_optimization).total_seconds() >= self.optimization_interval:
                    await self._optimize_connections()
                    self.last_optimization = now

            except Exception as e:
                logger.error(f"Error in periodic maintenance: {e}")

            # Run every 5 minutes
            await asyncio.sleep(300)

    async def _apply_weight_decay(self):
        """Apply weight decay to all synapses."""
        logger.debug("Applying weight decay")

        # Count of updated synapses
        updated_count = 0

        # Apply decay to all synapses
        for synapse_id, synapse in list(self.synapses.items()):
            # Skip very recently updated synapses
            if (datetime.now() - synapse.last_updated).total_seconds() < 3600:
                continue

            # Calculate decay amount based on time since last update
            time_factor = min(10, (datetime.now() - synapse.last_updated).total_seconds() / 86400)
            decay_amount = self.decay_rate * time_factor

            # Apply decay
            if synapse.weight > 0:
                previous_weight = synapse.weight
                previous_state = synapse.state

                synapse.weight = max(0, synapse.weight - decay_amount)

                # Update state if needed
                if synapse.weight < 0.2:
                    synapse.state = SynapseState.PRUNING
                elif synapse.weight < previous_weight:
                    synapse.state = SynapseState.WEAKENING

                # Update timestamp
                synapse.last_updated = datetime.now()

                # Count updated synapses
                updated_count += 1

                # Emit event if significant change
                if abs(synapse.weight - previous_weight) > 0.1 or synapse.state != previous_state:
                    await self._emit_synapse_state_changed(synapse, previous_state, previous_weight)

                # Remove if weight decayed to zero
                if synapse.weight == 0:
                    await self._remove_synapse(synapse_id)

        logger.debug(f"Applied weight decay to {updated_count} synapses")

    async def _optimize_connections(self):
        """Optimize network connections."""
        logger.info("Optimizing network connections")

        # Run optimization
        optimized_synapses = await self.connection_optimizer.optimize_network(self.synapses, self.connections)

        # Apply optimizations
        for synapse_id, weight_change in optimized_synapses.items():
            await self.update_synapse_weight(synapse_id, weight_change)

        logger.info(f"Optimized {len(optimized_synapses)} network connections")

    async def _remove_synapse(self, synapse_id: str):
        """
        Remove a synapse from the network.

        Args:
            synapse_id: Synapse ID
        """
        if synapse_id in self.synapses:
            synapse = self.synapses[synapse_id]

            # Remove from connection graph
            if synapse.from_node_id in self.connections:
                self.connections[synapse.from_node_id].discard(synapse.to_node_id)

                # Remove empty sets
                if not self.connections[synapse.from_node_id]:
                    del self.connections[synapse.from_node_id]

            # Remove synapse
            del self.synapses[synapse_id]

            logger.debug(f"Removed synapse: {synapse_id}")

    async def _emit_synapse_state_changed(
        self,
        synapse: Synapse,
        previous_state: Optional[SynapseState],
        previous_weight: float,
    ):
        """
        Emit synapse state changed event.

        Args:
            synapse: The synapse
            previous_state: Previous state
            previous_weight: Previous weight
        """
        # Create payload
        payload = SynapseStateChangedPayload(
            synapse_id=synapse.id,
            from_node_id=synapse.from_node_id,
            to_node_id=synapse.to_node_id,
            previous_state=previous_state or SynapseState.FORMING,
            new_state=synapse.state,
            weight_change=synapse.weight - previous_weight,
        )

        # Emit event
        await self.event_handler.emit_event(event_type="context.synapse.changed", payload=payload.dict())

    async def _load_synapses(self):
        """Load synapses from persistent storage."""
        # In a real implementation, this would load from database
        logger.info("Loading synapses from storage")

    async def _save_synapses(self):
        """Save synapses to persistent storage."""
        # In a real implementation, this would save to database
        logger.info("Saving synapses to storage")
