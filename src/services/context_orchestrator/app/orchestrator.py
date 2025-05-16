from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta

from neural_context_mesh_models.enums import ContextType
from neural_context_mesh_models.context import GlobalContextState

from .activation_manager import ActivationManager
from .context_router import ContextRouter
from .event_handlers import OrchestratorEventHandler

logger = logging.getLogger(__name__)


class NeuralContextOrchestrator:
    """
    Central orchestrator for the Neural Context Mesh.

    Coordinates context activation, routing, and synapse management
    across the entire mesh network.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the neural context orchestrator."""
        self.config = config
        self.activation_manager = ActivationManager(config)
        self.context_router = ContextRouter(config)
        self.event_handler = OrchestratorEventHandler(self)

        # Global state tracking
        self.global_state = GlobalContextState()

        # Track active context nodes
        self.active_nodes: Dict[str, float] = {}

        # Track recent activations for learning
        self.recent_activations: List[Dict[str, Any]] = []
        self.max_recent_history = config.get("max_recent_history", 100)

        # Optimization parameters
        self.optimization_interval = config.get("optimization_interval", 3600)  # 1 hour
        self.last_optimization = datetime.now()

        logger.info("Neural Context Orchestrator initialized")

    async def start(self):
        """Start the orchestrator and all its components."""
        logger.info("Starting Neural Context Orchestrator")

        # Start components
        await self.activation_manager.start()
        await self.context_router.start()
        await self.event_handler.start()

        # Start background tasks
        asyncio.create_task(self._run_periodic_optimization())
        asyncio.create_task(self._monitor_global_state())

        logger.info("Neural Context Orchestrator started successfully")

    async def stop(self):
        """Stop the orchestrator and all its components."""
        logger.info("Stopping Neural Context Orchestrator")

        # Stop components
        await self.event_handler.stop()
        await self.context_router.stop()
        await self.activation_manager.stop()

        logger.info("Neural Context Orchestrator stopped successfully")

    async def activate_context(
        self,
        query: str,
        context_type: Optional[ContextType] = None,
        min_activation: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Activate context nodes based on a query string.

        Args:
            query: The query string to activate context with
            context_type: Optional specific context type to query
            min_activation: Minimum activation threshold

        Returns:
            Dict containing activated nodes and their values
        """
        logger.debug(f"Activating context for query: {query}")

        # Generate query embedding
        query_vector = await self.activation_manager.embed_query(query)

        # Find matching contexts
        activations = await self.activation_manager.activate_nodes(query_vector, context_type, min_activation)

        # Update active nodes
        for node_id, activation_value in activations.items():
            self.active_nodes[node_id] = activation_value

        # Record for learning
        activation_record = {
            "timestamp": datetime.now(),
            "query": query,
            "context_type": context_type,
            "activations": activations.copy(),
        }
        self._record_activation(activation_record)

        # Route to appropriate nodes via events
        await self.context_router.route_activations(activations, query_vector)

        return {
            "query": query,
            "activations": activations,
            "context_type": context_type,
        }

    def _record_activation(self, activation_record: Dict[str, Any]):
        """Record activation for learning and evolution."""
        self.recent_activations.append(activation_record)

        # Trim history if needed
        if len(self.recent_activations) > self.max_recent_history:
            self.recent_activations = self.recent_activations[-self.max_recent_history :]

    async def _run_periodic_optimization(self):
        """Run periodic optimization of the mesh network."""
        while True:
            try:
                now = datetime.now()
                if (now - self.last_optimization).total_seconds() >= self.optimization_interval:
                    logger.info("Running periodic optimization")
                    await self._optimize_network()
                    self.last_optimization = now
            except Exception as e:
                logger.error(f"Error in periodic optimization: {e}")

            await asyncio.sleep(60)  # Check every minute

    async def _optimize_network(self):
        """Optimize the neural context mesh network."""
        # Identify patterns in activations
        co_activation_matrix = self._build_co_activation_matrix()

        # Update synapse weights based on co-activation
        await self.context_router.update_synapse_weights(co_activation_matrix)

        # Prune unused connections
        pruned_count = await self.context_router.prune_weak_synapses()

        logger.info(f"Network optimization complete. Pruned {pruned_count} weak synapses.")

    def _build_co_activation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build a co-activation matrix from recent activations."""
        # This tracks which nodes tend to activate together
        node_ids = set()
        for record in self.recent_activations:
            node_ids.update(record["activations"].keys())

        # Initialize empty matrix
        matrix = {node_id: {other_id: 0.0 for other_id in node_ids} for node_id in node_ids}

        # Compute co-activation strengths
        for record in self.recent_activations:
            activations = record["activations"]
            for node_id, value in activations.items():
                for other_id, other_value in activations.items():
                    if node_id != other_id:
                        # Nodes that activate strongly together get stronger connections
                        matrix[node_id][other_id] += value * other_value

        return matrix

    async def _monitor_global_state(self):
        """Monitor the global state of the mesh network."""
        while True:
            try:
                # Collect statistics
                active_node_count = len(self.active_nodes)

                # Update global state
                self.global_state.active_nodes = active_node_count
                self.global_state.last_updated = datetime.now()

                # Check for anomalies or health issues
                if active_node_count > self.config.get("max_active_nodes", 1000):
                    logger.warning(f"Excessive active nodes: {active_node_count}")

                # Clean up stale activations
                self._cleanup_stale_activations()

            except Exception as e:
                logger.error(f"Error monitoring global state: {e}")

            await asyncio.sleep(10)  # Every 10 seconds

    def _cleanup_stale_activations(self):
        """Clean up stale activations that haven't been used recently."""
        now = datetime.now()
        stale_cutoff = now - timedelta(seconds=self.config.get("activation_ttl", 300))

        # Remove stale activations from tracking
        stale_nodes = []
        for node_id, last_activation in self.active_nodes.items():
            if last_activation.timestamp < stale_cutoff:
                stale_nodes.append(node_id)

        for node_id in stale_nodes:
            del self.active_nodes[node_id]
