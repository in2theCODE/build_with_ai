from typing import Dict, List, Any, Tuple
import logging
import numpy as np
from datetime import datetime

from neural_context_mesh_models.enums import LearningStrategy
from neural_context_mesh_models.synapses import Synapse

logger = logging.getLogger(__name__)


class LearningService:
    """
    Service implementing learning algorithms for synapse adaptation.

    Provides different learning strategies for adapting synapse weights
    based on observed activations and patterns.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the learning service."""
        self.config = config

        # Learning parameters
        self.hebbian_rate = config.get("hebbian_learning_rate", 0.1)
        self.anti_hebbian_rate = config.get("anti_hebbian_learning_rate", 0.05)
        self.stdp_window = config.get("stdp_time_window", 1.0)  # in seconds
        self.reinforcement_learning_rate = config.get("reinforcement_learning_rate", 0.2)

        # History tracking for STDP
        self.activation_history: Dict[str, List[Dict[str, Any]]] = {}

        logger.info("Learning Service initialized")

    async def initialize(self):
        """Initialize the learning service."""
        logger.info("Initializing Learning Service")
        # Initialization tasks
        logger.info("Learning Service initialized")

    async def compute_weight_change(
        self,
        synapse: Synapse,
        activation_values: Tuple[float, float],
        strategy: LearningStrategy,
    ) -> float:
        """
        Compute weight change for a synapse based on learning strategy.

        Args:
            synapse: The synapse to update
            activation_values: (source_activation, target_activation)
            strategy: Learning strategy to use

        Returns:
            Weight change value
        """
        source_activation, target_activation = activation_values

        if strategy == LearningStrategy.HEBBIAN:
            return await self._hebbian_learning(synapse, source_activation, target_activation)

        elif strategy == LearningStrategy.ANTI_HEBBIAN:
            return await self._anti_hebbian_learning(synapse, source_activation, target_activation)

        elif strategy == LearningStrategy.STDP:
            return await self._stdp_learning(synapse, source_activation, target_activation)

        elif strategy == LearningStrategy.REINFORCEMENT:
            return await self._reinforcement_learning(synapse, source_activation, target_activation)

        else:
            # Default to Hebbian
            return await self._hebbian_learning(synapse, source_activation, target_activation)

    async def _hebbian_learning(self, synapse: Synapse, source_activation: float, target_activation: float) -> float:
        """
        Implement Hebbian learning: "Neurons that fire together, wire together."

        Args:
            synapse: The synapse to update
            source_activation: Activation value of source node
            target_activation: Activation value of target node

        Returns:
            Weight change value
        """
        # Basic Hebbian rule: weight change proportional to product of activations
        weight_change = self.hebbian_rate * source_activation * target_activation

        return weight_change

    async def _anti_hebbian_learning(
        self, synapse: Synapse, source_activation: float, target_activation: float
    ) -> float:
        """
        Implement anti-Hebbian learning: opposite of Hebbian.

        Args:
            synapse: The synapse to update
            source_activation: Activation value of source node
            target_activation: Activation value of target node

        Returns:
            Weight change value
        """
        # Anti-Hebbian rule: weight change proportional to negative product of activations
        weight_change = -self.anti_hebbian_rate * source_activation * target_activation

        return weight_change

    async def _stdp_learning(self, synapse: Synapse, source_activation: float, target_activation: float) -> float:
        """
        Implement spike-timing-dependent plasticity (STDP) learning.

        Args:
            synapse: The synapse to update
            source_activation: Activation value of source node
            target_activation: Activation value of target node

        Returns:
            Weight change value
        """
        # Record activations with timestamps
        now = datetime.now()

        # Record source activation
        if synapse.from_node_id not in self.activation_history:
            self.activation_history[synapse.from_node_id] = []

        self.activation_history[synapse.from_node_id].append({"value": source_activation, "timestamp": now})

        # Record target activation
        if synapse.to_node_id not in self.activation_history:
            self.activation_history[synapse.to_node_id] = []

        self.activation_history[synapse.to_node_id].append({"value": target_activation, "timestamp": now})

        # Prune old history entries
        self._prune_activation_history()

        # Compute STDP weight change
        weight_change = self._compute_stdp_weight_change(synapse)

        return weight_change

    def _compute_stdp_weight_change(self, synapse: Synapse) -> float:
        """
        Compute weight change based on STDP rule.

        Args:
            synapse: The synapse

        Returns:
            Weight change value
        """
        # Get activation histories
        source_history = self.activation_history.get(synapse.from_node_id, [])
        target_history = self.activation_history.get(synapse.to_node_id, [])

        if not source_history or not target_history:
            return 0.0

        # Get latest activations
        latest_source = source_history[-1]
        latest_target = target_history[-1]

        # Compute time difference
        time_diff = (latest_target["timestamp"] - latest_source["timestamp"]).total_seconds()

        # If target fires after source, strengthen connection
        # If source fires after target, weaken connection
        if abs(time_diff) > self.stdp_window:
            return 0.0  # Outside time window

        # Time factor - exponential decay with time difference
        time_factor = np.exp(-abs(time_diff) / self.stdp_window)

        # Activation factor - product of activations
        activation_factor = latest_source["value"] * latest_target["value"]

        # STDP weight change
        if time_diff > 0:
            # Target fired after source - strengthen
            weight_change = self.hebbian_rate * time_factor * activation_factor
        else:
            # Source fired after target - weaken
            weight_change = -self.anti_hebbian_rate * time_factor * activation_factor

        return weight_change

    def _prune_activation_history(self):
        """Prune old entries from activation history."""
        now = datetime.now()
        prune_threshold = now - np.timedelta64(int(self.stdp_window * 10), "s")

        for node_id in list(self.activation_history.keys()):
            # Keep only recent activations
            self.activation_history[node_id] = [
                activation
                for activation in self.activation_history[node_id]
                if activation["timestamp"] > prune_threshold
            ]

            # Limit history size
            if len(self.activation_history[node_id]) > 100:
                self.activation_history[node_id] = self.activation_history[node_id][-100:]

            # Remove empty histories
            if not self.activation_history[node_id]:
                del self.activation_history[node_id]

    async def _reinforcement_learning(
        self, synapse: Synapse, source_activation: float, target_activation: float
    ) -> float:
        """
        Implement reinforcement learning.

        Args:
            synapse: The synapse to update
            source_activation: Activation value of source node
            target_activation: Activation value of target node

        Returns:
            Weight change value
        """
        # Get reward signal (could come from success metrics, etc.)
        # For this example, we'll use a simple function of the activations
        reward = source_activation * target_activation * 2 - 0.5

        # Compute weight change based on reward
        if reward > 0:
            # Positive reward - strengthen connection
            weight_change = self.reinforcement_learning_rate * reward
        else:
            # Negative reward - weaken connection
            weight_change = self.reinforcement_learning_rate * reward * 0.5  # Less penalty

        return weight_change
