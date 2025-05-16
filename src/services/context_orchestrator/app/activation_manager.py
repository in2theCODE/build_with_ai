from typing import Dict, List, Any, Optional
import logging
import numpy as np
from datetime import datetime

from src.services.shared.models.enums import ContextType, ActivationFunction

logger = logging.getLogger(__name__)


class ActivationManager:
    """
    Manages activation of context nodes in the neural mesh.

    Handles embedding generation, activation functions, and
    propagation of activation through the network.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the activation manager."""
        self.config = config

        # Function for computing activation values
        self.activation_function_type = config.get("activation_function", ActivationFunction.SIGMOID)

        # Initialize embedding client
        # This would connect to your embedding service
        self.embedding_dimension = config.get("embedding_dimension", 1536)

        # Node activation cache
        self.node_activation_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("Activation Manager initialized")

    async def start(self):
        """Start the activation manager."""
        logger.info("Starting Activation Manager")
        # Initialize connections to node services
        # Set up any background tasks

    async def stop(self):
        """Stop the activation manager."""
        logger.info("Stopping Activation Manager")
        # Close connections
        # Cancel background tasks

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding vector for a query string.

        Args:
            query: The query string to embed

        Returns:
            Embedding vector as a list of floats
        """
        # In production, this would call your embedding service
        # For now, generate a random vector of appropriate dimension
        # This is just a placeholder - you would replace with actual embedding call
        vector = np.random.normal(0, 1, self.embedding_dimension).tolist()
        return vector

    async def activate_nodes(
        self,
        query_vector: List[float],
        context_type: Optional[ContextType] = None,
        min_activation: float = 0.5,
    ) -> Dict[str, float]:
        """
        Activate context nodes based on query vector similarity.

        Args:
            query_vector: The query embedding vector
            context_type: Optional context type filter
            min_activation: Minimum activation threshold

        Returns:
            Dict mapping node IDs to activation values
        """
        # In production, this would query your vector database
        # For simulation, we'll generate some synthetic activations

        # These would be replaced with actual vector DB query results
        # showing which nodes match the query vector
        mock_results = [
            {
                "node_id": f"node-{i}",
                "similarity": max(0.1, min(0.99, 0.5 + np.random.normal(0, 0.2))),
                "context_type": np.random.choice(list(ContextType)).value,
            }
            for i in range(10)
        ]

        # Filter by context type if specified
        if context_type:
            mock_results = [r for r in mock_results if r["context_type"] == context_type.value]

        # Apply activation function and threshold
        activations = {}
        for result in mock_results:
            activation_value = self._compute_activation(result["similarity"])
            if activation_value >= min_activation:
                activations[result["node_id"]] = activation_value

                # Cache activation for learning
                self.node_activation_cache[result["node_id"]] = {
                    "value": activation_value,
                    "timestamp": datetime.now(),
                    "query_vector": query_vector,
                    "context_type": result["context_type"],
                }

        return activations

    def _compute_activation(self, similarity: float) -> float:
        """
        Compute activation value using the configured activation function.

        Args:
            similarity: Cosine similarity or other similarity measure

        Returns:
            Activation value between 0 and 1
        """
        if self.activation_function_type == ActivationFunction.SIGMOID:
            # Sigmoid function: 1 / (1 + exp(-k * (x - x0)))
            k = self.config.get("sigmoid_steepness", 10)
            x0 = self.config.get("sigmoid_midpoint", 0.5)
            return 1.0 / (1.0 + np.exp(-k * (similarity - x0)))

        elif self.activation_function_type == ActivationFunction.RELU:
            # ReLU with scaling
            threshold = self.config.get("relu_threshold", 0.5)
            scale = self.config.get("relu_scale", 1.0)
            return max(0, scale * (similarity - threshold))

        elif self.activation_function_type == ActivationFunction.THRESHOLD:
            # Simple threshold function
            threshold = self.config.get("threshold_value", 0.5)
            return 1.0 if similarity >= threshold else 0.0

        elif self.activation_function_type == ActivationFunction.TANH:
            # Tanh function scaled to [0,1]
            scale = self.config.get("tanh_scale", 2.0)
            return (np.tanh(scale * (similarity - 0.5)) + 1) / 2

        else:
            # Default linear function
            return max(0, min(1, similarity))
