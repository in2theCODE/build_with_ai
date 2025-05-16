from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import uuid


from .pattern_storage import PatternStorage
from .pattern_embedder import PatternEmbedder
from .pattern_matcher import PatternMatcher
from .event_handlers import PatternNodeEventHandler

logger = logging.getLogger(__name__)


class PatternNodeService:
    """
    Service managing code pattern nodes in the neural mesh.

    Stores, retrieves, and matches code patterns and templates with
    vector-based similarity and tree structure awareness.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the pattern node service."""
        self.config = config
        self.pattern_storage = PatternStorage(config)
        self.pattern_embedder = PatternEmbedder(config)
        self.pattern_matcher = PatternMatcher(config)
        self.event_handler = PatternNodeEventHandler(self)

        # Active patterns tracking
        self.active_patterns: Dict[str, Dict[str, Any]] = {}

        logger.info("Pattern Node Service initialized")

    async def start(self):
        """Start the pattern node service."""
        logger.info("Starting Pattern Node Service")

        # Initialize storage
        await self.pattern_storage.initialize()

        # Initialize embedder
        await self.pattern_embedder.initialize()

        # Start event handler
        await self.event_handler.start()

        logger.info("Pattern Node Service started")

    async def stop(self):
        """Stop the pattern node service."""
        logger.info("Stopping Pattern Node Service")

        # Stop event handler
        await self.event_handler.stop()

        # Close storage
        await self.pattern_storage.close()

        logger.info("Pattern Node Service stopped")

    async def store_pattern(self, pattern_code: str, metadata: Dict[str, Any]) -> str:
        """
        Store a new code pattern in the system.

        Args:
            pattern_code: The code pattern to store
            metadata: Pattern metadata

        Returns:
            Pattern ID
        """
        try:
            # Generate ID
            pattern_id = str(uuid.uuid4())

            # Generate embedding
            embedding = await self.pattern_embedder.embed_pattern(pattern_code)

            # Extract tree structure
            tree_structure = await self.pattern_embedder.extract_tree_structure(pattern_code)

            # Store pattern
            await self.pattern_storage.store_pattern(
                pattern_id=pattern_id,
                pattern_code=pattern_code,
                embedding=embedding,
                tree_structure=tree_structure,
                metadata=metadata,
            )

            logger.info(f"Stored pattern {pattern_id}")
            return pattern_id

        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
            raise

    async def find_similar_patterns(
        self, query: str, limit: int = 10, min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find patterns similar to a query.

        Args:
            query: The query string or code snippet
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar patterns with scores
        """
        try:
            # Generate query embedding
            query_embedding = await self.pattern_embedder.embed_pattern(query)

            # Extract tree structure if it's code
            tree_structure = None
            if self.pattern_embedder.is_code(query):
                tree_structure = await self.pattern_embedder.extract_tree_structure(query)

            # Find similar patterns
            results = await self.pattern_storage.find_similar_patterns(
                embedding=query_embedding,
                tree_structure=tree_structure,
                limit=limit,
                min_similarity=min_similarity,
            )

            # Track activations
            for result in results:
                self.active_patterns[result["id"]] = {
                    "similarity": result["similarity"],
                    "activated_at": datetime.now(),
                }

            return results

        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            raise

    async def handle_activation(
        self,
        node_id: str,
        activation_value: float,
        query_vector: Optional[List[float]] = None,
    ):
        """
        Handle activation of a pattern node.

        Args:
            node_id: ID of the node being activated
            activation_value: Activation value
            query_vector: Optional query vector that caused activation
        """
        try:
            # Get pattern details
            pattern = await self.pattern_storage.get_pattern(node_id)
            if not pattern:
                logger.warning(f"Activated unknown pattern node: {node_id}")
                return

            # Track activation
            self.active_patterns[node_id] = {
                "activation": activation_value,
                "activated_at": datetime.now(),
            }

            # If it's from a direct query, no need to match again
            if query_vector:
                return

            # For propagated activations, find similar patterns
            results = await self.pattern_storage.find_similar_patterns(
                embedding=pattern["embedding"],
                tree_structure=pattern.get("tree_structure"),
                limit=5,
                min_similarity=0.8,
            )

            # Only include those not already active
            new_activations = [r for r in results if r["id"] != node_id and r["id"] not in self.active_patterns]

            # Emit activation events for these
            for result in new_activations:
                # Decay activation value by similarity
                propagated_value = activation_value * result["similarity"]

                if propagated_value >= 0.5:  # Only propagate significant activations
                    await self.event_handler.emit_pattern_activation(
                        pattern_id=result["id"],
                        activation_value=propagated_value,
                        source_node_id=node_id,
                    )

        except Exception as e:
            logger.error(f"Error handling pattern activation: {e}")
