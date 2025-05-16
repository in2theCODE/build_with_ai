from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import uuid


from .knowledge_store import KnowledgeStore
from .retrieval_service import RetrievalService
from .event_handlers import KnowledgeNodeEventHandler

logger = logging.getLogger(__name__)


class KnowledgeNodeService:
    """
    Service managing knowledge nodes in the neural mesh.

    Stores, retrieves, and manages knowledge such as code examples,
    documentation, and other context-relevant information.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the knowledge node service."""
        self.config = config
        self.knowledge_store = KnowledgeStore(config)
        self.retrieval_service = RetrievalService(config)
        self.event_handler = KnowledgeNodeEventHandler(self)

        # Active knowledge tracking
        self.active_knowledge: Dict[str, Dict[str, Any]] = {}

        logger.info("Knowledge Node Service initialized")

    async def start(self):
        """Start the knowledge node service."""
        logger.info("Starting Knowledge Node Service")

        # Initialize storage
        await self.knowledge_store.initialize()

        # Initialize retrieval service
        await self.retrieval_service.initialize()

        # Start event handler
        await self.event_handler.start()

        logger.info("Knowledge Node Service started")

    async def stop(self):
        """Stop the knowledge node service."""
        logger.info("Stopping Knowledge Node Service")

        # Stop event handler
        await self.event_handler.stop()

        # Close storage
        await self.knowledge_store.close()

        logger.info("Knowledge Node Service stopped")

    async def store_knowledge(self, content: str, content_type: str, metadata: Dict[str, Any]) -> str:
        """
        Store a new knowledge item in the system.

        Args:
            content: The knowledge content to store
            content_type: Type of content (e.g., code, documentation)
            metadata: Knowledge metadata

        Returns:
            Knowledge ID
        """
        try:
            # Generate ID
            knowledge_id = str(uuid.uuid4())

            # Generate embedding
            embedding = await self.retrieval_service.generate_embedding(content)

            # Store knowledge
            await self.knowledge_store.store_item(
                knowledge_id=knowledge_id,
                content=content,
                content_type=content_type,
                embedding=embedding,
                metadata=metadata,
            )

            logger.info(f"Stored knowledge {knowledge_id}")
            return knowledge_id

        except Exception as e:
            logger.error(f"Error storing knowledge: {e}")
            raise

    async def retrieve_knowledge(
        self,
        query: str,
        content_type: Optional[str] = None,
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge based on a query.

        Args:
            query: The query string
            content_type: Optional filter for content type
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of knowledge items with scores
        """
        try:
            # Generate query embedding
            query_embedding = await self.retrieval_service.generate_embedding(query)

            # Find relevant knowledge
            results = await self.knowledge_store.find_similar_items(
                embedding=query_embedding,
                content_type=content_type,
                limit=limit,
                min_similarity=min_similarity,
            )

            # Track activations
            for result in results:
                self.active_knowledge[result["id"]] = {
                    "similarity": result["similarity"],
                    "activated_at": datetime.now(),
                }

            return results

        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            raise

    async def handle_activation(
        self,
        node_id: str,
        activation_value: float,
        query_vector: Optional[List[float]] = None,
    ):
        """
        Handle activation of a knowledge node.

        Args:
            node_id: ID of the node being activated
            activation_value: Activation value
            query_vector: Optional query vector that caused activation
        """
        try:
            # Get knowledge item details
            knowledge = await self.knowledge_store.get_item(node_id)
            if not knowledge:
                logger.warning(f"Activated unknown knowledge node: {node_id}")
                return

            # Track activation
            self.active_knowledge[node_id] = {
                "activation": activation_value,
                "activated_at": datetime.now(),
            }

            # Retrieve related knowledge
            if "embedding" in knowledge:
                results = await self.knowledge_store.find_similar_items(
                    embedding=knowledge["embedding"],
                    content_type=knowledge.get("content_type"),
                    limit=5,
                    min_similarity=0.8,
                )

                # Emit activation events for related knowledge
                for result in results:
                    if result["id"] != node_id and result["id"] not in self.active_knowledge:
                        # Decay activation value by similarity
                        propagated_value = activation_value * result["similarity"]

                        if propagated_value >= 0.5:  # Only propagate significant activations
                            await self.event_handler.emit_knowledge_activation(
                                knowledge_id=result["id"],
                                activation_value=propagated_value,
                                source_node_id=node_id,
                            )

        except Exception as e:
            logger.error(f"Error handling knowledge activation: {e}")
