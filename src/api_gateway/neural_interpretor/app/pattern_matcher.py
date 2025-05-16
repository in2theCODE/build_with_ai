import asyncio
import logging
from typing import Any, Dict, List, Optional
import uuid

from neo4j import AsyncGraphDatabase


logger = logging.getLogger(__name__)


class PatternMatcher:
    """
    Pattern matcher using Neo4j for storing and matching patterns.
    Uses graph relationships and vector similarity for fast pattern matching.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize the pattern matcher with Neo4j connection

        Args:
            neo4j_uri: URI for Neo4j database
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize database and create constraints/indexes"""
        async with self.driver.session() as session:
            # Create constraints
            await session.run("CREATE CONSTRAINT pattern_id IF NOT EXISTS FOR (p:Pattern) REQUIRE p.id IS UNIQUE")

            # Create vector index if Neo4j supports it (Neo4j 5.0+)
            try:
                await session.run(
                    "CALL db.index.vector.createNodeIndex("
                    "'pattern_embedding', "
                    "'Pattern', "
                    "'embedding', "
                    "384, "  # Dimension of embedding vectors
                    "'cosine'"
                    ")"
                )
            except Exception as e:
                logger.warning(f"Could not create vector index: {e}")

    async def store_pattern(self, pattern_text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> str:
        """
        Store a pattern with its embedding for later matching

        Args:
            pattern_text: The text pattern to store
            embedding: Vector embedding of the pattern
            metadata: Additional metadata for the pattern

        Returns:
            pattern_id: ID of the stored pattern
        """
        pattern_id = str(uuid.uuid4())
        metadata = metadata or {}

        async with self.driver.session() as session:
            await session.run(
                """
                CREATE (p:Pattern {
                    id: $id,
                    text: $text,
                    embedding: $embedding,
                    metadata: $metadata,
                    created_at: datetime()
                })
                """,
                id=pattern_id,
                text=pattern_text,
                embedding=embedding,
                metadata=metadata,
            )

        return pattern_id

    async def match(
        self,
        query_text: str,
        embedding: List[float],
        threshold: float = 0.85,
        limit: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        Match query against stored patterns using vector similarity

        Args:
            query_text: Text to match
            embedding: Vector embedding of the query
            threshold: Similarity threshold
            limit: Maximum number of matches to return

        Returns:
            best_match: Best matching pattern or None if no match above threshold
        """
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (p:Pattern)
                WHERE p.text IS NOT NULL
                WITH p, gds.similarity.cosine(p.embedding, $embedding) AS score
                WHERE score >= $threshold
                RETURN p.id AS id, p.text AS text, p.metadata AS metadata, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                embedding=embedding,
                threshold=threshold,
                limit=limit,
            )

            matches = [record async for record in result]

            if not matches:
                return None

            best_match = matches[0]
            return {
                "id": best_match["id"],
                "text": best_match["text"],
                "score": best_match["score"],
                "metadata": best_match["metadata"],
            }

    async def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern by its ID"""
        async with self.driver.session() as session:
            result = await session.run(
                "MATCH (p:Pattern {id: $id}) DELETE p RETURN count(p) as deleted",
                id=pattern_id,
            )
            record = await result.single()
            return record and record["deleted"] > 0

    async def list_patterns(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all stored patterns"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (p:Pattern)
                RETURN p.id AS id, p.text AS text, p.metadata AS metadata
                LIMIT $limit
                """,
                limit=limit,
            )

            return [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": record["metadata"],
                }
                async for record in result
            ]

    async def close(self):
        """Close Neo4j connection"""
        await self.driver.close()
