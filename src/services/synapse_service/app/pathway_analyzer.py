"""
Pathway analyzer for the Neural Context Mesh.

This module analyzes and optimizes information pathways through
the neural mesh network.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import heapq
from datetime import datetime

from neural_context_mesh_models.synapses import Synapse, Pathway, PathwaySegment

logger = logging.getLogger(__name__)


class PathwayAnalyzer:
    """
    Analyzes and optimizes pathways through the neural mesh.

    Identifies optimal routes for information flow and maintains
    a library of successful pathways.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the pathway analyzer."""
        self.config = config

        # Pathway identification parameters
        self.max_pathway_length = config.get("max_pathway_length", 10)
        self.min_pathway_weight = config.get("min_pathway_weight", 0.3)

        # Pathway storage
        self.pathways: Dict[str, Pathway] = {}

        # Pathway usage statistics
        self.pathway_usage: Dict[str, Dict[str, Any]] = {}

        logger.info("Pathway Analyzer initialized")

    async def initialize(self):
        """Initialize the pathway analyzer."""
        logger.info("Initializing Pathway Analyzer")
        # Any initialization steps
        logger.info("Pathway Analyzer initialized")

    async def find_optimal_pathway(
        self,
        from_node_ids: List[str],
        to_node_ids: List[str],
        synapses: Dict[str, Synapse],
        connections: Dict[str, Set[str]],
    ) -> List[str]:
        """
        Find optimal pathway between sets of nodes.

        Args:
            from_node_ids: Source node IDs
            to_node_ids: Target node IDs
            synapses: Dict mapping synapse IDs to Synapse objects
            connections: Dict mapping node IDs to sets of connected node IDs

        Returns:
            List of node IDs representing optimal pathway
        """
        try:
            # Check for direct pathways first
            direct_pathway = self._check_direct_pathways(from_node_ids, to_node_ids, self.pathways)

            if direct_pathway:
                # Update usage statistics
                self._record_pathway_usage(direct_pathway)
                return direct_pathway.node_sequence

            # If no direct pathway found, search for optimal path
            optimal_path = None
            best_score = 0.0

            # Try from each source to each target
            for source in from_node_ids:
                for target in to_node_ids:
                    path, score = await self._find_path(source, target, synapses, connections)

                    if path and score > best_score:
                        optimal_path = path
                        best_score = score

            if not optimal_path:
                logger.warning(f"No pathway found between {from_node_ids} and {to_node_ids}")
                return []

            # Create pathway object
            pathway_id = f"pathway-{datetime.now().timestamp()}"

            # Create segments
            segments = []
            for i in range(len(optimal_path) - 1):
                from_id = optimal_path[i]
                to_id = optimal_path[i + 1]
                synapse_id = f"{from_id}_to_{to_id}"

                if synapse_id in synapses:
                    synapse = synapses[synapse_id]
                    segment = PathwaySegment(
                        synapse_id=synapse_id,
                        from_node_id=from_id,
                        to_node_id=to_id,
                        weight=synapse.weight,
                        traversal_time=1.0 / synapse.weight,  # Higher weight = faster traversal
                    )
                    segments.append(segment)

            # Calculate pathway metrics
            total_weight = sum(segment.weight for segment in segments) / len(segments)
            total_traversal_time = sum(segment.traversal_time for segment in segments)

            # Create pathway
            pathway = Pathway(
                id=pathway_id,
                segments=segments,
                start_node_id=optimal_path[0],
                end_node_id=optimal_path[-1],
                total_weight=total_weight,
                total_traversal_time=total_traversal_time,
                created_at=datetime.now(),
                usage_count=1,  # Initial usage
            )

            # Store pathway
            self.pathways[pathway_id] = pathway

            logger.info(f"Found optimal pathway: {optimal_path}")
            return optimal_path

        except Exception as e:
            logger.error(f"Error finding optimal pathway: {e}")
            return []

    def _check_direct_pathways(
        self,
        from_node_ids: List[str],
        to_node_ids: List[str],
        pathways: Dict[str, Pathway],
    ) -> Optional[Pathway]:
        """
        Check if there are existing direct pathways between the nodes.

        Args:
            from_node_ids: Source node IDs
            to_node_ids: Target node IDs
            pathways: Dict mapping pathway IDs to Pathway objects

        Returns:
            Direct pathway if found, None otherwise
        """
        # Check all pathways
        for pathway in pathways.values():
            if pathway.start_node_id in from_node_ids and pathway.end_node_id in to_node_ids:
                return pathway

        return None

    async def _find_path(
        self,
        from_node: str,
        to_node: str,
        synapses: Dict[str, Synapse],
        connections: Dict[str, Set[str]],
    ) -> Tuple[List[str], float]:
        """
        Find path from source to target using Dijkstra's algorithm.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            synapses: Dict mapping synapse IDs to Synapse objects
            connections: Dict mapping node IDs to sets of connected node IDs

        Returns:
            Tuple of (path, score)
        """
        # Initialize Dijkstra structures
        distances = {from_node: 0}
        previous = {}
        queue = [(0, from_node)]
        visited = set()

        while queue:
            # Get node with minimum distance
            dist, current = heapq.heappop(queue)

            # If we reached the target
            if current == to_node:
                break

            # Skip if already visited
            if current in visited:
                continue

            visited.add(current)

            # Get outgoing connections
            outgoing = connections.get(current, set())

            # Process each neighbor
            for neighbor in outgoing:
                # Skip if already visited
                if neighbor in visited:
                    continue

                # Get synapse
                synapse_id = f"{current}_to_{neighbor}"
                if synapse_id not in synapses:
                    continue

                synapse = synapses[synapse_id]

                # Skip weak connections
                if synapse.weight < self.min_pathway_weight:
                    continue

                # Calculate distance (inverse of weight)
                weight = synapse.weight
                distance = 1.0 / weight

                # Update distance if better
                alt_dist = distances[current] + distance
                if neighbor not in distances or alt_dist < distances[neighbor]:
                    distances[neighbor] = alt_dist
                    previous[neighbor] = current
                    heapq.heappush(queue, (alt_dist, neighbor))

        # Reconstruct path
        if to_node not in previous and from_node != to_node:
            return [], 0.0

        path = []
        current = to_node

        while current:
            path.append(current)
            current = previous.get(current)

        # Reverse path (from source to target)
        path.reverse()

        # Calculate path score
        if not path or len(path) < 2:
            return path, 0.0

        # Score is inversely proportional to path length and directly proportional to weights
        total_weight = 0.0
        for i in range(len(path) - 1):
            synapse_id = f"{path[i]}_to_{path[i + 1]}"
            if synapse_id in synapses:
                total_weight += synapses[synapse_id].weight

        avg_weight = total_weight / (len(path) - 1)
        length_factor = 1.0 / len(path)

        score = avg_weight * length_factor

        return path, score

    def _record_pathway_usage(self, pathway: Pathway):
        """
        Record usage of a pathway.

        Args:
            pathway: The pathway that was used
        """
        pathway_id = pathway.id

        # Initialize if not yet tracked
        if pathway_id not in self.pathway_usage:
            self.pathway_usage[pathway_id] = {
                "id": pathway_id,
                "count": 0,
                "last_used": None,
                "sequence": pathway.node_sequence,
            }

        # Update usage
        self.pathway_usage[pathway_id]["count"] += 1
        self.pathway_usage[pathway_id]["last_used"] = datetime.now()

        # Update pathway usage count
        pathway.usage_count += 1
