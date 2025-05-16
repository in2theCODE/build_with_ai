"""
Connection optimizer for the Neural Context Mesh.

This module provides optimization algorithms for the mesh network structure,
enhancing connectivity patterns for better information flow.
"""

import logging
from typing import Dict, List, Any, Set
import numpy as np
from datetime import datetime

from neural_context_mesh_models.synapses import Synapse

logger = logging.getLogger(__name__)


class ConnectionOptimizer:
    """
    Optimizes connection structure in the neural mesh.

    Identifies and optimizes connectivity patterns for better
    information flow and efficiency in the mesh network.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the connection optimizer."""
        self.config = config

        # Optimization parameters
        self.optimization_threshold = config.get("optimization_threshold", 0.3)
        self.reinforcement_factor = config.get("reinforcement_factor", 0.2)
        self.penalization_factor = config.get("penalization_factor", 0.1)
        self.pruning_threshold = config.get("pruning_threshold", 0.2)

        # Analysis metrics
        self.optimization_history = []

        logger.info("Connection Optimizer initialized")

    async def initialize(self):
        """Initialize the connection optimizer."""
        logger.info("Initializing Connection Optimizer")
        # Any initialization steps
        logger.info("Connection Optimizer initialized")

    async def optimize_network(
        self, synapses: Dict[str, Synapse], connections: Dict[str, Set[str]]
    ) -> Dict[str, float]:
        """
        Optimize the network connectivity structure.

        Args:
            synapses: Dict mapping synapse IDs to Synapse objects
            connections: Dict mapping node IDs to sets of connected node IDs

        Returns:
            Dict mapping synapse IDs to weight changes
        """
        try:
            # Start optimization
            logger.info("Starting network optimization")
            start_time = datetime.now()

            # Identify network motifs
            hub_nodes = self._identify_hub_nodes(connections)
            bottleneck_nodes = self._identify_bottleneck_nodes(connections)
            underconnected_nodes = self._identify_underconnected_nodes(connections)

            # Calculate optimization changes
            optimization_changes = {}

            # Strengthen connections to hub nodes
            for node_id in hub_nodes:
                for from_node in self._get_incoming_connections(node_id, connections):
                    synapse_id = f"{from_node}_to_{node_id}"
                    if synapse_id in synapses:
                        optimization_changes[synapse_id] = self.reinforcement_factor

            # Strengthen connections through bottleneck nodes
            for node_id in bottleneck_nodes:
                for from_node in self._get_incoming_connections(node_id, connections):
                    synapse_id = f"{from_node}_to_{node_id}"
                    if synapse_id in synapses:
                        optimization_changes[synapse_id] = self.reinforcement_factor * 1.5

            # Add connections to underconnected nodes
            for node_id in underconnected_nodes:
                # Find potential new connections
                for potential_source in connections:
                    if potential_source != node_id and node_id not in connections.get(potential_source, set()):
                        # Create weak connection
                        synapse_id = f"{potential_source}_to_{node_id}"
                        if synapse_id in synapses:
                            optimization_changes[synapse_id] = 0.1

            # Prune redundant connections
            redundant_connections = self._identify_redundant_connections(connections)
            for from_node, to_node in redundant_connections:
                synapse_id = f"{from_node}_to_{to_node}"
                if synapse_id in synapses:
                    optimization_changes[synapse_id] = -self.penalization_factor

            # Apply global stability adjustments
            stability_adjustments = await self._calculate_stability_adjustments(synapses, connections)

            for synapse_id, adjustment in stability_adjustments.items():
                if synapse_id in optimization_changes:
                    optimization_changes[synapse_id] += adjustment
                else:
                    optimization_changes[synapse_id] = adjustment

            # Record optimization metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.optimization_history.append(
                {
                    "timestamp": end_time.isoformat(),
                    "duration": duration,
                    "changes_count": len(optimization_changes),
                    "hub_nodes": len(hub_nodes),
                    "bottleneck_nodes": len(bottleneck_nodes),
                    "underconnected_nodes": len(underconnected_nodes),
                }
            )

            logger.info(f"Completed network optimization with {len(optimization_changes)} changes")
            return optimization_changes

        except Exception as e:
            logger.error(f"Error in network optimization: {e}")
            return {}

    def _identify_hub_nodes(self, connections: Dict[str, Set[str]]) -> List[str]:
        """
        Identify hub nodes in the network.

        Args:
            connections: Dict mapping node IDs to sets of connected node IDs

        Returns:
            List of hub node IDs
        """
        # Calculate incoming connection counts
        incoming_counts = {}

        for from_node, to_nodes in connections.items():
            for to_node in to_nodes:
                if to_node not in incoming_counts:
                    incoming_counts[to_node] = 0
                incoming_counts[to_node] += 1

        # Calculate mean and standard deviation
        if not incoming_counts:
            return []

        counts = list(incoming_counts.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        # Hub threshold: mean + 1.5 standard deviations
        hub_threshold = mean_count + 1.5 * std_count

        # Identify hubs
        hubs = [node_id for node_id, count in incoming_counts.items() if count > hub_threshold]

        logger.debug(f"Identified {len(hubs)} hub nodes")
        return hubs

    def _identify_bottleneck_nodes(self, connections: Dict[str, Set[str]]) -> List[str]:
        """
        Identify bottleneck nodes in the network.

        Args:
            connections: Dict mapping node IDs to sets of connected node IDs

        Returns:
            List of bottleneck node IDs
        """
        # Simplified bottleneck detection
        # In a real implementation, this would use more sophisticated network metrics

        # Find nodes with high betweenness
        bottlenecks = []

        for node_id in connections:
            incoming = len(self._get_incoming_connections(node_id, connections))
            outgoing = len(connections.get(node_id, set()))

            # A bottleneck has both incoming and outgoing connections
            if incoming > 0 and outgoing > 0:
                # Bottleneck score based on product of incoming and outgoing
                score = incoming * outgoing

                # Nodes with high scores are bottlenecks
                if score > 10:  # Arbitrary threshold
                    bottlenecks.append(node_id)

        logger.debug(f"Identified {len(bottlenecks)} bottleneck nodes")
        return bottlenecks

    def _identify_underconnected_nodes(self, connections: Dict[str, Set[str]]) -> List[str]:
        """
        Identify underconnected nodes in the network.

        Args:
            connections: Dict mapping node IDs to sets of connected node IDs

        Returns:
            List of underconnected node IDs
        """
        # Calculate total connection counts (incoming + outgoing)
        total_connections = {}

        # Add outgoing connections
        for from_node, to_nodes in connections.items():
            if from_node not in total_connections:
                total_connections[from_node] = 0
            total_connections[from_node] += len(to_nodes)

        # Add incoming connections
        for from_node, to_nodes in connections.items():
            for to_node in to_nodes:
                if to_node not in total_connections:
                    total_connections[to_node] = 0
                total_connections[to_node] += 1

        # Calculate mean and standard deviation
        if not total_connections:
            return []

        counts = list(total_connections.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        # Underconnected threshold: mean - 1 standard deviation
        underconnected_threshold = mean_count - 1.0 * std_count

        # Identify underconnected nodes
        underconnected = [
            node_id
            for node_id, count in total_connections.items()
            if count < underconnected_threshold and count > 0  # Must have at least one connection
        ]

        logger.debug(f"Identified {len(underconnected)} underconnected nodes")
        return underconnected

    def _identify_redundant_connections(self, connections: Dict[str, Set[str]]) -> List[tuple]:
        """
        Identify redundant connections in the network.

        Args:
            connections: Dict mapping node IDs to sets of connected node IDs

        Returns:
            List of (from_node, to_node) pairs for redundant connections
        """
        # Simplified redundancy detection
        # In a real implementation, this would use more sophisticated metrics

        redundant = []

        # Check for nodes with many outgoing connections to the same destinations
        for from_node, to_nodes in connections.items():
            if len(to_nodes) > 10:  # Arbitrary threshold
                # Find less important connections to prune
                # This is a very simplified approach
                to_nodes_list = list(to_nodes)
                # Consider the last few connections less important
                redundant.extend([(from_node, to_node) for to_node in to_nodes_list[-3:]])

        logger.debug(f"Identified {len(redundant)} redundant connections")
        return redundant

    def _get_incoming_connections(self, node_id: str, connections: Dict[str, Set[str]]) -> List[str]:
        """
        Get incoming connections to a node.

        Args:
            node_id: Target node ID
            connections: Dict mapping node IDs to sets of connected node IDs

        Returns:
            List of source node IDs
        """
        incoming = []

        for from_node, to_nodes in connections.items():
            if node_id in to_nodes:
                incoming.append(from_node)

        return incoming

    async def _calculate_stability_adjustments(
        self, synapses: Dict[str, Synapse], connections: Dict[str, Set[str]]
    ) -> Dict[str, float]:
        """
        Calculate adjustments to enhance network stability.

        Args:
            synapses: Dict mapping synapse IDs to Synapse objects
            connections: Dict mapping node IDs to sets of connected node IDs

        Returns:
            Dict mapping synapse IDs to adjustment values
        """
        # Simplified stability adjustment
        adjustments = {}

        # Smooth out extreme weights
        for synapse_id, synapse in synapses.items():
            # Adjust very high weights slightly downward
            if synapse.weight > 0.9:
                adjustments[synapse_id] = -0.05

            # Adjust very low weights that aren't meant to be pruned
            elif 0.3 > synapse.weight > self.pruning_threshold:
                adjustments[synapse_id] = 0.05

        return adjustments
