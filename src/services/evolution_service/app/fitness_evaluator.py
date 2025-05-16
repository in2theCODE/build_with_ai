"""
Fitness evaluator for template evolution in the Neural Context Mesh.

This module provides fitness evaluation for templates and patterns
to drive the evolutionary process.
"""

import logging
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """
    Evaluates fitness of templates and patterns for evolution.

    Applies various metrics and heuristics to determine how well
    templates are performing in the neural mesh.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the fitness evaluator."""
        self.config = config

        # Fitness metrics configuration
        self.usage_weight = config.get("fitness_usage_weight", 0.4)
        self.success_weight = config.get("fitness_success_weight", 0.4)
        self.novelty_weight = config.get("fitness_novelty_weight", 0.2)
        self.min_usage_threshold = config.get("min_usage_threshold", 5)

        # Fitness evaluation history
        self.evaluation_history = {}

        logger.info("Fitness Evaluator initialized")

    async def initialize(self):
        """Initialize the fitness evaluator."""
        logger.info("Initializing Fitness Evaluator")
        # Any initialization steps
        logger.info("Fitness Evaluator initialized")

    async def evaluate_fitness(
        self,
        template: Dict[str, Any],
        usage_data: Dict[str, Any],
        population: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate the fitness of a template.

        Args:
            template: The template to evaluate
            usage_data: Usage data for the template
            population: The current population for comparisons

        Returns:
            Fitness score between 0 and 1
        """
        template_id = template["id"]

        # Get usage statistics
        usage_count = usage_data.get("usage_count", 0)
        success_count = usage_data.get("success_sum", 0)

        # Calculate usage score - more usage is better
        if usage_count < self.min_usage_threshold:
            # Templates with very low usage get a baseline score
            usage_score = 0.3
        else:
            # Calculate relative to population average
            avg_usage = sum(usage_data.get(t["id"], {}).get("usage_count", 0) for t in population) / max(
                1, len(population)
            )

            usage_factor = min(3.0, usage_count / max(1, avg_usage))
            usage_score = min(1.0, usage_factor / 3.0 + 0.3)  # 0.3 to 1.0

        # Calculate success score - higher success rate is better
        if usage_count > 0:
            success_rate = success_count / usage_count
            success_score = success_rate
        else:
            # No usage yet
            success_score = 0.5  # Neutral score

        # Calculate novelty score - more unique templates are valued
        novelty_score = await self._calculate_novelty(template, population)

        # Combine scores with weights
        fitness = (
            self.usage_weight * usage_score + self.success_weight * success_score + self.novelty_weight * novelty_score
        )

        # Record evaluation
        self.evaluation_history[template_id] = {
            "timestamp": datetime.now(),
            "fitness": fitness,
            "usage_score": usage_score,
            "success_score": success_score,
            "novelty_score": novelty_score,
            "usage_count": usage_count,
        }

        logger.debug(f"Evaluated fitness for {template_id}: {fitness}")
        return fitness

    async def _calculate_novelty(self, template: Dict[str, Any], population: List[Dict[str, Any]]) -> float:
        """
        Calculate novelty score for a template.

        Args:
            template: The template to evaluate
            population: The current population for comparisons

        Returns:
            Novelty score between 0 and 1
        """
        # If no embedding, use a simplified approach
        if "embedding" not in template:
            return 0.5  # Neutral score

        # Calculate average similarity to other templates
        similarities = []

        for other in population:
            if other["id"] == template["id"]:
                continue  # Skip self

            if "embedding" in other:
                similarity = self._cosine_similarity(template["embedding"], other["embedding"])
                similarities.append(similarity)

        if not similarities:
            return 0.5  # Neutral score

        # Average similarity
        avg_similarity = np.mean(similarities)

        # Novelty is inverse of similarity
        novelty = 1.0 - avg_similarity

        return novelty

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity between 0 and 1
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return max(0, min(1, (dot_product / (norm1 * norm2) + 1) / 2))
