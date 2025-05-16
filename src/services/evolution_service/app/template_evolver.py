"""
Template evolver for the Neural Context Mesh.

This module handles the evolution of templates through various
evolutionary mechanisms like mutation, crossover, and selection.
"""

import logging
import random
from typing import Dict, List, Any
import copy
import uuid
from datetime import datetime

from neural_context_mesh_models.enums import EvolutionMechanism

logger = logging.getLogger(__name__)


class TemplateEvolver:
    """
    Handles evolution of templates in the neural mesh.

    Implements various evolutionary mechanisms to create and
    improve templates based on fitness and usage patterns.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the template evolver."""
        self.config = config

        # Evolution parameters
        self.mutation_rate = config.get("mutation_rate", 0.2)
        self.crossover_rate = config.get("crossover_rate", 0.3)
        self.selection_pressure = config.get("selection_pressure", 1.5)
        self.population_size = config.get("population_size", 100)
        self.elitism_count = config.get("elitism_count", 5)

        # Embedding service for template similarity
        self.embedding_service = None

        logger.info("Template Evolver initialized")

    async def initialize(self):
        """Initialize the template evolver."""
        logger.info("Initializing Template Evolver")
        # Any initialization steps
        logger.info("Template Evolver initialized")

    async def evolve_population(
        self, population: List[Dict[str, Any]], fitness_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Evolve a population of templates.

        Args:
            population: Current population of templates
            fitness_scores: Dict mapping template IDs to fitness scores

        Returns:
            New evolved population
        """
        if not population:
            logger.warning("Empty population for evolution")
            return []

        try:
            # Sort population by fitness
            sorted_population = sorted(population, key=lambda t: fitness_scores.get(t["id"], 0.0), reverse=True)

            # Create new population
            new_population = []

            # Add elite individuals (best performers)
            elites = sorted_population[: self.elitism_count]
            new_population.extend(copy.deepcopy(elites))

            # Fill the rest of the population with evolved templates
            while len(new_population) < len(population):
                # Randomly select evolution mechanism
                mechanism = self._select_evolution_mechanism()

                if mechanism == EvolutionMechanism.MUTATION:
                    # Select a parent based on fitness
                    parent = self._select_template(sorted_population, fitness_scores)

                    # Create mutated child
                    child = await self._mutate_template(parent)

                    # Add to new population
                    new_population.append(child)

                elif mechanism == EvolutionMechanism.CROSSOVER:
                    # Select two parents based on fitness
                    parent1 = self._select_template(sorted_population, fitness_scores)
                    parent2 = self._select_template(sorted_population, fitness_scores)

                    # Create crossover child
                    child = await self._crossover_templates(parent1, parent2)

                    # Add to new population
                    new_population.append(child)

                # Other evolution mechanisms could be added here

            # Ensure we don't exceed the desired population size
            new_population = new_population[: len(population)]

            logger.info(f"Evolved population: {len(new_population)} templates")
            return new_population

        except Exception as e:
            logger.error(f"Error evolving population: {e}")
            return population  # Return original population on error

    async def create_template_from_emergence(self, emergence_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new template from an emergent pattern.

        Args:
            emergence_pattern: The emergent pattern

        Returns:
            New template
        """
        try:
            # Extract pattern components
            pattern_type = emergence_pattern.get("type", "unknown")
            component_ids = emergence_pattern.get("patterns", [])
            description = emergence_pattern.get("description", "Emergent pattern")

            # Create template ID
            template_id = f"template-{str(uuid.uuid4())[:8]}"

            # Create template
            template = {
                "id": template_id,
                "name": f"Emergent {pattern_type.capitalize()} Template",
                "content": f"# Auto-generated from emergent pattern\n# {description}",
                "category": pattern_type,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": 1,
                "parent_ids": component_ids,
                "generation": 0,
                "fitness_score": 0.5,  # Initial neutral score
                "usage_count": 0,
                "success_count": 0,
                "metadata": {
                    "emergent": True,
                    "emergence_pattern_id": emergence_pattern.get("id"),
                    "emergence_confidence": emergence_pattern.get("confidence", 1.0),
                    "emergence_description": description,
                },
                "evolution_mechanism": EvolutionMechanism.EMERGENCE,
            }

            logger.info(f"Created template from emergence: {template_id}")
            return template

        except Exception as e:
            logger.error(f"Error creating template from emergence: {e}")
            # Create a simple fallback template
            return {
                "id": f"template-{str(uuid.uuid4())[:8]}",
                "name": "Fallback Template",
                "content": "# Fallback template",
                "category": "fallback",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": 1,
                "parent_ids": [],
                "generation": 0,
                "fitness_score": 0.1,
                "metadata": {"emergent": True, "error": str(e)},
            }

    def _select_evolution_mechanism(self) -> EvolutionMechanism:
        """
        Select an evolution mechanism based on configured rates.

        Returns:
            Selected evolution mechanism
        """
        # Calculate probabilities
        total = self.mutation_rate + self.crossover_rate
        mutation_prob = self.mutation_rate / total

        # Random selection
        r = random.random()

        if r < mutation_prob:
            return EvolutionMechanism.MUTATION
        else:
            return EvolutionMechanism.CROSSOVER

    def _select_template(self, population: List[Dict[str, Any]], fitness_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Select a template using tournament selection.

        Args:
            population: Population of templates
            fitness_scores: Dict mapping template IDs to fitness scores

        Returns:
            Selected template
        """
        # Tournament selection
        tournament_size = min(3, len(population))
        tournament = random.sample(population, tournament_size)

        # Select the best from tournament
        best = max(tournament, key=lambda t: fitness_scores.get(t["id"], 0.0))

        return best

    async def _mutate_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a mutated version of a template.

        Args:
            template: Parent template

        Returns:
            Mutated child template
        """
        # Create a copy of the template
        child = copy.deepcopy(template)

        # Update child properties
        child["id"] = f"template-{str(uuid.uuid4())[:8]}"
        child["name"] = f"{template['name']} (Mutated)"
        child["version"] = template.get("version", 1) + 1
        child["parent_ids"] = [template["id"]]
        child["generation"] = template.get("generation", 0) + 1
        child["created_at"] = datetime.now().isoformat()
        child["updated_at"] = datetime.now().isoformat()
        child["fitness_score"] = 0.0  # Reset fitness
        child["usage_count"] = 0
        child["success_count"] = 0
        child["evolution_mechanism"] = EvolutionMechanism.MUTATION

        # Simple mutation placeholder
        # In a real implementation, this would make meaningful changes to the template content
        child["content"] = f"# Mutated from parent: {template['id']}\n{template.get('content', '')}"

        # Record mutation in metadata
        child["metadata"] = child.get("metadata", {})
        child["metadata"]["mutation_from"] = template["id"]
        child["metadata"]["mutation_time"] = datetime.now().isoformat()

        logger.debug(f"Mutated template: {child['id']} from {template['id']}")
        return child

    async def _crossover_templates(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a child template by crossing over two parents.

        Args:
            parent1: First parent template
            parent2: Second parent template

        Returns:
            Child template
        """
        # Create a new template
        child = {
            "id": f"template-{str(uuid.uuid4())[:8]}",
            "name": "Crossover Template",
            "version": 1,
            "parent_ids": [parent1["id"], parent2["id"]],
            "generation": max(parent1.get("generation", 0), parent2.get("generation", 0)) + 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "fitness_score": 0.0,  # Reset fitness
            "usage_count": 0,
            "success_count": 0,
            "category": parent1.get("category", "crossover"),
            "evolution_mechanism": EvolutionMechanism.CROSSOVER,
        }

        # Simple crossover placeholder
        # In a real implementation, this would combine content from both parents intelligently
        child["content"] = (
            f"# Crossover from parents: {parent1['id']} and {parent2['id']}\n"
            f"# Parent 1 content:\n{parent1.get('content', '')}\n\n"
            f"# Parent 2 content:\n{parent2.get('content', '')}"
        )

        # Combine metadata
        child["metadata"] = {
            "crossover_from": [parent1["id"], parent2["id"]],
            "crossover_time": datetime.now().isoformat(),
        }

        logger.debug(f"Created crossover template: {child['id']} from {parent1['id']} and {parent2['id']}")
        return child
