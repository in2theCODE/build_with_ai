from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime
import uuid
import random

from neural_context_mesh_models.enums import EvolutionMechanism
from neural_context_mesh_models.events import EvolutionEventPayload

from .template_evolver import TemplateEvolver
from .fitness_evaluator import FitnessEvaluator
from .emergence_detector import EmergenceDetector
from .event_handlers import EvolutionNodeEventHandler

logger = logging.getLogger(__name__)


class EvolutionService:
    """
    Service managing template evolution in the neural mesh.

    Handles learning from usage patterns, evolving templates,
    and detecting emergent properties in the system.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the evolution service."""
        self.config = config
        self.template_evolver = TemplateEvolver(config)
        self.fitness_evaluator = FitnessEvaluator(config)
        self.emergence_detector = EmergenceDetector(config)
        self.event_handler = EvolutionNodeEventHandler(self)

        # Track populations and generations
        self.template_populations: Dict[str, List[Dict[str, Any]]] = {}
        self.current_generation = 0
        self.evolution_interval = config.get("evolution_interval", 3600)  # seconds
        self.last_evolution = datetime.now()

        # Track pattern usage for fitness
        self.pattern_usage: Dict[str, Dict[str, Any]] = {}

        logger.info("Evolution Service initialized")

    async def start(self):
        """Start the evolution service."""
        logger.info("Starting Evolution Service")

        # Initialize components
        await self.template_evolver.initialize()
        await self.fitness_evaluator.initialize()
        await self.emergence_detector.initialize()

        # Start event handler
        await self.event_handler.start()

        # Start background tasks
        asyncio.create_task(self._periodic_evolution())
        asyncio.create_task(self._scan_for_emergence())

        logger.info("Evolution Service started")

    async def stop(self):
        """Stop the evolution service."""
        logger.info("Stopping Evolution Service")

        # Stop event handler
        await self.event_handler.stop()

        logger.info("Evolution Service stopped")

    async def record_pattern_usage(
        self,
        pattern_id: str,
        context: Dict[str, Any],
        success_rating: Optional[float] = None,
    ):
        """
        Record pattern usage for fitness calculation.

        Args:
            pattern_id: ID of the pattern used
            context: Context of the usage
            success_rating: Optional rating of success (0-1)
        """
        try:
            # Initialize if not yet tracked
            if pattern_id not in self.pattern_usage:
                self.pattern_usage[pattern_id] = {
                    "usage_count": 0,
                    "success_sum": 0,
                    "usage_contexts": [],
                    "last_used": None,
                }

            # Update tracking
            self.pattern_usage[pattern_id]["usage_count"] += 1
            self.pattern_usage[pattern_id]["last_used"] = datetime.now()

            # Store success rating if provided
            if success_rating is not None:
                self.pattern_usage[pattern_id]["success_sum"] += success_rating

            # Store context sample
            if len(self.pattern_usage[pattern_id]["usage_contexts"]) < 10:
                self.pattern_usage[pattern_id]["usage_contexts"].append(context)
            else:
                # Replace a random context to maintain diversity
                idx = random.randint(0, 9)
                self.pattern_usage[pattern_id]["usage_contexts"][idx] = context

            logger.debug(f"Recorded usage of pattern {pattern_id}")

        except Exception as e:
            logger.error(f"Error recording pattern usage: {e}")

    async def evolve_templates(self, population_id: str) -> List[str]:
        """
        Evolve a population of templates.

        Args:
            population_id: ID of the template population

        Returns:
            IDs of newly evolved templates
        """
        try:
            # Check if population exists
            if population_id not in self.template_populations:
                logger.warning(f"Template population {population_id} not found")
                return []

            # Get the current population
            population = self.template_populations[population_id]

            # Calculate fitness for each template
            fitness_scores = await self._calculate_fitness(population)

            # Evolution step
            evolved_templates = await self.template_evolver.evolve_population(population, fitness_scores)

            # Store new population
            self.template_populations[population_id] = evolved_templates

            # Increment generation
            self.current_generation += 1

            # Return IDs of evolved templates
            evolved_ids = [t["id"] for t in evolved_templates if t.get("generation") == self.current_generation]

            # Emit events for each evolved template
            for template in evolved_templates:
                if template.get("generation") == self.current_generation:
                    await self._emit_evolution_event(template)

            logger.info(f"Evolved population {population_id}: {len(evolved_ids)} new templates")
            return evolved_ids

        except Exception as e:
            logger.error(f"Error evolving templates: {e}")
            return []

    async def _calculate_fitness(self, population: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate fitness scores for a population of templates.

        Args:
            population: List of templates

        Returns:
            Dict mapping template IDs to fitness scores
        """
        fitness_scores = {}

        for template in population:
            template_id = template["id"]

            # Get usage statistics
            usage = self.pattern_usage.get(template_id, {"usage_count": 0, "success_sum": 0})

            # Calculate fitness based on usage and success
            if usage["usage_count"] > 0:
                # Success rate
                success_rate = usage["success_sum"] / usage["usage_count"]

                # Usage frequency relative to population average
                avg_usage = sum(self.pattern_usage.get(t["id"], {}).get("usage_count", 0) for t in population) / len(
                    population
                )
                usage_factor = usage["usage_count"] / max(1, avg_usage)

                # Combine factors
                fitness = success_rate * 0.7 + min(1.0, usage_factor) * 0.3
            else:
                # New templates get a modest default fitness
                fitness = 0.5

            fitness_scores[template_id] = fitness

        return fitness_scores

    async def _periodic_evolution(self):
        """Run evolution periodically on all template populations."""
        while True:
            try:
                now = datetime.now()
                if (now - self.last_evolution).total_seconds() >= self.evolution_interval:
                    logger.info(f"Running periodic evolution on {len(self.template_populations)} populations")

                    for population_id in self.template_populations:
                        await self.evolve_templates(population_id)

                    self.last_evolution = now
            except Exception as e:
                logger.error(f"Error in periodic evolution: {e}")

            await asyncio.sleep(60)  # Check every minute

    async def _scan_for_emergence(self):
        """Periodically scan for emergent patterns in the system."""
        while True:
            try:
                # Run emergence detection
                emergent_patterns = await self.emergence_detector.detect_emergence(self.pattern_usage)

                # Process any detected emergent patterns
                for pattern in emergent_patterns:
                    logger.info(f"Detected emergent pattern: {pattern['id']}")

                    # Create a new template from the emergent pattern
                    new_template = await self.template_evolver.create_template_from_emergence(pattern)

                    # Add to appropriate population
                    category = new_template.get("category", "default")
                    if category not in self.template_populations:
                        self.template_populations[category] = []

                    self.template_populations[category].append(new_template)

                    # Emit emergence event
                    await self._emit_emergence_event(new_template, pattern)

            except Exception as e:
                logger.error(f"Error scanning for emergence: {e}")

            await asyncio.sleep(300)  # Check every 5 minutes

    async def _emit_evolution_event(self, template: Dict[str, Any]):
        """
        Emit an evolution event.

        Args:
            template: The evolved template
        """
        # Get parent templates
        parent_ids = template.get("parent_ids", [])

        # Determine evolution mechanism
        mechanism = template.get("evolution_mechanism", EvolutionMechanism.MUTATION)

        # Create payload
        payload = EvolutionEventPayload(
            evolution_id=str(uuid.uuid4()),
            mechanism=mechanism,
            parent_templates=parent_ids,
            child_template=template["id"],
            fitness_change=template.get("fitness_change", 0.0),
        )

        # Emit event
        await self.event_handler.emit_event(event_type="context.evolution", payload=payload.dict())

    async def _emit_emergence_event(self, template: Dict[str, Any], pattern: Dict[str, Any]):
        """
        Emit an emergence event.

        Args:
            template: The new template
            pattern: The emergent pattern
        """
        # Create payload
        payload = EvolutionEventPayload(
            evolution_id=str(uuid.uuid4()),
            mechanism=EvolutionMechanism.EMERGENCE,
            parent_templates=[],  # Emergent patterns don't have direct parents
            child_template=template["id"],
            fitness_change=0.0,
        )

        # Add emergence specific data
        payload_dict = payload.dict()
        payload_dict["metadata"] = {
            "emergence_pattern": pattern["id"],
            "emergence_confidence": pattern.get("confidence", 1.0),
            "emergence_description": pattern.get("description", "Emergent pattern"),
        }

        # Emit event
        await self.event_handler.emit_event(event_type="context.emergence", payload=payload_dict)
