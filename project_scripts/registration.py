#!/usr/bin/env python3
"""
Component registration script for the Program Synthesis System.

This script registers all the services in the component factory,
ensuring they're available for use in the system.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.models.enums import ComponentType
from app.components.component_factory import ComponentFactory
from app.utils.advanced_logger import configure_logging, get_logger

# Import services
from app.components.neural_code_generator.neural_code_generator import NeuralCodeGenerator
from app.components.neural_code_generator.enhanced_neural_code_generator import EnhancedNeuralCodeGenerator

# Import the new services
from app.components.constraint_relaxer.constraint_relaxer import ConstraintRelaxer
from app.components.feedback_collector.feedback_collector import FeedbackCollector
from app.components.incremental_synthesis.incremental_synthesis import IncrementalSynthesis
from app.components.language_interop.language_interop import LanguageInterop
from app.components.meta_learner.meta_learner import MetaLearner
from app.components.spec_inference.spec_inference import SpecInference
from app.components.synthesis_engine.synthesis_engine import SynthesisEngine
from app.components.version_manager.version_manager import VersionManager


def register_components():
    """Register all services in the component factory."""
    # Configure logging
    configure_logging({
        "level": logging.INFO,
        "console": True,
        "file": True,
        "use_colors": True,
        "directory": "logs"
    })

    logger = get_logger("component_registration")
    logger.info("Registering services")

    # Create component factory
    factory = ComponentFactory()

    # Register services
    factory.register_component(
        ComponentType.CODE_GENERATOR,
        "neural_code_generator",
        NeuralCodeGenerator
    )

    factory.register_component(
        ComponentType.CODE_GENERATOR,
        "enhanced_neural_code_generator",
        EnhancedNeuralCodeGenerator
    )

    # Register new services
    factory.register_component(
        ComponentType.CONSTRAINT_RELAXER,
        "constraint_relaxer",
        ConstraintRelaxer
    )

    factory.register_component(
        ComponentType.FEEDBACK_COLLECTOR,
        "feedback_collector",
        FeedbackCollector
    )

    factory.register_component(
        ComponentType.INCREMENTAL_SYNTHESIS,
        "incremental_synthesis",
        IncrementalSynthesis
    )

    factory.register_component(
        ComponentType.LANGUAGE_INTEROP,
        "language_interop",
        LanguageInterop
    )

    factory.register_component(
        ComponentType.META_LEARNER,
        "meta_learner",
        MetaLearner
    )

    factory.register_component(
        ComponentType.SPEC_INFERENCE,
        "spec_inference",
        SpecInference
    )

    factory.register_component(
        ComponentType.SYNTHESIS_ENGINE,
        "synthesis_engine",
        SynthesisEngine
    )

    factory.register_component(
        ComponentType.VERSION_MANAGER,
        "version_manager",
        VersionManager
    )

    logger.info("All services registered successfully")

    return factory


if __name__ == "__main__":
    factory = register_components()