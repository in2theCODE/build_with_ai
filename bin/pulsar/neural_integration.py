#!/usr/bin/env python3
"""
Integration script for the Neural Code Generator component.

This script demonstrates how to integrate and use the advanced neural
code generation techniques with the existing program synthesis system.
"""

import argparse
import logging
from pathlib import Path
import sys

from program_synthesis_system.src.components.component_factory import (
    ComponentFactory,
)
from program_synthesis_system.src.components.neural_code_generator.neural_code_generator import (
    NeuralCodeGenerator,
)
from program_synthesis_system.src.shared.enums import ComponentType
import yaml


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent  # Assuming bin is two levels down from project root
default_config_path = project_root / "configs" / "neural_system_configs.yaml"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Neural Code Generator Integration")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/system_config.yaml",
        help="Path to system configuration file",
    )
    parser.add_argument(
        "--spec", "-s", required=True, help="Path to specification file or specification string"
    )
    parser.add_argument("--output", "-o", help="Output file for generated code (default: stdout)")
    parser.add_argument("--language", "-l", default="python", help="Target programming language")
    parser.add_argument(
        "--technique",
        "-t",
        choices=["attention", "tree", "hierarchical", "hybrid", "all"],
        default="all",
        help="Neural generation technique to use",
    )
    parser.add_argument(
        "--beam-width", "-b", type=int, default=5, help="Beam width for syntax-aware search"
    )
    parser.add_argument(
        "--kb-path", "-k", help="Path to knowledge base for retrieval-augmented generation"
    )
    parser.add_argument("--model-path", "-m", help="Path to pre-trained models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    return parser.parse_args()


# Update the load_configuration function
def load_configuration(config_path):
    """Load configuration from a YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent  # Go up one level from bin
        alt_path = project_dir / "configs" / config_path
        if alt_path.exists():
            config_path = alt_path
        else:
            # If still not found, try the literal path
            alt_path = project_dir / config_path
            if alt_path.exists():
                config_path = alt_path

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_specification(spec_path_or_string):
    """Load specification from a file or use it directly as a string."""
    try:
        with open(spec_path_or_string, "r") as f:
            return f.read()
    except FileNotFoundError:
        # If file not found, assume it's a direct specification string
        return spec_path_or_string


def setup_neural_code_generator(args, config):
    """Set up and configure the neural code generator."""
    # Override config with command-line arguments
    params = {
        "target_language": args.language,
        "use_retrieval_augmentation": args.kb_path is not None,
        "use_tree_transformers": args.technique in ["tree", "all"],
        "use_hierarchical_generation": args.technique in ["hierarchical", "all"],
        "use_syntax_aware_search": True,
        "use_hybrid_grammar_neural": args.technique in ["hybrid", "all"],
        "beam_width": args.beam_width,
    }

    # If knowledge base path specified, add it to params
    if args.kb_path:
        params["file_storage_path"] = args.kb_path

    # Create the component factory
    factory = ComponentFactory()

    # Create the neural code generator
    neural_generator = factory.create_component(
        ComponentType.CODE_GENERATOR, "neural_code_generator", params
    )

    # If model path specified, load pre-trained models
    if args.model_path and neural_generator:
        neural_generator.import_model(args.model_path)

    return neural_generator


def main():
    """Main entry point for the integration script."""
    # Parse command-line arguments
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("neural_integration")

    # Load configuration
    try:
        config = load_configuration(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Load specification
    try:
        spec_string = load_specification(args.spec)
        logger.info(f"Loaded specification from {args.spec}")
    except Exception as e:
        logger.error(f"Failed to load specification: {e}")
        sys.exit(1)

    # Set up the neural code generator
    neural_generator = setup_neural_code_generator(args, config)
    if not neural_generator:
        logger.error("Failed to set up neural code generator")
        sys.exit(1)

    # Create other required services using the factory
    factory = ComponentFactory()
    components = factory.create_from_config(config)

    # Make sure we have the necessary services
    if "specification_parser" not in components:
        logger.error("Specification parser not found in configuration")
        sys.exit(1)

    spec_parser = components["specification_parser"]

    # Parse the specification
    try:
        logger.info("Parsing specification")
        formal_spec = spec_parser.parse(spec_string)
    except Exception as e:
        logger.error(f"Failed to parse specification: {e}")
        sys.exit(1)

    # Generate code using the neural code generator
    try:
        logger.info(f"Generating code using {args.technique} technique")
        synthesis_result = neural_generator.generate(formal_spec)
    except Exception as e:
        logger.error(f"Failed to generate code: {e}")
        sys.exit(1)

    # Convert the AST to code
    try:
        if "ast_code_generator" in components:
            logger.info("Converting AST to code")
            code = components["ast_code_generator"].generate(synthesis_result)
        else:
            # Fallback: Just print the AST
            logger.warning("AST code generator not found, printing AST structure")
            import json

            code = json.dumps(synthesis_result.program_ast, indent=2)
    except Exception as e:
        logger.error(f"Failed to convert AST to code: {e}")
        sys.exit(1)

    # Output the generated code
    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(code)
            logger.info(f"Code written to {args.output}")
        except Exception as e:
            logger.error(f"Failed to write code to {args.output}: {e}")
            sys.exit(1)
    else:
        # Print to stdout
        print("\n=== GENERATED CODE ===\n")
        print(code)
        print("\n=== END OF CODE ===\n")
        print(f"Generation strategy: {synthesis_result.strategy}")
        print(f"Confidence score: {synthesis_result.confidence_score:.4f}")
        print(f"Time taken: {synthesis_result.time_taken:.2f} seconds")


if __name__ == "__main__":
    main()
