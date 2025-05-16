#!/usr/bin/env python3
"""
Standalone module for the Neural Code Generator.

This module allows running the neural code generator directly for a single
specification without requiring Apache Pulsar for messaging.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys

import torch


# Setup logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("neural_code_generator_standalone")

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

# Import the enhanced neural code generator
try:
    from src.services.neural_code_generator.enhanced_neural_code_generator import (
        EnhancedNeuralCodeGenerator,
    )
except ImportError:
    logger.error("Failed to import EnhancedNeuralCodeGenerator")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Neural Code Generator - Standalone Mode"
    )
    parser.add_argument(
        "--spec",
        "-s",
        required=True,
        help="Path to specification file or specification string",
    )
    parser.add_argument(
        "--output", "-o", help="Output file for generated code (default: stdout)"
    )
    parser.add_argument(
        "--language", "-l", default="python", help="Target programming language"
    )
    parser.add_argument(
        "--technique",
        "-t",
        choices=["attention", "tree", "hierarchical", "hybrid", "all"],
        default="all",
        help="Neural generation technique to use",
    )
    parser.add_argument(
        "--beam-width",
        "-b",
        type=int,
        default=5,
        help="Beam width for syntax-aware search",
    )
    parser.add_argument(
        "--kb-path",
        "-k",
        help="Path to knowledge base for retrieval-augmented generation",
    )
    parser.add_argument("--model-path", "-m", help="Path to pre-trained app")
    parser.add_argument(
        "--quantization",
        "-q",
        choices=["none", "int8", "int4"],
        default="int8",
        help="Model quantization level",
    )
    parser.add_argument(
        "--no-flash-attention", action="store_true", help="Disable Flash Attention"
    )
    parser.add_argument(
        "--no-mixed-precision", action="store_true", help="Disable mixed precision"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "code", "ast"],
        default="code",
        help="Output format",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def load_specification(spec_path_or_string):
    """Load specification from a file or use it directly as a string."""
    try:
        # Check if it's a path to a file
        path = Path(spec_path_or_string)
        if path.exists() and path.is_file():
            with open(path, "r") as f:
                content = f.read()

            # Try to parse as JSON if the file has a .json extension
            if path.suffix.lower() == ".json":
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse specification as JSON, using raw content"
                    )
                    return content
            else:
                return content
        else:
            # Not a valid file path, treat as a direct specification string
            return spec_path_or_string
    except Exception as e:
        logger.error(f"Error loading specification: {e}")
        sys.exit(1)


def setup_neural_code_generator(args):
    """Set up and configure the neural code generator."""
    # Prepare parameters
    params = {
        # Core parameters
        "model_path": args.model_path,
        "target_language": args.language,
        "quantization": args.quantization if args.quantization != "none" else None,
        "use_flash_attention": not args.no_flash_attention,
        "mixed_precision": not args.no_mixed_precision,
        # Advanced technique parameters based on selected technique
        "use_retrieval_augmentation": args.kb_path is not None
        and (args.technique in ["all"]),
        "use_tree_transformers": args.technique in ["tree", "all"],
        "use_hierarchical_generation": args.technique in ["hierarchical", "all"],
        "use_syntax_aware_search": True,
        "use_hybrid_grammar_neural": args.technique in ["hybrid", "all"],
        # Beam search parameters
        "beam_width": args.beam_width,
        # Pulsar parameters (disabled in standalone mode)
        "pulsar_enabled": False,
    }

    # Add knowledge base path if specified
    if args.kb_path:
        params["file_storage_path"] = args.kb_path

    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create the neural code generator
    try:
        neural_generator = EnhancedNeuralCodeGenerator(**params)
        logger.info(
            f"Neural code generator initialized with {args.technique} technique"
        )
        return neural_generator
    except Exception as e:
        logger.error(f"Failed to initialize neural code generator: {e}")
        sys.exit(1)


def save_output(result, args):
    """Save or print the generation result."""
    if args.format == "json":
        # Convert result to JSON
        if hasattr(result, "to_dict"):
            output = json.dumps(result.to_dict(), indent=2)
        else:
            output = json.dumps(result, indent=2)
    elif args.format == "ast":
        # Output the AST
        if hasattr(result, "program_ast"):
            output = json.dumps(result.program_ast, indent=2)
        else:
            output = json.dumps(result.get("program_ast", {}), indent=2)
    else:
        # Output the code
        if hasattr(result, "program_ast") and "code" in result.program_ast:
            output = result.program_ast["code"]
        elif (
            isinstance(result, dict)
            and "program_ast" in result
            and "code" in result["program_ast"]
        ):
            output = result["program_ast"]["code"]
        else:
            logger.error("No code found in generation result")
            output = json.dumps(result, indent=2)

    # Save to file or print to stdout
    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(output)
            logger.info(f"Output saved to {args.output}")
        except Exception as e:
            logger.error(f"Failed to write output to {args.output}: {e}")
            print(output)
    else:
        print(output)


def main():
    """Main entry point for the standalone module."""
    # Parse arguments
    args = parse_args()

    # Load specification
    spec = load_specification(args.spec)
    logger.info(f"Loaded specification from {args.spec}")

    # Set up the neural code generator
    neural_generator = setup_neural_code_generator(args)

    # Generate code
    try:
        logger.info(f"Generating code using {args.technique} technique")
        result = neural_generator.generate(spec)
        logger.info("Code generation complete")

        # Show generation stats
        if (
            hasattr(result, "confidence_score")
            and hasattr(result, "time_taken")
            and hasattr(result, "strategy")
        ):
            logger.info(f"Strategy: {result.strategy}")
            logger.info(f"Confidence: {result.confidence_score:.2f}")
            logger.info(f"Time taken: {result.time_taken:.2f} seconds")
        elif isinstance(result, dict):
            logger.info(f"Strategy: {result.get('strategy', 'unknown')}")
            logger.info(f"Confidence: {result.get('confidence_score', 0):.2f}")
            logger.info(f"Time taken: {result.get('time_taken', 0):.2f} seconds")

        # Save or print the result
        save_output(result, args)

        return 0

    except Exception as e:
        logger.error(f"Failed to generate code: {e}")
        return 1


if __name__ == "__main__":
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run the main function
    sys.exit(main())
