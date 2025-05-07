#!/usr/bin/env python3
"""
Command-line interface for the Program Synthesis System.
"""

import argparse
import json
from pathlib import Path
import sys


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.orchestration import SynthesisSystem


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Program Synthesis System CLI")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/system_config.yaml",
        help="Path to system configuration file",
    )
    parser.add_argument(
        "--spec", "-s", required=True, help="Path to specification file or specification string"
    )
    parser.add_argument("--context", "-x", help="Path to context JSON file")
    parser.add_argument("--output", "-o", help="Output file for generated code (default: stdout)")
    parser.add_argument("--language", "-l", default="python", help="Target programming language")
    parser.add_argument(
        "--best-effort",
        "-b",
        action="store_true",
        help="Allow best-effort results if verification fails",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main():
    """Main entry point for the synthesis CLI."""
    args = parse_args()

    # Set up logging based on verbosity
    import logging

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Initialize synthesis system
    try:
        system = SynthesisSystem("config_path")
    except Exception as e:
        print(f"Error initializing synthesis system: {e}")
        sys.exit(1)

    # Load specification
    spec_path = Path(args.spec)
    if spec_path.exists():
        with open(spec_path, "r") as f:
            specification = f.read()
    else:
        # Assume it's a direct specification string
        specification = args.spec

    # Load context if provided
    context = None
    if args.context:
        context_path = Path(args.context)
        if not context_path.exists():
            print(f"Error: Context file not found: {context_path}")
            sys.exit(1)

        with open(context_path, "r") as f:
            context = json.load(f)
    else:
        context = {}

    # Add command line options to context
    context["target_language"] = args.language
    context["allow_best_effort"] = args.best_effort

    try:
        # Generate code from specification
        generated_code, metadata = system.generate_from_spec(specification, context)

        # Output the generated code
        if args.output:
            with open(args.output, "w") as f:
                f.write(generated_code)
            print(f"Generated code written to: {args.output}")
        else:
            print("\n=== GENERATED CODE ===\n")
            print(generated_code)
            print("\n=== METADATA ===\n")
            print(json.dumps(metadata, indent=2))

    except Exception as e:
        print(f"Error during synthesis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
