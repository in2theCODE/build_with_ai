#!/usr/bin/env python3
"""
Service module for the Neural Code Generator.

This module runs the neural code generator as a service that listens to
requests via Apache Pulsar and sends responses back to specified topics.
"""

import asyncio
import logging
import os
import signal
import sys


# Setup logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("neural_code_generator_service")

# Ensure the module can be found in the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

# Import the enhanced neural code generator
try:
    from program_synthesis_system.components.neural_code_generator.enhanced_neural_code_generator import (
        EnhancedNeuralCodeGenerator,
    )
except ImportError:
    logger.error("Failed to import EnhancedNeuralCodeGenerator")
    sys.exit(1)


async def main():
    """Main entry point for the service."""
    logger.info("Starting Neural Code Generator Service")

    # Get configuration from environment variables
    pulsar_service_url = os.environ.get("PULSAR_SERVICE_URL", "pulsar://localhost:6650")
    input_topic = os.environ.get("INPUT_TOPIC", "code-generation-requests")
    output_topic = os.environ.get("OUTPUT_TOPIC", "code-generation-results")
    subscription_name = os.environ.get("SUBSCRIPTION_NAME", "code-generator-worker")

    # Create parameter dictionary for the neural code generator
    params = {
        "pulsar_service_url": pulsar_service_url,
        "input_topic": input_topic,
        "output_topic": output_topic,
        "subscription_name": subscription_name,
        "pulsar_enabled": True,
        # Model configuration
        "model_path": os.environ.get("MODEL_PATH", "~/.app/deepseek-coder-6.7b"),
        "target_language": os.environ.get("TARGET_LANGUAGE", "python"),
        "max_context_length": int(os.environ.get("MAX_CONTEXT_LENGTH", "8192")),
        "quantization": os.environ.get("QUANTIZATION", "int8"),
        "use_flash_attention": os.environ.get("USE_FLASH_ATTENTION", "true").lower()
        == "true",
        # Technique configuration
        "use_retrieval_augmentation": os.environ.get("USE_RETRIEVAL", "true").lower()
        == "true",
        "use_tree_transformers": os.environ.get("USE_TREE_TRANSFORMERS", "true").lower()
        == "true",
        "use_hierarchical_generation": os.environ.get(
            "USE_HIERARCHICAL", "true"
        ).lower()
        == "true",
        "use_syntax_aware_search": os.environ.get("USE_SYNTAX_AWARE", "true").lower()
        == "true",
        "use_hybrid_grammar_neural": os.environ.get("USE_HYBRID", "true").lower()
        == "true",
        # Performance configuration
        "batch_size": int(os.environ.get("BATCH_SIZE", "1")),
        "mixed_precision": os.environ.get("MIXED_PRECISION", "true").lower() == "true",
        "low_cpu_mem_usage": os.environ.get("LOW_CPU_MEM", "true").lower() == "true",
        # Monitoring configuration
        "enable_tracing": os.environ.get("ENABLE_TRACING", "true").lower() == "true",
        "trace_sample_rate": float(os.environ.get("TRACE_SAMPLE_RATE", "0.1")),
    }

    # Initialize the neural code generator
    try:
        code_generator = EnhancedNeuralCodeGenerator(**params)
        logger.info("Neural Code Generator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Neural Code Generator: {e}")
        sys.exit(1)

    # Set up signal handling for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(shutdown(code_generator, loop))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Start the service
    try:
        await code_generator.start_service()

        # Keep the service running until interrupted
        while True:
            await asyncio.sleep(3600)  # Sleep for an hour at a time

    except Exception as e:
        logger.error(f"Service error: {e}")
        await shutdown(code_generator, loop)


async def shutdown(code_generator, loop):
    """Gracefully shut down the service."""
    logger.info("Shutting down Neural Code Generator Service")
    try:
        await code_generator.stop_service()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    # Stop the event loop
    loop.stop()


if __name__ == "__main__":
    # Run the main function
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Service stopped")
    finally:
        loop.close()
