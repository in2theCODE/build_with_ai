#!/usr/bin/env python3
"""
Production-ready speculative decoding accelerator using Arctic Inference and Arctic Training.

This module implements a robust, configurable service for accelerating LLM inference
with state-of-the-art speculative decoding techniques from Snowflake AI Research.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    import arctic_inference
    from arctic_inference.plugins.vllm import configure_arctic_speculative

    HAS_ARCTIC = True
except ImportError:
    HAS_ARCTIC = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("speculative_accelerator.log"),
    ],
)
logger = logging.getLogger("speculative_accelerator")


class SpeculationMethod(str, Enum):
    """Available speculative decoding methods."""

    NONE = "none"
    SUFFIX = "suffix"
    DRAFT = "draft"
    HYBRID = "hybrid"


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking inference acceleration."""

    total_tokens: int = 0
    total_time_ms: float = 0
    tokens_per_second: float = 0.0
    acceptance_rate: float = 0.0
    requests_served: int = 0

    def update(self, tokens: int, time_ms: float, acceptance_rate: float):
        """Update metrics with new request data."""
        self.total_tokens += tokens
        self.total_time_ms += time_ms
        self.requests_served += 1
        self.tokens_per_second = self.total_tokens / (self.total_time_ms / 1000)
        # Exponential moving average of acceptance rate
        self.acceptance_rate = 0.9 * self.acceptance_rate + 0.1 * acceptance_rate

    def as_dict(self) -> Dict:
        """Convert metrics to a dictionary for reporting."""
        return {
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "acceptance_rate": self.acceptance_rate,
            "requests_served": self.requests_served,
        }


class LLMRequest(BaseModel):
    """API request model for LLM inference."""

    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: Optional[List[str]] = None


class SpeculativeAccelerator:
    """
    Production-ready LLM inference accelerator using Arctic Inference and Arctic Training.

    This class provides a high-performance, robust implementation of speculative decoding
    techniques for accelerating LLM inference in production environments.
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the accelerator with the given configuration.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self._validate_dependencies()

        self.model_name = self.config["model"]["name"]
        self.method = SpeculationMethod(self.config["speculation"]["method"])
        self.tensor_parallel_size = self.config["model"]["tensor_parallel_size"]
        self.quantization = self.config["model"].get("quantization", None)

        # Performance tracking
        self.metrics = PerformanceMetrics()

        # Configuration for Arctic Inference
        self.speculative_config = self._build_speculative_config()

        # Initialize vLLM engine
        self.engine = self._init_engine()

        logger.info(f"Initialized SpeculativeAccelerator with {self.model_name}")
        logger.info(f"Speculation method: {self.method}")
        logger.info(f"Using tensor parallelism: {self.tensor_parallel_size}")
        logger.info(f"Quantization: {self.quantization or 'none'}")

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load and validate the configuration file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate required sections
            required_sections = ["model", "speculation", "server"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Configuration missing required section: {section}")

            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _validate_dependencies(self):
        """Validate that required dependencies are installed."""
        if not HAS_VLLM:
            raise ImportError("vLLM is required but not installed. Install with: pip install vllm")

        if not HAS_ARCTIC:
            raise ImportError(
                "Arctic Inference is required but not installed. "
                "Install with: pip install git+https://github.com/snowflakedb/ArcticInference.git#egg=arctic-inference[vllm]"
            )

        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Performance will be severely limited.")

    def _build_speculative_config(self) -> Dict:
        """Build the speculative decoding configuration for vLLM."""
        spec_config = {}

        if self.method == SpeculationMethod.NONE:
            return None

        if self.method in [SpeculationMethod.DRAFT, SpeculationMethod.HYBRID]:
            spec_config["method"] = "arctic"
            spec_config["model"] = self.config["speculation"]["draft_model"]
            spec_config["num_speculative_tokens"] = self.config["speculation"]["num_tokens"]

            if self.method == SpeculationMethod.HYBRID:
                spec_config["enable_suffix_decoding"] = True

        elif self.method == SpeculationMethod.SUFFIX:
            spec_config["method"] = "arctic"
            spec_config["enable_suffix_decoding"] = True

        # Add advanced options if configured
        if "advanced" in self.config["speculation"]:
            for key, value in self.config["speculation"]["advanced"].items():
                spec_config[key] = value

        return spec_config

    def _init_engine(self) -> AsyncLLMEngine:
        """Initialize the vLLM engine with Arctic Inference configuration."""
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization=self.quantization,
            max_model_len=self.config["model"].get("max_model_len", 8192),
            trust_remote_code=self.config["model"].get("trust_remote_code", True),
            gpu_memory_utilization=self.config["model"].get("gpu_memory_utilization", 0.9),
            dtype=self.config["model"].get("dtype", "auto"),
            disable_log_stats=False,
        )

        # Configure Arctic Inference if using speculation
        if self.speculative_config:
            configure_arctic_speculative(engine_args, self.speculative_config)

        # Initialize the engine
        return AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self, request: LLMRequest) -> Dict:
        """
        Generate text using the LLM with speculative acceleration.

        Args:
            request: The API request containing generation parameters

        Returns:
            A dictionary containing the generated text and metadata
        """
        start_time = time.time()

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )

        # Generate response
        result = await self.engine.generate(request.prompt, sampling_params)

        # Extract output
        generated_text = result.outputs[0].text
        prompt_tokens = result.prompt_token_ids.shape[0]
        completion_tokens = len(result.outputs[0].token_ids)
        total_tokens = prompt_tokens + completion_tokens

        # Calculate stats
        generation_time_ms = (time.time() - start_time) * 1000
        tokens_per_second = completion_tokens / (generation_time_ms / 1000)

        # Get acceptance rate if available
        acceptance_rate = 0.0
        if hasattr(result, "acceptance_rate"):
            acceptance_rate = result.acceptance_rate

        # Update metrics
        self.metrics.update(
            tokens=completion_tokens, time_ms=generation_time_ms, acceptance_rate=acceptance_rate
        )

        # Log performance
        logger.info(
            f"Generated {completion_tokens} tokens in {generation_time_ms:.2f}ms "
            f"({tokens_per_second:.2f} tokens/sec)"
        )

        # Prepare response
        response = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "performance": {
                "generation_time_ms": generation_time_ms,
                "tokens_per_second": tokens_per_second,
                "acceptance_rate": acceptance_rate,
            },
        }

        return response

    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        return self.metrics.as_dict()


def create_app(config_path: Union[str, Path]) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Speculative Decoding Accelerator",
        description="High-performance LLM inference using Arctic Inference and Arctic Training",
        version="1.0.0",
    )

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["server"].get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize accelerator
    accelerator = SpeculativeAccelerator(config_path)

    @app.post("/v1/generate")
    async def generate(request: LLMRequest):
        """Generate text using speculative acceleration."""
        try:
            result = await accelerator.generate(request)
            return result
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/metrics")
    async def metrics():
        """Get current performance metrics."""
        return accelerator.get_metrics()

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


def main():
    """Run the speculative accelerator as a standalone service."""
    parser = argparse.ArgumentParser(description="Speculative Decoding Accelerator")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--host", type=str, help="Host to bind the server to")
    parser.add_argument("--port", type=int, help="Port to bind the server to")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create app
    app = create_app(args.config)

    # Get host and port from arguments or config
    host = args.host or config["server"].get("host", "0.0.0.0")
    port = args.port or config["server"].get("port", 8000)

    # Run server
    import uvicorn

    logger.info(f"Starting speculative accelerator server at {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
