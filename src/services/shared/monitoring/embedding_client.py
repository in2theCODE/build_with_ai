# src/models/embedding_client.py

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
import time

# You might need to adjust this import path based on your project structure
from metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

class EmbeddingClient:
    """Client for generating text embeddings for semantic analysis."""

    def __init__(self, config: Dict[str, Any], metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize the embedding client.

        Args:
            config: Configuration dictionary with embedding settings
            metrics_collector: Optional metrics collector for instrumentation
        """
        self.config = config
        self.metrics_collector = metrics_collector
        self.model_name = config.get("embedding_model", "default-embedding-model")
        self.dimension = config.get("embedding_dimension", 1536)  # Default for many embedding models
        self.batch_size = config.get("batch_size", 32)
        self.logger = logging.getLogger(__name__)

        # Initialize any embedding models or clients here
        self._initialize_models()

    class Timer:
        """A simple timer class for measuring execution time."""

        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            """Start the timer."""
            self.start_time = time.time()
            return self

        def stop(self):
            """Stop the timer and return the duration."""
            self.end_time = time.time()
            return self.duration

        @property
        def duration(self):
            """Return the duration in seconds."""
            if self.start_time is None:
                return 0
            if self.end_time is None:
                # If timer hasn't been stopped, return current duration
                return time.time() - self.start_time
            return self.end_time - self.start_time


    def _initialize_models(self):
        """Initialize embedding models based on configuration."""

        model_type = self.config.get("embedding_type", "default")

        if self.metrics_collector:
            timer = self.metrics_collector.start_model_loading_timer(f"embedding_{model_type}")

        try:
            # In a real implementation, you would initialize your embedding model here
            # For example:
            # if model_type == "openai":
            #     import openai
            #     openai.api_key = self.config["openai_api_key"]
            # elif model_type == "tensorflow":
            #     import tensorflow as tf
            #     self.model = tf.saved_model.load(self.config["model_path"])

            self.logger.info(f"Initialized {model_type} embedding model")

            # For this example, we'll just simulate the model
            self.initialized = True

        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            self.initialized = False
            if self.metrics_collector:
                self.metrics_collector.record_error("embedding_initialization_error")

        finally:
            if self.metrics_collector and timer:
                # The timer created by start_model_loading_timer automatically records when it goes out of scope
                pass

    async def embed_text(self, text: str) -> list[float] | None:
        """
        Generate an embedding vector for the provided text.

        Args:
            text: The text to embed

        Returns:
            A list of floating point values representing the embedding vector
        """
        if not self.initialized:
            self.logger.error("Embedding model not initialized")
            if self.metrics_collector:
                self.metrics_collector.record_error("embedding_model_not_initialized")
            return [0.0] * self.dimension

        if not text:
            return [0.0] * self.dimension

        start_time = time.time()

        try:
            # In a real implementation, you would call your embedding model here
            # For example:
            # if hasattr(self, "openai"====):
            #     response = await openai.Embedding.acreate(
            #         input=text,
            #         model=self.model_name
            #     )
            #     embedding = response['data'][0]['embedding']

            # For this example, we'll generate a random embedding
            # In a real implementation, replace this with your actual embedding code
            embedding = self._simulate_embedding(text)

            # Record metrics if available
            if self.metrics_collector:
                tokens = len(text.split())
                self.metrics_collector.record_tokens("embedding_input", tokens)

            return embedding

        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_error("embedding_generation_error")
            return [0.0] * self.dimension

        finally:
            # Record duration
            duration = time.time() - start_time
            if self.metrics_collector:
                self.metrics_collector.record_latency("embedding_generation_time", duration)
                pass

    def _simulate_embedding(self, text: str) -> List[float]:
        """
        Simulate an embedding for testing purposes.
        In a real implementation, this would be replaced with actual embedding logic.
        """
        # Use a deterministic seed based on the text
        text_seed = sum(ord(c) for c in text)
        np.random.seed(text_seed)

        # Generate a random vector with values between -1 and 1
        embedding = np.random.uniform(-1, 1, self.dimension).tolist()

        # Normalize to unit length (common in embedding vectors)
        norm = np.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        results = []
        # Process in batches to avoid overloading the API or model
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await asyncio.gather(*[self.embed_text(text) for text in batch])
            results.extend(batch_embeddings)
        return results