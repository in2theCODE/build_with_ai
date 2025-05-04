"""Vector knowledge base implementation for code storage and retrieval."""

import logging
import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import os

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import qdrant_client
    from qdrant_client.http import models
except ImportError:
    logging.warning("Vector DB dependencies not installed. Run: pip install sentence-transformers qdrant-client")

from src.services.shared.constants.base_component import BaseComponent


class VectorKnowledgeBase(BaseComponent):
    """Knowledge base that stores code with vector embeddings for similarity search."""

    def __init__(self, **params):
        """Initialize the vector knowledge base with connection parameters."""
        super().__init__(**params)
        self.connection_string = self.get_param("connection_string", "")
        self.embedding_model = self.get_param("embedding_model", "all-mpnet-base-v2")
        self.similarity_threshold = self.get_param("similarity_threshold", 0.85)
        self.collection_name = self.get_param("collection_name", "code_embeddings")
        self.vector_size = self.get_param("vector_size", 768)  # Default for many models
        self.logger = logging.getLogger(self.__class__.__name__)

        # Determine storage type based on connection string
        if self.connection_string.startswith("postgresql://"):
            self.storage_type = "postgres"
        elif self.connection_string.startswith("mongodb://"):
            self.storage_type = "mongodb"
        elif self.connection_string:
            self.storage_type = "qdrant"
        else:
            self.storage_type = "file"
            self.file_storage_path = self.get_param("file_storage_path", "knowledge_base")
            os.makedirs(self.file_storage_path, exist_ok=True)

        self._initialize_storage()
        self._initialize_embedding_model()

    def _initialize_storage(self):
        """Initialize the storage backend."""
        try:
            if self.storage_type == "qdrant":
                # Parse connection string
                if ":" in self.connection_string:
                    host, port = self.connection_string.split(":")
                    port = int(port)
                else:
                    host = self.connection_string
                    port = 6333

                self.client = qdrant_client.QdrantClient(host=host, port=port)

                # Create collection if it doesn't exist
                collections = self.client.get_collections().collections
                collection_names = [c.name for c in collections]

                if self.collection_name not in collection_names:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=self.vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                self.logger.info(f"Connected to Qdrant at {self.connection_string}")

            elif self.storage_type == "postgres":
                self.logger.info("PostgreSQL storage selected but not implemented yet")
                # Fallback to file storage
                self.storage_type = "file"
                self.file_storage_path = "knowledge_base"
                os.makedirs(self.file_storage_path, exist_ok=True)

            elif self.storage_type == "mongodb":
                self.logger.info("MongoDB storage selected but not implemented yet")
                # Fallback to file storage
                self.storage_type = "file"
                self.file_storage_path = "knowledge_base"
                os.makedirs(self.file_storage_path, exist_ok=True)

            elif self.storage_type == "file":
                self.logger.info(f"Using file storage at {self.file_storage_path}")
                # Create metadata index file if it doesn't exist
                index_path = Path(self.file_storage_path) / "index.json"
                if not index_path.exists():
                    with open(index_path, 'w') as f:
                        json.dump({"entries": []}, f)

        except Exception as e:
            self.logger.error(f"Failed to initialize storage: {e}")
            # Fallback to file storage
            self.storage_type = "file"
            self.file_storage_path = "knowledge_base"
            os.makedirs(self.file_storage_path, exist_ok=True)

    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            self.model = SentenceTransformer(self.embedding_model)
            self.logger.info(f"Loaded embedding model: {self.embedding_model}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.model = None

    def store(self, key: str, data: Dict[str, Any]) -> bool:
        """
        Store data in the knowledge base with vector embedding.

        Args:
            key: Unique identifier for the data
            data: Dictionary containing code and metadata

        Returns:
            True if storage was successful
        """
        try:
            # Create embedding for the code
            code = data.get("code", "")
            metadata = data.get("metadata", {})

            if self.model is not None:
                embedding = self.model.encode(code)
            else:
                # Fallback to random embedding if model not available
                embedding = np.random.rand(self.vector_size).astype(np.float32)

            if self.storage_type == "qdrant":
                # Store in Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=int(hashlib.md5(key.encode()).hexdigest(), 16) % (10 ** 10),
                            vector=embedding.tolist(),
                            payload={
                                "key": key,
                                "code": code,
                                "metadata": metadata
                            }
                        )
                    ]
                )

            elif self.storage_type == "file":
                # Store in file system
                # 1. Save the data file
                data_path = Path(self.file_storage_path) / f"{key}.json"
                with open(data_path, 'w') as f:
                    json.dump({
                        "key": key,
                        "code": code,
                        "metadata": metadata,
                        "embedding": embedding.tolist()
                    }, f)

                # 2. Update the index
                index_path = Path(self.file_storage_path) / "index.json"
                with open(index_path, 'r') as f:
                    index = json.load(f)

                # Add or update entry
                entry = {"key": key, "path": str(data_path)}

                # Check if entry already exists
                for i, existing in enumerate(index["entries"]):
                    if existing["key"] == key:
                        index["entries"][i] = entry
                        break
                else:
                    index["entries"].append(entry)

                # Write updated index
                with open(index_path, 'w') as f:
                    json.dump(index, f)

            self.logger.info(f"Stored entry with key: {key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store data: {e}")
            return False

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data by exact key.

        Args:
            key: The unique identifier

        Returns:
            The stored data if found, None otherwise
        """
        try:
            if self.storage_type == "qdrant":
                # Search by key in payload
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="key",
                                match=models.MatchValue(value=key)
                            )
                        ]
                    ),
                    limit=1
                )

                if results and results[0]:
                    point = results[0][0]  # First result
                    return {
                        "code": point.payload.get("code", ""),
                        "metadata": point.payload.get("metadata", {})
                    }
                return None

            elif self.storage_type == "file":
                # Check if file exists
                data_path = Path(self.file_storage_path) / f"{key}.json"
                if data_path.exists():
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                        return {
                            "code": data.get("code", ""),
                            "metadata": data.get("metadata", {})
                        }
                return None

        except Exception as e:
            self.logger.error(f"Failed to retrieve data: {e}")
            return None

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar code using vector similarity.

        Args:
            query: The search query
            limit: Maximum number of results

        Returns:
            List of matching entries
        """
        try:
            if self.model is None:
                raise ValueError("Embedding model not available")

            # Create embedding for the query
            query_embedding = self.model.encode(query)

            if self.storage_type == "qdrant":
                # Search in Qdrant
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=limit
                )

                return [
                    {
                        "key": point.payload.get("key", ""),
                        "code": point.payload.get("code", ""),
                        "metadata": point.payload.get("metadata", {}),
                        "score": point.score
                    }
                    for point in results
                    if point.score >= self.similarity_threshold
                ]

            elif self.storage_type == "file":
                # Load index
                index_path = Path(self.file_storage_path) / "index.json"
                with open(index_path, 'r') as f:
                    index = json.load(f)

                # Load all embeddings and compute similarities
                results = []

                for entry in index["entries"]:
                    data_path = entry["path"]
                    with open(data_path, 'r') as f:
                        data = json.load(f)

                    if "embedding" in data:
                        stored_embedding = np.array(data["embedding"])
                        # Compute cosine similarity
                        score = np.dot(query_embedding, stored_embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                        )

                        if score >= self.similarity_threshold:
                            results.append({
                                "key": data.get("key", ""),
                                "code": data.get("code", ""),
                                "metadata": data.get("metadata", {}),
                                "score": float(score)
                            })

                # Sort by similarity score
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:limit]

            return []

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def delete(self, key: str) -> bool:
        """
        Delete an entry from the knowledge base.

        Args:
            key: The unique identifier

        Returns:
            True if deletion was successful
        """
        try:
            if self.storage_type == "qdrant":
                # Delete from Qdrant
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="key",
                                match=models.MatchValue(value=key)
                            )
                        ]
                    )
                )
                self.logger.info(f"Deleted entry with key: {key}")
                return True

            elif self.storage_type == "file":
                # Delete file
                data_path = Path(self.file_storage_path) / f"{key}.json"
                if data_path.exists():
                    os.remove(data_path)

                # Update index
                index_path = Path(self.file_storage_path) / "index.json"
                with open(index_path, 'r') as f:
                    index = json.load(f)

                index["entries"] = [entry for entry in index["entries"] if entry["key"] != key]

                with open(index_path, 'w') as f:
                    json.dump(index, f)

                self.logger.info(f"Deleted entry with key: {key}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to delete data: {e}")
            return False