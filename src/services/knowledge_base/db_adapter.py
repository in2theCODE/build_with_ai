def _init_redis(self):
    """Initialize Redis connection with enterprise configuration."""
    try:
        import redis
        from redis.cluster import RedisCluster

        # Extract configuration
        host = self.cache_config.get("host", "redis")
        port = self.cache_config.get("port", 6379)
        db = self.cache_config.get("db", 0)
        password = self.cache_config.get("password", None)
        cluster_mode = self.cache_config.get("cluster_mode", False)
        sentinel_mode = self.cache_config.get("sentinel_mode", False)
        connection_pool_size = self.cache_config.get("connection_pool_size", 100)
        socket_timeout = self.cache_config.get("socket_timeout", 5)
        socket_connect_timeout = self.cache_config.get("socket_connect_timeout", 2)
        health_check_interval = self.cache_config.get("health_check_interval", 30)
        ssl = self.cache_config.get("ssl", False)

        # Enterprise Redis features
        if cluster_mode:
            # Redis Cluster configuration for high availability
            cluster_nodes = self.cache_config.get("cluster_nodes", [(host, port)])

            client = RedisCluster(
                startup_nodes=cluster_nodes,
                password=password,
                decode_responses=True,
                ssl=ssl,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                health_check_interval=health_check_interval,
                max_connections=connection_pool_size
            )

            logger.info(f"Connected to Redis Cluster with {len(cluster_nodes)} nodes")
            return {"type": "redis_cluster", "client": client}

        elif sentinel_mode:
            # Redis Sentinel configuration for high availability
            from redis.sentinel import Sentinel

            sentinel_hosts = self.cache_config.get("sentinel_hosts", [(host, port)])
            sentinel_master = self.cache_config.get("sentinel_master", "mymaster")

            sentinel = Sentinel(
                sentinel_hosts,
                socket_timeout=socket_timeout,
                password=password,
                sentinel_kwargs={"password": password} if password else None,
                ssl=ssl
            )

            master = sentinel.master_for(
                sentinel_master,
                socket_timeout=socket_timeout,
                password=password,
                db=db,
                decode_responses=True
            )

            slave = sentinel.slave_for(
                sentinel_master,
                socket_timeout=socket_timeout,
                password=password,
                db=db,
                decode_responses=True
            )

            logger.info(f"Connected to Redis Sentinel with master '{sentinel_master}'")
            return {
                "type": "redis_sentinel",
                "sentinel": sentinel,
                "master": master,
                "slave": slave
            }

        else:
            # Standard Redis connection with optimized parameters
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                health_check_interval=health_check_interval,
                max_connections=connection_pool_size,
                ssl=ssl
            )

            client = redis.Redis(connection_pool=pool)

            # Test connection
            client.ping()

            # Configure advanced features if available
            if self.cache_config.get("enable_notifications", False):
                # Subscribe to invalidation channel
                pubsub = client.pubsub()
                pubsub.subscribe("cache_invalidations")

                # Start pubsub thread
                pubsub_thread = pubsub.run_in_thread(sleep_time=0.5)

                # Store thread for cleanup
                self.pubsub_thread = pubsub_thread

            logger.info(f"Connected to Redis at {host}:{port}, db: {db}")
            return {"type": "redis", "client": client, "connection_pool": pool}

    except ImportError:
        logger.error("Redis client not available. Install with: pip install redis")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        return None    async def store_generation_history(self,
                                                          key: str,
                                                          specification: str,
                                                          generated_code: str,
                                                          strategy: str,
                                                          confidence: float,
                                                          generation_time: float,
                                                          metadata: Dict[str, Any]) -> bool:
    """
    Store generation history in the relational database.

    Args:
        key: Unique identifier
        specification: Code specification/prompt
        generated_code: Generated code
        strategy: Generation strategy used
        confidence: Confidence score
        generation_time: Time taken for generation
        metadata: Additional metadata

    Returns:
        True if successful, False otherwise
    """
    if not self.relational_db:
        logger.error("Relational database not initialized")
        return False

    try:
        db_type = self.relational_db["type"]

        if db_type == "postgresql":
            conn = self.relational_db["connection"]

            # Serialize metadata to JSON
            metadata_json = json.dumps(metadata)

            # Insert generation history
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO generation_history 
                    (id, specification, generated_code, strategy, confidence, generation_time, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """, (key, specification, generated_code, strategy, confidence, generation_time, metadata_json))

                conn.commit()

            return True

        elif db_type == "sqlite":
            conn = self.relational_db["connection"]

            # Serialize metadata to JSON
            metadata_json = json.dumps(metadata)

            # Insert generation history
            with conn:
                # Check if entry exists
                cursor = conn.execute("SELECT id FROM generation_history WHERE id = ?", (key,))
                if not cursor.fetchone():
                    # Insert new entry
                    conn.execute("""
                        INSERT INTO generation_history
                        (id, specification, generated_code, strategy, confidence, generation_time, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (key, specification, generated_code, strategy, confidence, generation_time, metadata_json))

            return True

        else:
            logger.error(f"Unsupported relational database type: {db_type}")
            return False

    except Exception as e:
        logger.error(f"Failed to store generation history: {e}")
        return False

async def get_generation_history(self, key: str) -> Optional[Dict[str, Any]]:
    """
    Get generation history from the relational database.

    Args:
        key: Unique identifier

    Returns:
        Generation history or None if not found
    """
    # Check cache first
    cached_result = await self.get_from_cache(f"generation_history:{key}")
    if cached_result:
        # Record cache hit in metrics
        if self.metrics_collector:
            self.metrics_collector.record_cache_hit("generation_history")
        return cached_result

    if not self.relational_db:
        logger.error("Relational database not initialized")
        return None

    try:
        db_type = self.relational_db["type"]

        if db_type == "postgresql":
            conn = self.relational_db["connection"]

            # Get generation history
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, specification, generated_code, strategy, confidence, generation_time, created_at, metadata
                    FROM generation_history
                    WHERE id = %s
                    """, (key,))

                row = cur.fetchone()

                if row:
                    result = {
                        "id": row[0],
                        "specification": row[1],
                        "generated_code": row[2],
                        "strategy": row[3],
                        "confidence": row[4],
                        "generation_time": row[5],
                        "created_at": row[6].isoformat() if row[6] else None,
                        "metadata": json.loads(row[7]) if row[7] else {}
                    }

                    # Cache the result
                    await self.store_in_cache(f"generation_history:{key}", result, ttl=3600)

                    # Record cache miss in metrics
                    if self.metrics_collector:
                        self.metrics_collector.record_cache_miss("generation_history")

                    return result

            return None

        elif db_type == "sqlite":
            conn = self.relational_db["connection"]

            # Get generation history
            cursor = conn.execute("""
                SELECT id, specification, generated_code, strategy, confidence, generation_time, created_at, metadata
                FROM generation_history
                WHERE id = ?
                """, (key,))

            row = cursor.fetchone()

            if row:
                result = {
                    "id": row[0],
                    "specification": row[1],
                    "generated_code": row[2],
                    "strategy": row[3],
                    "confidence": row[4],
                    "generation_time": row[5],
                    "created_at": row[6],
                    "metadata": json.loads(row[7]) if row[7] else {}
                }

                # Cache the result
                await self.store_in_cache(f"generation_history:{key}", result, ttl=3600)

                # Record cache miss in metrics
                if self.metrics_collector:
                    self.metrics_collector.record_cache_miss("generation_history")

                return result

            return None

        else:
            logger.error(f"Unsupported relational database type: {db_type}")
            return None

    except Exception as e:
        logger.error(f"Failed to get generation history: {e}")
        return None

async def search_generation_history(self,
                                    criteria: Dict[str, Any],
                                    limit: int = 10,
                                    offset: int = 0) -> List[Dict[str, Any]]:
    """
    Search generation history in the relational database.

    Args:
        criteria: Search criteria
        limit: Maximum number of results
        offset: Offset for pagination

    Returns:
        List of matching generation history entries
    """
    if not self.relational_db:
        logger.error("Relational database not initialized")
        return []

    try:
        db_type = self.relational_db["type"]

        if db_type == "postgresql":
            conn = self.relational_db["connection"]

            # Build WHERE clause
            where_clauses = []
            params = []

            for key, value in criteria.items():
                if key == "specification":
                    where_clauses.append("specification ILIKE %s")
                    params.append(f"%{value}%")
                elif key == "strategy":
                    where_clauses.append("strategy = %s")
                    params.append(value)
                elif key == "min_confidence":
                    where_clauses.append("confidence >= %s")
                    params.append(value)
                # Add more criteria as needed

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            # Execute search
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id, specification, generated_code, strategy, confidence, generation_time, created_at, metadata
                    FROM generation_history
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """, params + [limit, offset])

                rows = cur.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "id": row[0],
                        "specification": row[1],
                        "generated_code": row[2],
                        "strategy": row[3],
                        "confidence": row[4],
                        "generation_time": row[5],
                        "created_at": row[6].isoformat() if row[6] else None,
                        "metadata": json.loads(row[7]) if row[7] else {}
                    })

                return results

        elif db_type == "sqlite":
            conn = self.relational_db["connection"]

            # Build WHERE clause
            where_clauses = []
            params = []

            for key, value in criteria.items():
                if key == "specification":
                    where_clauses.append("specification LIKE ?")
                    params.append(f"%{value}%")
                elif key == "strategy":
                    where_clauses.append("strategy = ?")
                    params.append(value)
                elif key == "min_confidence":
                    where_clauses.append("confidence >= ?")
                    params.append(value)
                # Add more criteria as needed

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            # Execute search
            cursor = conn.execute(f"""
                SELECT id, specification, generated_code, strategy, confidence, generation_time, created_at, metadata
                FROM generation_history
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """, params + [limit, offset])

            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "specification": row[1],
                    "generated_code": row[2],
                    "strategy": row[3],
                    "confidence": row[4],
                    "generation_time": row[5],
                    "created_at": row[6],
                    "metadata": json.loads(row[7]) if row[7] else {}
                })

            return results

        else:
            logger.error(f"Unsupported relational database type: {db_type}")
            return []

    except Exception as e:
        logger.error(f"Failed to search generation history: {e}")
        return []

async def store_in_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
    """
    Store a value in the cache.

    Args:
        key: Cache key
        value: Value to store
        ttl: Time to live in seconds

    Returns:
        True if successful, False otherwise
    """
    if not self.cache:
        return False

    try:
        cache_type = self.cache["type"]

        if cache_type == "redis":
            client = self.cache["client"]

            # Serialize value to JSON
            value_json = json.dumps(value)

            # Store in Redis with TTL
            client.setex(key, ttl, value_json)

            return True

        elif cache_type == "memory":
            # Serialize value to avoid reference issues
            value_copy = json.loads(json.dumps(value))

            # Store in memory cache with expiry time
            self.cache["data"][key] = value_copy
            self.cache["expiry"][key] = time.time() + ttl

            # Clean up expired items
            self._clean_memory_cache()

            return True

        else:
            logger.error(f"Unsupported cache type: {cache_type}")
            return False

    except Exception as e:
        logger.error(f"Failed to store in cache: {e}")
        return False

async def get_from_cache(self, key: str) -> Optional[Any]:
    """
    Get a value from the cache.

    Args:
        key: Cache key

    Returns:
        Cached value or None if not found or expired
    """
    if not self.cache:
        return None

    try:
        cache_type = self.cache["type"]

        if cache_type == "redis":
            client = self.cache["client"]

            # Get from Redis
            value_json = client.get(key)

            if value_json:
                # Deserialize JSON
                return json.loads(value_json)

            return None

        elif cache_type == "memory":
            # Clean up expired items
            self._clean_memory_cache()

            # Check if key exists and not expired
            if key in self.cache["data"] and key in self.cache["expiry"]:
                if self.cache["expiry"][key] > time.time():
                    return self.cache["data"][key]

            return None

        else:
            logger.error(f"Unsupported cache type: {cache_type}")
            return None

    except Exception as e:
        logger.error(f"Failed to get from cache: {e}")
        return None

async def invalidate_cache(self, key: str) -> bool:
    """
    Invalidate a cache entry.

    Args:
        key: Cache key

    Returns:
        True if successful, False otherwise
    """
    if not self.cache:
        return False

    try:
        cache_type = self.cache["type"]

        if cache_type == "redis":
            client = self.cache["client"]

            # Delete from Redis
            client.delete(key)

            return True

        elif cache_type == "memory":
            # Remove from memory cache
            if key in self.cache["data"]:
                del self.cache["data"][key]

            if key in self.cache["expiry"]:
                del self.cache["expiry"][key]

            return True

        else:
            logger.error(f"Unsupported cache type: {cache_type}")
            return False

    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
        return False

def _clean_memory_cache(self):
    """Clean up expired items in memory cache."""
    if self.cache and self.cache["type"] == "memory":
        current_time = time.time()

        # Find expired keys
        expired_keys = []
        for key, expiry in self.cache["expiry"].items():
            if expiry < current_time:
                expired_keys.append(key)

        # Remove expired items
        for key in expired_keys:
            self.cache["data"].pop(key, None)
            self.cache["expiry"].pop(key, None)
"""Direct database adapter for the neural code generator container.

This module provides direct database access for the neural code generator,
allowing it to work with its dedicated vector database and relational database
for optimal performance.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


logger = logging.getLogger(__name__)

class DatabaseAdapter:
    """
    Direct database adapter for the neural code generator.

    This class provides abstracted access to the container's dedicated databases:
    - Vector database for embeddings and similarity search (Milvus/Qdrant)
    - Relational database for metadata and structured data (PostgreSQL)
    - Redis for caching
    """

    def __init__(self,
                 vector_db_config: Dict[str, Any],
                 relational_db_config: Dict[str, Any],
                 cache_config: Dict[str, Any],
                 metrics_collector = None):
        """
        Initialize the database adapter.

        Args:
            vector_db_config: Configuration for vector database
            relational_db_config: Configuration for relational database
            cache_config: Configuration for cache
            metrics_collector: Optional metrics collector
        """
        self.vector_db_config = vector_db_config
        self.relational_db_config = relational_db_config
        self.cache_config = cache_config
        self.metrics_collector = metrics_collector

        # Initialize connections
        self.vector_db = self._init_vector_db()
        self.relational_db = self._init_relational_db()
        self.cache = self._init_cache()

        logger.info("Initialized database adapter")

    def _init_vector_db(self):
        """Initialize vector database connection."""
        db_type = self.vector_db_config.get("type", "milvus").lower()

        try:
            if db_type == "milvus":
                return self._init_milvus()
            elif db_type == "qdrant":
                return self._init_qdrant()
            elif db_type == "file":
                logger.warning("Using file-based vector storage. This is not recommended for production.")
                return self._init_file_vector_db()
            else:
                logger.error(f"Unsupported vector database type: {db_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            # Fall back to file-based vector DB
            return self._init_file_vector_db()

    def _init_milvus(self):
        """Initialize Milvus connection with enterprise-grade configuration."""
        try:
            from pymilvus import Collection
            from pymilvus import CollectionSchema
            from pymilvus import connections
            from pymilvus import DataType
            from pymilvus import FieldSchema
            from pymilvus import utility

            # Extract configuration
            host = self.vector_db_config.get("host", "milvus")
            port = self.vector_db_config.get("port", 19530)
            user = self.vector_db_config.get("user", "")
            password = self.vector_db_config.get("password", "")
            collection_name = self.vector_db_config.get("collection", "neural_code_embeddings")
            vector_dim = self.vector_db_config.get("vector_dim", 1536)
            replica_number = self.vector_db_config.get("replica_number", 3)  # For high availability
            shards_num = self.vector_db_config.get("shards_num", 5)  # For scalability

            # Connect to Milvus with authentication if provided
            connect_kwargs = {
                "alias": "default",
                "host": host,
                "port": port
            }

            if user and password:
                connect_kwargs["user"] = user
                connect_kwargs["password"] = password

            connections.connect(**connect_kwargs)

            # Check if collection exists
            if not utility.has_collection(collection_name):
                # Create collection with optimized fields and schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100,
                                description="Unique identifier for the code snippet"),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim,
                                description="Embedding vector for the code snippet"),
                    FieldSchema(name="code_type", dtype=DataType.VARCHAR, max_length=50,
                                description="Type of code (function, class, module, etc.)"),
                    FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=50,
                                description="Programming language of the code"),
                    FieldSchema(name="timestamp", dtype=DataType.DOUBLE,
                                description="Timestamp when the code was added"),
                    FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=16,
                                description="Tags associated with the code snippet")
                ]

                schema = CollectionSchema(
                    fields=fields,
                    description="Neural code embeddings for code generation",
                    enable_dynamic_field=True  # Allow dynamic fields for flexibility
                )

                # Create collection with advanced configuration
                collection = Collection(
                    name=collection_name,
                    schema=schema,
                    shards_num=shards_num,  # Enable sharding for better performance
                    consistency_level="Strong"  # Ensure strong consistency
                )

                # Create optimized index for vector field using HNSW
                index_params = {
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {
                        "M": 64,  # Higher M value for better recall at the cost of more memory
                        "efConstruction": 500,  # Higher efConstruction for better index quality
                        "ef": 200  # Runtime search parameter for accuracy/speed trade-off
                    }
                }

                collection.create_index(field_name="vector", index_params=index_params)

                # Create scalar field indices for faster filtering
                collection.create_index(
                    field_name="code_type",
                    index_name="code_type_idx"
                )

                collection.create_index(
                    field_name="language",
                    index_name="language_idx"
                )

                # Load collection into memory for faster queries
                collection.load(replica_number=replica_number)

                logger.info(f"Created Milvus collection '{collection_name}' with optimized configuration")
            else:
                # Get existing collection
                collection = Collection(name=collection_name)

                # Ensure collection is loaded with proper replicas
                collection.load(replica_number=replica_number)

            logger.info(f"Connected to Milvus at {host}:{port}, collection: {collection_name}")
            return {
                "type": "milvus",
                "connection": connections,
                "collection": collection,
                "schema": collection.schema,
                "replica_number": replica_number,
                "shards_num": shards_num
            }

        except ImportError:
            logger.error("Milvus client not available. Install with: pip install pymilvus")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {e}")
            return None

    def _init_qdrant(self):
        """Initialize Qdrant connection with enterprise configuration."""
        try:
            import qdrant_client
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            # Extract configuration
            host = self.vector_db_config.get("host", "qdrant")
            port = self.vector_db_config.get("port", 6333)
            grpc_port = self.vector_db_config.get("grpc_port", 6334)
            prefer_grpc = self.vector_db_config.get("prefer_grpc", True)
            https = self.vector_db_config.get("https", True)
            api_key = self.vector_db_config.get("api_key", None)
            collection_name = self.vector_db_config.get("collection", "neural_code_embeddings")
            vector_dim = self.vector_db_config.get("vector_dim", 1536)

            # Configure client options
            client_config = {
                "host": host,
                "port": port,
                "prefer_grpc": prefer_grpc,
            }

            # Add GRPC port if using GRPC
            if prefer_grpc:
                client_config["grpc_port"] = grpc_port

            # Add HTTPS configuration
            if https:
                client_config["https"] = https

            # Add API key if provided
            if api_key:
                client_config["api_key"] = api_key

            # Connect to Qdrant
            client = QdrantClient(**client_config)

            # Check if collection exists
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name not in collection_names:
                # Create collection with optimized configuration
                vector_params = models.VectorParams(
                    size=vector_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True  # Store vectors on disk for large collections
                )

                # Advanced shard configuration for enterprise deployments
                shard_number = self.vector_db_config.get("shard_number", 3)
                replication_factor = self.vector_db_config.get("replication_factor", 2)

                # Create optimized collection
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vector_params,
                    shard_number=shard_number,
                    replication_factor=replication_factor,
                    on_disk_payload=True,  # Store payload on disk for large collections
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000,  # Optimize for large-scale indexing
                        memmap_threshold=100000    # Optimize memory usage
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=64,                 # Higher connectivity for better recall
                        ef_construct=512,     # Higher build-time accuracy
                        full_scan_threshold=10000  # When to use brute force search
                    )
                )

                # Create payload indexes for faster filtering
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="code_type",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="language",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="tags",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                logger.info(f"Created Qdrant collection '{collection_name}' with optimized configuration")

            logger.info(f"Connected to Qdrant at {host}:{port}, collection: {collection_name}")
            return {
                "type": "qdrant",
                "client": client,
                "collection_name": collection_name,
                "vector_dim": vector_dim,
                "shard_number": self.vector_db_config.get("shard_number", 3),
                "replication_factor": self.vector_db_config.get("replication_factor", 2)
            }

        except ImportError:
            logger.error("Qdrant client not available. Install with: pip install qdrant-client")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            return None

    def _init_faiss(self):
        """FAISS has been removed in favor of enterprise-grade vector databases."""
        logger.error("FAISS is no longer supported. Please use Milvus or Qdrant instead.")
        return None

    def _init_file_vector_db(self):
        """Initialize file-based vector database as fallback."""
        try:
            import numpy as np

            # Extract configuration
            storage_path = self.vector_db_config.get("storage_path", "vector_db")

            # Create directory if it doesn't exist
            os.makedirs(storage_path, exist_ok=True)

            # Create index file if it doesn't exist
            index_path = os.path.join(storage_path, "index.json")
            if not os.path.exists(index_path):
                with open(index_path, 'w') as f:
                    json.dump({"vectors": []}, f)

            logger.info(f"Initialized file-based vector database at {storage_path}")
            return {"type": "file", "storage_path": storage_path}

        except Exception as e:
            logger.error(f"Failed to initialize file-based vector database: {e}")
            return None

    def _init_relational_db(self):
        """Initialize relational database connection."""
        db_type = self.relational_db_config.get("type", "postgresql").lower()

        try:
            if db_type == "postgresql":
                return self._init_postgresql()
            elif db_type == "sqlite":
                return self._init_sqlite()
            else:
                logger.error(f"Unsupported relational database type: {db_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize relational database: {e}")
            # Fall back to SQLite
            return self._init_sqlite()

    def _init_postgresql(self):
        """Initialize PostgreSQL connection with enterprise-grade configuration."""
        try:
            import psycopg2
            from psycopg2.extras import DictCursor
            from psycopg2.extras import Json
            from psycopg2.extras import register_default_jsonb

            # Extract configuration
            host = self.relational_db_config.get("host", "postgres")
            port = self.relational_db_config.get("port", 5432)
            database = self.relational_db_config.get("database", "neural_code_generator")
            user = self.relational_db_config.get("user", "postgres")
            password = self.relational_db_config.get("password", "")
            application_name = self.relational_db_config.get("application_name", "neural_code_generator")
            min_connections = self.relational_db_config.get("min_connections", 5)
            max_connections = self.relational_db_config.get("max_connections", 20)

            # Connection pooling setup
            from psycopg2 import pool

            # Create a threaded connection pool
            connection_pool = pool.ThreadedConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                application_name=application_name,
                # Additional connection parameters for production
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=60,
                keepalives_interval=10,
                keepalives_count=5
            )

            # Get a connection from the pool
            conn = connection_pool.getconn()

            # Register JSONB adaptation for better handling of JSON data
            register_default_jsonb(conn)

            # Create tables with advanced features
            with conn.cursor() as cur:
                # Code metadata table with optimized schema
                cur.execute("""
                CREATE TABLE IF NOT EXISTS code_metadata (
                    id VARCHAR(100) PRIMARY KEY,
                    code_type VARCHAR(50) NOT NULL,
                    language VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    version INT NOT NULL DEFAULT 1,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT[] DEFAULT '{}'::TEXT[]
                )
                """)

                # Add indexes for performance
                cur.execute("""
                CREATE INDEX IF NOT EXISTS code_metadata_code_type_idx ON code_metadata (code_type);
                CREATE INDEX IF NOT EXISTS code_metadata_language_idx ON code_metadata (language);
                CREATE INDEX IF NOT EXISTS code_metadata_tags_idx ON code_metadata USING GIN (tags);
                CREATE INDEX IF NOT EXISTS code_metadata_metadata_idx ON code_metadata USING GIN (metadata);
                """)

                # Generation history table with optimized schema
                cur.execute("""
                CREATE TABLE IF NOT EXISTS generation_history (
                    id VARCHAR(100) PRIMARY KEY,
                    specification TEXT NOT NULL,
                    generated_code TEXT NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    confidence REAL NOT NULL,
                    generation_time REAL NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    tags TEXT[] DEFAULT '{}'::TEXT[],
                    parameters JSONB
                )
                """)

                # Add indexes for performance
                cur.execute("""
                CREATE INDEX IF NOT EXISTS generation_history_strategy_idx ON generation_history (strategy);
                CREATE INDEX IF NOT EXISTS generation_history_confidence_idx ON generation_history (confidence);
                CREATE INDEX IF NOT EXISTS generation_history_created_at_idx ON generation_history (created_at);
                CREATE INDEX IF NOT EXISTS generation_history_tags_idx ON generation_history USING GIN (tags);
                CREATE INDEX IF NOT EXISTS generation_history_metadata_idx ON generation_history USING GIN (metadata);
                """)

                # Add text search capabilities
                cur.execute("""
                ALTER TABLE generation_history 
                ADD COLUMN IF NOT EXISTS search_vector tsvector
                    GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', COALESCE(specification, '')), 'A') || 
                        setweight(to_tsvector('english', COALESCE(generated_code, '')), 'B')
                    ) STORED;

                CREATE INDEX IF NOT EXISTS generation_history_search_idx 
                ON generation_history USING GIN (search_vector);
                """)

                # Add monitoring table for query performance
                cur.execute("""
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id SERIAL PRIMARY KEY,
                    query_type VARCHAR(50) NOT NULL,
                    execution_time REAL NOT NULL,
                    query_text TEXT,
                    execution_plan JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """)

                # Add cache invalidation trigger function
                cur.execute("""
                CREATE OR REPLACE FUNCTION notify_code_update() RETURNS TRIGGER AS $
                BEGIN
                    PERFORM pg_notify('code_updates', NEW.id::text);
                    RETURN NEW;
                END;
                $ LANGUAGE plpgsql;

                DROP TRIGGER IF EXISTS code_metadata_update_trigger ON code_metadata;
                CREATE TRIGGER code_metadata_update_trigger
                    AFTER INSERT OR UPDATE ON code_metadata
                    FOR EACH ROW EXECUTE FUNCTION notify_code_update();
                """)

                conn.commit()

            # Return connection to the pool
            connection_pool.putconn(conn)

            logger.info(f"Connected to PostgreSQL at {host}:{port}, database: {database}")
            return {
                "type": "postgresql",
                "pool": connection_pool,
                "db_params": {
                    "host": host,
                    "port": port,
                    "database": database,
                    "user": user
                }
            }

        except ImportError:
            logger.error("psycopg2 not available. Install with: pip install psycopg2-binary")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            return None

    def _init_sqlite(self):
        """Initialize SQLite connection as fallback."""
        try:
            import sqlite3

            # Extract configuration
            db_path = self.relational_db_config.get("db_path", "neural_code_generator.db")

            # Connect to SQLite
            conn = sqlite3.connect(db_path)

            # Create tables if they don't exist
            with conn:
                # Code metadata table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS code_metadata (
                    id TEXT PRIMARY KEY,
                    code_type TEXT NOT NULL,
                    language TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

                # Generation history table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_history (
                    id TEXT PRIMARY KEY,
                    specification TEXT NOT NULL,
                    generated_code TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    generation_time REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
                """)

            logger.info(f"Connected to SQLite database at {db_path}")
            return {"type": "sqlite", "connection": conn}

        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            return None

    def _init_cache(self):
        """Initialize cache."""
        cache_type = self.cache_config.get("type", "memory").lower()

        try:
            if cache_type == "redis":
                return self._init_redis()
            elif cache_type == "memory":
                return self._init_memory_cache()
            else:
                logger.error(f"Unsupported cache type: {cache_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            # Fall back to memory cache
            return self._init_memory_cache()

    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            import redis

            # Extract configuration
            host = self.cache_config.get("host", "localhost")
            port = self.cache_config.get("port", 6379)
            db = self.cache_config.get("db", 0)
            password = self.cache_config.get("password", None)

            # Connect to Redis
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )

            # Test connection
            client.ping()

            logger.info(f"Connected to Redis at {host}:{port}, db: {db}")
            return {"type": "redis", "client": client}

        except ImportError:
            logger.error("Redis client not available. Install with: pip install redis")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            return None

    def _init_memory_cache(self):
        """Initialize in-memory cache as fallback."""
        # Simple in-memory cache with TTL
        return {
            "type": "memory",
            "data": {},
            "expiry": {}
        }

    async def close(self):
        """Close all database connections."""
        logger.info("Closing database connections")

        # Close vector database connection
        if self.vector_db:
            try:
                if self.vector_db["type"] == "milvus":
                    self.vector_db["connection"].disconnect("default")
                elif self.vector_db["type"] == "qdrant":
                    self.vector_db["client"].close()
                # FAISS and file-based don't need to be closed

                logger.info("Closed vector database connection")
            except Exception as e:
                logger.error(f"Error closing vector database connection: {e}")

        # Close relational database connection
        if self.relational_db:
            try:
                if self.relational_db["type"] in ["postgresql", "sqlite"]:
                    self.relational_db["connection"].close()

                logger.info("Closed relational database connection")
            except Exception as e:
                logger.error(f"Error closing relational database connection: {e}")

        # Close cache connection
        if self.cache and self.cache["type"] == "redis":
            try:
                self.cache["client"].close()
                logger.info("Closed Redis connection")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")

    async def store_vector(self,
                           key: str,
                           vector: List[float],
                           metadata: Dict[str, Any]) -> bool:
        """
        Store a vector in the vector database.

        Args:
            key: Unique identifier
            vector: Vector embedding
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.vector_db:
            logger.error("Vector database not initialized")
            return False

        # Start timer for metrics
        timer = None
        if self.metrics_collector:
            timer = self.metrics_collector.start_vector_db_timer("insert")

        try:
            db_type = self.vector_db["type"]

            if db_type == "milvus":
                collection = self.vector_db["collection"]

                # Prepare data
                data = [
                    [key],  # id
                    [vector],  # vector
                    [metadata.get("code_type", "function")],  # code_type
                    [metadata.get("language", "python")],  # language
                    [time.time()]  # timestamp
                ]

                # Insert data
                collection.insert(data)

                # Record operation in metrics
                if self.metrics_collector:
                    self.metrics_collector.record_vector_db_operation("insert", "success")

                return True

            elif db_type == "qdrant":
                client = self.vector_db["client"]
                collection_name = self.vector_db["collection_name"]

                # Convert key to integer for Qdrant
                import hashlib

                from qdrant_client.http import models
                point_id = int(hashlib.md5(key.encode()).hexdigest(), 16) % (10 ** 10)

                # Prepare point
                point = models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "key": key,
                        "code_type": metadata.get("code_type", "function"),
                        "language": metadata.get("language", "python"),
                        **metadata
                    }
                )

                # Upsert point
                client.upsert(
                    collection_name=collection_name,
                    points=[point]
                )

                # Record operation in metrics
                if self.metrics_collector:
                    self.metrics_collector.record_vector_db_operation("insert", "success")

                return True

            elif db_type == "faiss":
                # Convert vector to numpy array
                vector_np = np.array([vector], dtype=np.float32)

                # Add vector to index
                self.vector_db["index"].add(vector_np)

                # Store metadata
                self.vector_db["vectors"].append(key)
                self.vector_db["metadata"].append(metadata)

                # Record operation in metrics
                if self.metrics_collector:
                    self.metrics_collector.record_vector_db_operation("insert", "success")

                return True

            elif db_type == "file":
                storage_path = self.vector_db["storage_path"]

                # Load index file
                index_path = os.path.join(storage_path, "index.json")
                with open(index_path, 'r') as f:
                    index = json.load(f)

                # Check if vector already exists
                for i, entry in enumerate(index["vectors"]):
                    if entry["key"] == key:
                        # Update existing vector
                        index["vectors"][i] = {
                            "key": key,
                            "metadata": metadata,
                            "vector_path": os.path.join(storage_path, f"{key}.npy")
                        }
                        break
                else:
                    # Add new vector
                    index["vectors"].append({
                        "key": key,
                        "metadata": metadata,
                        "vector_path": os.path.join(storage_path, f"{key}.npy")
                    })

                # Save index file
                with open(index_path, 'w') as f:
                    json.dump(index, f)

                # Save vector
                # Save vector
                vector_np = np.array(vector, dtype=np.float32)
                np.save(os.path.join(storage_path, f"{key}.npy"), vector_np)

                # Record operation in metrics
                if self.metrics_collector:
                    self.metrics_collector.record_vector_db_operation("insert", "success")

                return True

            else:
                logger.error(f"Unsupported vector database type: {db_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to store vector: {e}")

            # Record operation in metrics
            if self.metrics_collector:
                self.metrics_collector.record_vector_db_operation("insert", "failure")

            return False

        finally:
            # Stop timer for metrics
            if timer:
                timer()

    async def search_vectors(self,
                             query_vector: List[float],
                             limit: int = 5,
                             filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the vector database.

        Args:
            query_vector: Query vector embedding
            limit: Maximum number of results
            filter_criteria: Optional filter criteria

        Returns:
            List of matching results with metadata
        """
        if not self.vector_db:
            logger.error("Vector database not initialized")
            return []

        # Start timer for metrics
        timer = None
        if self.metrics_collector:
            timer = self.metrics_collector.start_vector_db_timer("search")

        try:
            db_type = self.vector_db["type"]

            if db_type == "milvus":
                collection = self.vector_db["collection"]

                # Prepare search parameters
                search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

                # Prepare filter if provided
                expr = None
                if filter_criteria:
                    conditions = []
                    for key, value in filter_criteria.items():
                        if isinstance(value, str):
                            conditions.append(f"{key} == '{value}'")
                        else:
                            conditions.append(f"{key} == {value}")

                    if conditions:
                        expr = " && ".join(conditions)

                # Execute search
                results = collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param=search_params,
                    limit=limit,
                    expr=expr,
                    output_fields=["code_type", "language", "timestamp"]
                )

                # Format results
                formatted_results = []
                for hits in results:
                    for hit in hits:
                        formatted_results.append({
                            "id": hit.id,
                            "score": hit.score,
                            "code_type": hit.entity.get("code_type"),
                            "language": hit.entity.get("language"),
                            "timestamp": hit.entity.get("timestamp")
                        })

                # Record operation in metrics
                if self.metrics_collector:
                    self.metrics_collector.record_vector_db_operation("search", "success")

                return formatted_results

            elif db_type == "qdrant":
                client = self.vector_db["client"]
                collection_name = self.vector_db["collection_name"]

                from qdrant_client.http import models

                # Prepare filter if provided
                filter_obj = None
                if filter_criteria:
                    must_conditions = []
                    for key, value in filter_criteria.items():
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )

                    if must_conditions:
                        filter_obj = models.Filter(must=must_conditions)

                # Execute search
                results = client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=filter_obj
                )

                # Format results
                formatted_results = []
                for hit in results:
                    formatted_results.append({
                        "id": hit.payload.get("key"),
                        "score": hit.score,
                        "code_type": hit.payload.get("code_type"),
                        "language": hit.payload.get("language"),
                        **{k: v for k, v in hit.payload.items() if k not in ["key", "code_type", "language"]}
                    })

                # Record operation in metrics
                if self.metrics_collector:
                    self.metrics_collector.record_vector_db_operation("search", "success")

                return formatted_results

            elif db_type == "faiss":
                # Convert query vector to numpy array
                query_np = np.array([query_vector], dtype=np.float32)

                # Execute search
                distances, indices = self.vector_db["index"].search(query_np, limit)

                # Format results
                formatted_results = []
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(self.vector_db["vectors"]):
                        key = self.vector_db["vectors"][idx]
                        metadata = self.vector_db["metadata"][idx]
                        formatted_results.append({
                            "id": key,
                            "score": float(1.0 - distances[0][i]),  # Convert distance to similarity score
                            **metadata
                        })

                # Apply filters if provided
                if filter_criteria:
                    filtered_results = []
                    for result in formatted_results:
                        match = True
                        for key, value in filter_criteria.items():
                            if key in result and result[key] != value:
                                match = False
                                break

                        if match:
                            filtered_results.append(result)

                    formatted_results = filtered_results

                # Record operation in metrics
                if self.metrics_collector:
                    self.metrics_collector.record_vector_db_operation("search", "success")

                return formatted_results

            elif db_type == "file":
                storage_path = self.vector_db["storage_path"]

                # Load index file
                index_path = os.path.join(storage_path, "index.json")
                with open(index_path, 'r') as f:
                    index = json.load(f)

                # Convert query vector to numpy array
                query_np = np.array(query_vector, dtype=np.float32)

                # Calculate similarities
                results = []
                for entry in index["vectors"]:
                    # Load vector
                    vector_path = entry["vector_path"]
                    if os.path.exists(vector_path):
                        vector_np = np.load(vector_path)

                        # Calculate cosine similarity
                        similarity = np.dot(query_np, vector_np) / (np.linalg.norm(query_np) * np.linalg.norm(vector_np))

                        # Apply filters if provided
                        if filter_criteria:
                            match = True
                            for key, value in filter_criteria.items():
                                if key in entry["metadata"] and entry["metadata"][key] != value:
                                    match = False
                                    break

                            if not match:
                                continue

                        results.append({
                            "id": entry["key"],
                            "score": float(similarity),
                            **entry["metadata"]
                        })

                # Sort by similarity and limit results
                results.sort(key=lambda x: x["score"], reverse=True)
                results = results[:limit]

                # Record operation in metrics
                if self.metrics_collector:
                    self.metrics_collector.record_vector_db_operation("search", "success")

                return results

            else:
                logger.error(f"Unsupported vector database type: {db_type}")
                return []

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")

            # Record operation in metrics
            if self.metrics_collector:
                self.metrics_collector.record_vector_db_operation("search", "failure")

            return []

        finally:
            # Stop timer for metrics
            if timer:
                timer()

    async def delete_vector(self, key: str) -> bool:
        """
        Delete a vector from the vector database.

        Args:
            key: Unique identifier

        Returns:
            True if successful, False otherwise
        """
        if not self.vector_db:
            logger.error("Vector database not initialized")
            return False

        # Start timer for metrics
        timer = None
        if self.metrics_collector:
            timer = self.metrics_collector.start_vector_db_timer("delete")

        try:
            db_type = self.vector_db["type"]

            if db_type == "milvus":
                collection = self.vector_db["collection"]

                # Execute delete
                collection.delete(f"id == '{key}'")

                # Record operation in metrics
                if self.metrics_collector:
                    self.metrics_collector.record_vector_db_operation("delete", "success")

                return True

            elif db_type == "qdrant":
                client = self.vector_db["client"]
                collection_name = self.vector_db["collection_name"]

                from qdrant_client.http import models

                # Delete point by key in payload
                filter_obj = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="key",
                            match=models.MatchValue(value=key)
                        )
                    ]
                )

                client.delete(
                    collection_name=collection_name,
                    points_selector=filter_obj
                )

                # Record operation in metrics
                if self.metrics_collector:
                    self.metrics_collector.record_vector_db_operation("delete", "success")

                return True

            elif db_type == "faiss":
                # FAISS doesn't support deletion directly
                # We need to rebuild the index without the deleted vector

                # Find the index of the key
                if key in self.vector_db["vectors"]:
                    idx = self.vector_db["vectors"].index(key)

                    # Create a new index
                    import faiss
                    vector_dim = self.vector_db["index"].d
                    new_index = faiss.IndexFlatIP(vector_dim)

                    # Add all vectors except the one to delete
                    vectors = []
                    new_keys = []
                    new_metadata = []

                    for i, k in enumerate(self.vector_db["vectors"]):
                        if i != idx:
                            vectors.append(self.vector_db["index"].reconstruct(i))
                            new_keys.append(k)
                            new_metadata.append(self.vector_db["metadata"][i])

                    # Add vectors to new index
                    if vectors:
                        vectors_np = np.vstack(vectors)
                        new_index.add(vectors_np)

                    # Replace old index
                    self.vector_db["index"] = new_index
                    self.vector_db["vectors"] = new_keys
                    self.vector_db["metadata"] = new_metadata

                    # Record operation in metrics
                    if self.metrics_collector:
                        self.metrics_collector.record_vector_db_operation("delete", "success")

                    return True

                return False

            elif db_type == "file":
                storage_path = self.vector_db["storage_path"]

                # Load index file
                index_path = os.path.join(storage_path, "index.json")
                with open(index_path, 'r') as f:
                    index = json.load(f)

                # Find and remove entry
                for i, entry in enumerate(index["vectors"]):
                    if entry["key"] == key:
                        # Remove vector file
                        vector_path = entry["vector_path"]
                        if os.path.exists(vector_path):
                            os.remove(vector_path)

                        # Remove from index
                        index["vectors"].pop(i)

                        # Save updated index
                        with open(index_path, 'w') as f:
                            json.dump(index, f)

                        # Record operation in metrics
                        if self.metrics_collector:
                            self.metrics_collector.record_vector_db_operation("delete", "success")

                        return True

                return False

            else:
                logger.error(f"Unsupported vector database type: {db_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete vector: {e}")

            # Record operation in metrics
            if self.metrics_collector:
                self.metrics_collector.record_vector_db_operation("delete", "failure")

            return False

        finally:
            # Stop timer for metrics
            if timer:
                timer()

    async def store_code_metadata(self,
                                  key: str,
                                  code_type: str,
                                  language: str,
                                  content: str,
                                  metadata: Dict[str, Any]) -> bool:
        """
        Store code metadata in the relational database.

        Args:
            key: Unique identifier
            code_type: Type of code (function, class, module)
            language: Programming language
            content: Code content
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.relational_db:
            logger.error("Relational database not initialized")
            return False

        try:
            db_type = self.relational_db["type"]

            if db_type == "postgresql":
                conn = self.relational_db["connection"]

                # Serialize metadata to JSON
                metadata_json = json.dumps(metadata)

                # Insert or update code metadata
                with conn.cursor() as cur:
                    cur.execute("""
                    INSERT INTO code_metadata (id, code_type, language, content, metadata, updated_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO UPDATE
                    SET code_type = EXCLUDED.code_type,
                        language = EXCLUDED.language,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                    """, (key, code_type, language, content, metadata_json))

                    conn.commit()

                return True

            elif db_type == "sqlite":
                conn = self.relational_db["connection"]

                # Serialize metadata to JSON
                metadata_json = json.dumps(metadata)

                # Insert or update code metadata
                with conn:
                    # Check if entry exists
                    cursor = conn.execute("SELECT id FROM code_metadata WHERE id = ?", (key,))
                    if cursor.fetchone():
                        # Update existing entry
                        conn.execute("""
                        UPDATE code_metadata
                        SET code_type = ?, language = ?, content = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """, (code_type, language, content, metadata_json, key))
                    else:
                        # Insert new entry
                        conn.execute("""
                        INSERT INTO code_metadata (id, code_type, language, content, metadata, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (key, code_type, language, content, metadata_json))

                return True

            else:
                logger.error(f"Unsupported relational database type: {db_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to store code metadata: {e}")
            return False

    async def get_code_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get code metadata from the relational database.

        Args:
            key: Unique identifier

        Returns:
            Code metadata or None if not found
        """
        # Check cache first
        cached_result = await self.get_from_cache(f"code_metadata:{key}")
        if cached_result:
            # Record cache hit in metrics
            if self.metrics_collector:
                self.metrics_collector.record_cache_hit("code_metadata")
            return cached_result

        if not self.relational_db:
            logger.error("Relational database not initialized")
            return None

        try:
            db_type = self.relational_db["type"]

            if db_type == "postgresql":
                conn = self.relational_db["connection"]

                # Get code metadata
                with conn.cursor() as cur:
                    cur.execute("""
                    SELECT id, code_type, language, content, metadata, created_at, updated_at
                    FROM code_metadata
                    WHERE id = %s
                    """, (key,))

                    row = cur.fetchone()

                    if row:
                        result = {
                            "id": row[0],
                            "code_type": row[1],
                            "language": row[2],
                            "content": row[3],
                            "metadata": json.loads(row[4]) if row[4] else {},
                            "created_at": row[5].isoformat() if row[5] else None,
                            "updated_at": row[6].isoformat() if row[6] else None
                        }

                        # Cache the result
                        await self.store_in_cache(f"code_metadata:{key}", result, ttl=3600)

                        # Record cache miss in metrics
                        if self.metrics_collector:
                            self.metrics_collector.record_cache_miss("code_metadata")

                        return result

                return None

            elif db_type == "sqlite":
                conn = self.relational_db["connection"]

                # Get code metadata
                cursor = conn.execute("""
                SELECT id, code_type, language, content, metadata, created_at, updated_at
                FROM code_metadata
                WHERE id = ?
                """, (key,))

                row = cursor.fetchone()

                if row:
                    result = {
                        "id": row[0],
                        "code_type": row[1],
                        "language": row[2],
                        "content": row[3],
                        "metadata": json.loads(row[4]) if row[4] else {},
                        "created_at": row[5],
                        "updated_at": row[6]
                    }

                    # Cache the result
                    await self.store_in_cache(f"code_metadata:{key}", result, ttl=3600)

                    # Record cache miss in metrics
                    if self.metrics_collector:
                        self.metrics_collector.record_cache_miss("code_metadata")

                    return result

                return None

            else:
                logger.error(f"Unsupported relational database type: {db_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to get code metadata: {e}")
            return None