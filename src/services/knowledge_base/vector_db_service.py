import time

from src.services.shared.constants.base_component import BaseComponent
from src.services.shared.logging.logger import get_logger


class VectorDBService(BaseComponent):
    """Service that handles vector DB operations through Pulsar events."""

    def __init__(self, **params):
        """Initialize the Vector DB service."""
        super().__init__(**params)
        self.logger = get_logger(self.__class__.__name__)
        self.vector_db_client = self._initialize_vector_db()
        self.running = False

    def _initialize_vector_db(self):
        """Initialize the vector database client."""
        vector_db_type = self.get_param("vector_db_type", "milvus")
        connection_params = self.get_param("vector_db_connection", {})

        if vector_db_type == "milvus":
            from pymilvus import connections, Collection, utility

            # Connect to Milvus
            connections.connect(
                alias="default",
                host=connection_params.get("host", "localhost"),
                port=connection_params.get("port", 19530)
            )

            self.logger.info(f"Connected to Milvus at {connection_params.get('host', 'localhost')}:{connection_params.get('port', 19530)}")

            return MilvusVectorDBClient(self.logger)

        elif vector_db_type == "qdrant":
            from qdrant_client import QdrantClient

            client = QdrantClient(
                host=connection_params.get("host", "localhost"),
                port=connection_params.get("port", 6333)
            )

            self.logger.info(f"Connected to Qdrant at {connection_params.get('host', 'localhost')}:{connection_params.get('port', 6333)}")

            return QdrantVectorDBClient(client, self.logger)

        elif vector_db_type == "pinecone":
            import pinecone

            pinecone.init(
                api_key=connection_params.get("api_key", ""),
                environment=connection_params.get("environment", "")
            )

            self.logger.info(f"Connected to Pinecone in {connection_params.get('environment', '')}")

            return PineconeVectorDBClient(self.logger)

        else:
            self.logger.warning(f"Unknown vector DB type: {vector_db_type}, using in-memory fallback")
            return InMemoryVectorDBClient(self.logger)

    async def start(self):
        """Start listening for vector DB operation events."""
        if self.running:
            self.logger.info("VectorDBService is already running")
            return

        self.running = True

        # Subscribe to vector DB operation events
        await self.event_listener.subscribe(
            topics=["vector_db.query", "vector_db.store", "vector_db.delete"],
            callback=self._handle_vector_db_event
        )

        self.logger.info("VectorDBService started successfully")

    async def stop(self):
        """Stop the Vector DB service."""
        if not self.running:
            return

        self.running = False

        # Unsubscribe from events
        await self.event_listener.unsubscribe(
            topics=["vector_db.query", "vector_db.store", "vector_db.delete"]
        )

        # Close vector DB connection
        if hasattr(self.vector_db_client, "close"):
            await self.vector_db_client.close()

        self.logger.info("VectorDBService stopped")

    async def _handle_vector_db_event(self, event):
        """Handle vector DB operation events."""
        operation = event.event_type.split(".")[-1]  # "query", "store", or "delete"
        collection = event.data.get("collection")
        correlation_id = event.correlation_id

        if not collection:
            await self._emit_error_response(
                operation, correlation_id, "Missing collection name"
            )
            return

        try:
            if operation == "query":
                await self._handle_query(event, collection, correlation_id)
            elif operation == "store":
                await self._handle_store(event, collection, correlation_id)
            elif operation == "delete":
                await self._handle_delete(event, collection, correlation_id)
            else:
                await self._emit_error_response(
                    operation, correlation_id, f"Unknown operation: {operation}"
                )
        except Exception as e:
            self.logger.error(f"Error handling {operation} operation: {str(e)}", exc_info=True)
            await self._emit_error_response(operation, correlation_id, str(e))

    async def _handle_query(self, event, collection, correlation_id):
        """Handle query operation."""
        vector = event.data.get("vector")
        limit = event.data.get("limit", 5)
        filter_criteria = event.data.get("filter", {})

        if not vector:
            await self._emit_error_response(
                "query", correlation_id, "Missing vector for query"
            )
            return

        start_time = time.time()
        results = await self.vector_db_client.query_vectors(
            collection, vector, limit=limit, filter_criteria=filter_criteria
        )
        query_time = time.time() - start_time

        # Log performance metric
        self.logger.performance(
            f"Vector DB query completed for {collection}",
            extra={
                "structured_data": {
                    "metric": "vector_db_query_time",
                    "value": query_time,
                    "tags": {
                        "collection": collection,
                        "result_count": len(results),
                        "limit": limit
                    }
                }
            }
        )

        # Emit response event
        await self.event_emitter.emit_event(
            event_type="vector_db.query.response",
            correlation_id=correlation_id,
            data={"results": results}
        )

    async def _handle_store(self, event, collection, correlation_id):
        """Handle store operation."""
        vector = event.data.get("vector")
        metadata = event.data.get("metadata", {})
        id = event.data.get("id")

        if not vector:
            await self._emit_error_response(
                "store", correlation_id, "Missing vector for store operation"
            )
            return

        start_time = time.time()
        result = await self.vector_db_client.store_vector(
            collection, vector, metadata, id=id
        )
        store_time = time.time() - start_time

        # Log performance metric
        self.logger.performance(
            f"Vector DB store completed for {collection}",
            extra={
                "structured_data": {
                    "metric": "vector_db_store_time",
                    "value": store_time,
                    "tags": {
                        "collection": collection
                    }
                }
            }
        )

        # Emit response event
        await self.event_emitter.emit_event(
            event_type="vector_db.store.response",
            correlation_id=correlation_id,
            data={"result": result}
        )

    async def _handle_delete(self, event, collection, correlation_id):
        """Handle delete operation."""
        id = event.data.get("id")
        filter_criteria = event.data.get("filter", {})

        if not id and not filter_criteria:
            await self._emit_error_response(
                "delete", correlation_id, "Missing id or filter criteria for delete operation"
            )
            return

        start_time = time.time()
        result = await self.vector_db_client.delete_vectors(
            collection, id=id, filter_criteria=filter_criteria
        )
        delete_time = time.time() - start_time

        # Log performance metric
        self.logger.performance(
            f"Vector DB delete completed for {collection}",
            extra={
                "structured_data": {
                    "metric": "vector_db_delete_time",
                    "value": delete_time,
                    "tags": {
                        "collection": collection
                    }
                }
            }
        )

        # Emit response event
        await self.event_emitter.emit_event(
            event_type="vector_db.delete.response",
            correlation_id=correlation_id,
            data={"result": result}
        )

    async def _emit_error_response(self, operation, correlation_id, error_message):
        """Emit an error response event."""
        await self.event_emitter.emit_event(
            event_type=f"vector_db.{operation}.error",
            correlation_id=correlation_id,
            data={"error": error_message}
        )