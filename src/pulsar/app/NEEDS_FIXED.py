import asyncio
from datetime import datetime
from datetime import timedelta
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pulsar
from uuid6 import uuid7
import uvicorn
import yaml

from .api.models import ErrorResponse
from .api.models import HealthResponse
from .api.models import IntentAnalysis
from .api.models import PatternCreateRequest
from .api.models import PatternListResponse
from .api.models import PatternResponse
from .api.models import QueryRequest
from .api.models import QueryResponse
from .core.intent_analyzer import IntentAnalyzer
from .core.intent_analyzer import ProcessingMode
from .core.pattern_matcher import PatternMatcher
from .events.publisher.events import EventPublisher
from .models.embedding_client import EmbeddingClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("neural-interpreter")


# Load config
def load_config():
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Fall back to default config
        return {
            "embedding_service": {"url": "http://embedding-service:8001"},
            "intent_analysis": {
                "complexity_threshold": 0.7,
                "fast_path_confidence_threshold": 0.85,
                "max_token_threshold": 200,
            },
            "neo4j": {"uri": "neo4j://neo4j:7687", "user": "neo4j", "password": "password"},
            "pulsar": {
                "service_url": "pulsar://pulsar-proxy:6650",
                "inference_topic": "persistent://public/default/inference-requests",
                "workflow_topic": "persistent://public/default/workflow-executions",
            },
        }


# Application state
class AppState:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = time.time()
        self.embedding_client = None
        self.pattern_matcher = None
        self.intent_analyzer = None
        self.event_publisher = None

    async def initialize(self):
        """Initialize application services"""
        # Create embedding client
        self.embedding_client = EmbeddingClient(self.config["embedding_service"]["url"])

        # Create pattern matcher with Neo4j
        self.pattern_matcher = PatternMatcher(
            neo4j_uri=self.config["neo4j"]["uri"],
            neo4j_user=self.config["neo4j"]["user"],
            neo4j_password=self.config["neo4j"]["password"],
        )
        await self.pattern_matcher.initialize()

        # Create intent analyzer
        self.intent_analyzer = IntentAnalyzer(
            config=self.config,
            embedding_client=self.embedding_client,
            pattern_matcher=self.pattern_matcher,
        )

        # Create event publisher
        self.event_publisher = EventPublisher(
            pulsar_service_url=self.config["pulsar"]["service_url"],
            inference_topic=self.config["pulsar"]["inference_topic"],
            workflow_topic=self.config["pulsar"]["workflow_topic"],
        )
        await self.event_publisher.initialize()

    async def shutdown(self):
        """Shutdown application services"""
        if self.pattern_matcher:
            await self.pattern_matcher.close()

        if self.event_publisher:
            await self.event_publisher.close()


# Create FastAPI application
config = load_config()
app = FastAPI(
    title="Neural Interpreter Service",
    description="Query analysis, pattern matching, and intent classification",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create application state
app_state = AppState(config)


# Dependency to get app services
async def get_intent_analyzer():
    if app_state.intent_analyzer is None:
        raise HTTPException(
            status_code=503, detail="Service initializing, please try again shortly"
        )
    return app_state.intent_analyzer


async def get_pattern_matcher():
    if app_state.pattern_matcher is None:
        raise HTTPException(
            status_code=503, detail="Service initializing, please try again shortly"
        )
    return app_state.pattern_matcher


async def get_event_publisher():
    if app_state.event_publisher is None:
        raise HTTPException(
            status_code=503, detail="Service initializing, please try again shortly"
        )
    return app_state.event_publisher


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Neural Interpreter Service")
    await app_state.initialize()
    logger.info("Neural Interpreter Service started")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Neural Interpreter Service")
    await app_state.shutdown()
    logger.info("Neural Interpreter Service stopped")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    neo4j_status = "healthy"
    embedding_status = "healthy"

    # Check Neo4j connection
    try:
        if app_state.pattern_matcher:
            await app_state.pattern_matcher.list_patterns(limit=1)
        else:
            neo4j_status = "initializing"
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        neo4j_status = "unhealthy"

    # Check embedding service
    try:
        if app_state.embedding_client:
            test_embedding = await app_state.embedding_client.embed_text("test")
            if not test_embedding or len(test_embedding) < 10:
                embedding_status = "degraded"
        else:
            embedding_status = "initializing"
    except Exception as e:
        logger.error(f"Embedding service health check failed: {e}")
        embedding_status = "unhealthy"

    uptime = int(time.time() - app_state.start_time)

    return HealthResponse(
        status=(
            "healthy" if neo4j_status == "healthy" and embedding_status == "healthy" else "degraded"
        ),
        version="1.0.0",
        uptime_seconds=uptime,
        database_status=neo4j_status,
        embedding_service_status=embedding_status,
    )


# Query analysis endpoint
@app.post("/interpret", response_model=QueryResponse)
async def interpret_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    intent_analyzer: IntentAnalyzer = Depends(get_intent_analyzer),
    event_publisher: EventPublisher = Depends(get_event_publisher),
):
    start_time = time.time()
    request_id = request.request_id or str(uuid7())

    try:
        # Analyze the query intent
        intent = await intent_analyzer.analyze_query(
            query=request.prompt,
            system_message=request.system_message,
            require_delegation=request.require_deliberation,
        )

        # Convert to response model
        intent_analysis = IntentAnalysis(
            processing_mode=intent.processing_mode,
            complexity_score=intent.complexity_score,
            confidence=(
                1.0
                if intent.processing_mode == ProcessingMode.DELIBERATIVE
                else (intent.fast_path_match["score"] if intent.fast_path_match else 0.7)
            ),
            pattern_match=intent.fast_path_match,
            elapsed_ms=int((time.time() - start_time) * 1000),
        )

        # Publish events in background to avoid blocking response
        events_published = []
        if intent.processing_mode == ProcessingMode.REACTIVE:
            # Fast path - publish inference request
            background_tasks.add_task(
                event_publisher.publish_inference_request,
                query=request.prompt,
                system_message=request.system_message,
                session_id=request.session_id,
                request_id=request_id,
                pattern_match=intent.fast_path_match,
            )
            events_published.append(config["pulsar"]["inference_topic"])
        else:
            # Complex path - publish workflow execution request
            background_tasks.add_task(
                event_publisher.publish_workflow_request,
                query=request.prompt,
                system_message=request.system_message,
                session_id=request.session_id,
                request_id=request_id,
                complexity_score=intent.complexity_score,
            )
            events_published.append(config["pulsar"]["workflow_topic"])

        return QueryResponse(
            request_id=request_id,
            session_id=request.session_id,
            intent_analysis=intent_analysis,
            events_published=events_published,
        )

    except Exception as e:
        logger.error(f"Error interpreting query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=str(e), error_code="INTERPRETATION_ERROR", request_id=request_id
            ).dict(),
        )


# Pattern management endpoints
@app.post("/patterns", response_model=PatternResponse)
async def create_pattern(
    request: PatternCreateRequest,
    pattern_matcher: PatternMatcher = Depends(get_pattern_matcher),
    embedding_client: EmbeddingClient = Depends(lambda: app_state.embedding_client),
):
    try:
        # Get embedding if not provided
        embedding = request.embedding
        if embedding is None:
            embedding = await embedding_client.embed_text(request.text)

        # Store pattern
        pattern_id = await pattern_matcher.store_pattern(
            pattern_text=request.text, embedding=embedding, metadata=request.metadata
        )

        return PatternResponse(
            id=pattern_id,
            text=request.text,
            created_at=datetime.utcnow(),
            metadata=request.metadata,
        )

    except Exception as e:
        logger.error(f"Error creating pattern: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns", response_model=PatternListResponse)
async def list_patterns(
    limit: int = 100, pattern_matcher: PatternMatcher = Depends(get_pattern_matcher)
):
    try:
        patterns = await pattern_matcher.list_patterns(limit=limit)

        return PatternListResponse(
            patterns=[
                PatternResponse(
                    id=p["id"],
                    text=p["text"],
                    created_at=datetime.utcnow() - timedelta(days=1),  # Placeholder
                    metadata=p["metadata"],
                )
                for p in patterns
            ],
            count=len(patterns),
        )

    except Exception as e:
        logger.error(f"Error listing patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/patterns/{pattern_id}", response_model=Dict[str, bool])
async def delete_pattern(
    pattern_id: str, pattern_matcher: PatternMatcher = Depends(get_pattern_matcher)
):
    try:
        success = await pattern_matcher.delete_pattern(pattern_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting pattern: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"error": str(exc), "error_code": "INTERNAL_ERROR"}
    )


# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
