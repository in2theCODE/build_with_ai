"""
Health check API for Neural Code Generator.

This module provides a simple HTTP server for container health checks
and basic monitoring of the Neural Code Generator.
"""

import logging
import os
import threading
import time
from typing import Any, Dict

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
import psutil
from pydantic import BaseModel
import torch
import uvicorn


# Configure logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("health_check")

# Create FastAPI application
app = FastAPI(
    title="Neural Code Generator Health Check",
    description="Health check and monitoring API for the Neural Code Generator service",
    version="1.0.0",
)

# Define global variables for status tracking
start_time = time.time()
last_activity = time.time()
request_count = 0
success_count = 0
error_count = 0
avg_processing_time = 0.0

# Thread lock for updating stats
stats_lock = threading.Lock()


class HealthStatus(BaseModel):
    """Health status response model."""

    status: str
    uptime: float
    memory_usage: Dict[str, float]
    gpu_usage: Dict[str, Any]
    request_stats: Dict[str, Any]


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Return the current health status of the service."""
    try:
        # Calculate uptime
        uptime = time.time() - start_time

        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage = {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
        }

        # Get GPU usage if available
        gpu_usage = {"available": False}
        if torch.cuda.is_available():
            gpu_usage = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                "max_memory_mb": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024),
            }

        # Get request statistics
        with stats_lock:
            request_stats = {
                "total_requests": request_count,
                "successful_requests": success_count,
                "error_requests": error_count,
                "avg_processing_time": avg_processing_time,
                "last_activity": time.time() - last_activity,
            }

        # Return the health status
        return HealthStatus(
            status="healthy",
            uptime=uptime,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            request_stats=request_stats,
        )
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint redirecting to health check."""
    return {
        "message": "Neural Code Generator Health Check API",
        "health_endpoint": "/health",
    }


@app.get("/readiness")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    # Check if enough memory is available
    try:
        memory = psutil.virtual_memory()
        if memory.available < 1 * 1024 * 1024 * 1024:  # 1 GB
            return Response(content="Low memory", status_code=503)

        # Check if GPU is available when required
        if os.environ.get("REQUIRE_GPU", "false").lower() == "true" and not torch.cuda.is_available():
            return Response(content="GPU not available", status_code=503)

        # Check for recent activity (if service has been running for a while)
        uptime = time.time() - start_time
        if uptime > 300 and time.time() - last_activity > 600:
            return Response(content="Service inactive", status_code=503)

        return Response(content="Ready", status_code=200)
    except Exception as e:
        logger.error(f"Error in readiness check: {e}")
        return Response(content=str(e), status_code=503)


@app.get("/liveness")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return Response(content="Alive", status_code=200)


def update_stats(processing_time, success=True):
    """Update the request statistics."""
    global request_count, success_count, error_count, avg_processing_time, last_activity

    with stats_lock:
        last_activity = time.time()
        request_count += 1

        if success:
            success_count += 1
        else:
            error_count += 1

        # Update average processing time with exponential moving average
        if avg_processing_time == 0.0:
            avg_processing_time = processing_time
        else:
            avg_processing_time = 0.9 * avg_processing_time + 0.1 * processing_time


def start_server():
    """Start the health check server."""
    port = int(os.environ.get("HEALTH_CHECK_PORT", "8000"))
    host = os.environ.get("HEALTH_CHECK_HOST", "0.0.0.0")

    logger.info(f"Starting health check server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
