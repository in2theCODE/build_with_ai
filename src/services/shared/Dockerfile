FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install only required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY shared-models/src/services/aggregators/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code with proper directory structure
COPY shared-models/src /app/src

# Run the service using module notation
CMD ["python",  "src.services.aggregators.app.response_aggregator.py"]

# Health check to verify service is running
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import socket; socket.socket().connect(('localhost', 8080))" || exit 1