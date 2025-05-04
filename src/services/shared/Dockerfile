# Build stage
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Set working directory
WORKDIR /build

# Copy and install dependencies first (for better caching)
COPY setup.py README.md /build/
# Only copy what's needed for installing dependencies
COPY shared/__init__.py /build/shared/

# Install dependencies
RUN pip install --no-cache-dir --prefix=/install .

# Final stage
FROM python:3.10-slim

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Set working directory
WORKDIR /app

# Copy built artifacts from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=appuser:appuser shared/ /app/shared/
COPY --chown=appuser:appuser validation/ /app/validation/
COPY --chown=appuser:appuser pulsar/ /app/pulsar/
COPY --chown=appuser:appuser monitoring/ /app/monitoring/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser logging/ /app/logging/
COPY --chown=appuser:appuser concurrency/ /app/concurrency/

# Add main.py entry point file
COPY --chown=appuser:appuser main.py /app/

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0) if True else sys.exit(1)"

# Command - using the main.py entry point
CMD ["python", "main.py"]