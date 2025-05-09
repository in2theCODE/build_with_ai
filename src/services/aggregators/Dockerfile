FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Environment variables
ENV PYTHONPATH=/app

# Run the service with a non-root user
RUN useradd -m appuser
USER appuser

# Command to run the service
CMD ["python", "-m", "src.services.response_aggregator.main"]