FROM python:3.11-slim

# Set environment variables for flexibility
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/llm_volume/models/deepseek-coder-6.7b-instruct \
    DEVICE_TYPE=auto

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ML dependencies (adjust versions as needed)
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    transformers>=4.34.0 \
    sentence-transformers>=2.2.2 \
    accelerate>=0.21.0 \
    bitsandbytes>=0.40.0 \
    pulsar-client>=3.1.0 \
    astunparse>=1.6.3

# Copy the necessary files
COPY src/services/neural_code_generator/app/enhanced_neural_code_generator/
COPY components/knowledge_base/vector_knowledge_base.py /components/knowledge_base/

# Create start script with hardware detection
RUN echo '#!/bin/bash\n\
# Detect hardware and set device type\n\
if [ -d "/proc/driver/nvidia" ] || [ -d "/dev/nvidia0" ]; then\n\
  echo "NVIDIA GPU detected, using CUDA"\n\
  export DEVICE_TYPE="cuda"\n\
elif [ "$(uname -m)" = "arm64" ] && [ "$(uname)" = "Darwin" ]; then\n\
  echo "Apple Silicon detected, using MPS"\n\
  export DEVICE_TYPE="mps"\n\
else\n\
  echo "No GPU detected, using CPU"\n\
  export DEVICE_TYPE="cpu"\n\
fi\n\
\n\
# Start neural code generator service using config file\n\
if [ "$USE_ENHANCED" = "true" ]; then\n\
  python -m services.neural_code_generator.enhanced_neural_code_generator "$@"\n\
else\n\
  python -m services.neural_code_generator.neural_code_generator "$@"\n\
fi' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Set up non-root user for better security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set environment variables based on config file
ENV USE_ENHANCED=true \
    PULSAR_SERVICE_URL=pulsar://pulsar:6650 \
    INPUT_TOPIC=code-generation-requests \
    OUTPUT_TOPIC=code-generation-results \
    SUBSCRIPTION_NAME=code-generator-worker \
    KNOWLEDGE_BASE_PATH=/app/knowledge_base

ENTRYPOINT ["/app/entrypoint.sh"]