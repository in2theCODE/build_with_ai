FROM nvidia/cuda:12.3.0-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Install PyTorch with CUDA
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install vLLM and Arctic Inference
RUN pip install --no-cache-dir \
    "vllm==0.8.4" \
    "git+https://github.com/snowflakedb/ArcticInference.git#egg=arctic-inference[vllm]" \
    pyyaml==6.0.1 \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic==2.5.3 \
    prometheus-client==0.19.0

# Create application directory
WORKDIR /app

# Copy application files
COPY speculative_accelerator.py .
COPY config.yaml .

# Set entrypoint and default command
ENTRYPOINT ["python", "speculative_accelerator.py"]
CMD ["--config", "config.yaml"]

# Expose API port
EXPOSE 8000
EXPOSE 8001

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1