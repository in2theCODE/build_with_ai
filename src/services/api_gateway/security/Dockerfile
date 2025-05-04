FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the HMAC signer code
COPY hmac_signer.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Expose port for the service
EXPOSE 8080

# Environment variables for configuration
ENV PYTHONUNBUFFERED=1
ENV HMAC_SECRET_KEY=""
ENV HMAC_ALGORITHM="sha256"
ENV SERVICE_PORT=8080

# Run the HMAC signer service
CMD ["python", "hmac_signer.py"]