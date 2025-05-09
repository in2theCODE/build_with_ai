FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY . /app/

# Create storage directory
RUN mkdir -p /app/storage

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# Run the service
CMD ["python", "main.py"]