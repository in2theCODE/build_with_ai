FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for meta-learning data
RUN mkdir -p /app/meta_learning_data

# Copy application code
COPY . /app/

# Set Python path for imports
ENV PYTHONPATH=/app

# Run the meta_learner service
CMD ["python", "meta_learner.py"]