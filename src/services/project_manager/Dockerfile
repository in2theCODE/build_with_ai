FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install shared schemas package
# We set this as a build argument so we can use different versions in different environments
ARG SCHEMAS_VERSION=1.0.0
RUN pip install --no-cache-dir specs-schemas==${SCHEMAS_VERSION}

# Copy service code
COPY . /app/

# Create storage directory
RUN mkdir -p /app/storage

# Run the service
CMD ["python", "spec_registry.py"]