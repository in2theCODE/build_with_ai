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

# Create directories
RUN mkdir -p /app/templates

# Environment variables (will be overridden by docker-compose)
ENV PULSAR_URL=pulsar://pulsar:6650
ENV TEMPLATE_ROOT=/app/templates
ENV TEMPLATE_REQUEST_TOPIC=persistent://public/default/template_requests
ENV TEMPLATE_RESPONSE_TOPIC=persistent://public/default/template_responses
ENV METRICS_TOPIC=persistent://public/default/metrics

# Run the service
CMD ["python", "template_discovery_service.py"]