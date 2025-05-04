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

# Environment variables (will be overridden by docker-compose)
ENV PULSAR_URL=pulsar://pulsar:6650
ENV STORAGE_DIR=/app/storage
ENV PROJECT_EVENTS_TOPIC=persistent://public/default/project_events
ENV SPEC_SHEET_EVENTS_TOPIC=persistent://public/default/spec_sheet_events
ENV CODE_GEN_EVENTS_TOPIC=persistent://public/default/code_generation_events
ENV WORKFLOW_COMMANDS_TOPIC=persistent://public/default/workflow_commands

# Run the service
CMD ["python", "workflow_orchestrator.py"]