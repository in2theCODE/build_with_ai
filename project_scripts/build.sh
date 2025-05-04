#!/bin/bash
#============================================================================
# Template Registry System for Spec-Driven AI Code Generation Platform
#
# This script sets up a comprehensive template registry system with:
# - Apache Pulsar for event-driven architecture
# - Git-based version control for templates
# - Redis caching for high performance
# - Self-evolving templates through usage analytics
# - Robust monitoring and security features
#
# Author: Claude
# Date: 2025-04-18
#============================================================================

set -e

echo "======================================================================"
echo "Setting up Template Registry System for Spec-Driven AI Code Generation"
echo "======================================================================"

# Set up log file
LOG_FILE="./template-registry-build.log"
touch $LOG_FILE

# Function to log messages to both console and log file
log() {
    echo "$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
}

# Function to check system requirements
check_requirements() {
    log "Checking system requirements..."

    # Check Python version
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log "Found Python version: $PYTHON_VERSION"

        # Check if Python version is 3.8 or higher
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            log "Python version requirement met."
        else
            log "ERROR: Python 3.8 or higher is required."
            exit 1
        fi
    else
        log "ERROR: Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi

    # Check Git
    if command -v git &>/dev/null; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        log "Found Git version: $GIT_VERSION"
    else
        log "ERROR: Git not found. Please install Git."
        exit 1
    fi

    # Check Docker (optional)
    if command -v docker &>/dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
        log "Found Docker version: $DOCKER_VERSION"

        # Check Docker Compose
        if command -v docker-compose &>/dev/null; then
            DOCKER_COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | tr -d ',')
            log "Found Docker Compose version: $DOCKER_COMPOSE_VERSION"
        else
            log "WARNING: Docker Compose not found. It will be needed for local deployment."
        fi
    else
        log "WARNING: Docker not found. It will be needed for local deployment."
    fi

    log "System requirement check completed."
}

# Create directory structure
echo "[1/8] Creating directory structure..."
mkdir -p ./template-registry/{src,models,config,storage,scripts,tests,docs}
mkdir -p ./template-registry/src/{core,event_handlers,validators,adapters,utils,security,metrics}
mkdir -p ./template-registry/models/{templates,schema,events,analytics}
mkdir -p ./template-registry/storage/{git,cache,registry}
mkdir -p ./template-registry/tests/{unit,integration,performance}
mkdir -p ./template-registry/docs/{api,architecture,deployment}
mkdir -p ./template-registry/kubernetes

echo "Directory structure created successfully."

# Create core configuration files
echo "[2/8] Creating configuration files and documentation..."

# Create README file
cat > ./template-registry/README.md << 'EOF'
# Template Registry System

A comprehensive, self-evolving template registry system for spec-driven AI code generation platforms.

## Overview

This system provides a robust, event-driven architecture for managing, versioning, and evolving specification templates. It's designed to be:

- **Self-evolving:** Templates automatically improve based on usage patterns
- **Event-driven:** Built entirely on Apache Pulsar for asynchronous communication
- **Highly available:** Hybrid storage approach with Git, Redis, and Pulsar Schema Registry
- **Production-ready:** Includes Docker, Kubernetes configs, monitoring, and security

## Architecture

The system follows a modular, microservice-based architecture with the following key components:

1. **Core Registry:** Manages templates, versions, and operations
2. **Event Bus:** Handles all communication using Apache Pulsar
3. **Storage Layer:** Hybrid approach with:
   - Git for version control and history
   - Redis for high-performance caching
   - Pulsar Schema Registry for validation
4. **Evolution Engine:** Analyzes usage patterns and suggests improvements
5. **Metrics & Monitoring:** Tracks system performance and template usage

## Directory Structure

```
template-registry/
├── config/             # Configuration files
├── docs/               # Documentation
├── kubernetes/         # Kubernetes deployment files
├── models/             # Data models
├── scripts/            # Utility scripts
├── src/                # Source code
│   ├── adapters/       # Storage adapters
│   ├── core/           # Core registry logic
│   ├── event_handlers/ # Event handling
│   ├── metrics/        # Metrics collection
│   ├── security/       # Security features
│   ├── utils/          # Utilities
│   └── validators/     # Validation logic
├── storage/            # Storage directories
│   ├── cache/          # Redis cache data
│   ├── git/            # Git repositories
│   └── registry/       # Schema registry data
└── tests/              # Tests
    ├── integration/    # Integration tests
    ├── performance/    # Performance tests
    └── unit/           # Unit tests
```

## Technologies Used

- **Apache Pulsar:** Event bus and schema registry
- **Redis:** High-performance caching
- **Git:** Version control for templates
- **Python 3.10+:** Core implementation language
- **Docker & Kubernetes:** Containerization and orchestration
- **Prometheus & Grafana:** Monitoring and visualization

## Self-Evolution Features

The template registry includes advanced self-evolution capabilities:

1. **Usage Analytics:** Tracks how templates are used in real-time
2. **Pattern Recognition:** Identifies common patterns and pain points
3. **Automated Suggestions:** Suggests improvements based on usage
4. **Versioning:** Maintains backward compatibility while evolving
5. **A/B Testing:** Tests new template versions with real users

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Git

### Installation

1. Clone the repository
2. Run the build script: `./build-template-registry.sh`
3. Start the services: `cd template-registry && docker-compose up -d`

### Configuration

Edit the `config/config.yaml` file to customize the system:

- Pulsar connection settings
- Redis cache configuration
- Git repository settings
- Evolution parameters
- Security settings

## Documentation

For more detailed documentation, see:

- [Architecture](./docs/architecture/)
- [API Reference](./docs/api/)
- [Deployment Guide](./docs/deployment/)
- [Evolution Design](./docs/evolution/)

## License

MIT License
EOF

# Create architecture documentation
mkdir -p ./template-registry/docs/architecture
cat > ./template-registry/docs/architecture/overview.md << 'EOF'
# Template Registry Architecture Overview

## System Architecture

The Template Registry system is built as a modern, cloud-native application with a focus on:

1. **Event-driven communication**
2. **Distributed storage**
3. **Self-evolution**
4. **High availability**
5. **Observability**

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       Template Registry                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
┌─────────────▼─────┐  ┌────────▼────────┐  ┌─────▼─────────────┐
│  Event Bus (Pulsar)│  │  Storage Layer  │  │  Evolution Engine │
└─────────────┬─────┘  └────────┬────────┘  └─────┬─────────────┘
              │                 │                 │
              │        ┌────────┼────────┐        │
              │        │        │        │        │
      ┌───────▼──┐ ┌───▼───┐ ┌──▼───┐ ┌──▼───┐ ┌──▼───┐
      │Producers │ │  Git  │ │Redis │ │Schema│ │ AI   │
      │Consumers │ │       │ │Cache │ │Registry│ │Models│
      └──────────┘ └───────┘ └──────┘ └──────┘ └──────┘
```

## Event-Driven Architecture

The entire system is designed around an event-driven architecture using Apache Pulsar:

1. **Events:** All operations are modeled as events (create, update, delete, etc.)
2. **Producers:** Components that initiate operations emit events
3. **Consumers:** Components that process operations consume events
4. **Topics:** Organized by event type and tenant

This approach provides:
- Loose coupling between components
- Scalability through independent scaling of producers and consumers
- Resilience to component failures
- Built-in event sourcing for auditability

## Storage Architecture

The Template Registry uses a hybrid storage approach:

1. **Git:** Primary storage for template definitions with complete version history
2. **Redis:** High-performance cache for frequently accessed templates
3. **Pulsar Schema Registry:** Schema storage for validation and compatibility checking

This hybrid approach provides:
- Durability and version control through Git
- Low-latency access through Redis caching
- Schema validation and evolution through Pulsar Schema Registry

## Self-Evolution Architecture

The self-evolution capabilities are built on:

1. **Analytics Collection:** Usage metrics and patterns are collected from template instances
2. **Analysis Engine:** Analyzes usage patterns to identify improvement opportunities
3. **Suggestion Generation:** AI-driven component suggests template improvements
4. **Evolution Management:** Manages the rollout of evolved templates while maintaining compatibility

## Security Architecture

Security is integrated throughout the system:

1. **Authentication:** Secure identity verification for all components
2. **Authorization:** Fine-grained access control for templates and operations
3. **Encryption:** Data encryption in transit and at rest
4. **Audit Logging:** Comprehensive logging for all security-relevant events

## Observability Architecture

The system includes comprehensive observability:

1. **Metrics:** Key performance indicators for all components
2. **Logging:** Structured logging with correlation IDs
3. **Tracing:** Distributed tracing across components
4. **Alerts:** Proactive alerting for anomalies and issues
5. **Dashboards:** Real-time visualization of system health and performance
EOF

cat > ./template-registry/docs/architecture/evolution.md << 'EOF'
# Template Evolution Architecture

## Overview

The Template Evolution system is a core differentiator that enables templates to automatically improve based on usage patterns and feedback. This document explains the architecture of this self-evolution capability.

## Evolution Process

The template evolution process follows these steps:

1. **Data Collection:** Usage data is gathered from template instances
2. **Analysis:** The data is analyzed to identify patterns and improvement opportunities
3. **Suggestion Generation:** Improvement suggestions are generated based on the analysis
4. **Validation:** Suggestions are validated for compatibility and safety
5. **Application:** Approved suggestions are applied to create new template versions
6. **Testing:** New versions are tested with real-world usage
7. **Promotion:** Successful versions are promoted to become the new default

## Components

### 1. Analytics Collector

Collects usage data from template instances:
- Field usage patterns
- Completion rates
- Error rates
- Completion times
- User paths through the template

### 2. Evolution Analyzer

Analyzes the collected data to identify patterns:
- Field correlations
- Common values
- Error patterns
- Completion bottlenecks
- Unused fields

### 3. Suggestion Generator

Generates template improvement suggestions based on the analysis:
- Adding/removing fields
- Reordering fields
- Modifying validation rules
- Adding/removing sections
- Changing default values

### 4. Evolution Manager

Manages the evolution process:
- Evaluates suggestions for compatibility
- Creates new template versions
- Manages A/B testing of new versions
- Tracks the performance of evolved templates
- Promotes successful evolutions

## Evolution Strategies

The system employs several strategies for template evolution:

### Field-Level Evolution

- **Field Removal:** Suggests removing rarely used fields
- **Field Addition:** Suggests adding commonly needed fields
- **Field Reordering:** Suggests more intuitive field ordering
- **Default Values:** Suggests better default values based on common inputs
- **Validation Rules:** Suggests improved validation rules based on error patterns

### Section-Level Evolution

- **Section Splitting:** Suggests splitting large sections
- **Section Merging:** Suggests merging related sections
- **Section Reordering:** Suggests more logical section ordering

### Template-Level Evolution

- **Template Splitting:** Suggests splitting complex templates
- **Template Merging:** Suggests merging similar templates
- **Template Specialization:** Suggests creating specialized variants

## Evolution Safeguards

To ensure safe evolution, the system includes several safeguards:

1. **Compatibility Checking:** Ensures backward compatibility with existing instances
2. **Confidence Thresholds:** Only high-confidence suggestions are considered
3. **Impact Analysis:** Analyzes the potential impact of each change
4. **Human Oversight:** Option for human review of significant changes
5. **Gradual Rollout:** New versions are initially used for a small percentage of users
6. **Performance Monitoring:** New versions are monitored for issues
7. **Rollback Capability:** Easy rollback to previous versions if needed

## AI-Driven Evolution

The system leverages AI for advanced evolution capabilities:

1. **Pattern Recognition:** Identifying non-obvious patterns in usage data
2. **Natural Language Understanding:** Analyzing field descriptions and values
3. **Predictive Modeling:** Predicting the impact of suggested changes
4. **Semantic Grouping:** Identifying semantically related fields and sections
5. **Intelligent Defaults:** Generating context-aware default values
EOF

# Main configuration file
cat > ./template-registry/config/config.yaml << 'EOF'
system:
  name: "Template Registry"
  version: "1.0.0"
  environment: "production"
  log_level: "INFO"

pulsar:
  service_url: "pulsar://localhost:6650"
  admin_url: "http://localhost:8080"
  tenant: "template-registry"
  namespace: "templates"
  functions_worker_url: "pulsar://localhost:6650"
  subscription_name: "template-registry-service"

  # Event topics
  event_topics:
    template_create: "template-registry-create"
    template_update: "template-registry-update"
    template_get: "template-registry-get"
    template_list: "template-registry-list"
    template_delete: "template-registry-delete"
    template_validate: "template-registry-validate"
    template_version: "template-registry-version"
    template_analyze: "template-registry-analyze"
    template_usage: "template-registry-usage"
    template_evolve: "template-registry-evolve"

  # Security settings
  authentication_enabled: false
  authentication_provider: ""
  authentication_params: ""
  tls_enabled: false
  tls_cert_file: ""
  tls_key_file: ""
  tls_trust_certs_file: ""

  # Performance settings
  producer_batch_enabled: true
  producer_batch_max_delay: 10 # ms
  producer_max_pending_messages: 1000
  consumer_receive_queue_size: 1000
  negative_ack_redelivery_delay: 60000 # ms

storage:
  git:
    repository_path: "./storage/git/templates"
    remote: ""
    branch: "main"
    push_on_update: true
    author_name: "Template Registry"
    author_email: "template-registry@example.com"

  cache:
    type: "redis"
    host: "localhost"
    port: 6379
    db: 0
    password: ""
    ttl: 3600 # seconds
    prefix: "template-registry:"

  schema_registry:
    type: "pulsar"
    cache_size: 100
    validation_enabled: true

security:
  encryption_key: "${ENCRYPTION_KEY:-default-development-key}"
  token_expiration: 86400 # seconds
  cors_allowed_origins: ["*"]
  rate_limit:
    enabled: true
    requests_per_minute: 60

evolution:
  enabled: true
  min_usage_count: 10
  analysis_interval: 86400 # seconds
  max_auto_versions: 5
  suggestion_threshold: 0.8

metrics:
  enabled: true
  prometheus_enabled: true
  prometheus_port: 8081
  statsd_enabled: false
  statsd_host: "localhost"
  statsd_port: 8125
  statsd_prefix: "template_registry."
  collection_interval: 60 # seconds
EOF

# Docker configuration
cat > ./template-registry/docker-compose.yaml << 'EOF'
version: '3.8'

services:
  pulsar:
    image: apachepulsar/pulsar:2.10.0
    ports:
      - "6650:6650"
      - "8080:8080"
    environment:
      PULSAR_MEM: "-Xms512m -Xmx512m"
    volumes:
      - pulsar-data:/pulsar/data
      - pulsar-conf:/pulsar/conf
    command: >
      /bin/bash -c "bin/apply-config-from-env.py conf/standalone.conf && bin/pulsar standalone"
    healthcheck:
      test: ["CMD", "bin/pulsar-admin", "brokers", "healthcheck"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  redis:
    image: redis:6.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  template-registry:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
      - "8081:8081"
    volumes:
      - ./config:/app/config
      - ./storage:/app/storage
    environment:
      - CONFIG_PATH=/app/config/config.yaml
      - PULSAR_SERVICE_URL=pulsar://pulsar:6650
      - PULSAR_ADMIN_URL=http://pulsar:8080
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - LOG_LEVEL=INFO
    depends_on:
      pulsar:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    depends_on:
      - template-registry

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus

volumes:
  pulsar-data:
  pulsar-conf:
  redis-data:
  prometheus-data:
  grafana-data:
EOF

# Dockerfile
cat > ./template-registry/Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p ./storage/git/templates ./storage/cache ./storage/registry

# Set environment variables
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config/config.yaml

# Expose ports
EXPOSE 8000 8081

# Run the application
CMD ["python", "-m", "src.main"]
EOF

# Kubernetes deployment
cat > ./template-registry/kubernetes/deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: template-registry
  labels:
    app: template-registry
spec:
  replicas: 2
  selector:
    matchLabels:
      app: template-registry
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: template-registry
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8081"
    spec:
      containers:
      - name: template-registry
        image: template-registry:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: api
        - containerPort: 8081
          name: metrics
        env:
        - name: CONFIG_PATH
          value: "/app/config/config.yaml"
        - name: PULSAR_SERVICE_URL
          valueFrom:
            configMapKeyRef:
              name: template-registry-config
              key: pulsar.service_url
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: template-registry-config
              key: redis.host
        - name: REDIS_PORT
          valueFrom:
            configMapKeyRef:
              name: template-registry-config
              key: redis.port
        - name: ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: template-registry-secrets
              key: encryption_key
        resources:
          limits:
            cpu: "1"
            memory: "1Gi"
          requests:
            cpu: "500m"
            memory: "512Mi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 20
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: storage-volume
          mountPath: /app/storage
      volumes:
      - name: config-volume
        configMap:
          name: template-registry-config
      - name: storage-volume
        persistentVolumeClaim:
          claimName: template-registry-storage
EOF

cat > ./template-registry/kubernetes/service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: template-registry
  labels:
    app: template-registry
spec:
  selector:
    app: template-registry
  ports:
  - port: 8000
    targetPort: 8000
    name: api
  - port: 8081
    targetPort: 8081
    name: metrics
  type: ClusterIP
EOF

cat > ./template-registry/kubernetes/configmap.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: template-registry-config
data:
  config.yaml: |
    system:
      name: "Template Registry"
      version: "1.0.0"
      environment: "production"
      log_level: "INFO"

    pulsar:
      service_url: "pulsar://pulsar:6650"
      admin_url: "http://pulsar-admin:8080"
      tenant: "template-registry"
      namespace: "templates"
      subscription_name: "template-registry-service"
      event_topics:
        template_create: "template-registry-create"
        template_update: "template-registry-update"
        template_get: "template-registry-get"
        template_list: "template-registry-list"
        template_delete: "template-registry-delete"
        template_validate: "template-registry-validate"
        template_version: "template-registry-version"
        template_analyze: "template-registry-analyze"
        template_usage: "template-registry-usage"
        template_evolve: "template-registry-evolve"

    storage:
      git:
        repository_path: "./storage/git/templates"
        remote: ""
        branch: "main"
        push_on_update: true

      cache:
        type: "redis"
        host: "redis"
        port: 6379
        db: 0
        ttl: 3600

  pulsar.service_url: "pulsar://pulsar:6650"
  redis.host: "redis"
  redis.port: "6379"
EOF

cat > ./template-registry/kubernetes/secrets.yaml << 'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: template-registry-secrets
type: Opaque
data:
  encryption_key: ZGVmYXVsdC1wcm9kdWN0aW9uLWtleQ==  # Base64 encoded "default-production-key"
EOF

cat > ./template-registry/kubernetes/pvc.yaml << 'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: template-registry-storage
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

echo "Configuration files created successfully."

# Create Python Models
# Create requirements.txt
cat > ./template-registry/requirements.txt << 'EOF'
# Core dependencies
pyyaml>=6.0
python-dotenv>=1.0.0
click>=8.1.3
pydantic>=2.0.0
fastapi>=0.100.0
uvicorn>=0.22.0
httpx>=0.24.1

# Pulsar
pulsar-client>=3.1.0
aiormq>=6.7.7

# Storage
redis>=4.5.5
GitPython>=3.1.37
jsonschema>=4.17.3

# Security
cryptography>=41.0.0
pyjwt>=2.8.0
passlib>=1.7.4
bcrypt>=4.0.1

# Metrics & Monitoring
prometheus-client>=0.17.0
opentelemetry-api>=1.18.0
opentelemetry-sdk>=1.18.0
opentelemetry-exporter-prometheus>=1.18.0

# Utilities
tenacity>=8.2.2
python-dateutil>=2.8.2
structlog>=23.1.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.1
pytest-mock>=3.11.1
EOF

# Create Prometheus configuration
cat > ./template-registry/config/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'template-registry'
    static_configs:
      - targets: ['template-registry:8081']
EOF

# Create Grafana dashboards
mkdir -p ./template-registry/config/grafana/provisioning/dashboards
mkdir -p ./template-registry/config/grafana/provisioning/datasources

cat > ./template-registry/config/grafana/provisioning/datasources/datasource.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
    access: proxy
    editable: false
EOF

cat > ./template-registry/config/grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'Template Registry'
    orgId: 1
    folder: 'Template Registry'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
      foldersFromFilesStructure: true
EOF

cat > ./template-registry/config/grafana/provisioning/dashboards/template-registry-dashboard.json << 'EOF'
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": {
          "type

# Create directory for schema models
mkdir -p ./template-registry/models/schema

# Create schema model files
cat > ./template-registry/models/schema/schema_registry.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Schema Registry Models

This module defines models for the schema registry.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Union


class SchemaType(str, Enum):
    """Types of schemas in the registry"""
    JSON = "json"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class SchemaCompatibilityType(str, Enum):
    """Schema compatibility types"""
    BACKWARD = "backward"
    FORWARD = "forward"
    FULL = "full"
    NONE = "none"


@dataclass
class SchemaVersion:
    """Version information for a schema"""
    schema_id: str
    version: int
    schema_type: SchemaType
    schema_definition: str
    compatibility: SchemaCompatibilityType
    created_at: int
    created_by: str
    is_deleted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaVersion':
        """Create from dictionary"""
        return cls(
            schema_id=data["schema_id"],
            version=data["version"],
            schema_type=SchemaType(data["schema_type"]),
            schema_definition=data["schema_definition"],
            compatibility=SchemaCompatibilityType(data["compatibility"]),
            created_at=data["created_at"],
            created_by=data["created_by"],
            is_deleted=data.get("is_deleted", False),
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'SchemaVersion':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class SchemaInfo:
    """Information about a schema in the registry"""
    schema_id: str
    name: str
    description: str
    schema_type: SchemaType
    latest_version: int
    total_versions: int
    created_at: int
    updated_at: int
    created_by: str
    updated_by: str
    compatibility: SchemaCompatibilityType = SchemaCompatibilityType.BACKWARD
    is_deleted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaInfo':
        """Create from dictionary"""
        return cls(
            schema_id=data["schema_id"],
            name=data["name"],
            description=data["description"],
            schema_type=SchemaType(data["schema_type"]),
            latest_version=data["latest_version"],
            total_versions=data["total_versions"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            created_by=data["created_by"],
            updated_by=data["updated_by"],
            compatibility=SchemaCompatibilityType(data.get("compatibility", "backward")),
            is_deleted=data.get("is_deleted", False),
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'SchemaInfo':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class SchemaValidationResult:
    """Result of schema validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaValidationResult':
        """Create from dictionary"""
        return cls(
            is_valid=data["is_valid"],
            errors=data.get("errors", []),
            warnings=data.get("warnings", [])
        )


@dataclass
class SchemaCompatibilityResult:
    """Result of schema compatibility check"""
    is_compatible: bool
    schema_id: str
    version1: int
    version2: int
    compatibility_type: SchemaCompatibilityType
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaCompatibilityResult':
        """Create from dictionary"""
        return cls(
            is_compatible=data["is_compatible"],
            schema_id=data["schema_id"],
            version1=data["version1"],
            version2=data["version2"],
            compatibility_type=SchemaCompatibilityType(data["compatibility_type"]),
            errors=data.get("errors", [])
        )
EOF

# Create analytics models
cat > ./template-registry/models/analytics/usage.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Usage Analytics Models

This module defines models for template usage analytics.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Union


@dataclass
class FieldUsageStats:
    """Usage statistics for a field"""
    field_path: str  # Format: "section_name.field_name"
    section_name: str
    field_name: str
    completion_rate: float = 0.0  # Percentage of instances where field is filled
    error_rate: float = 0.0       # Percentage of instances where field has validation errors
    avg_fill_time: float = 0.0    # Average time to fill the field in seconds
    common_values: List[Any] = field(default_factory=list)  # Most common values
    common_errors: List[str] = field(default_factory=list)  # Most common validation errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldUsageStats':
        """Create from dictionary"""
        return cls(
            field_path=data["field_path"],
            section_name=data["section_name"],
            field_name=data["field_name"],
            completion_rate=data.get("completion_rate", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_fill_time=data.get("avg_fill_time", 0.0),
            common_values=data.get("common_values", []),
            common_errors=data.get("common_errors", [])
        )


@dataclass
class SectionUsageStats:
    """Usage statistics for a section"""
    section_name: str
    completion_rate: float = 0.0  # Percentage of instances where section is filled
    error_rate: float = 0.0       # Percentage of instances where section has validation errors
    avg_fill_time: float = 0.0    # Average time to fill the section in seconds
    field_stats: Dict[str, FieldUsageStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result["field_stats"] = {k: v.to_dict() for k, v in self.field_stats.items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SectionUsageStats':
        """Create from dictionary"""
        field_stats = {}
        for k, v in data.get("field_stats", {}).items():
            field_stats[k] = FieldUsageStats.from_dict(v)

        return cls(
            section_name=data["section_name"],
            completion_rate=data.get("completion_rate", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_fill_time=data.get("avg_fill_time", 0.0),
            field_stats=field_stats
        )


@dataclass
class CompletionPathStats:
    """Statistics about the path users take to complete a template"""
    total_instances: int = 0
    avg_completion_time: float = 0.0
    section_order: List[str] = field(default_factory=list)  # Most common section fill order
    field_order: List[str] = field(default_factory=list)    # Most common field fill order
    common_start_sections: List[str] = field(default_factory=list)
    common_end_sections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompletionPathStats':
        """Create from dictionary"""
        return cls(
            total_instances=data.get("total_instances", 0),
            avg_completion_time=data.get("avg_completion_time", 0.0),
            section_order=data.get("section_order", []),
            field_order=data.get("field_order", []),
            common_start_sections=data.get("common_start_sections", []),
            common_end_sections=data.get("common_end_sections", [])
        )


@dataclass
class TemplateUsageAnalytics:
    """Comprehensive usage analytics for a template"""
    template_id: str
    template_version: str
    analysis_timestamp: int = field(default_factory=lambda: int(time.time()))
    total_instances: int = 0
    completed_instances: int = 0
    completion_rate: float = 0.0
    avg_completion_time: float = 0.0
    validation_success_rate: float = 0.0
    section_stats: Dict[str, SectionUsageStats] = field(default_factory=dict)
    completion_path: CompletionPathStats = field(default_factory=CompletionPathStats)
    user_segments: Dict[str, Any] = field(default_factory=dict)
    generation_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "template_id": self.template_id,
            "template_version": self.template_version,
            "analysis_timestamp": self.analysis_timestamp,
            "total_instances": self.total_instances,
            "completed_instances": self.completed_instances,
            "completion_rate": self.completion_rate,
            "avg_completion_time": self.avg_completion_time,
            "validation_success_rate": self.validation_success_rate,
            "section_stats": {k: v.to_dict() for k, v in self.section_stats.items()},
            "completion_path": self.completion_path.to_dict(),
            "user_segments": self.user_segments,
            "generation_stats": self.generation_stats
        }
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateUsageAnalytics':
        """Create from dictionary"""
        section_stats = {}
        for k, v in data.get("section_stats", {}).items():
            section_stats[k] = SectionUsageStats.from_dict(v)

        completion_path = CompletionPathStats.from_dict(
            data.get("completion_path", {})
        )

        return cls(
            template_id=data["template_id"],
            template_version=data["template_version"],
            analysis_timestamp=data.get("analysis_timestamp", int(time.time())),
            total_instances=data.get("total_instances", 0),
            completed_instances=data.get("completed_instances", 0),
            completion_rate=data.get("completion_rate", 0.0),
            avg_completion_time=data.get("avg_completion_time", 0.0),
            validation_success_rate=data.get("validation_success_rate", 0.0),
            section_stats=section_stats,
            completion_path=completion_path,
            user_segments=data.get("user_segments", {}),
            generation_stats=data.get("generation_stats", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateUsageAnalytics':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
EOF

cat > ./template-registry/models/analytics/evolution.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Evolution Analytics Models

This module defines models for template evolution analytics and suggestions.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Union


class EvolutionSuggestionType(str, Enum):
    """Types of template evolution suggestions"""
    ADD_FIELD = "add_field"
    REMOVE_FIELD = "remove_field"
    MODIFY_FIELD = "modify_field"
    REORDER_FIELDS = "reorder_fields"
    ADD_SECTION = "add_section"
    REMOVE_SECTION = "remove_section"
    MODIFY_SECTION = "modify_section"
    REORDER_SECTIONS = "reorder_sections"
    ADD_VALIDATION = "add_validation"
    REMOVE_VALIDATION = "remove_validation"
    MODIFY_VALIDATION = "modify_validation"
    SPLIT_TEMPLATE = "split_template"
    MERGE_TEMPLATES = "merge_templates"


@dataclass
class EvolutionSuggestion:
    """A suggestion for evolving a template"""
    suggestion_id: str
    suggestion_type: EvolutionSuggestionType
    template_id: str
    template_version: str
    description: str
    confidence: float
    impact_score: float
    created_at: int = field(default_factory=lambda: int(time.time()))
    changes: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    applied: bool = False
    applied_at: Optional[int] = None
    applied_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionSuggestion':
        """Create from dictionary"""
        return cls(
            suggestion_id=data["suggestion_id"],
            suggestion_type=EvolutionSuggestionType(data["suggestion_type"]),
            template_id=data["template_id"],
            template_version=data["template_version"],
            description=data["description"],
            confidence=data["confidence"],
            impact_score=data["impact_score"],
            created_at=data.get("created_at", int(time.time())),
            changes=data.get("changes", {}),
            rationale=data.get("rationale", ""),
            applied=data.get("applied", False),
            applied_at=data.get("applied_at"),
            applied_version=data.get("applied_version")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'EvolutionSuggestion':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class FieldEvolutionData:
    """Evolution data for a field"""
    field_path: str
    field_name: str
    section_name: str
    usage_count: int = 0
    completion_rate: float = 0.0
    error_rate: float = 0.0
    avg_fill_time: float = 0.0
    common_values: List[Any] = field(default_factory=list)
    value_patterns: List[Dict[str, Any]] = field(default_factory=list)
    correlations: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldEvolutionData':
        """Create from dictionary"""
        return cls(
            field_path=data["field_path"],
            field_name=data["field_name"],
            section_name=data["section_name"],
            usage_count=data.get("usage_count", 0),
            completion_rate=data.get("completion_rate", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_fill_time=data.get("avg_fill_time", 0.0),
            common_values=data.get("common_values", []),
            value_patterns=data.get("value_patterns", []),
            correlations=data.get("correlations", {})
        )


@dataclass
class SectionEvolutionData:
    """Evolution data for a section"""
    section_name: str
    usage_count: int = 0
    completion_rate: float = 0.0
    error_rate: float = 0.0
    avg_fill_time: float = 0.0
    field_evolution: Dict[str, FieldEvolutionData] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "section_name": self.section_name,
            "usage_count": self.usage_count,
            "completion_rate": self.completion_rate,
            "error_rate": self.error_rate,
            "avg_fill_time": self.avg_fill_time,
            "field_evolution": {k: v.to_dict() for k, v in self.field_evolution.items()}
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SectionEvolutionData':
        """Create from dictionary"""
        field_evolution = {}
        for k, v in data.get("field_evolution", {}).items():
            field_evolution[k] = FieldEvolutionData.from_dict(v)

        return cls(
            section_name=data["section_name"],
            usage_count=data.get("usage_count", 0),
            completion_rate=data.get("completion_rate", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_fill_time=data.get("avg_fill_time", 0.0),
            field_evolution=field_evolution
        )


@dataclass
class TemplateEvolutionAnalysis:
    """Comprehensive evolution analysis for a template"""
    template_id: str
    template_version: str
    analysis_timestamp: int = field(default_factory=lambda: int(time.time()))
    total_instances: int = 0
    evolution_score: float = 0.0  # Overall score indicating evolution need (0-1)
    section_evolution: Dict[str, SectionEvolutionData] = field(default_factory=dict)
    field_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    suggestions: List[EvolutionSuggestion] = field(default_factory=list)
    common_patterns: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "template_id": self.template_id,
            "template_version": self.template_version,
            "analysis_timestamp": self.analysis_timestamp,
            "total_instances": self.total_instances,
            "evolution_score": self.evolution_score,
            "section_evolution": {k: v.to_dict() for k, v in self.section_evolution.items()},
            "field_correlations": self.field_correlations,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "common_patterns": self.common_patterns
        }
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateEvolutionAnalysis':
        """Create from dictionary"""
        section_evolution = {}
        for k, v in data.get("section_evolution", {}).items():
            section_evolution[k] = SectionEvolutionData.from_dict(v)

        suggestions = []
        for s in data.get("suggestions", []):
            suggestions.append(EvolutionSuggestion.from_dict(s))

        return cls(
            template_id=data["template_id"],
            template_version=data["template_version"],
            analysis_timestamp=data.get("analysis_timestamp", int(time.time())),
            total_instances=data.get("total_instances", 0),
            evolution_score=data.get("evolution_score", 0.0),
            section_evolution=section_evolution,
            field_correlations=data.get("field_correlations", {}),
            suggestions=suggestions,
            common_patterns=data.get("common_patterns", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateEvolutionAnalysis':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
EOF

# Template models
cat > ./template-registry/models/templates/base.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Base Models

This module defines the core data models for the template registry system.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Union


class TemplateCategory(str, Enum):
    """Categories for templates"""
    API = "api"
    DATABASE = "database"
    UI = "ui"
    WORKFLOW = "workflow"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class TemplateLanguage(str, Enum):
    """Programming languages supported by templates"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    ANY = "any"


class TemplateStatus(str, Enum):
    """Status of a template"""
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class FieldType(str, Enum):
    """Types of fields in a template"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    ARRAY = "array"
    OBJECT = "object"
    CODE = "code"
    DATE = "date"
    DATETIME = "datetime"
    REFERENCE = "reference"


class ValidationRuleType(str, Enum):
    """Types of validation rules"""
    REGEX = "regex"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    REQUIRED = "required"
    ENUM_VALUES = "enum_values"
    FORMAT = "format"
    DEPENDENCY = "dependency"
    CUSTOM = "custom"


class CompatibilityType(str, Enum):
    """Schema compatibility types"""
    BACKWARD = "backward"
    FORWARD = "forward"
    FULL = "full"
    NONE = "none"


@dataclass
class ValidationRule:
    """Definition of a validation rule for a field"""
    rule_type: ValidationRuleType
    expression: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class FieldDefinition:
    """Definition of a field in a template"""
    name: str
    type: FieldType
    description: str
    required: bool = False
    default_value: Any = None
    validation_rules: List[ValidationRule] = field(default_factory=list)
    options: List[str] = field(default_factory=list)
    nested_fields: List['FieldDefinition'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        if self.nested_fields:
            result['nested_fields'] = [field.to_dict() for field in self.nested_fields]
        if self.validation_rules:
            result['validation_rules'] = [rule.to_dict() for rule in self.validation_rules]
        return result

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format"""
        schema = {
            "type": self._map_type_to_json_schema(),
            "description": self.description
        }

        if self.default_value is not None:
            schema["default"] = self.default_value

        if self.type == FieldType.ENUM and self.options:
            schema["enum"] = self.options

        if self.type == FieldType.ARRAY and self.nested_fields:
            schema["items"] = self.nested_fields[0].to_json_schema()

        if self.type == FieldType.OBJECT and self.nested_fields:
            schema["properties"] = {
                field.name: field.to_json_schema() for field in self.nested_fields
            }
            if any(field.required for field in self.nested_fields):
                schema["required"] = [
                    field.name for field in self.nested_fields if field.required
                ]

        # Add additional validation constraints
        for rule in self.validation_rules:
            if rule.rule_type == ValidationRuleType.MIN_LENGTH:
                schema["minLength"] = int(rule.expression)
            elif rule.rule_type == ValidationRuleType.MAX_LENGTH:
                schema["maxLength"] = int(rule.expression)
            elif rule.rule_type == ValidationRuleType.MIN_VALUE:
                schema["minimum"] = float(rule.expression)
            elif rule.rule_type == ValidationRuleType.MAX_VALUE:
                schema["maximum"] = float(rule.expression)
            elif rule.rule_type == ValidationRuleType.REGEX:
                schema["pattern"] = rule.expression
            elif rule.rule_type == ValidationRuleType.FORMAT:
                schema["format"] = rule.expression

        return schema

    def _map_type_to_json_schema(self) -> str:
        """Map field type to JSON schema type"""
        mapping = {
            FieldType.STRING: "string",
            FieldType.INTEGER: "integer",
            FieldType.FLOAT: "number",
            FieldType.BOOLEAN: "boolean",
            FieldType.ENUM: "string",
            FieldType.ARRAY: "array",
            FieldType.OBJECT: "object",
            FieldType.CODE: "string",
            FieldType.DATE: "string",
            FieldType.DATETIME: "string",
            FieldType.REFERENCE: "string"
        }
        return mapping.get(self.type, "string")


@dataclass
class SectionDefinition:
    """Definition of a section in a template"""
    name: str
    description: str
    fields: List[FieldDefinition] = field(default_factory=list)
    order: int = 0
    conditional_display: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['fields'] = [field.to_dict() for field in self.fields]
        return result


@dataclass
class UsageMetrics:
    """Usage metrics for a template"""
    usage_count: int = 0
    completion_rate: float = 0.0
    avg_completion_time: float = 0.0
    last_used: int = 0
    error_count: int = 0
    popular_fields: List[str] = field(default_factory=list)
    common_values: Dict[str, List[Any]] = field(default_factory=dict)
    success_rate: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class TemplateMetadata:
    """Metadata for a template"""
    created_at: int = field(default_factory=lambda: int(time.time()))
    created_by: str = "system"
    updated_at: int = field(default_factory=lambda: int(time.time()))
    updated_by: str = "system"
    tags: List[str] = field(default_factory=list)
    domain: str = "general"
    complexity: str = "medium"
    estimated_completion_time: int = 0  # minutes
    version_history: List[str] = field(default_factory=list)
    usage: UsageMetrics = field(default_factory=UsageMetrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['usage'] = self.usage.to_dict()
        return result


@dataclass
class TemplateDefinition:
    """Definition of a template"""
    id: str
    name: str
    description: str
    version: str
    category: TemplateCategory
    status: TemplateStatus = TemplateStatus.DRAFT
    languages: List[TemplateLanguage] = field(default_factory=lambda: [TemplateLanguage.ANY])
    sections: List[SectionDefinition] = field(default_factory=list)
    metadata: TemplateMetadata = field(default_factory=TemplateMetadata)
    compatibility: CompatibilityType = CompatibilityType.BACKWARD
    is_system: bool = False
    parent_id: Optional[str] = None
    evolution_source: Optional[str] = None

    def __post_init__(self):
        """Validate after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['sections'] = [section.to_dict() for section in self.sections]
        result['metadata'] = self.metadata.to_dict()
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema"""
        properties = {}
        required = []

        for section in self.sections:
            section_properties = {}
            section_required = []

            for field in section.fields:
                section_properties[field.name] = field.to_json_schema()
                if field.required:
                    section_required.append(field.name)

            properties[section.name] = {
                "type": "object",
                "description": section.description,
                "properties": section_properties
            }

            if section_required:
                properties[section.name]["required"] = section_required

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": self.name,
            "description": self.description,
            "type": "object",
            "properties": properties,
            "required": [section.name for section in self.sections],
            "additionalProperties": False,
            "$id": f"template:{self.id}:{self.version}"
        }

        return schema

    def increment_usage(self) -> None:
        """Increment usage count"""
        self.metadata.usage.usage_count += 1
        self.metadata.usage.last_used = int(time.time())

    def record_completion(self, success: bool, time_taken: float) -> None:
        """Record completion metrics"""
        usage = self.metadata.usage
        total_completions = usage.completion_rate * usage.usage_count
        total_successes = total_completions * usage.success_rate

        # Add new completion
        total_completions += 1
        if success:
            total_successes += 1

        # Update metrics
        usage.success_rate = total_successes / total_completions if total_completions > 0 else 1.0
        usage.completion_rate = total_completions / usage.usage_count if usage.usage_count > 0 else 0.0

        # Update average completion time
        if success:
            old_avg = usage.avg_completion_time
            usage.avg_completion_time = ((old_avg * (total_successes - 1)) + time_taken) / total_successes
        else:
            usage.error_count += 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateDefinition':
        """Create from dictionary"""
        # Process sections and fields
        sections = []
        for section_data in data.get("sections", []):
            fields = []
            for field_data in section_data.get("fields", []):
                # Process validation rules
                validation_rules = []
                for rule_data in field_data.get("validation_rules", []):
                    validation_rules.append(ValidationRule(
                        rule_type=ValidationRuleType(rule_data["rule_type"]),
                        expression=rule_data["expression"],
                        error_message=rule_data.get("error_message")
                    ))

                # Process nested fields
                nested_fields = []
                for nested_field_data in field_data.get("nested_fields", []):
                    nested_fields.append(FieldDefinition(
                        name=nested_field_data["name"],
                        type=FieldType(nested_field_data["type"]),
                        description=nested_field_data["description"],
                        required=nested_field_data.get("required", False),
                        default_value=nested_field_data.get("default_value"),
                        validation_rules=[],
                        options=nested_field_data.get("options", []),
                        nested_fields=[],
                        metadata=nested_field_data.get("metadata", {})
                    ))

                # Create field
                fields.append(FieldDefinition(
                    name=field_data["name"],
                    type=FieldType(field_data["type"]),
                    description=field_data["description"],
                    required=field_data.get("required", False),
                    default_value=field_data.get("default_value"),
                    validation_rules=validation_rules,
                    options=field_data.get("options", []),
                    nested_fields=nested_fields,
                    metadata=field_data.get("metadata", {})
                ))

            # Create section
            sections.append(SectionDefinition(
                name=section_data["name"],
                description=section_data["description"],
                fields=fields,
                order=section_data.get("order", 0),
                conditional_display=section_data.get("conditional_display"),
                metadata=section_data.get("metadata", {})
            ))

        # Process metadata
        metadata_data = data.get("metadata", {})
        usage_data = metadata_data.get("usage", {})
        usage = UsageMetrics(
            usage_count=usage_data.get("usage_count", 0),
            completion_rate=usage_data.get("completion_rate", 0.0),
            avg_completion_time=usage_data.get("avg_completion_time", 0.0),
            last_used=usage_data.get("last_used", 0),
            error_count=usage_data.get("error_count", 0),
            popular_fields=usage_data.get("popular_fields", []),
            common_values=usage_data.get("common_values", {}),
            success_rate=usage_data.get("success_rate", 1.0)
        )

        metadata = TemplateMetadata(
            created_at=metadata_data.get("created_at", int(time.time())),
            created_by=metadata_data.get("created_by", "system"),
            updated_at=metadata_data.get("updated_at", int(time.time())),
            updated_by=metadata_data.get("updated_by", "system"),
            tags=metadata_data.get("tags", []),
            domain=metadata_data.get("domain", "general"),
            complexity=metadata_data.get("complexity", "medium"),
            estimated_completion_time=metadata_data.get("estimated_completion_time", 0),
            version_history=metadata_data.get("version_history", []),
            usage=usage
        )

        # Create template
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data["version"],
            category=TemplateCategory(data["category"]),
            status=TemplateStatus(data.get("status", TemplateStatus.DRAFT)),
            languages=[TemplateLanguage(lang) for lang in data.get("languages", ["any"])],
            sections=sections,
            metadata=metadata,
            compatibility=CompatibilityType(data.get("compatibility", "backward")),
            is_system=data.get("is_system", False),
            parent_id=data.get("parent_id"),
            evolution_source=data.get("evolution_source")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateDefinition':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TemplateVersion:
    """Information about a template version"""
    template_id: str
    version: str
    commit_id: str
    timestamp: int
    author: str
    message: str
    schema_id: Optional[str] = None
    changes: List[str] = field(default_factory=list)
    compatibility_type: CompatibilityType = CompatibilityType.BACKWARD
    is_breaking_change: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TemplateSearchQuery:
    """Query for searching templates"""
    keyword: Optional[str] = None
    categories: List[TemplateCategory] = field(default_factory=list)
    languages: List[TemplateLanguage] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    complexity: Optional[str] = None
    status: Optional[TemplateStatus] = None
    sort_by: str = "updated_at"
    sort_order: str = "desc"
    limit: int = 20
    offset: int = 0
    include_system: bool = False
    include_metadata: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
EOF

cat > ./template-registry/models/templates/instance.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Instance Models

This module defines the models for template instances.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Union


@dataclass
class FieldValue:
    """Value of a field in a template instance"""
    name: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SectionValues:
    """Values for a section in a template instance"""
    name: str
    fields: List[FieldValue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['fields'] = [field.to_dict() for field in self.fields]
        return result


@dataclass
class ValidationError:
    """Validation error for a template instance field"""
    section: str
    field: str
    message: str
    rule_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class TemplateInstance:
    """Instance of a filled template"""
    id: str
    template_id: str
    template_version: str
    name: str
    sections: List[SectionValues] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    created_by: str = "system"
    updated_by: str = "system"
    project_id: Optional[str] = None
    completed: bool = False
    validated: bool = False
    validation_errors: List[ValidationError] = field(default_factory=list)
    generation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['sections'] = [section.to_dict() for section in self.sections]
        result['validation_errors'] = [error.to_dict() for error in self.validation_errors]
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def get_field_value(self, section_name: str, field_name: str) -> Any:
        """Get the value of a field"""
        for section in self.sections:
            if section.name == section_name:
                for field in section.fields:
                    if field.name == field_name:
                        return field.value
        return None

    def set_field_value(self, section_name: str, field_name: str, value: Any) -> bool:
        """Set the value of a field"""
        for section in self.sections:
            if section.name == section_name:
                for field in section.fields:
                    if field.name == field_name:
                        field.value = value
                        self.updated_at = int(time.time())
                        return True

                # Field not found, create it
                section.fields.append(FieldValue(name=field_name, value=value))
                self.updated_at = int(time.time())
                return True

        # Section not found, create it with the field
        self.sections.append(SectionValues(
            name=section_name,
            fields=[FieldValue(name=field_name, value=value)]
        ))
        self.updated_at = int(time.time())
        return True

    def add_validation_error(self, section: str, field: str, message: str, rule_type: Optional[str] = None) -> None:
        """Add a validation error"""
        self.validation_errors.append(ValidationError(
            section=section,
            field=field,
            message=message,
            rule_type=rule_type
        ))
        self.validated = len(self.validation_errors) == 0

    def clear_validation_errors(self) -> None:
        """Clear all validation errors"""
        self.validation_errors = []
        self.validated = True

    def mark_completed(self, completed: bool = True) -> None:
        """Mark the instance as completed"""
        self.completed = completed
        self.updated_at = int(time.time())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateInstance':
        """Create from dictionary"""
        # Process sections and fields
        sections = []
        for section_data in data.get("sections", []):
            fields = []
            for field_data in section_data.get("fields", []):
                fields.append(FieldValue(
                    name=field_data["name"],
                    value=field_data["value"],
                    metadata=field_data.get("metadata", {})
                ))

            sections.append(SectionValues(
                name=section_data["name"],
                fields=fields
            ))

        # Process validation errors
        validation_errors = []
        for error_data in data.get("validation_errors", []):
            validation_errors.append(ValidationError(
                section=error_data["section"],
                field=error_data["field"],
                message=error_data["message"],
                rule_type=error_data.get("rule_type")
            ))

        return cls(
            id=data["id"],
            template_id=data["template_id"],
            template_version=data["template_version"],
            name=data["name"],
            sections=sections,
            created_at=data.get("created_at", int(time.time())),
            updated_at=data.get("updated_at", int(time.time())),
            created_by=data.get("created_by", "system"),
            updated_by=data.get("updated_by", "system"),
            project_id=data.get("project_id"),
            completed=data.get("completed", False),
            validated=data.get("validated", False),
            validation_errors=validation_errors,
            generation_id=data.get("generation_id"),
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateInstance':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
EOF

# Events models
cat > ./template-registry/models/events/base.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Event Models

This module defines the event models for the template registry system.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union


class EventType(str, Enum):
    """Types of events in the system"""
    # Template events
    TEMPLATE_CREATED = "template.created"
    TEMPLATE_UPDATED = "template.updated"
    TEMPLATE_DELETED = "template.deleted"
    TEMPLATE_PUBLISHED = "template.published"
    TEMPLATE_DEPRECATED = "template.deprecated"
    TEMPLATE_ARCHIVED = "template.archived"

    # Template instance events
    INSTANCE_CREATED = "instance.created"
    INSTANCE_UPDATED = "instance.updated"
    INSTANCE_COMPLETED = "instance.completed"
    INSTANCE_VALIDATED = "instance.validated"
    INSTANCE_DELETED = "instance.deleted"

    # Code generation events
    CODE_GENERATION_REQUESTED = "code.generation.requested"
    CODE_GENERATION_COMPLETED = "code.generation.completed"
    CODE_GENERATION_FAILED = "code.generation.failed"

    # Template evolution events
    TEMPLATE_ANALYSIS_REQUESTED = "template.analysis.requested"
    TEMPLATE_ANALYSIS_COMPLETED = "template.analysis.completed"
    TEMPLATE_EVOLUTION_SUGGESTED = "template.evolution.suggested"
    TEMPLATE_EVOLUTION_APPLIED = "template.evolution.applied"

    # System events
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_INFO = "system.info"


class EventPriority(int, Enum):
    """Priority levels for events"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BaseEvent:
    """Base event class for all events"""
    event_id: str
    event_type: EventType
    timestamp: int
    source: str
    priority: EventPriority = EventPriority.MEDIUM
    correlation_id: Optional[str] = None

    def __post_init__(self):
        """Validate after initialization"""
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = int(time.time() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "source": self.source,
            "priority": self.priority,
            "correlation_id": self.correlation_id
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvent':
        """Create from dictionary"""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=EventType(data["event_type"]),
            timestamp=data.get("timestamp", int(time.time() * 1000)),
            source=data["source"],
            priority=EventPriority(data.get("priority", EventPriority.MEDIUM)),
            correlation_id=data.get("correlation_id")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'BaseEvent':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TemplateEvent(BaseEvent):
    """Base class for template-related events"""
    template_id: str
    template_version: Optional[str] = None
    template_name: Optional[str] = None
    template_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "template_id": self.template_id,
            "template_version": self.template_version,
            "template_name": self.template_name,
            "template_category": self.template_category
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateEvent':
        """Create from dictionary"""
        base_event = BaseEvent.from_dict(data)
        return cls(
            event_id=base_event.event_id,
            event_type=base_event.event_type,
            timestamp=base_event.timestamp,
            source=base_event.source,
            priority=base_event.priority,
            correlation_id=base_event.correlation_id,
            template_id=data["template_id"],
            template_version=data.get("template_version"),
            template_name=data.get("template_name"),
            template_category=data.get("template_category")
        )


@dataclass
class TemplateInstanceEvent(BaseEvent):
    """Base class for template instance-related events"""
    instance_id: str
    template_id: str
    template_version: str
    project_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "instance_id": self.instance_id,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "project_id": self.project_id
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateInstanceEvent':
        """Create from dictionary"""
        base_event = BaseEvent.from_dict(data)
        return cls(
            event_id=base_event.event_id,
            event_type=base_event.event_type,
            timestamp=base_event.timestamp,
            source=base_event.source,
            priority=base_event.priority,
            correlation_id=base_event.correlation_id,
            instance_id=data["instance_id"],
            template_id=data["template_id"],
            template_version=data["template_version"],
            project_id=data.get("project_id")
        )


@dataclass
class CodeGenerationEvent(BaseEvent):
    """Base class for code generation-related events"""
    generation_id: str
    instance_id: str
    template_id: str
    template_version: str
    project_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "generation_id": self.generation_id,
            "instance_id": self.instance_id,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "project_id": self.project_id
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeGenerationEvent':
        """Create from dictionary"""
        base_event = BaseEvent.from_dict(data)
        return cls(
            event_id=base_event.event_id,
            event_type=base_event.event_type,
            timestamp=base_event.timestamp,
            source=base_event.source,
            priority=base_event.priority,
            correlation_id=base_event.correlation_id,
            generation_id=data["generation_id"],
            instance_id=data["instance_id"],
            template_id=data["template_id"],
            template_version=data["template_version"],
            project_id=data.get("project_id")
        )


@dataclass
class SystemEvent(BaseEvent):
    """Base class for system events"""
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "message": self.message,
            "details": self.details
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemEvent':
        """Create from dictionary"""
        base_event = BaseEvent.from_dict(data)
        return cls(
            event_id=base_event.event_id,
            event_type=base_event.event_type,
            timestamp=base_event.timestamp,
            source=base_event.source,
            priority=base_event.priority,
            correlation_id=base_event.correlation_id,
            message=data["message"],
            details=data.get("details")
        )
EOF

cat > ./template-registry/models/events/template_events.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Template Event Models

This module defines specific template event models.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union

from models.events.base import (
    EventType, EventPriority, BaseEvent, TemplateEvent
)


@dataclass
class TemplateCreatedEvent(TemplateEvent):
    """Event emitted when a template is created"""
    author: str = "system"
    is_system: bool = False

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.TEMPLATE_CREATED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "author": self.author,
            "is_system": self.is_system
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateCreatedEvent':
        """Create from dictionary"""
        template_event = TemplateEvent.from_dict(data)
        return cls(
            event_id=template_event.event_id,
            event_type=EventType.TEMPLATE_CREATED,  # Force correct type
            timestamp=template_event.timestamp,
            source=template_event.source,
            priority=template_event.priority,
            correlation_id=template_event.correlation_id,
            template_id=template_event.template_id,
            template_version=template_event.template_version,
            template_name=template_event.template_name,
            template_category=template_event.template_category,
            author=data.get("author", "system"),
            is_system=data.get("is_system", False)
        )


@dataclass
class TemplateUpdatedEvent(TemplateEvent):
    """Event emitted when a template is updated"""
    author: str = "system"
    previous_version: Optional[str] = None
    changes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.TEMPLATE_UPDATED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "author": self.author,
            "previous_version": self.previous_version,
            "changes": self.changes
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateUpdatedEvent':
        """Create from dictionary"""
        template_event = TemplateEvent.from_dict(data)
        return cls(
            event_id=template_event.event_id,
            event_type=EventType.TEMPLATE_UPDATED,  # Force correct type
            timestamp=template_event.timestamp,
            source=template_event.source,
            priority=template_event.priority,
            correlation_id=template_event.correlation_id,
            template_id=template_event.template_id,
            template_version=template_event.template_version,
            template_name=template_event.template_name,
            template_category=template_event.template_category,
            author=data.get("author", "system"),
            previous_version=data.get("previous_version"),
            changes=data.get("changes", [])
        )


@dataclass
class TemplateDeletedEvent(TemplateEvent):
    """Event emitted when a template is deleted"""
    deleted_by: str = "system"

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.TEMPLATE_DELETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "deleted_by": self.deleted_by
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateDeletedEvent':
        """Create from dictionary"""
        template_event = TemplateEvent.from_dict(data)
        return cls(
            event_id=template_event.event_id,
            event_type=EventType.TEMPLATE_DELETED,  # Force correct type
            timestamp=template_event.timestamp,
            source=template_event.source,
            priority=template_event.priority,
            correlation_id=template_event.correlation_id,
            template_id=template_event.template_id,
            template_version=template_event.template_version,
            template_name=template_event.template_name,
            template_category=template_event.template_category,
            deleted_by=data.get("deleted_by", "system")
        )


@dataclass
class TemplateStatusChangedEvent(TemplateEvent):
    """Event emitted when a template's status changes"""
    previous_status: str
    new_status: str
    changed_by: str = "system"
    reason: Optional[str] = None

    def __post_init__(self):
        """Initialize with correct event type based on status"""
        super().__post_init__()
        if self.new_status == "published":
            self.event_type = EventType.TEMPLATE_PUBLISHED
        elif self.new_status == "deprecated":
            self.event_type = EventType.TEMPLATE_DEPRECATED
        elif self.new_status == "archived":
            self.event_type = EventType.TEMPLATE_ARCHIVED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "changed_by": self.changed_by,
            "reason": self.reason
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateStatusChangedEvent':
        """Create from dictionary"""
        template_event = TemplateEvent.from_dict(data)
        return cls(
            event_id=template_event.event_id,
            event_type=template_event.event_type,
            timestamp=template_event.timestamp,
            source=template_event.source,
            priority=template_event.priority,
            correlation_id=template_event.correlation_id,
            template_id=template_event.template_id,
            template_version=template_event.template_version,
            template_name=template_event.template_name,
            template_category=template_event.template_category,
            previous_status=data["previous_status"],
            new_status=data["new_status"],
            changed_by=data.get("changed_by", "system"),
            reason=data.get("reason")
        )


@dataclass
class TemplateAnalysisRequestedEvent(TemplateEvent):
    """Event emitted when template analysis is requested"""
    min_usage_count: int = 10
    requested_by: str = "system"

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.TEMPLATE_ANALYSIS_REQUESTED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "min_usage_count": self.min_usage_count,
            "requested_by": self.requested_by
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateAnalysisRequestedEvent':
        """Create from dictionary"""
        template_event = TemplateEvent.from_dict(data)
        return cls(
            event_id=template_event.event_id,
            event_type=EventType.TEMPLATE_ANALYSIS_REQUESTED,  # Force correct type
            timestamp=template_event.timestamp,
            source=template_event.source,
            priority=template_event.priority,
            correlation_id=template_event.correlation_id,
            template_id=template_event.template_id,
            template_version=template_event.template_version,
            template_name=template_event.template_name,
            template_category=template_event.template_category,
            min_usage_count=data.get("min_usage_count", 10),
            requested_by=data.get("requested_by", "system")
        )


@dataclass
class TemplateEvolutionSuggestedEvent(TemplateEvent):
    """Event emitted when template evolution is suggested"""
    suggested_changes: Dict[str, Any]
    confidence: float
    analysis_data: Dict[str, Any]

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.TEMPLATE_EVOLUTION_SUGGESTED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "suggested_changes": self.suggested_changes,
            "confidence": self.confidence,
            "analysis_data": self.analysis_data
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateEvolutionSuggestedEvent':
        """Create from dictionary"""
        template_event = TemplateEvent.from_dict(data)
        return cls(
            event_id=template_event.event_id,
            event_type=EventType.TEMPLATE_EVOLUTION_SUGGESTED,  # Force correct type
            timestamp=template_event.timestamp,
            source=template_event.source,
            priority=template_event.priority,
            correlation_id=template_event.correlation_id,
            template_id=template_event.template_id,
            template_version=template_event.template_version,
            template_name=template_event.template_name,
            template_category=template_event.template_category,
            suggested_changes=data["suggested_changes"],
            confidence=data["confidence"],
            analysis_data=data["analysis_data"]
        )
EOF

cat > ./template-registry/models/events/code_events.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Code Generation Event Models

This module defines specific code generation event models.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union

from models.events.base import (
    EventType, EventPriority, BaseEvent, CodeGenerationEvent
)


@dataclass
class CodeGenerationRequestedEvent(CodeGenerationEvent):
    """Event emitted when code generation is requested"""
    requested_by: str = "system"
    language: str = "python"
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.CODE_GENERATION_REQUESTED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "requested_by": self.requested_by,
            "language": self.language,
            "options": self.options
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeGenerationRequestedEvent':
        """Create from dictionary"""
        code_event = CodeGenerationEvent.from_dict(data)
        return cls(
            event_id=code_event.event_id,
            event_type=EventType.CODE_GENERATION_REQUESTED,  # Force correct type
            timestamp=code_event.timestamp,
            source=code_event.source,
            priority=code_event.priority,
            correlation_id=code_event.correlation_id,
            generation_id=code_event.generation_id,
            instance_id=code_event.instance_id,
            template_id=code_event.template_id,
            template_version=code_event.template_version,
            project_id=code_event.project_id,
            requested_by=data.get("requested_by", "system"),
            language=data.get("language", "python"),
            options=data.get("options", {})
        )


@dataclass
class CodeGenerationCompletedEvent(CodeGenerationEvent):
    """Event emitted when code generation is completed"""
    completed_by: str = "system"
    output_path: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.CODE_GENERATION_COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "completed_by": self.completed_by,
            "output_path": self.output_path,
            "stats": self.stats,
            "generation_time": self.generation_time
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeGenerationCompletedEvent':
        """Create from dictionary"""
        code_event = CodeGenerationEvent.from_dict(data)
        return cls(
            event_id=code_event.event_id,
            event_type=EventType.CODE_GENERATION_COMPLETED,  # Force correct type
            timestamp=code_event.timestamp,
            source=code_event.source,
            priority=code_event.priority,
            correlation_id=code_event.correlation_id,
            generation_id=code_event.generation_id,
            instance_id=code_event.instance_id,
            template_id=code_event.template_id,
            template_version=code_event.template_version,
            project_id=code_event.project_id,
            completed_by=data.get("completed_by", "system"),
            output_path=data.get("output_path"),
            stats=data.get("stats", {}),
            generation_time=data.get("generation_time", 0.0)
        )


@dataclass
class CodeGenerationFailedEvent(CodeGenerationEvent):
    """Event emitted when code generation fails"""
    error_message: str
    error_type: str
    error_details: Optional[Dict[str, Any]] = None
    stacktrace: Optional[str] = None

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.CODE_GENERATION_FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "error_message": self.error_message,
            "error_type": self.error_type,
            "error_details": self.error_details,
            "stacktrace": self.stacktrace
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeGenerationFailedEvent':
        """Create from dictionary"""
        code_event = CodeGenerationEvent.from_dict(data)
        return cls(
            event_id=code_event.event_id,
            event_type=EventType.CODE_GENERATION_FAILED,  # Force correct type
            timestamp=code_event.timestamp,
            source=code_event.source,
            priority=code_event.priority,
            correlation_id=code_event.correlation_id,
            generation_id=code_event.generation_id,
            instance_id=code_event.instance_id,
            template_id=code_event.template_id,
            template_version=code_event.template_version,
            project_id=code_event.project_id,
            error_message=data["error_message"],
            error_type=data["error_type"],
            error_details=data.get("error_details"),
            stacktrace=data.get("stacktrace")
        )
EOF

cat > ./template-registry/models/events/system_events.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - System Event Models

This module defines specific system event models.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union

from models.events.base import (
    EventType, EventPriority, BaseEvent, SystemEvent
)


@dataclass
class SystemErrorEvent(SystemEvent):
    """Event emitted when a system error occurs"""
    error_type: str
    component: str
    stacktrace: Optional[str] = None

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.SYSTEM_ERROR

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "error_type": self.error_type,
            "component": self.component,
            "stacktrace": self.stacktrace
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemErrorEvent':
        """Create from dictionary"""
        system_event = SystemEvent.from_dict(data)
        return cls(
            event_id=system_event.event_id,
            event_type=EventType.SYSTEM_ERROR,  # Force correct type
            timestamp=system_event.timestamp,
            source=system_event.source,
            priority=system_event.priority,
            correlation_id=system_event.correlation_id,
            message=system_event.message,
            details=system_event.details,
            error_type=data["error_type"],
            component=data["component"],
            stacktrace=data.get("stacktrace")
        )


@dataclass
class SystemWarningEvent(SystemEvent):
    """Event emitted when a system warning occurs"""
    warning_type: str
    component: str

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.SYSTEM_WARNING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "warning_type": self.warning_type,
            "component": self.component
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemWarningEvent':
        """Create from dictionary"""
        system_event = SystemEvent.from_dict(data)
        return cls(
            event_id=system_event.event_id,
            event_type=EventType.SYSTEM_WARNING,  # Force correct type
            timestamp=system_event.timestamp,
            source=system_event.source,
            priority=system_event.priority,
            correlation_id=system_event.correlation_id,
            message=system_event.message,
            details=system_event.details,
            warning_type=data["warning_type"],
            component=data["component"]
        )


@dataclass
class SystemInfoEvent(SystemEvent):
    """Event emitted for system information"""
    info_type: str
    component: str

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.SYSTEM_INFO

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "info_type": self.info_type,
            "component": self.component
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemInfoEvent':
        """Create from dictionary"""
        system_event = SystemEvent.from_dict(data)
        return cls(
            event_id=system_event.event_id,
            event_type=EventType.SYSTEM_INFO,  # Force correct type
            timestamp=system_event.timestamp,
            source=system_event.source,
            priority=system_event.priority,
            correlation_id=system_event.correlation_id,
            message=system_event.message,
            details=system_event.details,
            info_type=data["info_type"],
            component=data["component"]
        )
EOF

cat > ./template-registry/models/events/instance_events.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Instance Event Models

This module defines specific template instance event models.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union

from models.events.base import (
    EventType, EventPriority, BaseEvent, TemplateInstanceEvent
)


@dataclass
class InstanceCreatedEvent(TemplateInstanceEvent):
    """Event emitted when a template instance is created"""
    created_by: str = "system"

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.INSTANCE_CREATED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "created_by": self.created_by
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstanceCreatedEvent':
        """Create from dictionary"""
        instance_event = TemplateInstanceEvent.from_dict(data)
        return cls(
            event_id=instance_event.event_id,
            event_type=EventType.INSTANCE_CREATED,  # Force correct type
            timestamp=instance_event.timestamp,
            source=instance_event.source,
            priority=instance_event.priority,
            correlation_id=instance_event.correlation_id,
            instance_id=instance_event.instance_id,
            template_id=instance_event.template_id,
            template_version=instance_event.template_version,
            project_id=instance_event.project_id,
            created_by=data.get("created_by", "system")
        )


@dataclass
class InstanceUpdatedEvent(TemplateInstanceEvent):
    """Event emitted when a template instance is updated"""
    updated_by: str = "system"
    updated_sections: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.INSTANCE_UPDATED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "updated_by": self.updated_by,
            "updated_sections": self.updated_sections
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstanceUpdatedEvent':
        """Create from dictionary"""
        instance_event = TemplateInstanceEvent.from_dict(data)
        return cls(
            event_id=instance_event.event_id,
            event_type=EventType.INSTANCE_UPDATED,  # Force correct type
            timestamp=instance_event.timestamp,
            source=instance_event.source,
            priority=instance_event.priority,
            correlation_id=instance_event.correlation_id,
            instance_id=instance_event.instance_id,
            template_id=instance_event.template_id,
            template_version=instance_event.template_version,
            project_id=instance_event.project_id,
            updated_by=data.get("updated_by", "system"),
            updated_sections=data.get("updated_sections", [])
        )


@dataclass
class InstanceCompletedEvent(TemplateInstanceEvent):
    """Event emitted when a template instance is completed"""
    completed_by: str = "system"
    completion_time: float = 0.0

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.INSTANCE_COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "completed_by": self.completed_by,
            "completion_time": self.completion_time
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstanceCompletedEvent':
        """Create from dictionary"""
        instance_event = TemplateInstanceEvent.from_dict(data)
        return cls(
            event_id=instance_event.event_id,
            event_type=EventType.INSTANCE_COMPLETED,  # Force correct type
            timestamp=instance_event.timestamp,
            source=instance_event.source,
            priority=instance_event.priority,
            correlation_id=instance_event.correlation_id,
            instance_id=instance_event.instance_id,
            template_id=instance_event.template_id,
            template_version=instance_event.template_version,
            project_id=instance_event.project_id,
            completed_by=data.get("completed_by", "system"),
            completion_time=data.get("completion_time", 0.0)
        )


@dataclass
class InstanceValidatedEvent(TemplateInstanceEvent):
    """Event emitted when a template instance is validated"""
    validated_by: str = "system"
    is_valid: bool = True
    error_count: int = 0
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.INSTANCE_VALIDATED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "validated_by": self.validated_by,
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "validation_errors": self.validation_errors
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstanceValidatedEvent':
        """Create from dictionary"""
        instance_event = TemplateInstanceEvent.from_dict(data)
        return cls(
            event_id=instance_event.event_id,
            event_type=EventType.INSTANCE_VALIDATED,  # Force correct type
            timestamp=instance_event.timestamp,
            source=instance_event.source,
            priority=instance_event.priority,
            correlation_id=instance_event.correlation_id,
            instance_id=instance_event.instance_id,
            template_id=instance_event.template_id,
            template_version=instance_event.template_version,
            project_id=instance_event.project_id,
            validated_by=data.get("validated_by", "system"),
            is_valid=data.get("is_valid", True),
            error_count=data.get("error_count", 0),
            validation_errors=data.get("validation_errors", [])
        )


@dataclass
class InstanceDeletedEvent(TemplateInstanceEvent):
    """Event emitted when a template instance is deleted"""
    deleted_by: str = "system"
    reason: Optional[str] = None

    def __post_init__(self):
        """Initialize with correct event type"""
        super().__post_init__()
        self.event_type = EventType.INSTANCE_DELETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "deleted_by": self.deleted_by,
            "reason": self.reason
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstanceDeletedEvent':
        """Create from dictionary"""
        instance_event = TemplateInstanceEvent.from_dict(data)
        return cls(
            event_id=instance_event.event_id,
            event_type=EventType.INSTANCE_DELETED,  # Force correct type
            timestamp=instance_event.timestamp,
            source=instance_event.source,
            priority=instance_event.priority,
            correlation_id=instance_event.correlation_id,
            instance_id=instance_event.instance_id,
            template_id=instance_event.template_id,
            template_version=instance_event.template_version,
            project_id=instance_event.project_id,
            deleted_by=data.get("deleted_by", "system"),
            reason=data.get("reason")
        )
EOF

#!/bin/bash
#============================================================================
# Template Registry System for Spec-Driven AI Code Generation Platform
#
# This script builds a comprehensive template registry system with:
# - Apache Pulsar for event-driven architecture
# - Git-based version control for templates
# - Redis caching for high performance
# - Self-evolving templates through usage analytics
# - Comprehensive security and monitoring features
#
# Author: Claude
# Date: 2025-04-18
#============================================================================

set -e

# Setup logging
LOG_FILE="build-template-registry.log"
echo "Build started at $(date)" > $LOG_FILE

# Function to log messages
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message"
    echo "[$timestamp] $message" >> $LOG_FILE
}

# Function to check requirements
check_requirements() {
    log "Checking system requirements..."

    # Check Python version
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log "Found Python version: $PYTHON_VERSION"

        # Check if Python version is 3.10 or higher
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
            log "Python version requirement met."
        else
            log "ERROR: Python 3.10 or higher is required."
            exit 1
        fi
    else
        log "ERROR: Python 3 not found. Please install Python 3.10 or higher."
        exit 1
    fi

    # Check Git
    if command -v git &>/dev/null; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        log "Found Git version: $GIT_VERSION"
    else
        log "ERROR: Git not found. Please install Git."
        exit 1
    fi

    # Check Docker (optional)
    if command -v docker &>/dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
        log "Found Docker version: $DOCKER_VERSION"

        # Check Docker Compose
        if command -v docker-compose &>/dev/null; then
            DOCKER_COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | tr -d ',')
            log "Found Docker Compose version: $DOCKER_COMPOSE_VERSION"
        else
            log "WARNING: Docker Compose not found. It will be needed for local deployment."
        fi
    else
        log "WARNING: Docker not found. It will be needed for local deployment."
    fi

    log "System requirement check completed."
}

# Function to create directory structure
create_directory_structure() {
    log "Creating directory structure..."

    # Main directories
    mkdir -p ./template-registry/{src,models,config,storage,scripts,tests,docs}

    # Source code directories
    mkdir -p ./template-registry/src/{core,event_handlers,validators,adapters,utils,security,metrics,evolution}

    # Model directories
    mkdir -p ./template-registry/models/{templates,schema,events,analytics}

    # Storage directories
    mkdir -p ./template-registry/storage/{git,cache,registry}

    # Test directories
    mkdir -p ./template-registry/tests/{unit,integration,performance}

    # Documentation directories
    mkdir -p ./template-registry/docs/{api,architecture,deployment}

    # Kubernetes deployment
    mkdir -p ./template-registry/kubernetes

    log "Directory structure created successfully."
}

# Function to create base configuration files
create_base_config() {
    log "Creating base configuration files..."

    # Main configuration file
    cat > ./template-registry/config/config.yaml << 'EOF'
system:
  name: "Template Registry"
  version: "1.0.0"
  environment: "development"
  log_level: "INFO"

pulsar:
  service_url: "pulsar://localhost:6650"
  admin_url: "http://localhost:8080"
  tenant: "template-registry"
  namespace: "templates"
  functions_worker_url: "pulsar://localhost:6650"
  subscription_name: "template-registry-service"

  # Event topics
  event_topics:
    template_create: "template-registry-create"
    template_update: "template-registry-update"
    template_get: "template-registry-get"
    template_list: "template-registry-list"
    template_delete: "template-registry-delete"
    template_validate: "template-registry-validate"
    template_version: "template-registry-version"
    template_analyze: "template-registry-analyze"
    template_usage: "template-registry-usage"
    template_evolve: "template-registry-evolve"

  # Security settings
  authentication_enabled: false
  authentication_provider: ""
  authentication_params: ""
  tls_enabled: false
  tls_cert_file: ""
  tls_key_file: ""
  tls_trust_certs_file: ""

  # Performance settings
  producer_batch_enabled: true
  producer_batch_max_delay: 10 # ms
  producer_max_pending_messages: 1000
  consumer_receive_queue_size: 1000
  negative_ack_redelivery_delay: 60000 # ms

storage:
  git:
    repository_path: "./storage/git/templates"
    remote: ""
    branch: "main"
    push_on_update: true
    author_name: "Template Registry"
    author_email: "template-registry@example.com"

  cache:
    type: "redis"
    host: "localhost"
    port: 6379
    db: 0
    password: ""
    ttl: 3600 # seconds
    prefix: "template-registry:"
    strategy: "cache_aside" # Options: cache_aside, write_through, write_behind

  schema_registry:
    type: "pulsar"
    cache_size: 100
    validation_enabled: true

security:
  encryption_key: "${ENCRYPTION_KEY:-default-development-key}"
  token_expiration: 86400 # seconds
  cors_allowed_origins: ["*"]
  rate_limit:
    enabled: true
    requests_per_minute: 60

evolution:
  enabled: true
  min_usage_count: 10
  analysis_interval: 86400 # seconds
  max_auto_versions: 5
  suggestion_threshold: 0.8
  automate_safe_changes: true

metrics:
  enabled: true
  prometheus_enabled: true
  prometheus_port: 8081
  statsd_enabled: false
  statsd_host: "localhost"
  statsd_port: 8125
  statsd_prefix: "template_registry."
  collection_interval: 60 # seconds
EOF

    # Environment variables
    cat > ./template-registry/.env.template << 'EOF'
# Template Registry Environment Variables
# Copy this file to .env and fill in appropriate values

# System
ENVIRONMENT=development
LOG_LEVEL=INFO

# Pulsar
PULSAR_SERVICE_URL=pulsar://localhost:6650
PULSAR_ADMIN_URL=http://localhost:8080
PULSAR_TENANT=template-registry
PULSAR_NAMESPACE=templates

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Security
ENCRYPTION_KEY=generate-a-secure-random-key-in-production
JWT_SECRET=generate-a-secure-random-key-in-production

# Evolution
EVOLUTION_ENABLED=true
EVOLUTION_MIN_USAGE_COUNT=10

# Metrics
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8081
EOF

    # .gitignore
    cat > ./template-registry/.gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
.coverage
htmlcov/
.tox/
.nox/
.hypothesis/
.pytest_cache/
.env
venv/
env/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo
.DS_Store

# Project specific
storage/git/*
storage/cache/*
!storage/git/.gitkeep
!storage/cache/.gitkeep
logs/
*.log
EOF

    # Create empty files to keep directories
    touch ./template-registry/storage/git/.gitkeep
    touch ./template-registry/storage/cache/.gitkeep

    log "Base configuration files created successfully."
}

# Function to create core model modules
create_model_modules() {
    log "Creating model modules..."

    # Base template models
    cat > ./template-registry/models/templates/base.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Base Models

This module defines the core data models for the template registry system.
"""

import json
import time
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set


class TemplateCategory(str, Enum):
    """Categories for templates"""
    API = "api"
    DATABASE = "database"
    UI = "ui"
    WORKFLOW = "workflow"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class TemplateLanguage(str, Enum):
    """Programming languages supported by templates"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    ANY = "any"


class TemplateStatus(str, Enum):
    """Status of a template"""
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class FieldType(str, Enum):
    """Types of fields in a template"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    ARRAY = "array"
    OBJECT = "object"
    CODE = "code"
    DATE = "date"
    DATETIME = "datetime"
    REFERENCE = "reference"


class ValidationRuleType(str, Enum):
    """Types of validation rules"""
    REGEX = "regex"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    REQUIRED = "required"
    ENUM_VALUES = "enum_values"
    FORMAT = "format"
    DEPENDENCY = "dependency"
    CUSTOM = "custom"


class CompatibilityType(str, Enum):
    """Schema compatibility types"""
    BACKWARD = "backward"
    FORWARD = "forward"
    FULL = "full"
    NONE = "none"


@dataclass
class ValidationRule:
    """Definition of a validation rule for a field"""
    rule_type: ValidationRuleType
    expression: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class FieldDefinition:
    """Definition of a field in a template"""
    name: str
    type: FieldType
    description: str
    required: bool = False
    default_value: Any = None
    validation_rules: List[ValidationRule] = field(default_factory=list)
    options: List[str] = field(default_factory=list)
    nested_fields: List['FieldDefinition'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        if self.nested_fields:
            result['nested_fields'] = [field.to_dict() for field in self.nested_fields]
        if self.validation_rules:
            result['validation_rules'] = [rule.to_dict() for rule in self.validation_rules]
        return result

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format"""
        schema = {
            "type": self._map_type_to_json_schema(),
            "description": self.description
        }

        if self.default_value is not None:
            schema["default"] = self.default_value

        if self.type == FieldType.ENUM and self.options:
            schema["enum"] = self.options

        if self.type == FieldType.ARRAY and self.nested_fields:
            schema["items"] = self.nested_fields[0].to_json_schema()

        if self.type == FieldType.OBJECT and self.nested_fields:
            schema["properties"] = {
                field.name: field.to_json_schema() for field in self.nested_fields
            }
            if any(field.required for field in self.nested_fields):
                schema["required"] = [
                    field.name for field in self.nested_fields if field.required
                ]

        # Add additional validation constraints
        for rule in self.validation_rules:
            if rule.rule_type == ValidationRuleType.MIN_LENGTH:
                schema["minLength"] = int(rule.expression)
            elif rule.rule_type == ValidationRuleType.MAX_LENGTH:
                schema["maxLength"] = int(rule.expression)
            elif rule.rule_type == ValidationRuleType.MIN_VALUE:
                schema["minimum"] = float(rule.expression)
            elif rule.rule_type == ValidationRuleType.MAX_VALUE:
                schema["maximum"] = float(rule.expression)
            elif rule.rule_type == ValidationRuleType.REGEX:
                schema["pattern"] = rule.expression
            elif rule.rule_type == ValidationRuleType.FORMAT:
                schema["format"] = rule.expression

        return schema

    def _map_type_to_json_schema(self) -> str:
        """Map field type to JSON schema type"""
        mapping = {
            FieldType.STRING: "string",
            FieldType.INTEGER: "integer",
            FieldType.FLOAT: "number",
            FieldType.BOOLEAN: "boolean",
            FieldType.ENUM: "string",
            FieldType.ARRAY: "array",
            FieldType.OBJECT: "object",
            FieldType.CODE: "string",
            FieldType.DATE: "string",
            FieldType.DATETIME: "string",
            FieldType.REFERENCE: "string"
        }
        return mapping.get(self.type, "string")


@dataclass
class SectionDefinition:
    """Definition of a section in a template"""
    name: str
    description: str
    fields: List[FieldDefinition] = field(default_factory=list)
    order: int = 0
    conditional_display: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['fields'] = [field.to_dict() for field in self.fields]
        return result


@dataclass
class UsageMetrics:
    """Usage metrics for a template"""
    usage_count: int = 0
    completion_rate: float = 0.0
    avg_completion_time: float = 0.0
    last_used: int = 0
    error_count: int = 0
    popular_fields: List[str] = field(default_factory=list)
    common_values: Dict[str, List[Any]] = field(default_factory=dict)
    success_rate: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class TemplateMetadata:
    """Metadata for a template"""
    created_at: int = field(default_factory=lambda: int(time.time()))
    created_by: str = "system"
    updated_at: int = field(default_factory=lambda: int(time.time()))
    updated_by: str = "system"
    tags: List[str] = field(default_factory=list)
    domain: str = "general"
    complexity: str = "medium"
    estimated_completion_time: int = 0  # minutes
    version_history: List[str] = field(default_factory=list)
    usage: UsageMetrics = field(default_factory=UsageMetrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['usage'] = self.usage.to_dict()
        return result


@dataclass
class TemplateDefinition:
    """Definition of a template"""
    id: str
    name: str
    description: str
    version: str
    category: TemplateCategory
    status: TemplateStatus = TemplateStatus.DRAFT
    languages: List[TemplateLanguage] = field(default_factory=lambda: [TemplateLanguage.ANY])
    sections: List[SectionDefinition] = field(default_factory=list)
    metadata: TemplateMetadata = field(default_factory=TemplateMetadata)
    compatibility: CompatibilityType = CompatibilityType.BACKWARD
    is_system: bool = False
    parent_id: Optional[str] = None
    evolution_source: Optional[str] = None

    def __post_init__(self):
        """Validate after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['sections'] = [section.to_dict() for section in self.sections]
        result['metadata'] = self.metadata.to_dict()
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema"""
        properties = {}
        required = []

        for section in self.sections:
            section_properties = {}
            section_required = []

            for field in section.fields:
                section_properties[field.name] = field.to_json_schema()
                if field.required:
                    section_required.append(field.name)

            properties[section.name] = {
                "type": "object",
                "description": section.description,
                "properties": section_properties
            }

            if section_required:
                properties[section.name]["required"] = section_required

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": self.name,
            "description": self.description,
            "type": "object",
            "properties": properties,
            "required": [section.name for section in self.sections],
            "additionalProperties": False,
            "$id": f"template:{self.id}:{self.version}"
        }

        return schema

    def get_hash(self) -> str:
        """Get a hash of the template structure for comparison"""
        # Create a deterministic representation of the template structure
        structure = {
            "sections": [
                {
                    "name": section.name,
                    "fields": [
                        {
                            "name": field.name,
                            "type": field.type,
                            "required": field.required
                        }
                        for field in section.fields
                    ]
                }
                for section in self.sections
            ]
        }

        structure_str = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(structure_str.encode()).hexdigest()

    def increment_usage(self) -> None:
        """Increment usage count"""
        self.metadata.usage.usage_count += 1
        self.metadata.usage.last_used = int(time.time())

    def record_completion(self, success: bool, time_taken: float) -> None:
        """Record completion metrics"""
        usage = self.metadata.usage
        total_completions = usage.completion_rate * usage.usage_count
        total_successes = total_completions * usage.success_rate

        # Add new completion
        total_completions += 1
        if success:
            total_successes += 1

        # Update metrics
        usage.success_rate = total_successes / total_completions if total_completions > 0 else 1.0
        usage.completion_rate = total_completions / usage.usage_count if usage.usage_count > 0 else 0.0

        # Update average completion time
        if success:
            old_avg = usage.avg_completion_time
            usage.avg_completion_time = ((old_avg * (total_successes - 1)) + time_taken) / total_successes
        else:
            usage.error_count += 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateDefinition':
        """Create from dictionary"""
        # Process sections and fields
        sections = []
        for section_data in data.get("sections", []):
            fields = []
            for field_data in section_data.get("fields", []):
                # Process validation rules
                validation_rules = []
                for rule_data in field_data.get("validation_rules", []):
                    validation_rules.append(ValidationRule(
                        rule_type=ValidationRuleType(rule_data["rule_type"]),
                        expression=rule_data["expression"],
                        error_message=rule_data.get("error_message")
                    ))

                # Process nested fields
                nested_fields = []
                for nested_field_data in field_data.get("nested_fields", []):
                    nested_fields.append(FieldDefinition(
                        name=nested_field_data["name"],
                        type=FieldType(nested_field_data["type"]),
                        description=nested_field_data["description"],
                        required=nested_field_data.get("required", False),
                        default_value=nested_field_data.get("default_value"),
                        validation_rules=[],
                        options=nested_field_data.get("options", []),
                        nested_fields=[],
                        metadata=nested_field_data.get("metadata", {})
                    ))

                # Create field
                fields.append(FieldDefinition(
                    name=field_data["name"],
                    type=FieldType(field_data["type"]),
                    description=field_data["description"],
                    required=field_data.get("required", False),
                    default_value=field_data.get("default_value"),
                    validation_rules=validation_rules,
                    options=field_data.get("options", []),
                    nested_fields=nested_fields,
                    metadata=field_data.get("metadata", {})
                ))

            # Create section
            sections.append(SectionDefinition(
                name=section_data["name"],
                description=section_data["description"],
                fields=fields,
                order=section_data.get("order", 0),
                conditional_display=section_data.get("conditional_display"),
                metadata=section_data.get("metadata", {})
            ))

        # Process metadata
        metadata_data = data.get("metadata", {})
        usage_data = metadata_data.get("usage", {})
        usage = UsageMetrics(
            usage_count=usage_data.get("usage_count", 0),
            completion_rate=usage_data.get("completion_rate", 0.0),
            avg_completion_time=usage_data.get("avg_completion_time", 0.0),
            last_used=usage_data.get("last_used", 0),
            error_count=usage_data.get("error_count", 0),
            popular_fields=usage_data.get("popular_fields", []),
            common_values=usage_data.get("common_values", {}),
            success_rate=usage_data.get("success_rate", 1.0)
        )

        metadata = TemplateMetadata(
            created_at=metadata_data.get("created_at", int(time.time())),
            created_by=metadata_data.get("created_by", "system"),
            updated_at=metadata_data.get("updated_at", int(time.time())),
            updated_by=metadata_data.get("updated_by", "system"),
            tags=metadata_data.get("tags", []),
            domain=metadata_data.get("domain", "general"),
            complexity=metadata_data.get("complexity", "medium"),
            estimated_completion_time=metadata_data.get("estimated_completion_time", 0),
            version_history=metadata_data.get("version_history", []),
            usage=usage
        )

        # Create template
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data["version"],
            category=TemplateCategory(data["category"]),
            status=TemplateStatus(data.get("status", TemplateStatus.DRAFT)),
            languages=[TemplateLanguage(lang) for lang in data.get("languages", ["any"])],
            sections=sections,
            metadata=metadata,
            compatibility=CompatibilityType(data.get("compatibility", "backward")),
            is_system=data.get("is_system", False),
            parent_id=data.get("parent_id"),
            evolution_source=data.get("evolution_source")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateDefinition':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TemplateVersion:
    """Information about a template version"""
    template_id: str
    version: str
    commit_id: str
    timestamp: int
    author: str
    message: str
    schema_id: Optional[str] = None
    changes: List[str] = field(default_factory=list)
    compatibility_type: CompatibilityType = CompatibilityType.BACKWARD
    is_breaking_change: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TemplateSearchQuery:
    """Query for searching templates"""
    keyword: Optional[str] = None
    categories: List[TemplateCategory] = field(default_factory=list)
    languages: List[TemplateLanguage] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    complexity: Optional[str] = None
    status: Optional[TemplateStatus] = None
    sort_by: str = "updated_at"
    sort_order: str = "desc"
    limit: int = 20
    offset: int = 0
    include_system: bool = False
    include_metadata: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
EOF

    # Template instance models
    cat > ./template-registry/models/templates/instance.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Instance Models

This module defines the models for template instances.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Union


@dataclass
class FieldValue:
    """Value of a field in a template instance"""
    name: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SectionValues:
    """Values for a section in a template instance"""
    name: str
    fields: List[FieldValue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['fields'] = [field.to_dict() for field in self.fields]
        return result


@dataclass
class ValidationError:
    """Validation error for a template instance field"""
    section: str
    field: str
    message: str
    rule_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class TemplateInstance:
    """Instance of a filled template"""
    id: str
    template_id: str
    template_version: str
    name: str
    sections: List[SectionValues] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    created_by: str = "system"
    updated_by: str = "system"
    project_id: Optional[str] = None
    completed: bool = False
    validated: bool = False
    validation_errors: List[ValidationError] = field(default_factory=list)
    generation_id: Optional[str] = None
    fill_order: List[str] = field(default_factory=list)  # Track field fill order for analytics
    fill_times: Dict[str, float] = field(default_factory=dict)  # Track fill times for analytics
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['sections'] = [section.to_dict() for section in self.sections]
        result['validation_errors'] = [error.to_dict() for error in self.validation_errors]
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def get_field_value(self, section_name: str, field_name: str) -> Any:
        """Get the value of a field"""
        for section in self.sections:
            if section.name == section_name:
                for field in section.fields:
                    if field.name == field_name:
                        return field.value
        return None

    def set_field_value(self, section_name: str, field_name: str, value: Any,
                        track_order: bool = True, track_time: Optional[float] = None) -> bool:
        """
        Set the value of a field

        Args:
            section_name: Name of the section
            field_name: Name of the field
            value: Value to set
            track_order: Whether to track the field fill order
            track_time: Time taken to fill the field (in seconds)

        Returns:
            True if successful, False otherwise
        """
        # Track field fill order for analytics
        if track_order:
            field_path = f"{section_name}.{field_name}"
            if field_path not in self.fill_order:
                self.fill_order.append(field_path)

            # Track fill time if provided
            if track_time is not None:
                self.fill_times[field_path] = track_time

        # Find and update the field
        for section in self.sections:
            if section.name == section_name:
                for field in section.fields:
                    if field.name == field_name:
                        field.value = value
                        self.updated_at = int(time.time())
                        return True

                # Field not found, create it
                section.fields.append(FieldValue(name=field_name, value=value))
                self.updated_at = int(time.time())
                return True

        # Section not found, create it with the field
        self.sections.append(SectionValues(
            name=section_name,
            fields=[FieldValue(name=field_name, value=value)]
        ))
        self.updated_at = int(time.time())
        return True

    def add_validation_error(self, section: str, field: str, message: str, rule_type: Optional[str] = None) -> None:
        """Add a validation error"""
        self.validation_errors.append(ValidationError(
            section=section,
            field=field,
            message=message,
            rule_type=rule_type
        ))
        self.validated = False

    def clear_validation_errors(self) -> None:
        """Clear all validation errors"""
        self.validation_errors = []
        self.validated = True

    def mark_completed(self, completed: bool = True) -> None:
        """Mark the instance as completed"""
        self.completed = completed
        self.updated_at = int(time.time())

    def is_section_completed(self, section_name: str) -> bool:
        """Check if a section is completed"""
        for section in self.sections:
            if section.name == section_name:
                # A section is completed if it has at least one field
                return len(section.fields) > 0
        return False

    def get_completion_stats(self) -> Dict[str, Any]:
        """Get completion statistics"""
        filled_sections = sum(1 for section in self.sections if len(section.fields) > 0)
        total_fields = sum(len(section.fields) for section in self.sections)

        return {
            "filled_sections": filled_sections,
            "total_fields": total_fields,
            "is_completed": self.completed,
            "is_validated": self.validated,
            "error_count": len(self.validation_errors),
            "field_order": self.fill_order,
            "fill_times": self.fill_times
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateInstance':
        """Create from dictionary"""
        # Process sections and fields
        sections = []
        for section_data in data.get("sections", []):
            fields = []
            for field_data in section_data.get("fields", []):
                fields.append(FieldValue(
                    name=field_data["name"],
                    value=field_data["value"],
                    metadata=field_data.get("metadata", {})
                ))

            sections.append(SectionValues(
                name=section_data["name"],
                fields=fields
            ))

        # Process validation errors
        validation_errors = []
        for error_data in data.get("validation_errors", []):
            validation_errors.append(ValidationError(
                section=error_data["section"],
                field=error_data["field"],
                message=error_data["message"],
                rule_type=error_data.get("rule_type")
            ))

        return cls(
            id=data["id"],
            template_id=data["template_id"],
            template_version=data["template_version"],
            name=data["name"],
            sections=sections,
            created_at=data.get("created_at", int(time.time())),
            updated_at=data.get("updated_at", int(time.time())),
            created_by=data.get("created_by", "system"),
            updated_by=data.get("updated_by", "system"),
            project_id=data.get("project_id"),
            completed=data.get("completed", False),
            validated=data.get("validated", False),
            validation_errors=validation_errors,
            generation_id=data.get("generation_id"),
            fill_order=data.get("fill_order", []),
            fill_times=data.get("fill_times", {}),
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateInstance':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
EOF

    # Event models
    cat > ./template-registry/models/events/base.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Event Models

This module defines the event models for the template registry system.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union


class EventType(str, Enum):
    """Types of events in the system"""
    # Template events
    TEMPLATE_CREATED = "template.created"
    TEMPLATE_UPDATED = "template.updated"
    TEMPLATE_DELETED = "template.deleted"
    TEMPLATE_PUBLISHED = "template.published"
    TEMPLATE_DEPRECATED = "template.deprecated"
    TEMPLATE_ARCHIVED = "template.archived"

    # Template instance events
    INSTANCE_CREATED = "instance.created"
    INSTANCE_UPDATED = "instance.updated"
    INSTANCE_COMPLETED = "instance.completed"
    INSTANCE_VALIDATED = "instance.validated"
    INSTANCE_DELETED = "instance.deleted"

    # Code generation events
    CODE_GENERATION_REQUESTED = "code.generation.requested"
    CODE_GENERATION_COMPLETED = "code.generation.completed"
    CODE_GENERATION_FAILED = "code.generation.failed"

    # Template evolution events
    TEMPLATE_ANALYSIS_REQUESTED = "template.analysis.requested"
    TEMPLATE_ANALYSIS_COMPLETED = "template.analysis.completed"
    TEMPLATE_EVOLUTION_SUGGESTED = "template.evolution.suggested"
    TEMPLATE_EVOLUTION_APPLIED = "template.evolution.applied"

    # System events
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_INFO = "system.info"


class EventPriority(int, Enum):
    """Priority levels for events"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BaseEvent:
    """Base event class for all events"""
    event_id: str
    event_type: EventType
    timestamp: int
    source: str
    priority: EventPriority = EventPriority.MEDIUM
    correlation_id: Optional[str] = None

    def __post_init__(self):
        """Validate after initialization"""
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = int(time.time() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "source": self.source,
            "priority": self.priority,
            "correlation_id": self.correlation_id
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvent':
        """Create from dictionary"""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=EventType(data["event_type"]),
            timestamp=data.get("timestamp", int(time.time() * 1000)),
            source=data["source"],
            priority=EventPriority(data.get("priority", EventPriority.MEDIUM)),
            correlation_id=data.get("correlation_id")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'BaseEvent':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TemplateEvent(BaseEvent):
    """Base class for template-related events"""
    template_id: str
    template_version: Optional[str] = None
    template_name: Optional[str] = None
    template_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "template_id": self.template_id,
            "template_version": self.template_version,
            "template_name": self.template_name,
            "template_category": self.template_category
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateEvent':
        """Create from dictionary"""
        base_event = BaseEvent.from_dict(data)
        return cls(
            event_id=base_event.event_id,
            event_type=base_event.event_type,
            timestamp=base_event.timestamp,
            source=base_event.source,
            priority=base_event.priority,
            correlation_id=base_event.correlation_id,
            template_id=data["template_id"],
            template_version=data.get("template_version"),
            template_name=data.get("template_name"),
            template_category=data.get("template_category")
        )


@dataclass
class TemplateInstanceEvent(BaseEvent):
    """Base class for template instance-related events"""
    instance_id: str
    template_id: str
    template_version: str
    project_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "instance_id": self.instance_id,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "project_id": self.project_id
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateInstanceEvent':
        """Create from dictionary"""
        base_event = BaseEvent.from_dict(data)
        return cls(
            event_id=base_event.event_id,
            event_type=base_event.event_type,
            timestamp=base_event.timestamp,
            source=base_event.source,
            priority=base_event.priority,
            correlation_id=base_event.correlation_id,
            instance_id=data["instance_id"],
            template_id=data["template_id"],
            template_version=data["template_version"],
            project_id=data.get("project_id")
        )


@dataclass
class CodeGenerationEvent(BaseEvent):
    """Base class for code generation-related events"""
    generation_id: str
    instance_id: str
    template_id: str
    template_version: str
    project_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "generation_id": self.generation_id,
            "instance_id": self.instance_id,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "project_id": self.project_id
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeGenerationEvent':
        """Create from dictionary"""
        base_event = BaseEvent.from_dict(data)
        return cls(
            event_id=base_event.event_id,
            event_type=base_event.event_type,
            timestamp=base_event.timestamp,
            source=base_event.source,
            priority=base_event.priority,
            correlation_id=base_event.correlation_id,
            generation_id=data["generation_id"],
            instance_id=data["instance_id"],
            template_id=data["template_id"],
            template_version=data["template_version"],
            project_id=data.get("project_id")
        )


@dataclass
class SystemEvent(BaseEvent):
    """Base class for system events"""
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = super().to_dict()
        result.update({
            "message": self.message,
            "details": self.details
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemEvent':
        """Create from dictionary"""
        base_event = BaseEvent.from_dict(data)
        return cls(
            event_id=base_event.event_id,
            event_type=base_event.event_type,
            timestamp=base_event.timestamp,
            source=base_event.source,
            priority=base_event.priority,
            correlation_id=base_event.correlation_id,
            message=data["message"],
            details=data.get("details")
        )
EOF

    # Analytics models
    cat > ./template-registry/models/analytics/usage.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Usage Analytics Models

This module defines models for template usage analytics.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Union


@dataclass
class FieldUsageStats:
    """Usage statistics for a field"""
    field_path: str  # Format: "section_name.field_name"
    section_name: str
    field_name: str
    completion_rate: float = 0.0  # Percentage of instances where field is filled
    error_rate: float = 0.0       # Percentage of instances where field has validation errors
    avg_fill_time: float = 0.0    # Average time to fill the field in seconds
    common_values: List[Any] = field(default_factory=list)  # Most common values
    common_errors: List[str] = field(default_factory=list)  # Most common validation errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldUsageStats':
        """Create from dictionary"""
        return cls(
            field_path=data["field_path"],
            section_name=data["section_name"],
            field_name=data["field_name"],
            completion_rate=data.get("completion_rate", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_fill_time=data.get("avg_fill_time", 0.0),
            common_values=data.get("common_values", []),
            common_errors=data.get("common_errors", [])
        )


@dataclass
class SectionUsageStats:
    """Usage statistics for a section"""
    section_name: str
    completion_rate: float = 0.0  # Percentage of instances where section is filled
    error_rate: float = 0.0       # Percentage of instances where section has validation errors
    avg_fill_time: float = 0.0    # Average time to fill the section in seconds
    field_stats: Dict[str, FieldUsageStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result["field_stats"] = {k: v.to_dict() for k, v in self.field_stats.items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SectionUsageStats':
        """Create from dictionary"""
        field_stats = {}
        for k, v in data.get("field_stats", {}).items():
            field_stats[k] = FieldUsageStats.from_dict(v)

        return cls(
            section_name=data["section_name"],
            completion_rate=data.get("completion_rate", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_fill_time=data.get("avg_fill_time", 0.0),
            field_stats=field_stats
        )


@dataclass
class CompletionPathStats:
    """Statistics about the path users take to complete a template"""
    total_instances: int = 0
    avg_completion_time: float = 0.0
    section_order: List[str] = field(default_factory=list)  # Most common section fill order
    field_order: List[str] = field(default_factory=list)    # Most common field fill order
    common_start_sections: List[str] = field(default_factory=list)
    common_end_sections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompletionPathStats':
        """Create from dictionary"""
        return cls(
            total_instances=data.get("total_instances", 0),
            avg_completion_time=data.get("avg_completion_time", 0.0),
            section_order=data.get("section_order", []),
            field_order=data.get("field_order", []),
            common_start_sections=data.get("common_start_sections", []),
            common_end_sections=data.get("common_end_sections", [])
        )


@dataclass
class TemplateUsageAnalytics:
    """Comprehensive usage analytics for a template"""
    template_id: str
    template_version: str
    analysis_timestamp: int = field(default_factory=lambda: int(time.time()))
    total_instances: int = 0
    completed_instances: int = 0
    completion_rate: float = 0.0
    avg_completion_time: float = 0.0
    validation_success_rate: float = 0.0
    section_stats: Dict[str, SectionUsageStats] = field(default_factory=dict)
    completion_path: CompletionPathStats = field(default_factory=CompletionPathStats)
    user_segments: Dict[str, Any] = field(default_factory=dict)
    generation_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "template_id": self.template_id,
            "template_version": self.template_version,
            "analysis_timestamp": self.analysis_timestamp,
            "total_instances": self.total_instances,
            "completed_instances": self.completed_instances,
            "completion_rate": self.completion_rate,
            "avg_completion_time": self.avg_completion_time,
            "validation_success_rate": self.validation_success_rate,
            "section_stats": {k: v.to_dict() for k, v in self.section_stats.items()},
            "completion_path": self.completion_path.to_dict(),
            "user_segments": self.user_segments,
            "generation_stats": self.generation_stats
        }
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateUsageAnalytics':
        """Create from dictionary"""
        section_stats = {}
        for k, v in data.get("section_stats", {}).items():
            section_stats[k] = SectionUsageStats.from_dict(v)

        completion_path = CompletionPathStats.from_dict(
            data.get("completion_path", {})
        )

        return cls(
            template_id=data["template_id"],
            template_version=data["template_version"],
            analysis_timestamp=data.get("analysis_timestamp", int(time.time())),
            total_instances=data.get("total_instances", 0),
            completed_instances=data.get("completed_instances", 0),
            completion_rate=data.get("completion_rate", 0.0),
            avg_completion_time=data.get("avg_completion_time", 0.0),
            validation_success_rate=data.get("validation_success_rate", 0.0),
            section_stats=section_stats,
            completion_path=completion_path,
            user_segments=data.get("user_segments", {}),
            generation_stats=data.get("generation_stats", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateUsageAnalytics':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
EOF

    # Evolution analytics models
    cat > ./template-registry/models/analytics/evolution.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry - Evolution Analytics Models

This module defines models for template evolution analytics and suggestions.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Union


class EvolutionSuggestionType(str, Enum):
    """Types of template evolution suggestions"""
    ADD_FIELD = "add_field"
    REMOVE_FIELD = "remove_field"
    MODIFY_FIELD = "modify_field"
    REORDER_FIELDS = "reorder_fields"
    ADD_SECTION = "add_section"
    REMOVE_SECTION = "remove_section"
    MODIFY_SECTION = "modify_section"
    REORDER_SECTIONS = "reorder_sections"
    ADD_VALIDATION = "add_validation"
    REMOVE_VALIDATION = "remove_validation"
    MODIFY_VALIDATION = "modify_validation"
    SPLIT_TEMPLATE = "split_template"
    MERGE_TEMPLATES = "merge_templates"


class EvolutionImpactLevel(str, Enum):
    """Impact level of evolution suggestions"""
    LOW = "low"         # Minimal impact, safe to apply automatically
    MEDIUM = "medium"   # Some impact, review recommended
    HIGH = "high"       # Significant impact, manual review required


@dataclass
class EvolutionSuggestion:
    """A suggestion for evolving a template"""
    suggestion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    suggestion_type: EvolutionSuggestionType = EvolutionSuggestionType.MODIFY_FIELD
    template_id: str = ""
    template_version: str = ""
    description: str = ""
    confidence: float = 0.0
    impact_level: EvolutionImpactLevel = EvolutionImpactLevel.MEDIUM
    impact_score: float = 0.0  # 0 (no impact) to 1.0 (high impact)
    created_at: int = field(default_factory=lambda: int(time.time()))
    changes: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    applied: bool = False
    applied_at: Optional[int] = None
    applied_version: Optional[str] = None

    def is_safe_for_auto_application(self) -> bool:
        """Check if the suggestion is safe for automatic application"""
        # Safe types are generally those that don't remove functionality
        safe_types = {
            EvolutionSuggestionType.ADD_FIELD,
            EvolutionSuggestionType.ADD_VALIDATION,
            EvolutionSuggestionType.REORDER_FIELDS,
            EvolutionSuggestionType.REORDER_SECTIONS
        }

        # For lower confidence changes, we're only safe with low impact level
        return (
            self.suggestion_type in safe_types and
            self.impact_level == EvolutionImpactLevel.LOW and
            self.confidence >= 0.9
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionSuggestion':
        """Create from dictionary"""
        return cls(
            suggestion_id=data.get("suggestion_id", str(uuid.uuid4())),
            suggestion_type=EvolutionSuggestionType(data["suggestion_type"]),
            template_id=data["template_id"],
            template_version=data["template_version"],
            description=data["description"],
            confidence=data["confidence"],
            impact_level=EvolutionImpactLevel(data.get("impact_level", "medium")),
            impact_score=data["impact_score"],
            created_at=data.get("created_at", int(time.time())),
            changes=data.get("changes", {}),
            rationale=data.get("rationale", ""),
            applied=data.get("applied", False),
            applied_at=data.get("applied_at"),
            applied_version=data.get("applied_version")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'EvolutionSuggestion':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class FieldEvolutionData:
    """Evolution data for a field"""
    field_path: str
    field_name: str
    section_name: str
    usage_count: int = 0
    completion_rate: float = 0.0
    error_rate: float = 0.0
    avg_fill_time: float = 0.0
    common_values: List[Any] = field(default_factory=list)
    value_patterns: List[Dict[str, Any]] = field(default_factory=list)
    correlations: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldEvolutionData':
        """Create from dictionary"""
        return cls(
            field_path=data["field_path"],
            field_name=data["field_name"],
            section_name=data["section_name"],
            usage_count=data.get("usage_count", 0),
            completion_rate=data.get("completion_rate", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_fill_time=data.get("avg_fill_time", 0.0),
            common_values=data.get("common_values", []),
            value_patterns=data.get("value_patterns", []),
            correlations=data.get("correlations", {})
        )


@dataclass
class SectionEvolutionData:
    """Evolution data for a section"""
    section_name: str
    usage_count: int = 0
    completion_rate: float = 0.0
    error_rate: float = 0.0
    avg_fill_time: float = 0.0
    field_evolution: Dict[str, FieldEvolutionData] = field(default_factory=dict)

    @classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'SectionEvolutionData':
    """Create from dictionary"""
    field_evolution = {}
    for k, v in data.get("field_evolution", {}).items():
        field_evolution[k] = FieldEvolutionData.from_dict(v)

    return cls(
        section_name=data["section_name"],
        usage_count=data.get("usage_count", 0),
        completion_rate=data.get("completion_rate", 0.0),
        error_rate=data.get("error_rate", 0.0),
        avg_fill_time=data.get("avg_fill_time", 0.0),
        field_evolution=field_evolution
    )


@dataclass
class TemplateEvolutionAnalysis:
    """Comprehensive evolution analysis for a template"""
    template_id: str
    template_version: str
    analysis_timestamp: int = field(default_factory=lambda: int(time.time()))
    total_instances: int = 0
    evolution_score: float = 0.0  # Overall score indicating evolution need (0-1)
    section_evolution: Dict[str, SectionEvolutionData] = field(default_factory=dict)
    field_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    suggestions: List[EvolutionSuggestion] = field(default_factory=list)
    common_patterns: Dict[str, Any] = field(default_factory=dict)
    auto_apply_threshold: float = 0.9  # Threshold for auto-applying suggestions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "template_id": self.template_id,
            "template_version": self.template_version,
            "analysis_timestamp": self.analysis_timestamp,
            "total_instances": self.total_instances,
            "evolution_score": self.evolution_score,
            "section_evolution": {k: v.to_dict() for k, v in self.section_evolution.items()},
            "field_correlations": self.field_correlations,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "common_patterns": self.common_patterns,
            "auto_apply_threshold": self.auto_apply_threshold
        }
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def get_auto_suggestions(self) -> List[EvolutionSuggestion]:
        """Get suggestions that can be auto-applied"""
        return [
            suggestion for suggestion in self.suggestions
            if suggestion.is_safe_for_auto_application() and
            suggestion.confidence >= self.auto_apply_threshold
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateEvolutionAnalysis':
        """Create from dictionary"""
        section_evolution = {}
        for k, v in data.get("section_evolution", {}).items():
            section_evolution[k] = SectionEvolutionData.from_dict(v)

        suggestions = []
        for s in data.get("suggestions", []):
            suggestions.append(EvolutionSuggestion.from_dict(s))

        return cls(
            template_id=data["template_id"],
            template_version=data["template_version"],
            analysis_timestamp=data.get("analysis_timestamp", int(time.time())),
            total_instances=data.get("total_instances", 0),
            evolution_score=data.get("evolution_score", 0.0),
            section_evolution=section_evolution,
            field_correlations=data.get("field_correlations", {}),
            suggestions=suggestions,
            common_patterns=data.get("common_patterns", {}),
            auto_apply_threshold=data.get("auto_apply_threshold", 0.9)
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateEvolutionAnalysis':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))



EOF