FieldDefinition(
                    name="name",
                    type=FieldType.STRING,
                    description="Parameter name",
                    required=True
                ),
                FieldDefinition(
                    name="type",
                    type=FieldType.ENUM,
                    description="Parameter type",
                    required=True,
                    options=["string", "integer", "float", "boolean", "array", "object"]
                ),
                FieldDefinition(
                    name="description",
                    type=FieldType.STRING,
                    description="Parameter description",
                    required=False
                ),
                FieldDefinition(
                    name="required",
                    type=FieldType.BOOLEAN,
                    description="Whether parameter is required",
                    required=False,
                    default_value=True
                )
            ]
        )
    ]

    # Define fields for response section
    response_fields = [
        FieldDefinition(
            name="response_format",
            type=FieldType.ENUM,
            description="Response format",
            required=True,
            options=["JSON", "XML", "TEXT", "BINARY"]
        ),
        FieldDefinition(
            name="return_type",
            type=FieldType.STRING,
            description="Return type of the function",
            required=True
        ),
        FieldDefinition(
            name="status_codes",
            type=FieldType.ARRAY,
            description="Possible status codes",
            required=True,
            nested_fields=[
                FieldDefinition(
                    name="code",
                    type=FieldType.INTEGER,
                    description="HTTP status code",
                    required=True
                ),
                FieldDefinition(
                    name="description",
                    type=FieldType.STRING,
                    description="Description of the status code",
                    required=True
                )
            ]
        )
    ]

    # Define fields for security section
    security_fields = [
        FieldDefinition(
            name="requires_auth",
            type=FieldType.BOOLEAN,
            description="Whether the endpoint requires authentication",
            required=True,
            default_value=True
        ),
        FieldDefinition(
            name="auth_type",
            type=FieldType.ENUM,
            description="Authentication type",
            required=False,
            options=["JWT", "API_KEY", "OAUTH2", "BASIC", "NONE"]
        ),
        FieldDefinition(
            name="required_roles",
            type=FieldType.ARRAY,
            description="Required roles for access",
            required=False
        )
    ]

    # Define sections
    sections = [
        SectionDefinition(
            name="Basic Information",
            description="Basic information about the endpoint",
            fields=basic_fields
        ),
        SectionDefinition(
            name="Request Parameters",
            description="Parameters for the request",
            fields=request_fields
        ),
        SectionDefinition(
            name="Response",
            description="Response information",
            fields=response_fields
        ),
        SectionDefinition(
            name="Security",
            description="Security information",
            fields=security_fields
        )
    ]

    # Define metadata
    metadata = TemplateMetadata(
        tags=["api", "rest", "endpoint"],
        domain="web",
        complexity="medium",
        estimated_completion_time=15
    )

    # Create template
    template = TemplateDefinition(
        id="api-rest-endpoint",
        name="REST API Endpoint",
        description="Specification for a REST API endpoint",
        version="1.0.0",
        category=TemplateCategory.API,
        languages=[TemplateLanguage.PYTHON, TemplateLanguage.JAVASCRIPT, TemplateLanguage.TYPESCRIPT],
        sections=sections,
        metadata=metadata,
        compatibility=CompatibilityType.BACKWARD
    )

    return template


def create_database_model_template() -> TemplateDefinition:
    """Create an example database model template"""
    # Define fields for basic information section
    basic_fields = [
        FieldDefinition(
            name="model_name",
            type=FieldType.STRING,
            description="Name of the model class",
            required=True,
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.REGEX,
                    expression=r"^[A-Z][a-zA-Z0-9]*$",
                    error_message="Name must start with uppercase letter and contain only letters and numbers"
                )
            ]
        ),
        FieldDefinition(
            name="table_name",
            type=FieldType.STRING,
            description="Name of the database table",
            required=True,
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.REGEX,
                    expression=r"^[a-z][a-z0-9_]*$",
                    error_message="Name must start with lowercase letter and contain only lowercase letters, numbers, and underscores"
                )
            ]
        ),
        FieldDefinition(
            name="description",
            type=FieldType.STRING,
            description="Description of the model",
            required=False
        )
    ]

    # Define fields for columns section
    column_fields = [
        FieldDefinition(
            name="columns",
            type=FieldType.ARRAY,
            description="Database columns",
            required=True,
            nested_fields=[
                FieldDefinition(
                    name="name",
                    type=FieldType.STRING,
                    description="Column name",
                    required=True,
                    validation_rules=[
                        ValidationRule(
                            rule_type=ValidationRuleType.REGEX,
                            expression=r"^[a-z][a-z0-9_]*$",
                            error_message="Name must start with lowercase letter and contain only lowercase letters, numbers, and underscores"
                        )
                    ]
                ),
                FieldDefinition(
                    name="type",
                    type=FieldType.ENUM,
                    description="Column data type",
                    required=True,
                    options=["string", "integer", "float", "boolean", "date", "datetime", "binary", "json", "array"]
                ),
                FieldDefinition(
                    name="description",
                    type=FieldType.STRING,
                    description="Column description",
                    required=False
                ),
                FieldDefinition(
                    name="required",
                    type=FieldType.BOOLEAN,
                    description="Whether the column is required (NOT NULL)",
                    required=True,
                    default_value=True
                ),
                FieldDefinition(
                    name="primary_key",
                    type=FieldType.BOOLEAN,
                    description="Whether the column is a primary key",
                    required=False,
                    default_value=False
                ),
                FieldDefinition(
                    name="unique",
                    type=FieldType.BOOLEAN,
                    description="Whether the column has a unique constraint",
                    required=False,
                    default_value=False
                ),
                FieldDefinition(
                    name="foreign_key",
                    type=FieldType.OBJECT,
                    description="Foreign key information",
                    required=False,
                    nested_fields=[
                        FieldDefinition(
                            name="table",
                            type=FieldType.STRING,
                            description="Referenced table",
                            required=True
                        ),
                        FieldDefinition(
                            name="column",
                            type=FieldType.STRING,
                            description="Referenced column",
                            required=True
                        ),
                        FieldDefinition(
                            name="on_delete",
                            type=FieldType.ENUM,
                            description="On delete action",
                            required=False,
                            options=["CASCADE", "RESTRICT", "SET NULL", "SET DEFAULT", "NO ACTION"]
                        )
                    ]
                ),
                FieldDefinition(
                    name="default_value",
                    type=FieldType.STRING,
                    description="Default value expression",
                    required=False
                )
            ]
        )
    ]

    # Define fields for indexes section
    index_fields = [
        FieldDefinition(
            name="indexes",
            type=FieldType.ARRAY,
            description="Database indexes",
            required=False,
            nested_fields=[
                FieldDefinition(
                    name="name",
                    type=FieldType.STRING,
                    description="Index name",
                    required=True,
                    validation_rules=[
                        ValidationRule(
                            rule_type=ValidationRuleType.REGEX,
                            expression=r"^[a-z][a-z0-9_]*$",
                            error_message="Name must start with lowercase letter and contain only lowercase letters, numbers, and underscores"
                        )
                    ]
                ),
                FieldDefinition(
                    name="columns",
                    type=FieldType.ARRAY,
                    description="Columns in the index",
                    required=True
                ),
                FieldDefinition(
                    name="unique",
                    type=FieldType.BOOLEAN,
                    description="Whether the index is unique",
                    required=False,
                    default_value=False
                ),
                FieldDefinition(
                    name="type",
                    type=FieldType.ENUM,
                    description="Index type",
                    required=False,
                    options=["BTREE", "HASH", "GIN", "GIST"]
                )
            ]
        )
    ]

    # Define fields for relationships section
    relationship_fields = [
        FieldDefinition(
            name="relationships",
            type=FieldType.ARRAY,
            description="Model relationships",
            required=False,
            nested_fields=[
                FieldDefinition(
                    name="name",
                    type=FieldType.STRING,
                    description="Relationship name",
                    required=True
                ),
                FieldDefinition(
                    name="type",
                    type=FieldType.ENUM,
                    description="Relationship type",
                    required=True,
                    options=["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
                ),
                FieldDefinition(
                    name="related_model",
                    type=FieldType.STRING,
                    description="Related model name",
                    required=True
                ),
                FieldDefinition(
                    name="local_field",
                    type=FieldType.STRING,
                    description="Local field name",
                    required=True
                ),
                FieldDefinition(
                    name="remote_field",
                    type=FieldType.STRING,
                    description="Remote field name",
                    required=True
                ),
                FieldDefinition(
                    name="cascade_delete",
                    type=FieldType.BOOLEAN,
                    description="Whether to cascade delete",
                    required=False,
                    default_value=False
                )
            ]
        )
    ]

    # Define sections
    sections = [
        SectionDefinition(
            name="Basic Information",
            description="Basic information about the model",
            fields=basic_fields
        ),
        SectionDefinition(
            name="Columns",
            description="Database columns",
            fields=column_fields
        ),
        SectionDefinition(
            name="Indexes",
            description="Database indexes",
            fields=index_fields
        ),
        SectionDefinition(
            name="Relationships",
            description="Model relationships",
            fields=relationship_fields
        )
    ]

    # Define metadata
    metadata = TemplateMetadata(
        tags=["database", "model", "orm"],
        domain="database",
        complexity="medium",
        estimated_completion_time=20
    )

    # Create template
    template = TemplateDefinition(
        id="database-model",
        name="Database Model",
        description="Specification for a database model",
        version="1.0.0",
        category=TemplateCategory.DATABASE,
        languages=[TemplateLanguage.PYTHON, TemplateLanguage.JAVASCRIPT, TemplateLanguage.TYPESCRIPT, TemplateLanguage.JAVA],
        sections=sections,
        metadata=metadata,
        compatibility=CompatibilityType.BACKWARD
    )

    return template


def create_ui_component_template() -> TemplateDefinition:
    """Create an example UI component template"""
    # Define fields for basic information section
    basic_fields = [
        FieldDefinition(
            name="component_name",
            type=FieldType.STRING,
            description="Name of the component",
            required=True,
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.REGEX,
                    expression=r"^[A-Z][a-zA-Z0-9]*$",
                    error_message="Name must start with uppercase letter and contain only letters and numbers"
                )
            ]
        ),
        FieldDefinition(
            name="description",
            type=FieldType.STRING,
            description="Description of the component",
            required=True
        ),
        FieldDefinition(
            name="type",
            type=FieldType.ENUM,
            description="Type of component",
            required=True,
            options=["functional", "class", "pure"]
        ),
        FieldDefinition(
            name="style_approach",
            type=FieldType.ENUM,
            description="Styling approach",
            required=True,
            options=["css", "scss", "styled-components", "tailwind", "emotion", "material-ui", "none"]
        )
    ]

    # Define fields for props section
    props_fields = [
        FieldDefinition(
            name="props",
            type=FieldType.ARRAY,
            description="Component props",
            required=True,
            nested_fields=[
                FieldDefinition(
                    name="name",
                    type=FieldType.STRING,
                    description="Prop name",
                    required=True,
                    validation_rules=[
                        ValidationRule(
                            rule_type=ValidationRuleType.REGEX,
                            expression=r"^[a-z][a-zA-Z0-9]*$",
                            error_message="Name must start with lowercase letter and contain only letters and numbers"
                        )
                    ]
                ),
                FieldDefinition(
                    name="type",
                    type=FieldType.STRING,
                    description="Prop type",
                    required=True
                ),
                FieldDefinition(
                    name="description",
                    type=FieldType.STRING,
                    description="Prop description",
                    required=False
                ),
                FieldDefinition(
                    name="required",
                    type=FieldType.BOOLEAN,
                    description="Whether the prop is required",
                    required=False,
                    default_value=False
                ),
                FieldDefinition(
                    name="default_value",
                    type=FieldType.STRING,
                    description="Default value",
                    required=False
                )
            ]
        )
    ]

    # Define fields for state section
    state_fields = [
        FieldDefinition(
            name="state_items",
            type=FieldType.ARRAY,
            description="Component state items",
            required=False,
            nested_fields=[
                FieldDefinition(
                    name="name",
                    type=FieldType.STRING,
                    description="State item name",
                    required=True,
                    validation_rules=[
                        ValidationRule(
                            rule_type=ValidationRuleType.REGEX,
                            expression=r"^[a-z][a-zA-Z0-9]*$",
                            error_message="Name must start with lowercase letter and contain only letters and numbers"
                        )
                    ]
                ),
                FieldDefinition(
                    name="type",
                    type=FieldType.STRING,
                    description="State item type",
                    required=True
                ),
                FieldDefinition(
                    name="description",
                    type=FieldType.STRING,
                    description="State item description",
                    required=False
                ),
                FieldDefinition(
                    name="initial_value",
                    type=FieldType.STRING,
                    description="Initial value",
                    required=False
                )
            ]
        )
    ]

    # Define fields for methods section
    methods_fields = [
        FieldDefinition(
            name="methods",
            type=FieldType.ARRAY,
            description="Component methods",
            required=False,
            nested_fields=[
                FieldDefinition(
                    name="name",
                    type=FieldType.STRING,
                    description="Method name",
                    required=True,
                    validation_rules=[
                        ValidationRule(
                            rule_type=ValidationRuleType.REGEX,
                            expression=r"^[a-z][a-zA-Z0-9]*$",
                            error_message="Name must start with lowercase letter and contain only letters and numbers"
                        )
                    ]
                ),
                FieldDefinition(
                    name="description",
                    type=FieldType.STRING,
                    description="Method description",
                    required=True
                ),
                FieldDefinition(
                    name="parameters",
                    type=FieldType.ARRAY,
                    description="Method parameters",
                    required=False,
                    nested_fields=[
                        FieldDefinition(
                            name="name",
                            type=FieldType.STRING,
                            description="Parameter name",
                            required=True
                        ),
                        FieldDefinition(
                            name="type",
                            type=FieldType.STRING,
                            description="Parameter type",
                            required=True
                        ),
                        FieldDefinition(
                            name="description",
                            type=FieldType.STRING,
                            description="Parameter description",
                            required=False
                        )
                    ]
                ),
                FieldDefinition(
                    name="return_type",
                    type=FieldType.STRING,
                    description="Return type",
                    required=False
                )
            ]
        )
    ]

    # Define fields for lifecycle section
    lifecycle_fields = [
        FieldDefinition(
            name="lifecycle_hooks",
            type=FieldType.ARRAY,
            description="Lifecycle hooks",
            required=False,
            nested_fields=[
                FieldDefinition(
                    name="name",
                    type=FieldType.ENUM,
                    description="Hook name",
                    required=True,
                    options=["mount", "unmount", "update", "error", "custom"]
                ),
                FieldDefinition(
                    name="description",
                    type=FieldType.STRING,
                    description="Hook description",
                    required=True
                ),
                FieldDefinition(
                    name="dependencies",
                    type=FieldType.ARRAY,
                    description="Hook dependencies",
                    required=False
                )
            ]
        )
    ]

    # Define sections
    sections = [
        SectionDefinition(
            name="Basic Information",
            description="Basic information about the component",
            fields=basic_fields
        ),
        SectionDefinition(
            name="Props",
            description="Component props",
            fields=props_fields
        ),
        SectionDefinition(
            name="State",
            description="Component state",
            fields=state_fields
        ),
        SectionDefinition(
            name="Methods",
            description="Component methods",
            fields=methods_fields
        ),
        SectionDefinition(
            name="Lifecycle",
            description="Component lifecycle",
            fields=lifecycle_fields
        )
    ]

    # Define metadata
    metadata = TemplateMetadata(
        tags=["ui", "component", "frontend"],
        domain="frontend",
        complexity="medium",
        estimated_completion_time=25
    )

    # Create template
    template = TemplateDefinition(
        id="ui-component",
        name="UI Component",
        description="Specification for a UI component",
        version="1.0.0",
        category=TemplateCategory.UI,
        languages=[TemplateLanguage.JAVASCRIPT, TemplateLanguage.TYPESCRIPT],
        sections=sections,
        metadata=metadata,
        compatibility=CompatibilityType.BACKWARD
    )

    return template


def bootstrap_registry():
    """Bootstrap the registry with example templates"""
    # Get registry instance
    registry = get_registry()

    # Create example templates
    templates = [
        create_api_template(),
        create_database_model_template(),
        create_ui_component_template()
    ]

    # Add templates to registry
    for template in templates:
        success, message, created = registry.create_template(template)
        if success:
            print(f"Created template: {template.id} (version {template.version})")
        else:
            print(f"Failed to create template {template.id}: {message}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Bootstrap registry
    bootstrap_registry()
EOF

# Create a main script to run the event-based server
cat > ./template-registry/main.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry Main

This script is the entry point for the template registry service.
"""

import os
import sys
import json
import logging
import argparse
import asyncio
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.template_registry import get_registry
from src.event_handlers.template_event_handler import TemplateEventHandler
from utils.bootstrap import bootstrap_registry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}


def main():
    """Main entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Template Registry Service")
    parser.add_argument(
        "--config",
        default="./config/config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Bootstrap registry with example templates"
    )

    args = parser.parse_args()

    # Set config path environment variable
    os.environ["TEMPLATE_REGISTRY_CONFIG"] = args.config

    # Load configuration
    config = load_config(args.config)

    # Get registry instance
    registry = get_registry(config)

    # Bootstrap registry if requested
    if args.bootstrap:
        logger.info("Bootstrapping registry with example templates")
        bootstrap_registry()

    # Create and start event handler
    event_handler = TemplateEventHandler(config, registry)
    event_handler.start()

    # Keep the script running
    try:
        logger.info("Template registry service running. Press Ctrl+C to exit.")
        while True:
            # Sleep a bit to avoid busy waiting
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping template registry service")
        event_handler.stop()
        registry.close()


if __name__ == "__main__":
    main()
EOF

# Create dockerfiles for the template registry
cat > ./template-registry/Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Redis and Pulsar
RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Download and extract Apache Pulsar
RUN curl -L https://archive.apache.org/dist/pulsar/pulsar-2.10.0/apache-pulsar-2.10.0-bin.tar.gz | tar -xz -C /opt \
    && ln -s /opt/apache-pulsar-2.10.0 /opt/pulsar

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create default directories
RUN mkdir -p ./storage/git/templates ./storage/cache

# Expose ports
EXPOSE 6650 8080

# Start services and application
CMD ["./entrypoint.sh"]
EOF

# Create entrypoint script for the Docker container
cat > ./template-registry/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Start Redis
echo "Starting Redis server..."
redis-server --daemonize yes

# Start Pulsar in standalone mode
echo "Starting Apache Pulsar in standalone mode..."
/opt/pulsar/bin/pulsar standalone > /var/log/pulsar.log 2>&1 &

# Wait for Pulsar to start
echo "Waiting for Pulsar to start..."
sleep 10

# Start the template registry
echo "Starting template registry..."
exec python main.py --bootstrap
EOF

# Make the entrypoint script executable
chmod +x ./template-registry/entrypoint.sh

# Create docker-compose.yml for easy deployment
cat > ./template-registry/docker-compose.yml << 'EOF'
version: '3.8'

services:
  template-registry:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "6650:6650"  # Pulsar
      - "8080:8080"  # Pulsar admin
    volumes:
      - ./config:/app/config
      - ./storage:/app/storage
    environment:
      - TEMPLATE_REGISTRY_CONFIG=/app/config/config.json
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
EOF

# Create requirements.txt for Python dependencies
cat > ./template-registry/requirements.txt << 'EOF'
pulsar-client>=3.1.0
redis>=4.5.4
jsonschema>=4.17.3
requests>=2.28.2
python-dateutil>=2.8.2
PyYAML>=6.0
EOF

# Create basic README file
cat > ./template-registry/README.md << 'EOF'
# Template Registry System

A comprehensive template registry system for spec-driven AI code generation platforms. The system stores, versions, categorizes, and validates specification templates.

## Features

- Git-based version control for templates with semantic versioning
- Redis caching for high performance
- Pulsar schema registry for schema validation and evolution
- Event-driven architecture through Apache Pulsar
- Support for versioning and compatibility checks
- Template categories, domains, and metadata
- Usage statistics tracking

## Architecture

The template registry is built with a layered architecture:

- **Core**: The central business logic for the template registry
- **Adapters**: Storage adapters for Git, Redis, and Pulsar
- **Event Handlers**: Event-based API for interacting with the registry
- **Models**: Data models for templates, fields, sections, etc.
- **Validators**: Schema validation utilities
- **Utils**: Utility functions and scripts

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Redis
- Apache Pulsar

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure the registry in `config/config.json`
4. Run the server: `python main.py --bootstrap`

### Docker

The registry can also be run using Docker:

```
docker-compose up -d
```

## Event-Driven API

The template registry provides an event-driven API through Apache Pulsar. The following event types are supported:

- `template-registry-create`: Create a template
- `template-registry-update`: Update a template
- `template-registry-get`: Get a template
- `template-registry-list`: List templates
- `template-registry-delete`: Delete a template
- `template-registry-validate`: Validate a template instance
- `template-registry-compare`: Compare template versions
- `template-registry-stats`: Get template statistics

## Configuration

The registry is configured using a JSON configuration file:

```json
{
    "storage": {
        "type": "hybrid",
        "gitRepository": {
            "path": "./storage/git/templates",
            "remote": "",
            "branch": "main",
            "pushOnUpdate": true
        },
        "cache": {
            "type": "redis",
            "host": "localhost",
            "port": 6379,
            "ttl": 3600,
            "prefix": "template-registry:"
        },
        "schemaStore": {
            "type": "pulsar",
            "serviceUrl": "pulsar://localhost:6650",
            "tenant": "public",
            "namespace": "template-registry",
            "topic": "schemas"
        }
    },
    "eventBus": {
        "serviceUrl": "pulsar://localhost:6650",
        "tenant": "public",
        "namespace": "template-registry",
        "subscriptionName": "template-registry-service",
        "responseTopicPrefix": "template-registry-response",
        "eventTopics": {
            "templateCreate": "template-registry-create",
            "templateUpdate": "template-registry-update",
            "templateGet": "template-registry-get",
            "templateList": "template-registry-list",
            "templateDelete": "template-registry-delete",
            "templateValidate": "template-registry-validate",
            "templateCompare": "template-registry-compare",
            "templateStats": "template-registry-stats"
        }
    },
    "metrics": {
        "enabled": true,
        "statsdHost": "localhost",
        "statsdPort": 8125,
        "prefix": "template_registry."
    },
    "validation": {
        "strictMode": true,
        "allowSchemaEvolution": true,
        "compatibilityStrategy": "BACKWARD"
    }
}
```

## License

This project is licensed under the MIT License.
EOF

# Create a test script
cat > ./template-registry/tests/test_template_registry.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry Tests

This module contains tests for the template registry.
"""

import os
import sys
import json
import unittest
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.template_models import (
    TemplateDefinition,
    TemplateCategory,
    TemplateLanguage,
    CompatibilityType,
    SectionDefinition,
    FieldDefinition,
    FieldType,
    ValidationRule,
    ValidationRuleType,
    TemplateMetadata
)


class TestTemplateRegistry(unittest.TestCase):
    """Test cases for the template registry"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a test template
        self.template = self._create_test_template()

    def _create_test_template(self) -> TemplateDefinition:
        """Create a test template"""
        # Define a basic template for testing
        fields = [
            FieldDefinition(
                name="name",
                type=FieldType.STRING,
                description="Name",
                required=True
            ),
            FieldDefinition(
                name="description",
                type=FieldType.STRING,
                description="Description",
                required=True
            )
        ]

        sections = [
            SectionDefinition(
                name="Basic",
                description="Basic information",
                fields=fields
            )
        ]

        metadata = TemplateMetadata(
            tags=["test"],
            domain="testing",
            complexity="simple"
        )

        template = TemplateDefinition(
            id="test-template",
            name="Test Template",
            description="Template for testing",
            version="1.0.0",
            category=TemplateCategory.OTHER,
            languages=[TemplateLanguage.ANY],
            sections=sections,
            metadata=metadata,
            compatibility=CompatibilityType.BACKWARD
        )

        return template

    def test_template_to_json_schema(self):
        """Test converting a template to JSON schema"""
        schema = self.template.to_json_schema()

        # Verify schema
        self.assertEqual(schema["title"], "Test Template")
        self.assertEqual(schema["description"], "Template for testing")
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("Basic", schema["properties"])
        self.assertIn("properties", schema["properties"]["Basic"])
        self.assertIn("name", schema["properties"]["Basic"]["properties"])
        self.assertIn("description", schema["properties"]["Basic"]["properties"])

    def test_template_to_dict(self):
        """Test converting a template to dictionary"""
        data = self.template.to_dict()

        # Verify dictionary
        self.assertEqual(data["id"], "test-template")
        self.assertEqual(data["name"], "Test Template")
        self.assertEqual(data["version"], "1.0.0")
        self.assertEqual(data["category"], "other")
        self.assertEqual(len(data["sections"]), 1)
        self.assertEqual(data["sections"][0]["name"], "Basic")
        self.assertEqual(len(data["sections"][0]["fields"]), 2)

    def test_template_from_dict(self):
        """Test creating a template from dictionary"""
        data = self.template.to_dict()
        template = TemplateDefinition.from_dict(data)

        # Verify template
        self.assertEqual(template.id, "test-template")
        self.assertEqual(template.name, "Test Template")
        self.assertEqual(template.version, "1.0.0")
        self.assertEqual(template.category, TemplateCategory.OTHER)
        self.assertEqual(len(template.sections), 1)
        self.assertEqual(template.sections[0].name, "Basic")
        self.assertEqual(len(template.sections[0].fields), 2)

    def test_template_to_json(self):
        """Test converting a template to JSON"""
        json_str = self.template.to_json()
        data = json.loads(json_str)

        # Verify JSON
        self.assertEqual(data["id"], "test-template")
        self.assertEqual(data["name"], "Test Template")
        self.assertEqual(data["version"], "1.0.0")
        self.assertEqual(data["category"], "other")

    def test_template_from_json(self):
        """Test creating a template from JSON"""
        json_str = self.template.to_json()
        template = TemplateDefinition.from_json(json_str)

        # Verify template
        self.assertEqual(template.id, "test-template")
        self.assertEqual(template.name, "Test Template")
        self.assertEqual(template.version, "1.0.0")
        self.assertEqual(template.category, TemplateCategory.OTHER)


if __name__ == "__main__":
    unittest.main()
EOF

# Create the PostgreSQL schema for advanced storage (optional alternative)
cat > ./template-registry/schemas/postgres_schema.sql << 'EOPSQL'
-- PostgreSQL Schema for Template Registry (Advanced Option)

-- Create templates table
CREATE TABLE IF NOT EXISTS templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    version TEXT NOT NULL,
    category TEXT NOT NULL,
    languages TEXT[] NOT NULL,
    compatibility TEXT NOT NULL,
    metadata JSONB NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    created_by TEXT NOT NULL,
    updated_by TEXT NOT NULL
);

-- Create template_versions table
CREATE TABLE IF NOT EXISTS template_versions (
    template_id TEXT NOT NULL,
    version TEXT NOT NULL,
    commit_id TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    author TEXT NOT NULL,
    message TEXT NOT NULL,
    changes TEXT[] NOT NULL,
    compatibility_type TEXT NOT NULL,
    is_breaking_change BOOLEAN NOT NULL,
    definition JSONB NOT NULL,
    PRIMARY KEY (template_id, version),
    FOREIGN KEY (template_id) REFERENCES templates(id) ON DELETE CASCADE
);

-- Create template_sections table
CREATE TABLE IF NOT EXISTS template_sections (
    template_id TEXT NOT NULL,
    version TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (template_id, version, name),
    FOREIGN KEY (template_id, version) REFERENCES template_versions(template_id, version) ON DELETE CASCADE
);

-- Create template_fields table
CREATE TABLE IF NOT EXISTS template_fields (
    template_id TEXT NOT NULL,
    version TEXT NOT NULL,
    section_name TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    description TEXT NOT NULL,
    required BOOLEAN NOT NULL DEFAULT FALSE,
    default_value TEXT,
    options TEXT[],
    position INTEGER NOT NULL,
    PRIMARY KEY (template_id, version, section_name, name),
    FOREIGN KEY (template_id, version, section_name) REFERENCES template_sections(template_id, version, name) ON DELETE CASCADE
);

-- Create template_validation_rules table
CREATE TABLE IF NOT EXISTS template_validation_rules (
    template_id TEXT NOT NULL,
    version TEXT NOT NULL,
    section_name TEXT NOT NULL,
    field_name TEXT NOT NULL,
    rule_type TEXT NOT NULL,
    expression TEXT NOT NULL,
    error_message TEXT,
    PRIMARY KEY (template_id, version, section_name, field_name, rule_type),
    FOREIGN KEY (template_id, version, section_name, field_name) REFERENCES template_fields(template_id, version, section_name, name) ON DELETE CASCADE
);

-- Create template_stats table
CREATE TABLE IF NOT EXISTS template_stats (
    template_id TEXT NOT NULL,
    version TEXT NOT NULL,
    usage_count INTEGER NOT NULL DEFAULT 0,
    completion_rate FLOAT NOT NULL DEFAULT 0.0,
    avg_completion_time FLOAT NOT NULL DEFAULT 0.0,
    last_used BIGINT NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (template_id, version),
    FOREIGN KEY (template_id, version) REFERENCES template_versions(template_id, version) ON DELETE CASCADE
);

-- Create indices for better performance
CREATE INDEX IF NOT EXISTS idx_templates_category ON templates(category);
CREATE INDEX IF NOT EXISTS idx_template_versions_timestamp ON template_versions(timestamp);
CREATE INDEX IF NOT EXISTS idx_template_stats_usage ON template_stats(usage_count DESC);

-- Create view for template summary
CREATE OR REPLACE VIEW template_summary AS
SELECT
    t.id,
    t.name,
    t.description,
    t.version,
    t.category,
    t.languages,
    t.compatibility,
    t.created_at,
    t.updated_at,
    t.created_by,
    t.updated_by,
    COALESCE(s.usage_count, 0) AS usage_count,
    COALESCE(s.completion_rate, 0) AS completion_rate,
    COALESCE(s.last_used, 0) AS last_used
FROM
    templates t
LEFT JOIN
    template_stats s ON t.id = s.template_id AND t.version = s.version;
EOPSQL

# Create GitHub workflow for CI/CD
mkdir -p ./template-registry/.github/workflows
cat > ./template-registry/.github/workflows/ci-cd.yml << 'EOF'
name: Template Registry CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis
        ports:
          - 6379:6379
      pulsar:
        image: apachepulsar/pulsar:2.10.0
        ports:
          - 6650:6650
          - 8080:8080
        options: --entrypoint /pulsar/bin/pulsar standalone

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest pytest-cov
    - name: Test with pytest
      run: |
        pytest --cov=./ --cov-report=xml
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-and-push:
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}/template-registry:latest
          ghcr.io/${{ github.repository }}/template-registry:${{ github.sha }}
EOF

# Make the build script and other scripts executable
chmod +x ./template-registry/entrypoint.sh
chmod +x ./template-registry/main.py
chmod +x ./template-registry/utils/bootstrap.py
chmod +x ./template-registry/tests/test_template_registry.py

echo "Template Registry System setup complete!"
echo "To start the system, run: cd template-registry && python main.py --bootstrap"
echo "To use with Docker, run: cd template-registry && docker-compose up -d"
EOF

# Make the build script executable
chmod +x ./build-template-registry.sh

echo "Build script created: ./build-template-registry.sh"
echo "Run this script to set up the Template Registry System."
    def _upload_schema(
        self,
        topic: str,
        schema_type: str,
        schema_definition: Dict[str, Any],
        properties: Dict[str, str] = None
    ) -> bool:
        """
        Upload a schema to Pulsar schema registry

        Args:
            topic: Pulsar topic name
            schema_type: Schema type (JSON, AVRO, etc.)
            schema_definition: Schema definition
            properties: Schema properties

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare payload
            payload = {
                "type": schema_type,
                "schema": json.dumps(schema_definition),
                "properties": properties or {}
            }

            # Upload schema
            response = requests.post(
                f"{self.admin_url}/admin/v2/schemas/{topic}/schema",
                json=payload
            )

            if response.status_code == 200 or response.status_code == 204:
                return True
            else:
                logger.error(f"Failed to upload schema: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to upload schema: {e}")
            return False

    def _create_schema_class(self, template: TemplateDefinition) -> Record:
        """
        Create a Pulsar schema class for a template

        Args:
            template: Template definition

        Returns:
            Pulsar schema class
        """
        # Define schema fields
        schema_fields = {}

        # Add fields for each section
        for section in template.sections:
            section_fields = {}

            # Add fields for each field in the section
            for field in section.fields:
                # Map field type to Pulsar schema type
                field_type = self._map_field_type(field.type)

                # Add field to section fields
                section_fields[field.name] = field_type

            # Create Record class for the section
            section_class = type(
                f"{section.name.replace(' ', '')}Record",
                (Record,),
                section_fields
            )

            # Add section to schema fields
            schema_fields[section.name.replace(' ', '_').lower()] = section_class

        # Create Record class for the template
        schema_class = type(
            f"{template.name.replace(' ', '')}Schema",
            (Record,),
            schema_fields
        )

        return schema_class

    def _map_field_type(self, field_type: str) -> Any:
        """
        Map template field type to Pulsar schema type

        Args:
            field_type: Template field type

        Returns:
            Pulsar schema type
        """
        mapping = {
            "string": String(),
            "integer": Integer(),
            "float": Float(),
            "boolean": Boolean(),
            "array": Array(),
            # Complex types would need custom Record classes
        }

        return mapping.get(field_type, String())

    def register_template_schema(self, template: TemplateDefinition) -> bool:
        """
        Register a template schema in Pulsar schema registry

        Args:
            template: Template definition

        Returns:
            True if successful, False otherwise
        """
        # Get topic name
        topic = self._get_topic_name(template.id)

        # Convert template to JSON schema
        schema_definition = template.to_json_schema()

        # Add additional properties
        properties = {
            "template.name": template.name,
            "template.version": template.version,
            "template.category": template.category.value,
            "template.compatibility": template.compatibility.value,
            "template.languages": ",".join(lang.value for lang in template.languages),
            "template.created_at": str(template.metadata.created_at),
            "template.updated_at": str(template.metadata.updated_at)
        }

        # Upload schema
        return self._upload_schema(
            topic,
            "JSON",
            schema_definition,
            properties
        )

    def get_template_schema(self, template_id: str) -> Dict[str, Any]:
        """
        Get a template schema from Pulsar schema registry

        Args:
            template_id: ID of the template

        Returns:
            Schema dictionary
        """
        # Get topic name
        topic = self._get_topic_name(template_id)

        # Get schema info
        schema_info = self._get_schema_info(topic)

        if not schema_info:
            return {}

        # Parse schema data
        try:
            schema_data = json.loads(schema_info.get("schema", "{}"))
            return schema_data
        except:
            return {}

    def get_template_schema_versions(self, template_id: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a template schema

        Args:
            template_id: ID of the template

        Returns:
            List of schema version dictionaries
        """
        # Get topic name
        topic = self._get_topic_name(template_id)

        # Get schema versions
        return self._get_schema_versions(topic)

    def is_schema_compatible(
        self,
        template_id: str,
        template: TemplateDefinition
    ) -> Tuple[bool, str]:
        """
        Check if a template schema is compatible with the latest version

        Args:
            template_id: ID of the template
            template: New template definition

        Returns:
            Tuple of (is_compatible, error_message)
        """
        # Get topic name
        topic = self._get_topic_name(template_id)

        # Convert template to JSON schema
        new_schema = template.to_json_schema()

        # Get latest schema
        schema_info = self._get_schema_info(topic)

        if not schema_info:
            # No existing schema, so it's compatible
            return True, ""

        # Parse latest schema
        try:
            latest_schema = json.loads(schema_info.get("schema", "{}"))
        except:
            # Invalid schema, assume incompatible
            return False, "Invalid schema format"

        # Check compatibility based on template's compatibility type
        if template.compatibility == CompatibilityType.NONE:
            # No compatibility checks
            return True, ""

        elif template.compatibility == CompatibilityType.BACKWARD:
            # New schema must be readable by old consumers
            # Check if new schema has all required fields from old schema
            return self._check_backward_compatibility(latest_schema, new_schema)

        elif template.compatibility == CompatibilityType.FORWARD:
            # Old data must be readable by new consumers
            # Check if old schema has all required fields from new schema
            return self._check_forward_compatibility(latest_schema, new_schema)

        elif template.compatibility == CompatibilityType.FULL:
            # Both backward and forward compatible
            backward_compatible, backward_error = self._check_backward_compatibility(
                latest_schema, new_schema
            )

            if not backward_compatible:
                return False, f"Not backward compatible: {backward_error}"

            forward_compatible, forward_error = self._check_forward_compatibility(
                latest_schema, new_schema
            )

            if not forward_compatible:
                return False, f"Not forward compatible: {forward_error}"

            return True, ""

        return False, f"Unknown compatibility type: {template.compatibility}"

    def _check_backward_compatibility(
        self,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check backward compatibility between schemas

        Args:
            old_schema: Old schema
            new_schema: New schema

        Returns:
            Tuple of (is_compatible, error_message)
        """
        # Check for removed required fields in new schema
        old_required = self._get_required_fields(old_schema)
        new_required = self._get_required_fields(new_schema)

        # All required fields in old schema must be required in new schema
        for path, field_type in old_required.items():
            if path not in new_required:
                return False, f"Required field removed: {path}"

            if new_required[path] != field_type:
                return False, f"Field type changed for required field: {path}"

        return True, ""

    def _check_forward_compatibility(
        self,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check forward compatibility between schemas

        Args:
            old_schema: Old schema
            new_schema: New schema

        Returns:
            Tuple of (is_compatible, error_message)
        """
        # Check for added required fields in new schema
        old_fields = self._get_all_fields(old_schema)
        new_required = self._get_required_fields(new_schema)

        # New required fields must exist in old schema
        for path, field_type in new_required.items():
            if path not in old_fields:
                return False, f"New required field added that doesn't exist in old schema: {path}"

            if old_fields[path] != field_type:
                return False, f"Field type changed for required field: {path}"

        return True, ""

    def _get_required_fields(
        self,
        schema: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Get all required fields from a schema

        Args:
            schema: JSON schema

        Returns:
            Dictionary of {field_path: field_type}
        """
        required_fields = {}

        # Check root required fields
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for field in required:
            if field in properties:
                field_schema = properties[field]
                field_type = field_schema.get("type", "string")
                required_fields[field] = field_type

                # Recursively check nested objects
                if field_type == "object":
                    nested_fields = self._get_required_fields(field_schema)
                    for nested_path, nested_type in nested_fields.items():
                        required_fields[f"{field}.{nested_path}"] = nested_type

        return required_fields

    def _get_all_fields(
        self,
        schema: Dict[str, Any],
        prefix: str = ""
    ) -> Dict[str, str]:
        """
        Get all fields from a schema

        Args:
            schema: JSON schema
            prefix: Field path prefix

        Returns:
            Dictionary of {field_path: field_type}
        """
        all_fields = {}

        properties = schema.get("properties", {})

        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type", "string")
            path = f"{prefix}{field_name}" if prefix else field_name
            all_fields[path] = field_type

            # Recursively check nested objects
            if field_type == "object":
                nested_fields = self._get_all_fields(
                    field_schema,
                    prefix=f"{path}."
                )
                all_fields.update(nested_fields)

        return all_fields

    def validate_template_instance(
        self,
        template_id: str,
        instance: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validate a template instance against its schema

        Args:
            template_id: ID of the template
            instance: Instance data

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get schema
        schema = self.get_template_schema(template_id)

        if not schema:
            return False, f"Schema not found for template {template_id}"

        # Import JSON schema validator
        try:
            from jsonschema import validate, ValidationError
        except ImportError:
            return False, "JSON schema validator not available"

        # Validate instance
        try:
            validate(instance=instance, schema=schema)
            return True, ""
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def delete_schema(self, template_id: str) -> bool:
        """
        Delete a template schema from Pulsar schema registry

        Args:
            template_id: ID of the template

        Returns:
            True if successful, False otherwise
        """
        # Get topic name
        topic = self._get_topic_name(template_id)

        try:
            # Delete schema
            response = requests.delete(
                f"{self.admin_url}/admin/v2/schemas/{topic}/schema"
            )

            if response.status_code == 200 or response.status_code == 204:
                return True
            else:
                logger.error(f"Failed to delete schema: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete schema: {e}")
            return False

    def close(self) -> None:
        """Close the Pulsar client"""
        if hasattr(self, 'client'):
            self.client.close()
EOF

# Create core registry class
cat > ./template-registry/src/core/template_registry.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry

This module provides the core template registry functionality, integrating
the various storage adapters and implementing the business logic.
"""

import json
import logging
import time
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from models.template_models import (
    TemplateDefinition,
    TemplateVersionInfo,
    TemplateInstance,
    TemplateStats,
    TemplateMetadata,
    TemplateCategory,
    TemplateLanguage,
    CompatibilityType,
    TemplateSearchQuery
)


logger = logging.getLogger(__name__)


class TemplateRegistry:
    """
    Core template registry implementation

    This class integrates the various storage adapters and implements
    the business logic for the template registry system.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the template registry

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._init_storage_adapters()

    def _init_storage_adapters(self):
        """Initialize storage adapters based on configuration"""
        storage_config = self.config.get("storage", {})
        self.storage_type = storage_config.get("type", "hybrid")

        # Git storage for version control
        from src.adapters.git_storage import GitStorageAdapter
        self.git_storage = GitStorageAdapter(
            storage_config.get("gitRepository", {})
        )

        # Redis cache for performance
        from src.adapters.redis_cache import RedisCacheAdapter
        self.cache = RedisCacheAdapter(
            storage_config.get("cache", {})
        )

        # Pulsar schema registry for schema validation
        from src.adapters.pulsar_schema_registry import PulsarSchemaRegistryAdapter
        self.schema_registry = PulsarSchemaRegistryAdapter(
            storage_config.get("schemaStore", {})
        )

        # Set up metrics collector if enabled
        if self.config.get("metrics", {}).get("enabled", False):
            # Import and initialize metrics collector
            pass

    def create_template(
        self,
        template: TemplateDefinition,
        author: str = "system"
    ) -> Tuple[bool, str, Optional[TemplateDefinition]]:
        """
        Create a new template

        Args:
            template: Template definition to create
            author: Author of the template

        Returns:
            Tuple of (success, message, created_template)
        """
        # Check if template already exists
        existing = self.get_template(template.id)
        if existing:
            return False, f"Template with ID {template.id} already exists", None

        # Set initial metadata
        if not template.metadata:
            template.metadata = TemplateMetadata()

        template.metadata.created_at = int(time.time())
        template.metadata.updated_at = int(time.time())
        template.metadata.created_by = author
        template.metadata.updated_by = author

        # Default version if not set
        if not template.version:
            template.version = "1.0.0"

        # Save template to git storage
        version_info = self.git_storage.save_template(
            template,
            message="Create new template",
            author=author
        )

        # Register schema in Pulsar
        schema_registered = self.schema_registry.register_template_schema(template)
        if not schema_registered:
            logger.warning(f"Failed to register schema for template {template.id}")

        # Cache the template
        self.cache.cache_template(template)

        # Initialize stats
        stats = TemplateStats(
            template_id=template.id,
            version=template.version,
            last_used=int(time.time())
        )
        self.cache.cache_stats(template.id, template.version, stats)

        return True, f"Template created successfully with version {template.version}", template

    def update_template(
        self,
        template: TemplateDefinition,
        author: str = "system",
        message: str = "Update template"
    ) -> Tuple[bool, str, Optional[TemplateDefinition]]:
        """
        Update an existing template

        Args:
            template: Template definition to update
            author: Author of the update
            message: Commit message

        Returns:
            Tuple of (success, message, updated_template)
        """
        # Check if template exists
        existing = self.get_template(template.id)
        if not existing:
            return False, f"Template with ID {template.id} does not exist", None

        # Check schema compatibility
        is_compatible, error_message = self.schema_registry.is_schema_compatible(
            template.id, template
        )

        if not is_compatible and template.compatibility != CompatibilityType.NONE:
            return (
                False,
                f"Schema is not compatible: {error_message}",
                None
            )

        # Update metadata
        template.metadata.updated_at = int(time.time())
        template.metadata.updated_by = author

        # Save template to git storage
        version_info = self.git_storage.save_template(
            template,
            message=message,
            author=author
        )

        # Register updated schema in Pulsar
        schema_registered = self.schema_registry.register_template_schema(template)
        if not schema_registered:
            logger.warning(f"Failed to register schema for template {template.id}")

        # Update cache
        self.cache.cache_template(template)

        # Initialize stats for new version
        stats = TemplateStats(
            template_id=template.id,
            version=template.version,
            last_used=int(time.time())
        )
        self.cache.cache_stats(template.id, template.version, stats)

        return True, f"Template updated successfully to version {template.version}", template

    def get_template(self, template_id: str) -> Optional[TemplateDefinition]:
        """
        Get a template by ID

        Args:
            template_id: ID of the template

        Returns:
            TemplateDefinition object or None if not found
        """
        # Try to get from cache first
        template = self.cache.get_cached_template(template_id)
        if template:
            return template

        # Fall back to git storage
        template = self.git_storage.load_template(template_id)
        if template:
            # Cache the template for future use
            self.cache.cache_template(template)
            return template

        return None

    def get_template_version(
        self,
        template_id: str,
        version: str
    ) -> Optional[TemplateDefinition]:
        """
        Get a specific version of a template

        Args:
            template_id: ID of the template
            version: Version to retrieve

        Returns:
            TemplateDefinition object or None if not found
        """
        # Try to get from cache first
        template = self.cache.get_cached_template_version(template_id, version)
        if template:
            return template

        # Fall back to git storage
        template = self.git_storage.load_template_version(template_id, version)
        if template:
            # Cache the template for future use
            self.cache.cache_template(template)
            return template

        return None

    def list_templates(
        self,
        query: TemplateSearchQuery = None
    ) -> List[TemplateDefinition]:
        """
        List all templates, with optional filtering

        Args:
            query: Optional search query for filtering

        Returns:
            List of TemplateDefinition objects
        """
        # Get all templates from git storage
        templates = self.git_storage.list_templates()

        # Filter if query provided
        if query:
            filtered = []
            for template in templates:
                # Filter by category
                if query.categories and template.category not in query.categories:
                    continue

                # Filter by language
                if query.languages and not any(lang in template.languages for lang in query.languages):
                    continue

                # Filter by domain
                if query.domain and template.metadata.domain != query.domain:
                    continue

                # Filter by complexity
                if query.complexity and template.metadata.complexity != query.complexity:
                    continue

                # Filter by tags
                if query.tags and not any(tag in template.metadata.tags for tag in query.tags):
                    continue

                # Filter by keywords (in name, description, or tags)
                if query.keywords:
                    keywords = query.keywords.lower()
                    if (keywords not in template.name.lower() and
                        keywords not in template.description.lower() and
                        not any(keywords in tag.lower() for tag in template.metadata.tags)):
                        continue

                filtered.append(template)

            templates = filtered

        # Sort templates
        if query and query.sort_by:
            reverse = query.sort_order.lower() == "desc"

            if query.sort_by == "name":
                templates.sort(key=lambda t: t.name, reverse=reverse)
            elif query.sort_by == "category":
                templates.sort(key=lambda t: t.category, reverse=reverse)
            elif query.sort_by == "updated_at":
                templates.sort(key=lambda t: t.metadata.updated_at, reverse=reverse)
            elif query.sort_by == "created_at":
                templates.sort(key=lambda t: t.metadata.created_at, reverse=reverse)

        # Apply limit and offset
        if query:
            start = query.offset
            end = query.offset + query.limit if query.limit > 0 else None
            templates = templates[start:end]

        return templates

    def list_template_versions(self, template_id: str) -> List[TemplateVersionInfo]:
        """
        List all versions of a template

        Args:
            template_id: ID of the template

        Returns:
            List of TemplateVersionInfo objects
        """
        # Try to get from cache first
        versions = self.cache.get_cached_template_versions(template_id)
        if versions:
            return versions

        # Fall back to git storage
        versions = self.git_storage.list_template_versions(template_id)
        if versions:
            # Cache the versions for future use
            self.cache.cache_template_versions(template_id, versions)

        return versions

    def delete_template(self, template_id: str) -> Tuple[bool, str]:
        """
        Delete a template

        Args:
            template_id: ID of the template to delete

        Returns:
            Tuple of (success, message)
        """
        # Delete from git storage
        deleted = self.git_storage.delete_template(template_id)
        if not deleted:
            return False, f"Failed to delete template {template_id} from storage"

        # Delete schema from Pulsar
        self.schema_registry.delete_schema(template_id)

        # Clear cache
        self.cache.clear_cache(template_id)

        return True, f"Template {template_id} deleted successfully"

    def validate_template_instance(
        self,
        template_id: str,
        instance: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a template instance against the schema

        Args:
            template_id: ID of the template
            instance: Template instance data

        Returns:
            Tuple of (is_valid, error_messages)
        """
        is_valid, error_message = self.schema_registry.validate_template_instance(
            template_id, instance
        )

        errors = [error_message] if error_message else []
        return is_valid, errors

    def get_template_stats(
        self,
        template_id: str,
        version: str = None
    ) -> Optional[TemplateStats]:
        """
        Get usage statistics for a template

        Args:
            template_id: ID of the template
            version: Optional specific version

        Returns:
            TemplateStats object or None if not found
        """
        if not version:
            # Get latest version
            template = self.get_template(template_id)
            if not template:
                return None
            version = template.version

        return self.cache.get_cached_stats(template_id, version)

    def record_template_usage(
        self,
        template_id: str,
        version: str = None
    ) -> None:
        """
        Record usage of a template

        Args:
            template_id: ID of the template
            version: Optional specific version
        """
        if not version:
            # Get latest version
            template = self.get_template(template_id)
            if not template:
                return
            version = template.version

        self.cache.increment_usage_count(template_id, version)

    def record_template_completion(
        self,
        template_id: str,
        version: str,
        time_seconds: float,
        success: bool = True
    ) -> None:
        """
        Record completion of a template

        Args:
            template_id: ID of the template
            version: Version of the template
            time_seconds: Time taken to complete in seconds
            success: Whether completion was successful
        """
        self.cache.record_completion_time(template_id, version, time_seconds, success)

    def compare_template_versions(
        self,
        template_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a template

        Args:
            template_id: ID of the template
            version1: First version to compare
            version2: Second version to compare

        Returns:
            Dictionary with differences
        """
        return self.git_storage.compare_template_versions(
            template_id, version1, version2
        )

    def get_categories(self) -> List[str]:
        """
        Get all available template categories

        Returns:
            List of category names
        """
        return [category.value for category in TemplateCategory]

    def get_languages(self) -> List[str]:
        """
        Get all available template languages

        Returns:
            List of language names
        """
        return [language.value for language in TemplateLanguage]

    def get_compatibility_types(self) -> List[str]:
        """
        Get all available compatibility types

        Returns:
            List of compatibility type names
        """
        return [compat.value for compat in CompatibilityType]

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the template registry

        Returns:
            Dictionary with registry statistics
        """
        templates = self.list_templates()

        # Calculate statistics
        stats = {
            "total_templates": len(templates),
            "categories": {},
            "languages": {},
            "domains": {},
            "last_updated": 0,
            "template_versions": 0
        }

        for template in templates:
            # Count by category
            category = template.category.value
            stats["categories"][category] = stats["categories"].get(category, 0) + 1

            # Count by language
            for lang in template.languages:
                lang_value = lang.value
                stats["languages"][lang_value] = stats["languages"].get(lang_value, 0) + 1

            # Count by domain
            domain = template.metadata.domain
            stats["domains"][domain] = stats["domains"].get(domain, 0) + 1

            # Track last updated
            if template.metadata.updated_at > stats["last_updated"]:
                stats["last_updated"] = template.metadata.updated_at

            # Count versions
            versions = self.list_template_versions(template.id)
            stats["template_versions"] += len(versions)

        # Add cache stats
        stats["cache"] = self.cache.get_cache_stats()

        return stats

    def close(self) -> None:
        """Close connections and clean up resources"""
        self.schema_registry.close()


# Create a global registry instance
registry = None

def get_registry(config: Dict[str, Any] = None) -> TemplateRegistry:
    """
    Get the global template registry instance

    Args:
        config: Optional configuration dictionary

    Returns:
        TemplateRegistry instance
    """
    global registry
    if registry is None:
        if config is None:
            # Load default configuration
            config_path = os.environ.get(
                "TEMPLATE_REGISTRY_CONFIG",
                "./config/config.json"
            )

            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                config = {}

        registry = TemplateRegistry(config)

    return registry
EOF

# Create the event handler system
cat > ./template-registry/src/event_handlers/template_event_handler.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Event Handler

This module provides event handlers for template registry events through
Apache Pulsar. It replaces the RESTful API approach with a purely event-driven
architecture as specified in the SSOT document.
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Awaitable

import pulsar

from models.template_models import (
    TemplateDefinition,
    TemplateInstance,
    TemplateVersionInfo,
    TemplateStats,
    TemplateSearchQuery,
    TemplateCategory,
    TemplateLanguage
)
from src.core.template_registry import get_registry, TemplateRegistry


logger = logging.getLogger(__name__)


class TemplateEventHandler:
    """
    Event handler for template registry events

    This class handles events for template registry operations through
    Apache Pulsar, implementing a fully event-driven architecture.
    """

    def __init__(self, config: Dict[str, Any], registry: Optional[TemplateRegistry] = None):
        """
        Initialize the template event handler

        Args:
            config: Configuration dictionary
            registry: Optional TemplateRegistry instance
        """
        self.config = config
        self.event_bus_config = config.get("eventBus", {})
        self.service_url = self.event_bus_config.get("serviceUrl", "pulsar://localhost:6650")
        self.tenant = self.event_bus_config.get("tenant", "public")
        self.namespace = self.event_bus_config.get("namespace", "template-registry")
        self.subscription_name = self.event_bus_config.get("subscriptionName", "template-registry-service")
        self.response_topic_prefix = self.event_bus_config.get("responseTopicPrefix", "template-registry-response")

        # Get event topics
        self.event_topics = self.event_bus_config.get("eventTopics", {
            "templateCreate": "template-registry-create",
            "templateUpdate": "template-registry-update",
            "templateGet": "template-registry-get",
            "templateList": "template-registry-list",
            "templateDelete": "template-registry-delete",
            "templateValidate": "template-registry-validate",
            "templateCompare": "template-registry-compare",
            "templateStats": "template-registry-stats"
        })

        # Initialize Pulsar client
        self.client = pulsar.Client(self.service_url)

        # Initialize response producer
        self.response_producer = self.client.create_producer(
            f"persistent://{self.tenant}/{self.namespace}/{self.response_topic_prefix}",
            schema=pulsar.schema.JsonSchema(),
            properties={
                "producer-name": f"template-registry-response-{uuid.uuid4()}",
                "producer-id": str(uuid.uuid4())
            }
        )

        # Initialize consumers
        self.consumers = {}

        # Get or create registry
        self.registry = registry or get_registry(config)

    def start(self):
        """Start handling events"""
        logger.info("Starting template event handler")

        # Subscribe to all event topics
        self._subscribe_to_events()

        logger.info("Template event handler started")

    def _subscribe_to_events(self):
        """Subscribe to all event topics"""
        # Map event types to handler methods
        handlers = {
            "templateCreate": self._handle_template_create,
            "templateUpdate": self._handle_template_update,
            "templateGet": self._handle_template_get,
            "templateList": self._handle_template_list,
            "templateDelete": self._handle_template_delete,
            "templateValidate": self._handle_template_validate,
            "templateCompare": self._handle_template_compare,
            "templateStats": self._handle_template_stats
        }

        # Subscribe to each event type
        for event_type, handler in handlers.items():
            topic = self.event_topics.get(event_type)
            if not topic:
                logger.warning(f"No topic configured for event type {event_type}")
                continue

            logger.info(f"Subscribing to {event_type} events on topic {topic}")

            # Create message listener
            def create_message_listener(handler_func):
                def message_listener(consumer, msg):
                    try:
                        # Parse payload
                        payload = json.loads(msg.data())

                        # Get correlation ID for response
                        correlation_id = payload.get("correlation_id")

                        # Handle event
                        result = handler_func(payload)

                        # Send response
                        self._send_response(result, correlation_id)

                        # Acknowledge message
                        consumer.acknowledge(msg)
                    except Exception as e:
                        logger.error(f"Error handling event: {e}")
                        # Send error response
                        self._send_error(str(e), correlation_id)
                        # Acknowledge message anyway to avoid redelivery
                        consumer.acknowledge(msg)

                return message_listener

            # Subscribe to topic
            full_topic = f"persistent://{self.tenant}/{self.namespace}/{topic}"
            try:
                consumer = self.client.subscribe(
                    topic=full_topic,
                    subscription_name=self.subscription_name,
                    consumer_type=pulsar.ConsumerType.Shared,
                    message_listener=create_message_listener(handler)
                )
                self.consumers[event_type] = consumer
                logger.info(f"Subscribed to {full_topic}")
            except Exception as e:
                logger.error(f"Failed to subscribe to {full_topic}: {e}")

    def _send_response(self, result: Dict[str, Any], correlation_id: Optional[str] = None):
        """
        Send response to the response topic

        Args:
            result: Response data
            correlation_id: Optional correlation ID for the client
        """
        try:
            # Prepare response
            response = {
                "timestamp": int(time.time() * 1000),
                "correlation_id": correlation_id,
                "status": "success",
                "data": result
            }

            # Send response
            self.response_producer.send(
                value=response,
                partition_key=correlation_id or str(uuid.uuid4())
            )
        except Exception as e:
            logger.error(f"Failed to send response: {e}")

    def _send_error(self, error: str, correlation_id: Optional[str] = None):
        """
        Send error response to the response topic

        Args:
            error: Error message
            correlation_id: Optional correlation ID for the client
        """
        try:
            # Prepare response
            response = {
                "timestamp": int(time.time() * 1000),
                "correlation_id": correlation_id,
                "status": "error",
                "error": error
            }

            # Send response
            self.response_producer.send(
                value=response,
                partition_key=correlation_id or str(uuid.uuid4())
            )
        except Exception as e:
            logger.error(f"Failed to send error response: {e}")

    def _handle_template_create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle template create event

        Args:
            payload: Event payload

        Returns:
            Response data
        """
        # Extract template data
        template_data = payload.get("template")
        author = payload.get("author", "system")

        if not template_data:
            raise ValueError("No template data provided")

        # Create template
        template = TemplateDefinition.from_dict(template_data)
        success, message, created = self.registry.create_template(template, author=author)

        # Return response
        return {
            "success": success,
            "message": message,
            "template": created.to_dict() if created else None
        }

    def _handle_template_update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle template update event

        Args:
            payload: Event payload

        Returns:
            Response data
        """
        # Extract template data
        template_data = payload.get("template")
        author = payload.get("author", "system")
        message = payload.get("message", "Update template")

        if not template_data:
            raise ValueError("No template data provided")

        # Update template
        template = TemplateDefinition.from_dict(template_data)
        success, message, updated = self.registry.update_template(
            template,
            author=author,
            message=message
        )

        # Return response
        return {
            "success": success,
            "message": message,
            "template": updated.to_dict() if updated else None
        }

    def _handle_template_get(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle template get event

        Args:
            payload: Event payload

        Returns:
            Response data
        """
        # Extract parameters
        template_id = payload.get("template_id")
        version = payload.get("version")

        if not template_id:
            raise ValueError("No template ID provided")

        # Get template
        if version:
            template = self.registry.get_template_version(template_id, version)
        else:
            template = self.registry.get_template(template_id)

        # Record usage if found
        if template:
            self.registry.record_template_usage(template_id, template.version)

        # Return response
        return {
            "success": template is not None,
            "message": "Template retrieved successfully" if template else f"Template {template_id} not found",
            "template": template.to_dict() if template else None
        }

    def _handle_template_list(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle template list event

        Args:
            payload: Event payload

        Returns:
            Response data
        """
        # Extract query parameters
        query_data = payload.get("query", {})

        # Create search query
        query = None
        if query_data:
            # Convert category and language lists
            categories = query_data.get("categories", [])
            if categories:
                categories = [TemplateCategory(c) for c in categories]

            languages = query_data.get("languages", [])
            if languages:
                languages = [TemplateLanguage(l) for l in languages]

            # Create query object
            query = TemplateSearchQuery(
                keywords=query_data.get("keywords"),
                categories=categories,
                languages=languages,
                tags=query_data.get("tags", []),
                domain=query_data.get("domain"),
                complexity=query_data.get("complexity"),
                sort_by=query_data.get("sort_by", "updated_at"),
                sort_order=query_data.get("sort_order", "desc"),
                limit=query_data.get("limit", 20),
                offset=query_data.get("offset", 0)
            )

        # List templates
        templates = self.registry.list_templates(query)

        # Return response
        return {
            "templates": [t.to_dict() for t in templates],
            "count": len(templates)
        }

    def _handle_template_delete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle template delete event

        Args:
            payload: Event payload

        Returns:
            Response data
        """
        # Extract parameters
        template_id = payload.get("template_id")

        if not template_id:
            raise ValueError("No template ID provided")

        # Delete template
        success, message = self.registry.delete_template(template_id)

        # Return response
        return {
            "success": success,
            "message": message
        }

    def _handle_template_validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle template validate event

        Args:
            payload: Event payload

        Returns:
            Response data
        """
        # Extract parameters
        template_id = payload.get("template_id")
        instance = payload.get("instance")

        if not template_id:
            raise ValueError("No template ID provided")

        if not instance:
            raise ValueError("No instance data provided")

        # Validate instance
        is_valid, errors = self.registry.validate_template_instance(template_id, instance)

        # Return response
        return {
            "is_valid": is_valid,
            "errors": errors
        }

    def _handle_template_compare(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle template compare event

        Args:
            payload: Event payload

        Returns:
            Response data
        """
        # Extract parameters
        template_id = payload.get("template_id")
        version1 = payload.get("version1")
        version2 = payload.get("version2")

        if not template_id:
            raise ValueError("No template ID provided")

        if not version1 or not version2:
            raise ValueError("Both version1 and version2 are required")

        # Compare versions
        differences = self.registry.compare_template_versions(
            template_id,
            version1,
            version2
        )

        # Return response
        return differences

    def _handle_template_stats(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle template stats event

        Args:
            payload: Event payload

        Returns:
            Response data
        """
        # Extract parameters
        template_id = payload.get("template_id")
        version = payload.get("version")

        # Get registry stats if no template ID
        if not template_id:
            return {
                "registry_stats": self.registry.get_registry_stats()
            }

        # Get template stats
        stats = self.registry.get_template_stats(template_id, version)

        # Return response
        if not stats:
            return {
                "stats": {
                    "template_id": template_id,
                    "version": version,
                    "usage_count": 0,
                    "completion_rate": 0.0,
                    "avg_completion_time": 0.0,
                    "last_used": 0,
                    "error_count": 0
                }
            }

        return {
            "stats": stats.to_dict()
        }

    def stop(self):
        """Stop handling events and clean up resources"""
        logger.info("Stopping template event handler")

        # Close all consumers
        for consumer in self.consumers.values():
            consumer.close()

        # Close producer
        self.response_producer.close()

        # Close client
        self.client.close()

        logger.info("Template event handler stopped")
EOF

# Create validator for template schemas
cat > ./template-registry/src/validators/schema_validator.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Schema Validator

This module provides validation functionality for template schemas.
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Optional, Union

import jsonschema
from jsonschema import Draft7Validator, validators

from models.template_models import (
    TemplateDefinition,
    ValidationRule,
    ValidationRuleType
)


logger = logging.getLogger(__name__)


def extend_with_default(validator_class):
    """
    Extend validator to fill in default values

    This function extends a JSON schema validator to fill in default values
    for missing properties.
    """
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in properties.items():
            if "default" in subschema and prop not in instance:
                instance[prop] = subschema["default"]

        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return validators.extend(
        validator_class, {"properties": set_defaults}
    )


DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)


class SchemaValidator:
    """
    Validator for JSON schemas

    This class provides validation of JSON schemas and instances against schemas.
    """

    def __init__(self):
        """Initialize the schema validator"""
        pass

    def validate_schema(self, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a JSON schema against the Draft 7 meta-schema

        Args:
            schema: JSON schema to validate

        Returns:
            Tuple of (is_valid, errors)
        """
        try:
            jsonschema.validators.validate(schema, jsonschema.validators.Draft7Validator.META_SCHEMA)
            return True, []
        except jsonschema.exceptions.ValidationError as e:
            return False, [str(e)]
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False, [str(e)]

    def validate_instance(
        self,
        instance: Dict[str, Any],
        schema: Dict[str, Any],
        fill_defaults: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate an instance against a schema

        Args:
            instance: Instance to validate
            schema: Schema to validate against
            fill_defaults: Whether to fill in default values

        Returns:
            Tuple of (is_valid, errors)
        """
        try:
            if fill_defaults:
                # Fill in defaults
                DefaultValidatingDraft7Validator(schema).validate(instance)
            else:
                # Standard validation without filling defaults
                jsonschema.validate(instance, schema)

            return True, []
        except jsonschema.exceptions.ValidationError as e:
            return False, [str(e)]
        except Exception as e:
            logger.error(f"Instance validation error: {e}")
            return False, [str(e)]

    def validate_template_instance(
        self,
        instance: Dict[str, Any],
        template: TemplateDefinition
    ) -> Tuple[bool, List[str]]:
        """
        Validate an instance against a template definition

        Args:
            instance: Instance to validate
            template: Template definition

        Returns:
            Tuple of (is_valid, errors)
        """
        # Convert template to JSON schema
        schema = template.to_json_schema()

        # Validate instance against schema
        is_valid, errors = self.validate_instance(instance, schema)

        # If valid, apply custom validation rules
        if is_valid:
            custom_errors = self._apply_custom_validation_rules(instance, template)
            if custom_errors:
                is_valid = False
                errors.extend(custom_errors)

        return is_valid, errors

    def _apply_custom_validation_rules(
        self,
        instance: Dict[str, Any],
        template: TemplateDefinition
    ) -> List[str]:
        """
        Apply custom validation rules not covered by JSON schema

        Args:
            instance: Instance to validate
            template: Template definition

        Returns:
            List of validation errors
        """
        errors = []

        # Process each section
        for section in template.sections:
            section_name = section.name
            if section_name not in instance:
                continue

            section_instance = instance[section_name]

            # Process each field
            for field in section.fields:
                field_name = field.name
                if field_name not in section_instance:
                    continue

                field_value = section_instance[field_name]

                # Apply validation rules
                for rule in field.validation_rules:
                    error = self._validate_rule(
                        rule, field_value, field_name, section_name
                    )
                    if error:
                        errors.append(error)

        return errors

    def _validate_rule(
        self,
        rule: ValidationRule,
        value: Any,
        field_name: str,
        section_name: str
    ) -> Optional[str]:
        """
        Validate a value against a rule

        Args:
            rule: Validation rule
            value: Value to validate
            field_name: Name of the field
            section_name: Name of the section

        Returns:
            Error message or None if valid
        """
        if rule.rule_type == ValidationRuleType.REGEX:
            import re
            if not re.match(rule.expression, str(value)):
                return rule.error_message or f"Field {section_name}.{field_name} does not match pattern: {rule.expression}"

        elif rule.rule_type == ValidationRuleType.MIN_LENGTH:
            min_length = int(rule.expression)
            if isinstance(value, (str, list, dict)) and len(value) < min_length:
                return rule.error_message or f"Field {section_name}.{field_name} must have length >= {min_length}"

        elif rule.rule_type == ValidationRuleType.MAX_LENGTH:
            max_length = int(rule.expression)
            if isinstance(value, (str, list, dict)) and len(value) > max_length:
                return rule.error_message or f"Field {section_name}.{field_name} must have length <= {max_length}"

        elif rule.rule_type == ValidationRuleType.MIN_VALUE:
            min_value = float(rule.expression)
            if isinstance(value, (int, float)) and value < min_value:
                return rule.error_message or f"Field {section_name}.{field_name} must be >= {min_value}"

        elif rule.rule_type == ValidationRuleType.MAX_VALUE:
            max_value = float(rule.expression)
            if isinstance(value, (int, float)) and value > max_value:
                return rule.error_message or f"Field {section_name}.{field_name} must be <= {max_value}"

        elif rule.rule_type == ValidationRuleType.ENUM_VALUES:
            allowed_values = rule.expression.split(',')
            if str(value) not in allowed_values:
                return rule.error_message or f"Field {section_name}.{field_name} must be one of: {', '.join(allowed_values)}"

        elif rule.rule_type == ValidationRuleType.FORMAT:
            if rule.expression == "date":
                from dateutil.parser import parse
                try:
                    parse(value)
                except:
                    return rule.error_message or f"Field {section_name}.{field_name} must be a valid date"

            elif rule.expression == "email":
                import re
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}#!/bin/bash
#============================================================================
# Template Registry System for Spec-Driven AI Code Generation Platform
#
# This script sets up a template registry system that handles storage,
# versioning, categorization, and validation of specification templates.
# The system leverages Apache Pulsar for event-driven architecture,
# Redis for high-performance caching, and Git for version control.
#
# Author: Claude
# Date: 2025-04-18
#============================================================================

set -e
echo "Setting up Template Registry System..."

# Create directory structure
mkdir -p ./template-registry/{src,models,storage,schemas,utils,tests,config}
mkdir -p ./template-registry/src/{core,event_handlers,validators,adapters}
mkdir -p ./template-registry/storage/{git,cache}
mkdir -p ./template-registry/schemas/{templates,definitions}

# Configuration
cat > ./template-registry/config/config.json << 'EOF'
{
    "storage": {
        "type": "hybrid",
        "gitRepository": {
            "path": "./storage/git/templates",
            "remote": "",
            "branch": "main",
            "pushOnUpdate": true
        },
        "cache": {
            "type": "redis",
            "host": "localhost",
            "port": 6379,
            "ttl": 3600,
            "prefix": "template-registry:"
        },
        "schemaStore": {
            "type": "pulsar",
            "serviceUrl": "pulsar://localhost:6650",
            "tenant": "public",
            "namespace": "template-registry",
            "topic": "schemas"
        }
    },
    "eventBus": {
        "serviceUrl": "pulsar://localhost:6650",
        "tenant": "public",
        "namespace": "template-registry",
        "subscriptionName": "template-registry-service",
        "responseTopicPrefix": "template-registry-response",
        "eventTopics": {
            "templateCreate": "template-registry-create",
            "templateUpdate": "template-registry-update",
            "templateGet": "template-registry-get",
            "templateList": "template-registry-list",
            "templateDelete": "template-registry-delete",
            "templateValidate": "template-registry-validate",
            "templateCompare": "template-registry-compare",
            "templateStats": "template-registry-stats"
        }
    },
    "metrics": {
        "enabled": true,
        "statsdHost": "localhost",
        "statsdPort": 8125,
        "prefix": "template_registry."
    },
    "validation": {
        "strictMode": true,
        "allowSchemaEvolution": true,
        "compatibilityStrategy": "BACKWARD"
    }
}
EOF

# Create the core models for template data structures
cat > ./template-registry/models/template_models.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry Models

This module defines the core data models for the template registry system.
It includes classes for templates, fields, sections, validation rules, and more.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set


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
    REFERENCE = "reference"
    DATETIME = "datetime"


class ValidationRuleType(str, Enum):
    """Types of validation rules"""
    REGEX = "regex"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    REQUIRED = "required"
    ENUM_VALUES = "enum_values"
    CUSTOM = "custom"
    DEPENDENCY = "dependency"
    FORMAT = "format"


class TemplateCategory(str, Enum):
    """Categories for templates"""
    API = "api"
    DATABASE = "database"
    UI = "ui"
    WORKFLOW = "workflow"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    INFRASTRUCTURE = "infrastructure"
    OTHER = "other"


class TemplateLanguage(str, Enum):
    """Programming languages for templates"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    ANY = "any"


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
    error_message: str


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
            FieldType.REFERENCE: "string",
            FieldType.DATETIME: "string"
        }
        return mapping.get(self.type, "string")


@dataclass
class SectionDefinition:
    """Definition of a section in a template"""
    name: str
    description: str
    fields: List[FieldDefinition] = field(default_factory=list)


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


@dataclass
class TemplateDefinition:
    """Definition of a template"""
    id: str
    name: str
    description: str
    version: str
    category: TemplateCategory
    languages: List[TemplateLanguage] = field(default_factory=lambda: [TemplateLanguage.ANY])
    sections: List[SectionDefinition] = field(default_factory=list)
    metadata: TemplateMetadata = field(default_factory=TemplateMetadata)
    compatibility: CompatibilityType = CompatibilityType.BACKWARD

    def __post_init__(self):
        """Validate after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateDefinition':
        """Create from dictionary"""
        sections = []
        for section_data in data.get("sections", []):
            fields = []
            for field_data in section_data.get("fields", []):
                validation_rules = []
                for rule_data in field_data.get("validation_rules", []):
                    validation_rules.append(ValidationRule(
                        rule_type=ValidationRuleType(rule_data["rule_type"]),
                        expression=rule_data["expression"],
                        error_message=rule_data["error_message"]
                    ))

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
                        nested_fields=[]
                    ))

                fields.append(FieldDefinition(
                    name=field_data["name"],
                    type=FieldType(field_data["type"]),
                    description=field_data["description"],
                    required=field_data.get("required", False),
                    default_value=field_data.get("default_value"),
                    validation_rules=validation_rules,
                    options=field_data.get("options", []),
                    nested_fields=nested_fields
                ))

            sections.append(SectionDefinition(
                name=section_data["name"],
                description=section_data["description"],
                fields=fields
            ))

        metadata_data = data.get("metadata", {})
        metadata = TemplateMetadata(
            created_at=metadata_data.get("created_at", int(time.time())),
            created_by=metadata_data.get("created_by", "system"),
            updated_at=metadata_data.get("updated_at", int(time.time())),
            updated_by=metadata_data.get("updated_by", "system"),
            tags=metadata_data.get("tags", []),
            domain=metadata_data.get("domain", "general"),
            complexity=metadata_data.get("complexity", "medium"),
            estimated_completion_time=metadata_data.get("estimated_completion_time", 0)
        )

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data["version"],
            category=TemplateCategory(data["category"]),
            languages=[TemplateLanguage(lang) for lang in data.get("languages", ["any"])],
            sections=sections,
            metadata=metadata,
            compatibility=CompatibilityType(data.get("compatibility", "backward"))
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateDefinition':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class FieldValue:
    """Value of a field"""
    name: str
    value: Any


@dataclass
class SectionValues:
    """Values for a section"""
    name: str
    fields: List[FieldValue] = field(default_factory=list)


@dataclass
class TemplateInstance:
    """Filled template instance"""
    id: str
    template_id: str
    template_version: str
    project_id: str
    name: str
    sections: List[SectionValues] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    completed: bool = False
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateInstance':
        """Create from dictionary"""
        sections = []
        for section_data in data.get("sections", []):
            fields = []
            for field_data in section_data.get("fields", []):
                fields.append(FieldValue(
                    name=field_data["name"],
                    value=field_data["value"]
                ))

            sections.append(SectionValues(
                name=section_data["name"],
                fields=fields
            ))

        return cls(
            id=data["id"],
            template_id=data["template_id"],
            template_version=data["template_version"],
            project_id=data["project_id"],
            name=data["name"],
            sections=sections,
            created_at=data.get("created_at", int(time.time())),
            updated_at=data.get("updated_at", int(time.time())),
            completed=data.get("completed", False),
            validated=data.get("validated", False),
            validation_errors=data.get("validation_errors", [])
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateInstance':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TemplateDependency:
    """Dependency between templates"""
    source_id: str
    target_id: str
    dependency_type: str
    description: str


@dataclass
class TemplateVersionInfo:
    """Information about a template version"""
    template_id: str
    version: str
    commit_id: str
    timestamp: int
    author: str
    message: str
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
class TemplateStats:
    """Statistics for a template"""
    template_id: str
    version: str
    usage_count: int = 0
    completion_rate: float = 0.0
    avg_completion_time: float = 0.0
    last_used: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TemplateSearchQuery:
    """Query for searching templates"""
    keywords: Optional[str] = None
    categories: List[TemplateCategory] = field(default_factory=list)
    languages: List[TemplateLanguage] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    complexity: Optional[str] = None
    sort_by: str = "updated_at"
    sort_order: str = "desc"
    limit: int = 20
    offset: int = 0
EOF

# Create storage adapters
cat > ./template-registry/src/adapters/git_storage.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Git Storage Adapter

This module provides a Git-based storage adapter for template versioning.
It commits changes to templates into a Git repository for version control,
history tracking, and collaboration.
"""

import os
import json
import time
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from models.template_models import (
    TemplateDefinition,
    TemplateVersionInfo,
    CompatibilityType
)


logger = logging.getLogger(__name__)


class GitStorageAdapter:
    """
    Git-based storage adapter for templates with proper versioning

    This adapter stores templates as JSON files in a Git repository,
    providing version control, history tracking, and semantic versioning.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Git storage adapter

        Args:
            config: Configuration for the storage adapter
        """
        self.repo_path = Path(config.get("path", "./storage/git/templates"))
        self.remote = config.get("remote", "")
        self.branch = config.get("branch", "main")
        self.push_on_update = config.get("pushOnUpdate", True)

        # Ensure the repository exists
        self._ensure_repository()

    def _ensure_repository(self) -> None:
        """Ensure the Git repository exists and is properly configured"""
        if not (self.repo_path / ".git").exists():
            logger.info(f"Creating Git repository at {self.repo_path}")
            os.makedirs(self.repo_path, exist_ok=True)

            # Initialize the repository
            self._run_git_command("init")
            self._run_git_command("checkout", "-b", self.branch)

            # Create initial structure
            template_dir = self.repo_path / "templates"
            os.makedirs(template_dir, exist_ok=True)

            # Create .gitignore
            with open(self.repo_path / ".gitignore", "w") as f:
                f.write("*.pyc\n__pycache__/\n.DS_Store\n")

            # Initial commit
            self._run_git_command("add", ".")
            self._run_git_command("commit", "-m", "Initial commit")

            # Configure remote if provided
            if self.remote:
                self._run_git_command("remote", "add", "origin", self.remote)

        elif self.remote:
            # Check if remote needs to be updated
            try:
                current_remote = subprocess.check_output(
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.repo_path
                ).decode().strip()

                if current_remote != self.remote:
                    self._run_git_command("remote", "set-url", "origin", self.remote)
            except subprocess.CalledProcessError:
                self._run_git_command("remote", "add", "origin", self.remote)

    def _run_git_command(self, *args) -> Tuple[int, str]:
        """
        Run a Git command

        Args:
            *args: Arguments to pass to Git

        Returns:
            Tuple of (return_code, output)
        """
        try:
            result = subprocess.run(
                ["git"] + list(args),
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode, result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            return e.returncode, e.output

    def _get_template_path(self, template_id: str) -> Path:
        """
        Get the path for a template file

        Args:
            template_id: ID of the template

        Returns:
            Path to the template file
        """
        return self.repo_path / "templates" / f"{template_id}.json"

    def _get_versions_path(self, template_id: str) -> Path:
        """
        Get the path for version history of a template

        Args:
            template_id: ID of the template

        Returns:
            Path to the versions directory
        """
        return self.repo_path / "versions" / template_id

    def _parse_semver(self, version: str) -> Tuple[int, int, int]:
        """
        Parse a semantic version string

        Args:
            version: Version string in semver format (e.g. "1.2.3")

        Returns:
            Tuple of (major, minor, patch) version numbers
        """
        try:
            parts = version.split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return (major, minor, patch)
        except (ValueError, IndexError):
            return (0, 0, 0)

    def _increment_version(
        self,
        current_version: str,
        compatibility: CompatibilityType
    ) -> str:
        """
        Increment version based on compatibility type

        Args:
            current_version: Current version string
            compatibility: Compatibility type that determines version increment

        Returns:
            New version string
        """
        major, minor, patch = self._parse_semver(current_version)

        if compatibility == CompatibilityType.NONE:
            # Breaking change, increment major version
            return f"{major + 1}.0.0"
        elif compatibility == CompatibilityType.BACKWARD:
            # Backward compatible, increment minor version
            return f"{major}.{minor + 1}.0"
        else:
            # Forward compatible or full compatible, increment patch version
            return f"{major}.{minor}.{patch + 1}"

    def load_template(self, template_id: str) -> Optional[TemplateDefinition]:
        """
        Load a template by ID

        Args:
            template_id: ID of the template to load

        Returns:
            TemplateDefinition object or None if not found
        """
        template_path = self._get_template_path(template_id)

        if not template_path.exists():
            return None

        try:
            with open(template_path, "r") as f:
                data = json.load(f)
                return TemplateDefinition.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load template {template_id}: {e}")
            return None

    def load_template_version(
        self,
        template_id: str,
        version: str
    ) -> Optional[TemplateDefinition]:
        """
        Load a specific version of a template

        Args:
            template_id: ID of the template
            version: Version to load

        Returns:
            TemplateDefinition object or None if not found
        """
        versions_path = self._get_versions_path(template_id)
        version_file = versions_path / f"{version}.json"

        if not version_file.exists():
            return None

        try:
            with open(version_file, "r") as f:
                data = json.load(f)
                return TemplateDefinition.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load template {template_id} version {version}: {e}")
            return None

    def save_template(
        self,
        template: TemplateDefinition,
        message: str = "Update template",
        author: str = "system"
    ) -> TemplateVersionInfo:
        """
        Save a template and create a new version

        Args:
            template: Template to save
            message: Commit message
            author: Author of the change

        Returns:
            TemplateVersionInfo with details about the new version
        """
        # Check if this is a new template
        is_new = not self._get_template_path(template.id).exists()

        # If existing, load current version for comparison
        current_template = None
        if not is_new:
            current_template = self.load_template(template.id)

        # Determine new version
        if is_new or not current_template:
            new_version = template.version or "1.0.0"
        else:
            new_version = self._increment_version(
                current_template.version,
                template.compatibility
            )

        # Set the new version on the template
        template.version = new_version
        template.metadata.updated_at = int(time.time())
        template.metadata.updated_by = author

        # Create directory structure if not exists
        os.makedirs(self.repo_path / "templates", exist_ok=True)
        os.makedirs(self._get_versions_path(template.id), exist_ok=True)

        # Save the template as the latest version
        with open(self._get_template_path(template.id), "w") as f:
            json.dump(template.to_dict(), f, indent=2)

        # Save as a versioned copy
        version_path = self._get_versions_path(template.id) / f"{new_version}.json"
        with open(version_path, "w") as f:
            json.dump(template.to_dict(), f, indent=2)

        # Commit the changes
        self._run_git_command("add", str(self._get_template_path(template.id)))
        self._run_git_command("add", str(version_path))

        commit_message = f"{message}\n\nTemplate: {template.name}\nVersion: {new_version}"
        self._run_git_command("commit", "-m", commit_message, "--author", f"{author} <{author}@example.com>")

        # Get the commit hash
        returncode, output = self._run_git_command("rev-parse", "HEAD")
        commit_id = output.strip() if returncode == 0 else "unknown"

        # Push if configured
        if self.push_on_update and self.remote:
            self._run_git_command("push", "origin", self.branch)

        # Generate version info
        changes = []
        if current_template:
            # Compare sections
            current_sections = {s.name for s in current_template.sections}
            new_sections = {s.name for s in template.sections}

            added_sections = new_sections - current_sections
            removed_sections = current_sections - new_sections

            for section in added_sections:
                changes.append(f"Added section: {section}")

            for section in removed_sections:
                changes.append(f"Removed section: {section}")

            # Compare fields in common sections
            for new_section in template.sections:
                if new_section.name in current_sections:
                    current_section = next(
                        (s for s in current_template.sections if s.name == new_section.name),
                        None
                    )

                    if current_section:
                        current_fields = {f.name for f in current_section.fields}
                        new_fields = {f.name for f in new_section.fields}

                        added_fields = new_fields - current_fields
                        removed_fields = current_fields - new_fields

                        for field in added_fields:
                            changes.append(f"Added field: {new_section.name}.{field}")

                        for field in removed_fields:
                            changes.append(f"Removed field: {new_section.name}.{field}")

            # Check for category or compatibility changes
            if current_template.category != template.category:
                changes.append(f"Changed category from {current_template.category} to {template.category}")

            if current_template.compatibility != template.compatibility:
                changes.append(f"Changed compatibility from {current_template.compatibility} to {template.compatibility}")
        else:
            changes.append("Initial template creation")

        version_info = TemplateVersionInfo(
            template_id=template.id,
            version=new_version,
            commit_id=commit_id,
            timestamp=template.metadata.updated_at,
            author=author,
            message=message,
            changes=changes,
            compatibility_type=template.compatibility,
            is_breaking_change=(template.compatibility == CompatibilityType.NONE)
        )

        return version_info

    def list_templates(self) -> List[TemplateDefinition]:
        """
        List all available templates

        Returns:
            List of TemplateDefinition objects
        """
        templates_dir = self.repo_path / "templates"
        if not templates_dir.exists():
            return []

        templates = []
        for template_file in templates_dir.glob("*.json"):
            try:
                with open(template_file, "r") as f:
                    data = json.load(f)
                    template = TemplateDefinition.from_dict(data)
                    templates.append(template)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load template {template_file}: {e}")

        return templates

    def list_template_versions(self, template_id: str) -> List[TemplateVersionInfo]:
        """
        List all versions of a template

        Args:
            template_id: ID of the template

        Returns:
            List of TemplateVersionInfo objects
        """
        versions_path = self._get_versions_path(template_id)
        if not versions_path.exists():
            return []

        version_files = list(versions_path.glob("*.json"))
        if not version_files:
            return []

        # Get git log information for this template
        template_path = self._get_template_path(template_id)
        returncode, output = self._run_git_command(
            "log",
            "--pretty=format:%H|%at|%an|%s",
            "--follow",
            "--",
            str(template_path)
        )

        if returncode != 0:
            logger.error(f"Failed to get git log for {template_id}")
            return []

        # Parse git log
        commit_info = {}
        for line in output.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                commit_hash, timestamp, author, message = parts
                commit_info[commit_hash] = {
                    "timestamp": int(timestamp),
                    "author": author,
                    "message": message
                }

        # Load each version
        versions = []
        for version_file in version_files:
            version = version_file.stem
            try:
                template = self.load_template_version(template_id, version)
                if template:
                    # Try to find the commit for this version
                    returncode, output = self._run_git_command(
                        "log",
                        "-1",
                        "--pretty=format:%H",
                        "--",
                        str(version_file)
                    )

                    commit_id = output.strip() if returncode == 0 else "unknown"
                    commit_data = commit_info.get(commit_id, {
                        "timestamp": template.metadata.updated_at,
                        "author": template.metadata.updated_by,
                        "message": "Update template"
                    })

                    version_info = TemplateVersionInfo(
                        template_id=template_id,
                        version=version,
                        commit_id=commit_id,
                        timestamp=commit_data["timestamp"],
                        author=commit_data["author"],
                        message=commit_data["message"],
                        compatibility_type=template.compatibility,
                        is_breaking_change=(template.compatibility == CompatibilityType.NONE)
                    )
                    versions.append(version_info)
            except Exception as e:
                logger.error(f"Failed to load template version {version_file}: {e}")

        # Sort by version (semver)
        versions.sort(
            key=lambda v: self._parse_semver(v.version),
            reverse=True
        )

        return versions

    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template

        Args:
            template_id: ID of the template to delete

        Returns:
            True if successful, False otherwise
        """
        template_path = self._get_template_path(template_id)
        versions_path = self._get_versions_path(template_id)

        if not template_path.exists():
            return False

        try:
            # Remove the files
            os.remove(template_path)
            if versions_path.exists():
                shutil.rmtree(versions_path)

            # Commit the changes
            self._run_git_command("add", str(template_path))
            if versions_path.exists():
                self._run_git_command("add", str(versions_path))

            self._run_git_command(
                "commit",
                "-m",
                f"Delete template {template_id}"
            )

            # Push if configured
            if self.push_on_update and self.remote:
                self._run_git_command("push", "origin", self.branch)

            return True
        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            return False

    def compare_template_versions(
        self,
        template_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a template

        Args:
            template_id: ID of the template
            version1: First version to compare
            version2: Second version to compare

        Returns:
            Dictionary with differences
        """
        template1 = self.load_template_version(template_id, version1)
        template2 = self.load_template_version(template_id, version2)

        if not template1 or not template2:
            return {"error": "One or both versions not found"}

        # Compare basic attributes
        differences = {
            "name": template1.name != template2.name,
            "description": template1.description != template2.description,
            "category": template1.category != template2.category,
            "compatibility": template1.compatibility != template2.compatibility,
            "sections": {},
            "metadata": {}
        }

        # Compare metadata
        for key in ["tags", "domain", "complexity"]:
            if getattr(template1.metadata, key) != getattr(template2.metadata, key):
                differences["metadata"][key] = {
                    "version1": getattr(template1.metadata, key),
                    "version2": getattr(template2.metadata, key)
                }

        # Compare sections
        sections1 = {s.name: s for s in template1.sections}
        sections2 = {s.name: s for s in template2.sections}

        # Find added and removed sections
        added_sections = set(sections2.keys()) - set(sections1.keys())
        removed_sections = set(sections1.keys()) - set(sections2.keys())
        common_sections = set(sections1.keys()) & set(sections2.keys())

        if added_sections:
            differences["sections"]["added"] = list(added_sections)

        if removed_sections:
            differences["sections"]["removed"] = list(removed_sections)

        # Compare common sections
        differences["sections"]["modified"] = {}
        for section_name in common_sections:
            section1 = sections1[section_name]
            section2 = sections2[section_name]

            # Check for description changes
            if section1.description != section2.description:
                differences["sections"]["modified"][section_name] = {
                    "description_changed": True
                }
            else:
                differences["sections"]["modified"][section_name] = {
                    "description_changed": False
                }

            # Compare fields
            fields1 = {f.name: f for f in section1.fields}
            fields2 = {f.name: f for f in section2.fields}

            added_fields = set(fields2.keys()) - set(fields1.keys())
            removed_fields = set(fields1.keys()) - set(fields2.keys())

            if added_fields:
                differences["sections"]["modified"][section_name]["added_fields"] = list(added_fields)

            if removed_fields:
                differences["sections"]["modified"][section_name]["removed_fields"] = list(removed_fields)

            # Compare common fields
            modified_fields = {}
            for field_name in set(fields1.keys()) & set(fields2.keys()):
                field1 = fields1[field_name]
                field2 = fields2[field_name]

                if (field1.type != field2.type or
                    field1.description != field2.description or
                    field1.required != field2.required or
                    field1.default_value != field2.default_value or
                    field1.options != field2.options):

                    modified_fields[field_name] = {
                        "type_changed": field1.type != field2.type,
                        "description_changed": field1.description != field2.description,
                        "required_changed": field1.required != field2.required,
                        "default_value_changed": field1.default_value != field2.default_value,
                        "options_changed": field1.options != field2.options
                    }

            if modified_fields:
                differences["sections"]["modified"][section_name]["modified_fields"] = modified_fields

        return differences
EOF

cat > ./template-registry/src/adapters/redis_cache.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Redis Cache Adapter

This module provides a Redis-based caching adapter for template registry
to improve performance and reduce load on the underlying storage.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Union

import redis

from models.template_models import (
    TemplateDefinition,
    TemplateVersionInfo,
    TemplateStats
)


logger = logging.getLogger(__name__)


class RedisCacheAdapter:
    """
    Redis-based caching adapter for template registry

    This adapter caches templates, version information, and stats in Redis
    to improve performance and reduce load on the primary storage.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Redis cache adapter

        Args:
            config: Configuration for the cache adapter
        """
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password")
        self.ttl = config.get("ttl", 3600)  # Default TTL: 1 hour
        self.prefix = config.get("prefix", "template-registry:")

        # Connect to Redis
        self.redis = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True
        )

        # Test connection
        try:
            self.redis.ping()
            logger.info("Connected to Redis cache")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _template_key(self, template_id: str) -> str:
        """
        Generate a key for a template

        Args:
            template_id: ID of the template

        Returns:
            Redis key for the template
        """
        return f"{self.prefix}template:{template_id}"

    def _template_version_key(self, template_id: str, version: str) -> str:
        """
        Generate a key for a template version

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            Redis key for the template version
        """
        return f"{self.prefix}template:{template_id}:version:{version}"

    def _template_versions_key(self, template_id: str) -> str:
        """
        Generate a key for template versions list

        Args:
            template_id: ID of the template

        Returns:
            Redis key for the template versions list
        """
        return f"{self.prefix}template:{template_id}:versions"

    def _templates_list_key(self) -> str:
        """
        Generate a key for the templates list

        Returns:
            Redis key for the templates list
        """
        return f"{self.prefix}templates"

    def _template_stats_key(self, template_id: str, version: str = None) -> str:
        """
        Generate a key for template stats

        Args:
            template_id: ID of the template
            version: Optional version of the template

        Returns:
            Redis key for the template stats
        """
        if version:
            return f"{self.prefix}stats:{template_id}:{version}"
        return f"{self.prefix}stats:{template_id}"

    def _categories_key(self) -> str:
        """
        Generate a key for the categories set

        Returns:
            Redis key for the categories set
        """
        return f"{self.prefix}categories"

    def _languages_key(self) -> str:
        """
        Generate a key for the languages set

        Returns:
            Redis key for the languages set
        """
        return f"{self.prefix}languages"

    def _domains_key(self) -> str:
        """
        Generate a key for the domains set

        Returns:
            Redis key for the domains set
        """
        return f"{self.prefix}domains"

    def _tags_key(self) -> str:
        """
        Generate a key for the tags set

        Returns:
            Redis key for the tags set
        """
        return f"{self.prefix}tags"

    def cache_template(self, template: TemplateDefinition) -> bool:
        """
        Cache a template

        Args:
            template: Template to cache

        Returns:
            True if successful, False otherwise
        """
        try:
            # Cache the template
            template_key = self._template_key(template.id)
            self.redis.set(
                template_key,
                template.to_json(),
                ex=self.ttl
            )

            # Cache as a version
            version_key = self._template_version_key(template.id, template.version)
            self.redis.set(
                version_key,
                template.to_json(),
                ex=self.ttl
            )

            # Add to templates list
            self.redis.sadd(self._templates_list_key(), template.id)

            # Add to versions list
            versions_key = self._template_versions_key(template.id)
            self.redis.zadd(
                versions_key,
                {template.version: time.time()},
                nx=True
            )
            self.redis.expire(versions_key, self.ttl)

            # Update category, languages, domain, and tags
            self.redis.sadd(self._categories_key(), template.category)
            for lang in template.languages:
                self.redis.sadd(self._languages_key(), lang)
            self.redis.sadd(self._domains_key(), template.metadata.domain)
            for tag in template.metadata.tags:
                self.redis.sadd(self._tags_key(), tag)

            return True
        except Exception as e:
            logger.error(f"Failed to cache template {template.id}: {e}")
            return False

    def get_cached_template(self, template_id: str) -> Optional[TemplateDefinition]:
        """
        Get a cached template

        Args:
            template_id: ID of the template

        Returns:
            TemplateDefinition object or None if not found
        """
        try:
            template_key = self._template_key(template_id)
            json_data = self.redis.get(template_key)

            if not json_data:
                return None

            return TemplateDefinition.from_json(json_data)
        except Exception as e:
            logger.error(f"Failed to get cached template {template_id}: {e}")
            return None

    def get_cached_template_version(
        self,
        template_id: str,
        version: str
    ) -> Optional[TemplateDefinition]:
        """
        Get a cached template version

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            TemplateDefinition object or None if not found
        """
        try:
            version_key = self._template_version_key(template_id, version)
            json_data = self.redis.get(version_key)

            if not json_data:
                return None

            return TemplateDefinition.from_json(json_data)
        except Exception as e:
            logger.error(f"Failed to get cached template version {template_id}/{version}: {e}")
            return None

    def cache_template_versions(
        self,
        template_id: str,
        versions: List[TemplateVersionInfo]
    ) -> bool:
        """
        Cache template versions

        Args:
            template_id: ID of the template
            versions: List of version info objects

        Returns:
            True if successful, False otherwise
        """
        try:
            versions_key = self._template_versions_key(template_id)

            # Store each version info
            pipe = self.redis.pipeline()

            for version_info in versions:
                # Store version info
                version_info_key = f"{self.prefix}template:{template_id}:version_info:{version_info.version}"
                pipe.set(
                    version_info_key,
                    json.dumps(version_info.to_dict()),
                    ex=self.ttl
                )

                # Add to versions sorted set
                pipe.zadd(
                    versions_key,
                    {version_info.version: version_info.timestamp},
                    nx=True
                )

            pipe.expire(versions_key, self.ttl)
            pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Failed to cache template versions for {template_id}: {e}")
            return False

    def get_cached_template_versions(
        self,
        template_id: str
    ) -> List[TemplateVersionInfo]:
        """
        Get cached template versions

        Args:
            template_id: ID of the template

        Returns:
            List of TemplateVersionInfo objects
        """
        try:
            versions_key = self._template_versions_key(template_id)
            versions = self.redis.zrange(
                versions_key,
                0,
                -1,
                desc=True,
                withscores=False
            )

            if not versions:
                return []

            # Get version info for each version
            result = []
            pipe = self.redis.pipeline()

            for version in versions:
                version_info_key = f"{self.prefix}template:{template_id}:version_info:{version}"
                pipe.get(version_info_key)

            version_info_data = pipe.execute()

            for version, info_data in zip(versions, version_info_data):
                if info_data:
                    try:
                        version_info = TemplateVersionInfo(**json.loads(info_data))
                        result.append(version_info)
                    except Exception as e:
                        logger.error(f"Failed to parse version info for {template_id}/{version}: {e}")

            return result
        except Exception as e:
            logger.error(f"Failed to get cached template versions for {template_id}: {e}")
            return []

    def cache_stats(
        self,
        template_id: str,
        version: str,
        stats: TemplateStats
    ) -> bool:
        """
        Cache template stats

        Args:
            template_id: ID of the template
            version: Version of the template
            stats: Stats to cache

        Returns:
            True if successful, False otherwise
        """
        try:
            stats_key = self._template_stats_key(template_id, version)
            self.redis.set(
                stats_key,
                json.dumps(stats.to_dict()),
                ex=self.ttl
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cache stats for {template_id}/{version}: {e}")
            return False

    def get_cached_stats(
        self,
        template_id: str,
        version: str
    ) -> Optional[TemplateStats]:
        """
        Get cached template stats

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            TemplateStats object or None if not found
        """
        try:
            stats_key = self._template_stats_key(template_id, version)
            json_data = self.redis.get(stats_key)

            if not json_data:
                return None

            return TemplateStats(**json.loads(json_data))
        except Exception as e:
            logger.error(f"Failed to get cached stats for {template_id}/{version}: {e}")
            return None

    def increment_usage_count(self, template_id: str, version: str) -> bool:
        """
        Increment usage count for a template

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            True if successful, False otherwise
        """
        try:
            stats_key = self._template_stats_key(template_id, version)
            usage_key = f"{stats_key}:usage_count"

            # Use HINCRBY for the usage count
            self.redis.hincrby(stats_key, "usage_count", 1)
            self.redis.hset(stats_key, "last_used", int(time.time()))
            self.redis.expire(stats_key, self.ttl)

            return True
        except Exception as e:
            logger.error(f"Failed to increment usage count for {template_id}/{version}: {e}")
            return False

    def record_completion_time(
        self,
        template_id: str,
        version: str,
        time_seconds: float,
        success: bool = True
    ) -> bool:
        """
        Record completion time for a template

        Args:
            template_id: ID of the template
            version: Version of the template
            time_seconds: Completion time in seconds
            success: Whether completion was successful

        Returns:
            True if successful, False otherwise
        """
        try:
            stats_key = self._template_stats_key(template_id, version)

            # Get current stats
            pipe = self.redis.pipeline()
            pipe.hget(stats_key, "completion_count")
            pipe.hget(stats_key, "success_count")
            pipe.hget(stats_key, "total_time")

            completion_count, success_count, total_time = pipe.execute()

            # Update stats
            completion_count = int(completion_count or 0) + 1
            success_count = int(success_count or 0) + (1 if success else 0)
            total_time = float(total_time or 0) + time_seconds

            # Calculate completion rate and average time
            completion_rate = success_count / completion_count if completion_count > 0 else 0
            avg_time = total_time / completion_count if completion_count > 0 else 0

            # Store updated stats
            pipe = self.redis.pipeline()
            pipe.hset(stats_key, "completion_count", completion_count)
            pipe.hset(stats_key, "success_count", success_count)
            pipe.hset(stats_key, "total_time", total_time)
            pipe.hset(stats_key, "completion_rate", completion_rate)
            pipe.hset(stats_key, "avg_completion_time", avg_time)
            pipe.expire(stats_key, self.ttl)
            pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Failed to record completion time for {template_id}/{version}: {e}")
            return False

    def clear_cache(self, template_id: str = None) -> bool:
        """
        Clear cache for a template or all templates

        Args:
            template_id: Optional ID of the template to clear cache for

        Returns:
            True if successful, False otherwise
        """
        try:
            if template_id:
                # Get all versions
                versions_key = self._template_versions_key(template_id)
                versions = self.redis.zrange(versions_key, 0, -1)

                # Delete template and all versions
                keys_to_delete = [
                    self._template_key(template_id),
                    versions_key
                ]

                for version in versions:
                    keys_to_delete.append(self._template_version_key(template_id, version))
                    keys_to_delete.append(self._template_stats_key(template_id, version))
                    keys_to_delete.append(f"{self.prefix}template:{template_id}:version_info:{version}")

                if keys_to_delete:
                    self.redis.delete(*keys_to_delete)
            else:
                # Clear all template cache
                pattern = f"{self.prefix}template:*"
                cursor = 0
                while True:
                    cursor, keys = self.redis.scan(cursor, pattern, 100)
                    if keys:
                        self.redis.delete(*keys)
                    if cursor == 0:
                        break

                # Clear all stats cache
                pattern = f"{self.prefix}stats:*"
                cursor = 0
                while True:
                    cursor, keys = self.redis.scan(cursor, pattern, 100)
                    if keys:
                        self.redis.delete(*keys)
                    if cursor == 0:
                        break

            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        try:
            # Get number of templates
            templates_count = self.redis.scard(self._templates_list_key())

            # Get memory usage
            memory_stats = self.redis.info("memory")

            # Get key count with prefix
            pattern = f"{self.prefix}*"
            key_count = 0
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, pattern, 100)
                key_count += len(keys)
                if cursor == 0:
                    break

            return {
                "templates_count": templates_count,
                "key_count": key_count,
                "memory_usage": memory_stats.get("used_memory_human", "unknown"),
                "peak_memory": memory_stats.get("used_memory_peak_human", "unknown"),
                "redis_version": self.redis.info().get("redis_version", "unknown")
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
EOF

cat > ./template-registry/src/adapters/pulsar_schema_registry.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pulsar Schema Registry Adapter

This module provides an adapter for Apache Pulsar Schema Registry.
It allows storing template schemas in Pulsar's schema registry for
validation and version management.
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple

import pulsar
from pulsar.schema import (
    JsonSchema,
    Schema,
    AvroSchema,
    KeyValueSchema,
    Record,
    String,
    Integer,
    Float,
    Boolean,
    Array
)
import requests

from models.template_models import (
    TemplateDefinition,
    CompatibilityType,
    TemplateCategory,
    TemplateLanguage
)


logger = logging.getLogger(__name__)


class PulsarSchemaRegistryAdapter:
    """
    Apache Pulsar Schema Registry adapter for template schemas

    This adapter stores template schemas in Pulsar's built-in schema registry
    for validation, version management, and compatibility checks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Pulsar schema registry adapter

        Args:
            config: Configuration for the schema registry adapter
        """
        self.service_url = config.get("serviceUrl", "pulsar://localhost:6650")
        self.admin_url = self.service_url.replace("pulsar://", "http://").replace("6650", "8080")
        self.tenant = config.get("tenant", "public")
        self.namespace = config.get("namespace", "template-registry")
        self.topic_prefix = config.get("topic", "schemas")

        # Create a Pulsar client
        self.client = pulsar.Client(self.service_url)

        # Ensure namespace exists
        self._ensure_namespace()

    def _ensure_namespace(self) -> None:
        """Ensure the namespace exists"""
        try:
            # Check if namespace exists
            response = requests.get(
                f"{self.admin_url}/admin/v2/namespaces/{self.tenant}/{self.namespace}"
            )

            if response.status_code == 404:
                # Create namespace
                requests.put(
                    f"{self.admin_url}/admin/v2/namespaces/{self.tenant}/{self.namespace}"
                )

                # Set schema compatibility policy
                requests.put(
                    f"{self.admin_url}/admin/v2/namespaces/{self.tenant}/{self.namespace}/schemaCompatibilityStrategy",
                    json="BACKWARD"
                )

                logger.info(f"Created namespace {self.tenant}/{self.namespace}")
        except Exception as e:
            logger.error(f"Failed to ensure namespace: {e}")

    def _get_topic_name(self, template_id: str) -> str:
        """
        Get the Pulsar topic name for a template

        Args:
            template_id: ID of the template

        Returns:
            Pulsar topic name
        """
        return f"persistent://{self.tenant}/{self.namespace}/{self.topic_prefix}-{template_id}"

    def _get_schema_info(self, topic: str) -> Dict[str, Any]:
        """
        Get schema info from Pulsar schema registry

        Args:
            topic: Pulsar topic name

        Returns:
            Schema info dictionary
        """
        try:
            response = requests.get(
                f"{self.admin_url}/admin/v2/schemas/{topic}/schema"
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get schema info: {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {}

    def _get_schema_versions(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get schema versions from Pulsar schema registry

        Args:
            topic: Pulsar topic name

        Returns:
            List of schema version dictionaries
        """
        try:
            response = requests.get(
                f"{self.admin_url}/admin/v2/schemas/{topic}/versions"
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get schema versions: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Failed to get schema versions: {e}")
            return []

    def _upload_schema(
        self,
        topic
                if not re.match(email_pattern, str(value)):
                    return rule.error_message or f"Field {section_name}.{field_name} must be a valid email"

            elif rule.expression == "url":
                import re
                url_pattern = r'^(http|https)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?#!/bin/bash
#============================================================================
# Template Registry System for Spec-Driven AI Code Generation Platform
#
# This script sets up a template registry system that handles storage,
# versioning, categorization, and validation of specification templates.
# The system leverages Apache Pulsar for event-driven architecture,
# Redis for high-performance caching, and Git for version control.
#
# Author: Claude
# Date: 2025-04-18
#============================================================================

set -e
echo "Setting up Template Registry System..."

# Create directory structure
mkdir -p ./template-registry/{src,models,storage,schemas,utils,tests,config}
mkdir -p ./template-registry/src/{core,event_handlers,validators,adapters}
mkdir -p ./template-registry/storage/{git,cache}
mkdir -p ./template-registry/schemas/{templates,definitions}

# Configuration
cat > ./template-registry/config/config.json << 'EOF'
{
    "storage": {
        "type": "hybrid",
        "gitRepository": {
            "path": "./storage/git/templates",
            "remote": "",
            "branch": "main",
            "pushOnUpdate": true
        },
        "cache": {
            "type": "redis",
            "host": "localhost",
            "port": 6379,
            "ttl": 3600,
            "prefix": "template-registry:"
        },
        "schemaStore": {
            "type": "pulsar",
            "serviceUrl": "pulsar://localhost:6650",
            "tenant": "public",
            "namespace": "template-registry",
            "topic": "schemas"
        }
    },
    "eventBus": {
        "serviceUrl": "pulsar://localhost:6650",
        "tenant": "public",
        "namespace": "template-registry",
        "subscriptionName": "template-registry-service",
        "responseTopicPrefix": "template-registry-response",
        "eventTopics": {
            "templateCreate": "template-registry-create",
            "templateUpdate": "template-registry-update",
            "templateGet": "template-registry-get",
            "templateList": "template-registry-list",
            "templateDelete": "template-registry-delete",
            "templateValidate": "template-registry-validate",
            "templateCompare": "template-registry-compare",
            "templateStats": "template-registry-stats"
        }
    },
    "metrics": {
        "enabled": true,
        "statsdHost": "localhost",
        "statsdPort": 8125,
        "prefix": "template_registry."
    },
    "validation": {
        "strictMode": true,
        "allowSchemaEvolution": true,
        "compatibilityStrategy": "BACKWARD"
    }
}
EOF

# Create the core models for template data structures
cat > ./template-registry/models/template_models.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry Models

This module defines the core data models for the template registry system.
It includes classes for templates, fields, sections, validation rules, and more.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set


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
    REFERENCE = "reference"
    DATETIME = "datetime"


class ValidationRuleType(str, Enum):
    """Types of validation rules"""
    REGEX = "regex"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    REQUIRED = "required"
    ENUM_VALUES = "enum_values"
    CUSTOM = "custom"
    DEPENDENCY = "dependency"
    FORMAT = "format"


class TemplateCategory(str, Enum):
    """Categories for templates"""
    API = "api"
    DATABASE = "database"
    UI = "ui"
    WORKFLOW = "workflow"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    INFRASTRUCTURE = "infrastructure"
    OTHER = "other"


class TemplateLanguage(str, Enum):
    """Programming languages for templates"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    ANY = "any"


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
    error_message: str


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
            FieldType.REFERENCE: "string",
            FieldType.DATETIME: "string"
        }
        return mapping.get(self.type, "string")


@dataclass
class SectionDefinition:
    """Definition of a section in a template"""
    name: str
    description: str
    fields: List[FieldDefinition] = field(default_factory=list)


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


@dataclass
class TemplateDefinition:
    """Definition of a template"""
    id: str
    name: str
    description: str
    version: str
    category: TemplateCategory
    languages: List[TemplateLanguage] = field(default_factory=lambda: [TemplateLanguage.ANY])
    sections: List[SectionDefinition] = field(default_factory=list)
    metadata: TemplateMetadata = field(default_factory=TemplateMetadata)
    compatibility: CompatibilityType = CompatibilityType.BACKWARD

    def __post_init__(self):
        """Validate after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateDefinition':
        """Create from dictionary"""
        sections = []
        for section_data in data.get("sections", []):
            fields = []
            for field_data in section_data.get("fields", []):
                validation_rules = []
                for rule_data in field_data.get("validation_rules", []):
                    validation_rules.append(ValidationRule(
                        rule_type=ValidationRuleType(rule_data["rule_type"]),
                        expression=rule_data["expression"],
                        error_message=rule_data["error_message"]
                    ))

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
                        nested_fields=[]
                    ))

                fields.append(FieldDefinition(
                    name=field_data["name"],
                    type=FieldType(field_data["type"]),
                    description=field_data["description"],
                    required=field_data.get("required", False),
                    default_value=field_data.get("default_value"),
                    validation_rules=validation_rules,
                    options=field_data.get("options", []),
                    nested_fields=nested_fields
                ))

            sections.append(SectionDefinition(
                name=section_data["name"],
                description=section_data["description"],
                fields=fields
            ))

        metadata_data = data.get("metadata", {})
        metadata = TemplateMetadata(
            created_at=metadata_data.get("created_at", int(time.time())),
            created_by=metadata_data.get("created_by", "system"),
            updated_at=metadata_data.get("updated_at", int(time.time())),
            updated_by=metadata_data.get("updated_by", "system"),
            tags=metadata_data.get("tags", []),
            domain=metadata_data.get("domain", "general"),
            complexity=metadata_data.get("complexity", "medium"),
            estimated_completion_time=metadata_data.get("estimated_completion_time", 0)
        )

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data["version"],
            category=TemplateCategory(data["category"]),
            languages=[TemplateLanguage(lang) for lang in data.get("languages", ["any"])],
            sections=sections,
            metadata=metadata,
            compatibility=CompatibilityType(data.get("compatibility", "backward"))
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateDefinition':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class FieldValue:
    """Value of a field"""
    name: str
    value: Any


@dataclass
class SectionValues:
    """Values for a section"""
    name: str
    fields: List[FieldValue] = field(default_factory=list)


@dataclass
class TemplateInstance:
    """Filled template instance"""
    id: str
    template_id: str
    template_version: str
    project_id: str
    name: str
    sections: List[SectionValues] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    completed: bool = False
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateInstance':
        """Create from dictionary"""
        sections = []
        for section_data in data.get("sections", []):
            fields = []
            for field_data in section_data.get("fields", []):
                fields.append(FieldValue(
                    name=field_data["name"],
                    value=field_data["value"]
                ))

            sections.append(SectionValues(
                name=section_data["name"],
                fields=fields
            ))

        return cls(
            id=data["id"],
            template_id=data["template_id"],
            template_version=data["template_version"],
            project_id=data["project_id"],
            name=data["name"],
            sections=sections,
            created_at=data.get("created_at", int(time.time())),
            updated_at=data.get("updated_at", int(time.time())),
            completed=data.get("completed", False),
            validated=data.get("validated", False),
            validation_errors=data.get("validation_errors", [])
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateInstance':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TemplateDependency:
    """Dependency between templates"""
    source_id: str
    target_id: str
    dependency_type: str
    description: str


@dataclass
class TemplateVersionInfo:
    """Information about a template version"""
    template_id: str
    version: str
    commit_id: str
    timestamp: int
    author: str
    message: str
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
class TemplateStats:
    """Statistics for a template"""
    template_id: str
    version: str
    usage_count: int = 0
    completion_rate: float = 0.0
    avg_completion_time: float = 0.0
    last_used: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TemplateSearchQuery:
    """Query for searching templates"""
    keywords: Optional[str] = None
    categories: List[TemplateCategory] = field(default_factory=list)
    languages: List[TemplateLanguage] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    complexity: Optional[str] = None
    sort_by: str = "updated_at"
    sort_order: str = "desc"
    limit: int = 20
    offset: int = 0
EOF

# Create storage adapters
cat > ./template-registry/src/adapters/git_storage.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Git Storage Adapter

This module provides a Git-based storage adapter for template versioning.
It commits changes to templates into a Git repository for version control,
history tracking, and collaboration.
"""

import os
import json
import time
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from models.template_models import (
    TemplateDefinition,
    TemplateVersionInfo,
    CompatibilityType
)


logger = logging.getLogger(__name__)


class GitStorageAdapter:
    """
    Git-based storage adapter for templates with proper versioning

    This adapter stores templates as JSON files in a Git repository,
    providing version control, history tracking, and semantic versioning.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Git storage adapter

        Args:
            config: Configuration for the storage adapter
        """
        self.repo_path = Path(config.get("path", "./storage/git/templates"))
        self.remote = config.get("remote", "")
        self.branch = config.get("branch", "main")
        self.push_on_update = config.get("pushOnUpdate", True)

        # Ensure the repository exists
        self._ensure_repository()

    def _ensure_repository(self) -> None:
        """Ensure the Git repository exists and is properly configured"""
        if not (self.repo_path / ".git").exists():
            logger.info(f"Creating Git repository at {self.repo_path}")
            os.makedirs(self.repo_path, exist_ok=True)

            # Initialize the repository
            self._run_git_command("init")
            self._run_git_command("checkout", "-b", self.branch)

            # Create initial structure
            template_dir = self.repo_path / "templates"
            os.makedirs(template_dir, exist_ok=True)

            # Create .gitignore
            with open(self.repo_path / ".gitignore", "w") as f:
                f.write("*.pyc\n__pycache__/\n.DS_Store\n")

            # Initial commit
            self._run_git_command("add", ".")
            self._run_git_command("commit", "-m", "Initial commit")

            # Configure remote if provided
            if self.remote:
                self._run_git_command("remote", "add", "origin", self.remote)

        elif self.remote:
            # Check if remote needs to be updated
            try:
                current_remote = subprocess.check_output(
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.repo_path
                ).decode().strip()

                if current_remote != self.remote:
                    self._run_git_command("remote", "set-url", "origin", self.remote)
            except subprocess.CalledProcessError:
                self._run_git_command("remote", "add", "origin", self.remote)

    def _run_git_command(self, *args) -> Tuple[int, str]:
        """
        Run a Git command

        Args:
            *args: Arguments to pass to Git

        Returns:
            Tuple of (return_code, output)
        """
        try:
            result = subprocess.run(
                ["git"] + list(args),
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode, result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            return e.returncode, e.output

    def _get_template_path(self, template_id: str) -> Path:
        """
        Get the path for a template file

        Args:
            template_id: ID of the template

        Returns:
            Path to the template file
        """
        return self.repo_path / "templates" / f"{template_id}.json"

    def _get_versions_path(self, template_id: str) -> Path:
        """
        Get the path for version history of a template

        Args:
            template_id: ID of the template

        Returns:
            Path to the versions directory
        """
        return self.repo_path / "versions" / template_id

    def _parse_semver(self, version: str) -> Tuple[int, int, int]:
        """
        Parse a semantic version string

        Args:
            version: Version string in semver format (e.g. "1.2.3")

        Returns:
            Tuple of (major, minor, patch) version numbers
        """
        try:
            parts = version.split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return (major, minor, patch)
        except (ValueError, IndexError):
            return (0, 0, 0)

    def _increment_version(
        self,
        current_version: str,
        compatibility: CompatibilityType
    ) -> str:
        """
        Increment version based on compatibility type

        Args:
            current_version: Current version string
            compatibility: Compatibility type that determines version increment

        Returns:
            New version string
        """
        major, minor, patch = self._parse_semver(current_version)

        if compatibility == CompatibilityType.NONE:
            # Breaking change, increment major version
            return f"{major + 1}.0.0"
        elif compatibility == CompatibilityType.BACKWARD:
            # Backward compatible, increment minor version
            return f"{major}.{minor + 1}.0"
        else:
            # Forward compatible or full compatible, increment patch version
            return f"{major}.{minor}.{patch + 1}"

    def load_template(self, template_id: str) -> Optional[TemplateDefinition]:
        """
        Load a template by ID

        Args:
            template_id: ID of the template to load

        Returns:
            TemplateDefinition object or None if not found
        """
        template_path = self._get_template_path(template_id)

        if not template_path.exists():
            return None

        try:
            with open(template_path, "r") as f:
                data = json.load(f)
                return TemplateDefinition.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load template {template_id}: {e}")
            return None

    def load_template_version(
        self,
        template_id: str,
        version: str
    ) -> Optional[TemplateDefinition]:
        """
        Load a specific version of a template

        Args:
            template_id: ID of the template
            version: Version to load

        Returns:
            TemplateDefinition object or None if not found
        """
        versions_path = self._get_versions_path(template_id)
        version_file = versions_path / f"{version}.json"

        if not version_file.exists():
            return None

        try:
            with open(version_file, "r") as f:
                data = json.load(f)
                return TemplateDefinition.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load template {template_id} version {version}: {e}")
            return None

    def save_template(
        self,
        template: TemplateDefinition,
        message: str = "Update template",
        author: str = "system"
    ) -> TemplateVersionInfo:
        """
        Save a template and create a new version

        Args:
            template: Template to save
            message: Commit message
            author: Author of the change

        Returns:
            TemplateVersionInfo with details about the new version
        """
        # Check if this is a new template
        is_new = not self._get_template_path(template.id).exists()

        # If existing, load current version for comparison
        current_template = None
        if not is_new:
            current_template = self.load_template(template.id)

        # Determine new version
        if is_new or not current_template:
            new_version = template.version or "1.0.0"
        else:
            new_version = self._increment_version(
                current_template.version,
                template.compatibility
            )

        # Set the new version on the template
        template.version = new_version
        template.metadata.updated_at = int(time.time())
        template.metadata.updated_by = author

        # Create directory structure if not exists
        os.makedirs(self.repo_path / "templates", exist_ok=True)
        os.makedirs(self._get_versions_path(template.id), exist_ok=True)

        # Save the template as the latest version
        with open(self._get_template_path(template.id), "w") as f:
            json.dump(template.to_dict(), f, indent=2)

        # Save as a versioned copy
        version_path = self._get_versions_path(template.id) / f"{new_version}.json"
        with open(version_path, "w") as f:
            json.dump(template.to_dict(), f, indent=2)

        # Commit the changes
        self._run_git_command("add", str(self._get_template_path(template.id)))
        self._run_git_command("add", str(version_path))

        commit_message = f"{message}\n\nTemplate: {template.name}\nVersion: {new_version}"
        self._run_git_command("commit", "-m", commit_message, "--author", f"{author} <{author}@example.com>")

        # Get the commit hash
        returncode, output = self._run_git_command("rev-parse", "HEAD")
        commit_id = output.strip() if returncode == 0 else "unknown"

        # Push if configured
        if self.push_on_update and self.remote:
            self._run_git_command("push", "origin", self.branch)

        # Generate version info
        changes = []
        if current_template:
            # Compare sections
            current_sections = {s.name for s in current_template.sections}
            new_sections = {s.name for s in template.sections}

            added_sections = new_sections - current_sections
            removed_sections = current_sections - new_sections

            for section in added_sections:
                changes.append(f"Added section: {section}")

            for section in removed_sections:
                changes.append(f"Removed section: {section}")

            # Compare fields in common sections
            for new_section in template.sections:
                if new_section.name in current_sections:
                    current_section = next(
                        (s for s in current_template.sections if s.name == new_section.name),
                        None
                    )

                    if current_section:
                        current_fields = {f.name for f in current_section.fields}
                        new_fields = {f.name for f in new_section.fields}

                        added_fields = new_fields - current_fields
                        removed_fields = current_fields - new_fields

                        for field in added_fields:
                            changes.append(f"Added field: {new_section.name}.{field}")

                        for field in removed_fields:
                            changes.append(f"Removed field: {new_section.name}.{field}")

            # Check for category or compatibility changes
            if current_template.category != template.category:
                changes.append(f"Changed category from {current_template.category} to {template.category}")

            if current_template.compatibility != template.compatibility:
                changes.append(f"Changed compatibility from {current_template.compatibility} to {template.compatibility}")
        else:
            changes.append("Initial template creation")

        version_info = TemplateVersionInfo(
            template_id=template.id,
            version=new_version,
            commit_id=commit_id,
            timestamp=template.metadata.updated_at,
            author=author,
            message=message,
            changes=changes,
            compatibility_type=template.compatibility,
            is_breaking_change=(template.compatibility == CompatibilityType.NONE)
        )

        return version_info

    def list_templates(self) -> List[TemplateDefinition]:
        """
        List all available templates

        Returns:
            List of TemplateDefinition objects
        """
        templates_dir = self.repo_path / "templates"
        if not templates_dir.exists():
            return []

        templates = []
        for template_file in templates_dir.glob("*.json"):
            try:
                with open(template_file, "r") as f:
                    data = json.load(f)
                    template = TemplateDefinition.from_dict(data)
                    templates.append(template)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load template {template_file}: {e}")

        return templates

    def list_template_versions(self, template_id: str) -> List[TemplateVersionInfo]:
        """
        List all versions of a template

        Args:
            template_id: ID of the template

        Returns:
            List of TemplateVersionInfo objects
        """
        versions_path = self._get_versions_path(template_id)
        if not versions_path.exists():
            return []

        version_files = list(versions_path.glob("*.json"))
        if not version_files:
            return []

        # Get git log information for this template
        template_path = self._get_template_path(template_id)
        returncode, output = self._run_git_command(
            "log",
            "--pretty=format:%H|%at|%an|%s",
            "--follow",
            "--",
            str(template_path)
        )

        if returncode != 0:
            logger.error(f"Failed to get git log for {template_id}")
            return []

        # Parse git log
        commit_info = {}
        for line in output.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                commit_hash, timestamp, author, message = parts
                commit_info[commit_hash] = {
                    "timestamp": int(timestamp),
                    "author": author,
                    "message": message
                }

        # Load each version
        versions = []
        for version_file in version_files:
            version = version_file.stem
            try:
                template = self.load_template_version(template_id, version)
                if template:
                    # Try to find the commit for this version
                    returncode, output = self._run_git_command(
                        "log",
                        "-1",
                        "--pretty=format:%H",
                        "--",
                        str(version_file)
                    )

                    commit_id = output.strip() if returncode == 0 else "unknown"
                    commit_data = commit_info.get(commit_id, {
                        "timestamp": template.metadata.updated_at,
                        "author": template.metadata.updated_by,
                        "message": "Update template"
                    })

                    version_info = TemplateVersionInfo(
                        template_id=template_id,
                        version=version,
                        commit_id=commit_id,
                        timestamp=commit_data["timestamp"],
                        author=commit_data["author"],
                        message=commit_data["message"],
                        compatibility_type=template.compatibility,
                        is_breaking_change=(template.compatibility == CompatibilityType.NONE)
                    )
                    versions.append(version_info)
            except Exception as e:
                logger.error(f"Failed to load template version {version_file}: {e}")

        # Sort by version (semver)
        versions.sort(
            key=lambda v: self._parse_semver(v.version),
            reverse=True
        )

        return versions

    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template

        Args:
            template_id: ID of the template to delete

        Returns:
            True if successful, False otherwise
        """
        template_path = self._get_template_path(template_id)
        versions_path = self._get_versions_path(template_id)

        if not template_path.exists():
            return False

        try:
            # Remove the files
            os.remove(template_path)
            if versions_path.exists():
                shutil.rmtree(versions_path)

            # Commit the changes
            self._run_git_command("add", str(template_path))
            if versions_path.exists():
                self._run_git_command("add", str(versions_path))

            self._run_git_command(
                "commit",
                "-m",
                f"Delete template {template_id}"
            )

            # Push if configured
            if self.push_on_update and self.remote:
                self._run_git_command("push", "origin", self.branch)

            return True
        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            return False

    def compare_template_versions(
        self,
        template_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a template

        Args:
            template_id: ID of the template
            version1: First version to compare
            version2: Second version to compare

        Returns:
            Dictionary with differences
        """
        template1 = self.load_template_version(template_id, version1)
        template2 = self.load_template_version(template_id, version2)

        if not template1 or not template2:
            return {"error": "One or both versions not found"}

        # Compare basic attributes
        differences = {
            "name": template1.name != template2.name,
            "description": template1.description != template2.description,
            "category": template1.category != template2.category,
            "compatibility": template1.compatibility != template2.compatibility,
            "sections": {},
            "metadata": {}
        }

        # Compare metadata
        for key in ["tags", "domain", "complexity"]:
            if getattr(template1.metadata, key) != getattr(template2.metadata, key):
                differences["metadata"][key] = {
                    "version1": getattr(template1.metadata, key),
                    "version2": getattr(template2.metadata, key)
                }

        # Compare sections
        sections1 = {s.name: s for s in template1.sections}
        sections2 = {s.name: s for s in template2.sections}

        # Find added and removed sections
        added_sections = set(sections2.keys()) - set(sections1.keys())
        removed_sections = set(sections1.keys()) - set(sections2.keys())
        common_sections = set(sections1.keys()) & set(sections2.keys())

        if added_sections:
            differences["sections"]["added"] = list(added_sections)

        if removed_sections:
            differences["sections"]["removed"] = list(removed_sections)

        # Compare common sections
        differences["sections"]["modified"] = {}
        for section_name in common_sections:
            section1 = sections1[section_name]
            section2 = sections2[section_name]

            # Check for description changes
            if section1.description != section2.description:
                differences["sections"]["modified"][section_name] = {
                    "description_changed": True
                }
            else:
                differences["sections"]["modified"][section_name] = {
                    "description_changed": False
                }

            # Compare fields
            fields1 = {f.name: f for f in section1.fields}
            fields2 = {f.name: f for f in section2.fields}

            added_fields = set(fields2.keys()) - set(fields1.keys())
            removed_fields = set(fields1.keys()) - set(fields2.keys())

            if added_fields:
                differences["sections"]["modified"][section_name]["added_fields"] = list(added_fields)

            if removed_fields:
                differences["sections"]["modified"][section_name]["removed_fields"] = list(removed_fields)

            # Compare common fields
            modified_fields = {}
            for field_name in set(fields1.keys()) & set(fields2.keys()):
                field1 = fields1[field_name]
                field2 = fields2[field_name]

                if (field1.type != field2.type or
                    field1.description != field2.description or
                    field1.required != field2.required or
                    field1.default_value != field2.default_value or
                    field1.options != field2.options):

                    modified_fields[field_name] = {
                        "type_changed": field1.type != field2.type,
                        "description_changed": field1.description != field2.description,
                        "required_changed": field1.required != field2.required,
                        "default_value_changed": field1.default_value != field2.default_value,
                        "options_changed": field1.options != field2.options
                    }

            if modified_fields:
                differences["sections"]["modified"][section_name]["modified_fields"] = modified_fields

        return differences
EOF

cat > ./template-registry/src/adapters/redis_cache.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Redis Cache Adapter

This module provides a Redis-based caching adapter for template registry
to improve performance and reduce load on the underlying storage.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Union

import redis

from models.template_models import (
    TemplateDefinition,
    TemplateVersionInfo,
    TemplateStats
)


logger = logging.getLogger(__name__)


class RedisCacheAdapter:
    """
    Redis-based caching adapter for template registry

    This adapter caches templates, version information, and stats in Redis
    to improve performance and reduce load on the primary storage.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Redis cache adapter

        Args:
            config: Configuration for the cache adapter
        """
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password")
        self.ttl = config.get("ttl", 3600)  # Default TTL: 1 hour
        self.prefix = config.get("prefix", "template-registry:")

        # Connect to Redis
        self.redis = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True
        )

        # Test connection
        try:
            self.redis.ping()
            logger.info("Connected to Redis cache")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _template_key(self, template_id: str) -> str:
        """
        Generate a key for a template

        Args:
            template_id: ID of the template

        Returns:
            Redis key for the template
        """
        return f"{self.prefix}template:{template_id}"

    def _template_version_key(self, template_id: str, version: str) -> str:
        """
        Generate a key for a template version

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            Redis key for the template version
        """
        return f"{self.prefix}template:{template_id}:version:{version}"

    def _template_versions_key(self, template_id: str) -> str:
        """
        Generate a key for template versions list

        Args:
            template_id: ID of the template

        Returns:
            Redis key for the template versions list
        """
        return f"{self.prefix}template:{template_id}:versions"

    def _templates_list_key(self) -> str:
        """
        Generate a key for the templates list

        Returns:
            Redis key for the templates list
        """
        return f"{self.prefix}templates"

    def _template_stats_key(self, template_id: str, version: str = None) -> str:
        """
        Generate a key for template stats

        Args:
            template_id: ID of the template
            version: Optional version of the template

        Returns:
            Redis key for the template stats
        """
        if version:
            return f"{self.prefix}stats:{template_id}:{version}"
        return f"{self.prefix}stats:{template_id}"

    def _categories_key(self) -> str:
        """
        Generate a key for the categories set

        Returns:
            Redis key for the categories set
        """
        return f"{self.prefix}categories"

    def _languages_key(self) -> str:
        """
        Generate a key for the languages set

        Returns:
            Redis key for the languages set
        """
        return f"{self.prefix}languages"

    def _domains_key(self) -> str:
        """
        Generate a key for the domains set

        Returns:
            Redis key for the domains set
        """
        return f"{self.prefix}domains"

    def _tags_key(self) -> str:
        """
        Generate a key for the tags set

        Returns:
            Redis key for the tags set
        """
        return f"{self.prefix}tags"

    def cache_template(self, template: TemplateDefinition) -> bool:
        """
        Cache a template

        Args:
            template: Template to cache

        Returns:
            True if successful, False otherwise
        """
        try:
            # Cache the template
            template_key = self._template_key(template.id)
            self.redis.set(
                template_key,
                template.to_json(),
                ex=self.ttl
            )

            # Cache as a version
            version_key = self._template_version_key(template.id, template.version)
            self.redis.set(
                version_key,
                template.to_json(),
                ex=self.ttl
            )

            # Add to templates list
            self.redis.sadd(self._templates_list_key(), template.id)

            # Add to versions list
            versions_key = self._template_versions_key(template.id)
            self.redis.zadd(
                versions_key,
                {template.version: time.time()},
                nx=True
            )
            self.redis.expire(versions_key, self.ttl)

            # Update category, languages, domain, and tags
            self.redis.sadd(self._categories_key(), template.category)
            for lang in template.languages:
                self.redis.sadd(self._languages_key(), lang)
            self.redis.sadd(self._domains_key(), template.metadata.domain)
            for tag in template.metadata.tags:
                self.redis.sadd(self._tags_key(), tag)

            return True
        except Exception as e:
            logger.error(f"Failed to cache template {template.id}: {e}")
            return False

    def get_cached_template(self, template_id: str) -> Optional[TemplateDefinition]:
        """
        Get a cached template

        Args:
            template_id: ID of the template

        Returns:
            TemplateDefinition object or None if not found
        """
        try:
            template_key = self._template_key(template_id)
            json_data = self.redis.get(template_key)

            if not json_data:
                return None

            return TemplateDefinition.from_json(json_data)
        except Exception as e:
            logger.error(f"Failed to get cached template {template_id}: {e}")
            return None

    def get_cached_template_version(
        self,
        template_id: str,
        version: str
    ) -> Optional[TemplateDefinition]:
        """
        Get a cached template version

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            TemplateDefinition object or None if not found
        """
        try:
            version_key = self._template_version_key(template_id, version)
            json_data = self.redis.get(version_key)

            if not json_data:
                return None

            return TemplateDefinition.from_json(json_data)
        except Exception as e:
            logger.error(f"Failed to get cached template version {template_id}/{version}: {e}")
            return None

    def cache_template_versions(
        self,
        template_id: str,
        versions: List[TemplateVersionInfo]
    ) -> bool:
        """
        Cache template versions

        Args:
            template_id: ID of the template
            versions: List of version info objects

        Returns:
            True if successful, False otherwise
        """
        try:
            versions_key = self._template_versions_key(template_id)

            # Store each version info
            pipe = self.redis.pipeline()

            for version_info in versions:
                # Store version info
                version_info_key = f"{self.prefix}template:{template_id}:version_info:{version_info.version}"
                pipe.set(
                    version_info_key,
                    json.dumps(version_info.to_dict()),
                    ex=self.ttl
                )

                # Add to versions sorted set
                pipe.zadd(
                    versions_key,
                    {version_info.version: version_info.timestamp},
                    nx=True
                )

            pipe.expire(versions_key, self.ttl)
            pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Failed to cache template versions for {template_id}: {e}")
            return False

    def get_cached_template_versions(
        self,
        template_id: str
    ) -> List[TemplateVersionInfo]:
        """
        Get cached template versions

        Args:
            template_id: ID of the template

        Returns:
            List of TemplateVersionInfo objects
        """
        try:
            versions_key = self._template_versions_key(template_id)
            versions = self.redis.zrange(
                versions_key,
                0,
                -1,
                desc=True,
                withscores=False
            )

            if not versions:
                return []

            # Get version info for each version
            result = []
            pipe = self.redis.pipeline()

            for version in versions:
                version_info_key = f"{self.prefix}template:{template_id}:version_info:{version}"
                pipe.get(version_info_key)

            version_info_data = pipe.execute()

            for version, info_data in zip(versions, version_info_data):
                if info_data:
                    try:
                        version_info = TemplateVersionInfo(**json.loads(info_data))
                        result.append(version_info)
                    except Exception as e:
                        logger.error(f"Failed to parse version info for {template_id}/{version}: {e}")

            return result
        except Exception as e:
            logger.error(f"Failed to get cached template versions for {template_id}: {e}")
            return []

    def cache_stats(
        self,
        template_id: str,
        version: str,
        stats: TemplateStats
    ) -> bool:
        """
        Cache template stats

        Args:
            template_id: ID of the template
            version: Version of the template
            stats: Stats to cache

        Returns:
            True if successful, False otherwise
        """
        try:
            stats_key = self._template_stats_key(template_id, version)
            self.redis.set(
                stats_key,
                json.dumps(stats.to_dict()),
                ex=self.ttl
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cache stats for {template_id}/{version}: {e}")
            return False

    def get_cached_stats(
        self,
        template_id: str,
        version: str
    ) -> Optional[TemplateStats]:
        """
        Get cached template stats

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            TemplateStats object or None if not found
        """
        try:
            stats_key = self._template_stats_key(template_id, version)
            json_data = self.redis.get(stats_key)

            if not json_data:
                return None

            return TemplateStats(**json.loads(json_data))
        except Exception as e:
            logger.error(f"Failed to get cached stats for {template_id}/{version}: {e}")
            return None

    def increment_usage_count(self, template_id: str, version: str) -> bool:
        """
        Increment usage count for a template

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            True if successful, False otherwise
        """
        try:
            stats_key = self._template_stats_key(template_id, version)
            usage_key = f"{stats_key}:usage_count"

            # Use HINCRBY for the usage count
            self.redis.hincrby(stats_key, "usage_count", 1)
            self.redis.hset(stats_key, "last_used", int(time.time()))
            self.redis.expire(stats_key, self.ttl)

            return True
        except Exception as e:
            logger.error(f"Failed to increment usage count for {template_id}/{version}: {e}")
            return False

    def record_completion_time(
        self,
        template_id: str,
        version: str,
        time_seconds: float,
        success: bool = True
    ) -> bool:
        """
        Record completion time for a template

        Args:
            template_id: ID of the template
            version: Version of the template
            time_seconds: Completion time in seconds
            success: Whether completion was successful

        Returns:
            True if successful, False otherwise
        """
        try:
            stats_key = self._template_stats_key(template_id, version)

            # Get current stats
            pipe = self.redis.pipeline()
            pipe.hget(stats_key, "completion_count")
            pipe.hget(stats_key, "success_count")
            pipe.hget(stats_key, "total_time")

            completion_count, success_count, total_time = pipe.execute()

            # Update stats
            completion_count = int(completion_count or 0) + 1
            success_count = int(success_count or 0) + (1 if success else 0)
            total_time = float(total_time or 0) + time_seconds

            # Calculate completion rate and average time
            completion_rate = success_count / completion_count if completion_count > 0 else 0
            avg_time = total_time / completion_count if completion_count > 0 else 0

            # Store updated stats
            pipe = self.redis.pipeline()
            pipe.hset(stats_key, "completion_count", completion_count)
            pipe.hset(stats_key, "success_count", success_count)
            pipe.hset(stats_key, "total_time", total_time)
            pipe.hset(stats_key, "completion_rate", completion_rate)
            pipe.hset(stats_key, "avg_completion_time", avg_time)
            pipe.expire(stats_key, self.ttl)
            pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Failed to record completion time for {template_id}/{version}: {e}")
            return False

    def clear_cache(self, template_id: str = None) -> bool:
        """
        Clear cache for a template or all templates

        Args:
            template_id: Optional ID of the template to clear cache for

        Returns:
            True if successful, False otherwise
        """
        try:
            if template_id:
                # Get all versions
                versions_key = self._template_versions_key(template_id)
                versions = self.redis.zrange(versions_key, 0, -1)

                # Delete template and all versions
                keys_to_delete = [
                    self._template_key(template_id),
                    versions_key
                ]

                for version in versions:
                    keys_to_delete.append(self._template_version_key(template_id, version))
                    keys_to_delete.append(self._template_stats_key(template_id, version))
                    keys_to_delete.append(f"{self.prefix}template:{template_id}:version_info:{version}")

                if keys_to_delete:
                    self.redis.delete(*keys_to_delete)
            else:
                # Clear all template cache
                pattern = f"{self.prefix}template:*"
                cursor = 0
                while True:
                    cursor, keys = self.redis.scan(cursor, pattern, 100)
                    if keys:
                        self.redis.delete(*keys)
                    if cursor == 0:
                        break

                # Clear all stats cache
                pattern = f"{self.prefix}stats:*"
                cursor = 0
                while True:
                    cursor, keys = self.redis.scan(cursor, pattern, 100)
                    if keys:
                        self.redis.delete(*keys)
                    if cursor == 0:
                        break

            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        try:
            # Get number of templates
            templates_count = self.redis.scard(self._templates_list_key())

            # Get memory usage
            memory_stats = self.redis.info("memory")

            # Get key count with prefix
            pattern = f"{self.prefix}*"
            key_count = 0
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, pattern, 100)
                key_count += len(keys)
                if cursor == 0:
                    break

            return {
                "templates_count": templates_count,
                "key_count": key_count,
                "memory_usage": memory_stats.get("used_memory_human", "unknown"),
                "peak_memory": memory_stats.get("used_memory_peak_human", "unknown"),
                "redis_version": self.redis.info().get("redis_version", "unknown")
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
EOF

cat > ./template-registry/src/adapters/pulsar_schema_registry.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pulsar Schema Registry Adapter

This module provides an adapter for Apache Pulsar Schema Registry.
It allows storing template schemas in Pulsar's schema registry for
validation and version management.
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple

import pulsar
from pulsar.schema import (
    JsonSchema,
    Schema,
    AvroSchema,
    KeyValueSchema,
    Record,
    String,
    Integer,
    Float,
    Boolean,
    Array
)
import requests

from models.template_models import (
    TemplateDefinition,
    CompatibilityType,
    TemplateCategory,
    TemplateLanguage
)


logger = logging.getLogger(__name__)


class PulsarSchemaRegistryAdapter:
    """
    Apache Pulsar Schema Registry adapter for template schemas

    This adapter stores template schemas in Pulsar's built-in schema registry
    for validation, version management, and compatibility checks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Pulsar schema registry adapter

        Args:
            config: Configuration for the schema registry adapter
        """
        self.service_url = config.get("serviceUrl", "pulsar://localhost:6650")
        self.admin_url = self.service_url.replace("pulsar://", "http://").replace("6650", "8080")
        self.tenant = config.get("tenant", "public")
        self.namespace = config.get("namespace", "template-registry")
        self.topic_prefix = config.get("topic", "schemas")

        # Create a Pulsar client
        self.client = pulsar.Client(self.service_url)

        # Ensure namespace exists
        self._ensure_namespace()

    def _ensure_namespace(self) -> None:
        """Ensure the namespace exists"""
        try:
            # Check if namespace exists
            response = requests.get(
                f"{self.admin_url}/admin/v2/namespaces/{self.tenant}/{self.namespace}"
            )

            if response.status_code == 404:
                # Create namespace
                requests.put(
                    f"{self.admin_url}/admin/v2/namespaces/{self.tenant}/{self.namespace}"
                )

                # Set schema compatibility policy
                requests.put(
                    f"{self.admin_url}/admin/v2/namespaces/{self.tenant}/{self.namespace}/schemaCompatibilityStrategy",
                    json="BACKWARD"
                )

                logger.info(f"Created namespace {self.tenant}/{self.namespace}")
        except Exception as e:
            logger.error(f"Failed to ensure namespace: {e}")

    def _get_topic_name(self, template_id: str) -> str:
        """
        Get the Pulsar topic name for a template

        Args:
            template_id: ID of the template

        Returns:
            Pulsar topic name
        """
        return f"persistent://{self.tenant}/{self.namespace}/{self.topic_prefix}-{template_id}"

    def _get_schema_info(self, topic: str) -> Dict[str, Any]:
        """
        Get schema info from Pulsar schema registry

        Args:
            topic: Pulsar topic name

        Returns:
            Schema info dictionary
        """
        try:
            response = requests.get(
                f"{self.admin_url}/admin/v2/schemas/{topic}/schema"
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get schema info: {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {}

    def _get_schema_versions(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get schema versions from Pulsar schema registry

        Args:
            topic: Pulsar topic name

        Returns:
            List of schema version dictionaries
        """
        try:
            response = requests.get(
                f"{self.admin_url}/admin/v2/schemas/{topic}/versions"
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get schema versions: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Failed to get schema versions: {e}")
            return []

    def _upload_schema(
        self,
        topic
                if not re.match(url_pattern, str(value)):
                    return rule.error_message or f"Field {section_name}.{field_name} must be a valid URL"

        return None
EOF

# Create utility module for Pulsar event emission
cat > ./template-registry/utils/pulsar_events.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pulsar Event Utilities

This module provides utilities for emitting events to Apache Pulsar.
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, Optional

import pulsar


logger = logging.getLogger(__name__)


class EventType:
    """Event types for the template registry"""
    TEMPLATE_CREATED = "template.created"
    TEMPLATE_UPDATED = "template.updated"
    TEMPLATE_DELETED = "template.deleted"
    TEMPLATE_VIEWED = "template.viewed"
    TEMPLATE_VALIDATED = "template.validated"
    TEMPLATE_COMPARED = "template.compared"
    TEMPLATE_STATS = "template.stats"


class PulsarEventEmitter:
    """
    Emitter for Apache Pulsar events

    This class provides functionality for emitting events to Apache Pulsar
    topics for integration with other components.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Pulsar event emitter

        Args:
            config: Configuration dictionary
        """
        self.service_url = config.get("serviceUrl", "pulsar://localhost:6650")
        self.tenant = config.get("tenant", "public")
        self.namespace = config.get("namespace", "template-registry")
        self.topic_prefix = config.get("topicPrefix", "events")

        # Create a Pulsar client
        self.client = pulsar.Client(self.service_url)

        # Producer cache
        self.producers = {}

    def _get_topic_name(self, event_type: str) -> str:
        """
        Get the Pulsar topic name for an event type

        Args:
            event_type: Type of event

        Returns:
            Pulsar topic name
        """
        return f"persistent://{self.tenant}/{self.namespace}/{self.topic_prefix}-{event_type}"

    def _get_producer(self, event_type: str) -> pulsar.Producer:
        """
        Get or create a producer for an event type

        Args:
            event_type: Type of event

        Returns:
            Pulsar producer
        """
        topic = self._get_topic_name(event_type)

        if topic not in self.producers:
            self.producers[topic] = self.client.create_producer(
                topic=topic,
                schema=pulsar.schema.JsonSchema(),
                properties={
                    "producer-name": f"template-registry-{uuid.uuid4()}",
                    "producer-id": str(uuid.uuid4())
                }
            )

        return self.producers[topic]

    def emit_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        key: Optional[str] = None
    ) -> None:
        """
        Emit an event to Pulsar

        Args:
            event_type: Type of event
            payload: Event payload
            key: Optional event key for partitioning
        """
        try:
            # Get producer
            producer = self._get_producer(event_type)

            # Create event object
            event = {
                "type": event_type,
                "timestamp": int(time.time() * 1000),
                "id": str(uuid.uuid4()),
                "payload": payload
            }

            # Send event
            if key:
                producer.send(value=event, partition_key=key)
            else:
                producer.send(value=event)

            logger.debug(f"Emitted event {event_type}: {event['id']}")
        except Exception as e:
            logger.error(f"Failed to emit event {event_type}: {e}")

    def emit_template_event(
        self,
        event_type: str,
        template_id: str,
        template_data: Dict[str, Any],
        additional_data: Dict[str, Any] = None
    ) -> None:
        """
        Emit a template event to Pulsar

        Args:
            event_type: Type of event
            template_id: ID of the template
            template_data: Template data
            additional_data: Additional event data
        """
        payload = {
            "template_id": template_id,
            "template": template_data
        }

        if additional_data:
            payload.update(additional_data)

        self.emit_event(event_type, payload, key=template_id)

    def close(self) -> None:
        """Close the Pulsar client and all producers"""
        for producer in self.producers.values():
            producer.close()

        self.client.close()
EOF

# Create a helper script for bootstrapping the registry with example templates
cat > ./template-registry/utils/bootstrap.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap Utility

This script bootstraps the template registry with example templates.
"""

import json
import logging
import os
import sys
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.template_models import (
    TemplateDefinition,
    TemplateCategory,
    TemplateLanguage,
    CompatibilityType,
    SectionDefinition,
    FieldDefinition,
    FieldType,
    ValidationRule,
    ValidationRuleType,
    TemplateMetadata
)
from src.core.template_registry import get_registry


logger = logging.getLogger(__name__)


def create_api_template() -> TemplateDefinition:
    """Create an example API template"""
    # Define fields for basic information section
    basic_fields = [
        FieldDefinition(
            name="endpoint_path",
            type=FieldType.STRING,
            description="URL path of the endpoint",
            required=True,
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.REGEX,
                    expression=r"^/[a-zA-Z0-9_/-]*$",
                    error_message="Path must start with / and contain only alphanumeric characters, underscores, and hyphens"
                )
            ]
        ),
        FieldDefinition(
            name="http_method",
            type=FieldType.ENUM,
            description="HTTP method",
            required=True,
            options=["GET", "POST", "PUT", "DELETE", "PATCH"]
        ),
        FieldDefinition(
            name="endpoint_name",
            type=FieldType.STRING,
            description="Name of the endpoint function",
            required=True,
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.REGEX,
                    expression=r"^[a-z][a-z0-9_]*$",
                    error_message="Name must start with lowercase letter and contain only lowercase letters, numbers, and underscores"
                )
            ]
        )
    ]

    # Define fields for request parameters section
    request_fields = [
        FieldDefinition(
            name="query_parameters",
            type=FieldType.ARRAY,
            description="Query parameters",
            required=False,
            nested_fields=[
                FieldDefinition(
                    name="name",
                    type=FieldType.STRING,
                    description="Parameter name",
                    required=True
                ),
                FieldDefinition(
                    name="type",
                    type=FieldType.ENUM,
                    description="Parameter type",
                    required=True,
                    options=["string", "integer", "float", "boolean", "array"]
                ),
                FieldDefinition(
                    name="description",
                    type=FieldType.STRING,
                    description="Parameter description",
                    required=False
                ),
                FieldDefinition(
                    name="required",
                    type=FieldType.BOOLEAN,
                    description="Whether parameter is required",
                    required=False,
                    default_value=False
                )
            ]
        ),
        FieldDefinition(
            name="body_parameters",
            type=FieldType.ARRAY,
            description="Body parameters",
            required=False,
            nested_fields=[
                FieldDefinition(
                    name="name",
                    type=FieldType.STRING,
                    description="Parameter name",
                    required=True
                ),#!/bin/bash
#============================================================================
# Template Registry System for Spec-Driven AI Code Generation Platform
#
# This script sets up a template registry system that handles storage,
# versioning, categorization, and validation of specification templates.
# The system leverages Apache Pulsar for event-driven architecture,
# Redis for high-performance caching, and Git for version control.
#
# Author: Claude
# Date: 2025-04-18
#============================================================================

set -e
echo "Setting up Template Registry System..."

# Create directory structure
mkdir -p ./template-registry/{src,models,storage,schemas,utils,tests,config}
mkdir -p ./template-registry/src/{core,event_handlers,validators,adapters}
mkdir -p ./template-registry/storage/{git,cache}
mkdir -p ./template-registry/schemas/{templates,definitions}

# Configuration
cat > ./template-registry/config/config.json << 'EOF'
{
    "storage": {
        "type": "hybrid",
        "gitRepository": {
            "path": "./storage/git/templates",
            "remote": "",
            "branch": "main",
            "pushOnUpdate": true
        },
        "cache": {
            "type": "redis",
            "host": "localhost",
            "port": 6379,
            "ttl": 3600,
            "prefix": "template-registry:"
        },
        "schemaStore": {
            "type": "pulsar",
            "serviceUrl": "pulsar://localhost:6650",
            "tenant": "public",
            "namespace": "template-registry",
            "topic": "schemas"
        }
    },
    "eventBus": {
        "serviceUrl": "pulsar://localhost:6650",
        "tenant": "public",
        "namespace": "template-registry",
        "subscriptionName": "template-registry-service",
        "responseTopicPrefix": "template-registry-response",
        "eventTopics": {
            "templateCreate": "template-registry-create",
            "templateUpdate": "template-registry-update",
            "templateGet": "template-registry-get",
            "templateList": "template-registry-list",
            "templateDelete": "template-registry-delete",
            "templateValidate": "template-registry-validate",
            "templateCompare": "template-registry-compare",
            "templateStats": "template-registry-stats"
        }
    },
    "metrics": {
        "enabled": true,
        "statsdHost": "localhost",
        "statsdPort": 8125,
        "prefix": "template_registry."
    },
    "validation": {
        "strictMode": true,
        "allowSchemaEvolution": true,
        "compatibilityStrategy": "BACKWARD"
    }
}
EOF

# Create the core models for template data structures
cat > ./template-registry/models/template_models.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Registry Models

This module defines the core data models for the template registry system.
It includes classes for templates, fields, sections, validation rules, and more.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set


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
    REFERENCE = "reference"
    DATETIME = "datetime"


class ValidationRuleType(str, Enum):
    """Types of validation rules"""
    REGEX = "regex"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    REQUIRED = "required"
    ENUM_VALUES = "enum_values"
    CUSTOM = "custom"
    DEPENDENCY = "dependency"
    FORMAT = "format"


class TemplateCategory(str, Enum):
    """Categories for templates"""
    API = "api"
    DATABASE = "database"
    UI = "ui"
    WORKFLOW = "workflow"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    INFRASTRUCTURE = "infrastructure"
    OTHER = "other"


class TemplateLanguage(str, Enum):
    """Programming languages for templates"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    ANY = "any"


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
    error_message: str


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
            FieldType.REFERENCE: "string",
            FieldType.DATETIME: "string"
        }
        return mapping.get(self.type, "string")


@dataclass
class SectionDefinition:
    """Definition of a section in a template"""
    name: str
    description: str
    fields: List[FieldDefinition] = field(default_factory=list)


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


@dataclass
class TemplateDefinition:
    """Definition of a template"""
    id: str
    name: str
    description: str
    version: str
    category: TemplateCategory
    languages: List[TemplateLanguage] = field(default_factory=lambda: [TemplateLanguage.ANY])
    sections: List[SectionDefinition] = field(default_factory=list)
    metadata: TemplateMetadata = field(default_factory=TemplateMetadata)
    compatibility: CompatibilityType = CompatibilityType.BACKWARD

    def __post_init__(self):
        """Validate after initialization"""
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateDefinition':
        """Create from dictionary"""
        sections = []
        for section_data in data.get("sections", []):
            fields = []
            for field_data in section_data.get("fields", []):
                validation_rules = []
                for rule_data in field_data.get("validation_rules", []):
                    validation_rules.append(ValidationRule(
                        rule_type=ValidationRuleType(rule_data["rule_type"]),
                        expression=rule_data["expression"],
                        error_message=rule_data["error_message"]
                    ))

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
                        nested_fields=[]
                    ))

                fields.append(FieldDefinition(
                    name=field_data["name"],
                    type=FieldType(field_data["type"]),
                    description=field_data["description"],
                    required=field_data.get("required", False),
                    default_value=field_data.get("default_value"),
                    validation_rules=validation_rules,
                    options=field_data.get("options", []),
                    nested_fields=nested_fields
                ))

            sections.append(SectionDefinition(
                name=section_data["name"],
                description=section_data["description"],
                fields=fields
            ))

        metadata_data = data.get("metadata", {})
        metadata = TemplateMetadata(
            created_at=metadata_data.get("created_at", int(time.time())),
            created_by=metadata_data.get("created_by", "system"),
            updated_at=metadata_data.get("updated_at", int(time.time())),
            updated_by=metadata_data.get("updated_by", "system"),
            tags=metadata_data.get("tags", []),
            domain=metadata_data.get("domain", "general"),
            complexity=metadata_data.get("complexity", "medium"),
            estimated_completion_time=metadata_data.get("estimated_completion_time", 0)
        )

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data["version"],
            category=TemplateCategory(data["category"]),
            languages=[TemplateLanguage(lang) for lang in data.get("languages", ["any"])],
            sections=sections,
            metadata=metadata,
            compatibility=CompatibilityType(data.get("compatibility", "backward"))
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateDefinition':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class FieldValue:
    """Value of a field"""
    name: str
    value: Any


@dataclass
class SectionValues:
    """Values for a section"""
    name: str
    fields: List[FieldValue] = field(default_factory=list)


@dataclass
class TemplateInstance:
    """Filled template instance"""
    id: str
    template_id: str
    template_version: str
    project_id: str
    name: str
    sections: List[SectionValues] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    completed: bool = False
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateInstance':
        """Create from dictionary"""
        sections = []
        for section_data in data.get("sections", []):
            fields = []
            for field_data in section_data.get("fields", []):
                fields.append(FieldValue(
                    name=field_data["name"],
                    value=field_data["value"]
                ))

            sections.append(SectionValues(
                name=section_data["name"],
                fields=fields
            ))

        return cls(
            id=data["id"],
            template_id=data["template_id"],
            template_version=data["template_version"],
            project_id=data["project_id"],
            name=data["name"],
            sections=sections,
            created_at=data.get("created_at", int(time.time())),
            updated_at=data.get("updated_at", int(time.time())),
            completed=data.get("completed", False),
            validated=data.get("validated", False),
            validation_errors=data.get("validation_errors", [])
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TemplateInstance':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TemplateDependency:
    """Dependency between templates"""
    source_id: str
    target_id: str
    dependency_type: str
    description: str


@dataclass
class TemplateVersionInfo:
    """Information about a template version"""
    template_id: str
    version: str
    commit_id: str
    timestamp: int
    author: str
    message: str
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
class TemplateStats:
    """Statistics for a template"""
    template_id: str
    version: str
    usage_count: int = 0
    completion_rate: float = 0.0
    avg_completion_time: float = 0.0
    last_used: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TemplateSearchQuery:
    """Query for searching templates"""
    keywords: Optional[str] = None
    categories: List[TemplateCategory] = field(default_factory=list)
    languages: List[TemplateLanguage] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    complexity: Optional[str] = None
    sort_by: str = "updated_at"
    sort_order: str = "desc"
    limit: int = 20
    offset: int = 0
EOF

# Create storage adapters
cat > ./template-registry/src/adapters/git_storage.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Git Storage Adapter

This module provides a Git-based storage adapter for template versioning.
It commits changes to templates into a Git repository for version control,
history tracking, and collaboration.
"""

import os
import json
import time
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from models.template_models import (
    TemplateDefinition,
    TemplateVersionInfo,
    CompatibilityType
)


logger = logging.getLogger(__name__)


class GitStorageAdapter:
    """
    Git-based storage adapter for templates with proper versioning

    This adapter stores templates as JSON files in a Git repository,
    providing version control, history tracking, and semantic versioning.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Git storage adapter

        Args:
            config: Configuration for the storage adapter
        """
        self.repo_path = Path(config.get("path", "./storage/git/templates"))
        self.remote = config.get("remote", "")
        self.branch = config.get("branch", "main")
        self.push_on_update = config.get("pushOnUpdate", True)

        # Ensure the repository exists
        self._ensure_repository()

    def _ensure_repository(self) -> None:
        """Ensure the Git repository exists and is properly configured"""
        if not (self.repo_path / ".git").exists():
            logger.info(f"Creating Git repository at {self.repo_path}")
            os.makedirs(self.repo_path, exist_ok=True)

            # Initialize the repository
            self._run_git_command("init")
            self._run_git_command("checkout", "-b", self.branch)

            # Create initial structure
            template_dir = self.repo_path / "templates"
            os.makedirs(template_dir, exist_ok=True)

            # Create .gitignore
            with open(self.repo_path / ".gitignore", "w") as f:
                f.write("*.pyc\n__pycache__/\n.DS_Store\n")

            # Initial commit
            self._run_git_command("add", ".")
            self._run_git_command("commit", "-m", "Initial commit")

            # Configure remote if provided
            if self.remote:
                self._run_git_command("remote", "add", "origin", self.remote)

        elif self.remote:
            # Check if remote needs to be updated
            try:
                current_remote = subprocess.check_output(
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.repo_path
                ).decode().strip()

                if current_remote != self.remote:
                    self._run_git_command("remote", "set-url", "origin", self.remote)
            except subprocess.CalledProcessError:
                self._run_git_command("remote", "add", "origin", self.remote)

    def _run_git_command(self, *args) -> Tuple[int, str]:
        """
        Run a Git command

        Args:
            *args: Arguments to pass to Git

        Returns:
            Tuple of (return_code, output)
        """
        try:
            result = subprocess.run(
                ["git"] + list(args),
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode, result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            return e.returncode, e.output

    def _get_template_path(self, template_id: str) -> Path:
        """
        Get the path for a template file

        Args:
            template_id: ID of the template

        Returns:
            Path to the template file
        """
        return self.repo_path / "templates" / f"{template_id}.json"

    def _get_versions_path(self, template_id: str) -> Path:
        """
        Get the path for version history of a template

        Args:
            template_id: ID of the template

        Returns:
            Path to the versions directory
        """
        return self.repo_path / "versions" / template_id

    def _parse_semver(self, version: str) -> Tuple[int, int, int]:
        """
        Parse a semantic version string

        Args:
            version: Version string in semver format (e.g. "1.2.3")

        Returns:
            Tuple of (major, minor, patch) version numbers
        """
        try:
            parts = version.split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return (major, minor, patch)
        except (ValueError, IndexError):
            return (0, 0, 0)

    def _increment_version(
        self,
        current_version: str,
        compatibility: CompatibilityType
    ) -> str:
        """
        Increment version based on compatibility type

        Args:
            current_version: Current version string
            compatibility: Compatibility type that determines version increment

        Returns:
            New version string
        """
        major, minor, patch = self._parse_semver(current_version)

        if compatibility == CompatibilityType.NONE:
            # Breaking change, increment major version
            return f"{major + 1}.0.0"
        elif compatibility == CompatibilityType.BACKWARD:
            # Backward compatible, increment minor version
            return f"{major}.{minor + 1}.0"
        else:
            # Forward compatible or full compatible, increment patch version
            return f"{major}.{minor}.{patch + 1}"

    def load_template(self, template_id: str) -> Optional[TemplateDefinition]:
        """
        Load a template by ID

        Args:
            template_id: ID of the template to load

        Returns:
            TemplateDefinition object or None if not found
        """
        template_path = self._get_template_path(template_id)

        if not template_path.exists():
            return None

        try:
            with open(template_path, "r") as f:
                data = json.load(f)
                return TemplateDefinition.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load template {template_id}: {e}")
            return None

    def load_template_version(
        self,
        template_id: str,
        version: str
    ) -> Optional[TemplateDefinition]:
        """
        Load a specific version of a template

        Args:
            template_id: ID of the template
            version: Version to load

        Returns:
            TemplateDefinition object or None if not found
        """
        versions_path = self._get_versions_path(template_id)
        version_file = versions_path / f"{version}.json"

        if not version_file.exists():
            return None

        try:
            with open(version_file, "r") as f:
                data = json.load(f)
                return TemplateDefinition.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load template {template_id} version {version}: {e}")
            return None

    def save_template(
        self,
        template: TemplateDefinition,
        message: str = "Update template",
        author: str = "system"
    ) -> TemplateVersionInfo:
        """
        Save a template and create a new version

        Args:
            template: Template to save
            message: Commit message
            author: Author of the change

        Returns:
            TemplateVersionInfo with details about the new version
        """
        # Check if this is a new template
        is_new = not self._get_template_path(template.id).exists()

        # If existing, load current version for comparison
        current_template = None
        if not is_new:
            current_template = self.load_template(template.id)

        # Determine new version
        if is_new or not current_template:
            new_version = template.version or "1.0.0"
        else:
            new_version = self._increment_version(
                current_template.version,
                template.compatibility
            )

        # Set the new version on the template
        template.version = new_version
        template.metadata.updated_at = int(time.time())
        template.metadata.updated_by = author

        # Create directory structure if not exists
        os.makedirs(self.repo_path / "templates", exist_ok=True)
        os.makedirs(self._get_versions_path(template.id), exist_ok=True)

        # Save the template as the latest version
        with open(self._get_template_path(template.id), "w") as f:
            json.dump(template.to_dict(), f, indent=2)

        # Save as a versioned copy
        version_path = self._get_versions_path(template.id) / f"{new_version}.json"
        with open(version_path, "w") as f:
            json.dump(template.to_dict(), f, indent=2)

        # Commit the changes
        self._run_git_command("add", str(self._get_template_path(template.id)))
        self._run_git_command("add", str(version_path))

        commit_message = f"{message}\n\nTemplate: {template.name}\nVersion: {new_version}"
        self._run_git_command("commit", "-m", commit_message, "--author", f"{author} <{author}@example.com>")

        # Get the commit hash
        returncode, output = self._run_git_command("rev-parse", "HEAD")
        commit_id = output.strip() if returncode == 0 else "unknown"

        # Push if configured
        if self.push_on_update and self.remote:
            self._run_git_command("push", "origin", self.branch)

        # Generate version info
        changes = []
        if current_template:
            # Compare sections
            current_sections = {s.name for s in current_template.sections}
            new_sections = {s.name for s in template.sections}

            added_sections = new_sections - current_sections
            removed_sections = current_sections - new_sections

            for section in added_sections:
                changes.append(f"Added section: {section}")

            for section in removed_sections:
                changes.append(f"Removed section: {section}")

            # Compare fields in common sections
            for new_section in template.sections:
                if new_section.name in current_sections:
                    current_section = next(
                        (s for s in current_template.sections if s.name == new_section.name),
                        None
                    )

                    if current_section:
                        current_fields = {f.name for f in current_section.fields}
                        new_fields = {f.name for f in new_section.fields}

                        added_fields = new_fields - current_fields
                        removed_fields = current_fields - new_fields

                        for field in added_fields:
                            changes.append(f"Added field: {new_section.name}.{field}")

                        for field in removed_fields:
                            changes.append(f"Removed field: {new_section.name}.{field}")

            # Check for category or compatibility changes
            if current_template.category != template.category:
                changes.append(f"Changed category from {current_template.category} to {template.category}")

            if current_template.compatibility != template.compatibility:
                changes.append(f"Changed compatibility from {current_template.compatibility} to {template.compatibility}")
        else:
            changes.append("Initial template creation")

        version_info = TemplateVersionInfo(
            template_id=template.id,
            version=new_version,
            commit_id=commit_id,
            timestamp=template.metadata.updated_at,
            author=author,
            message=message,
            changes=changes,
            compatibility_type=template.compatibility,
            is_breaking_change=(template.compatibility == CompatibilityType.NONE)
        )

        return version_info

    def list_templates(self) -> List[TemplateDefinition]:
        """
        List all available templates

        Returns:
            List of TemplateDefinition objects
        """
        templates_dir = self.repo_path / "templates"
        if not templates_dir.exists():
            return []

        templates = []
        for template_file in templates_dir.glob("*.json"):
            try:
                with open(template_file, "r") as f:
                    data = json.load(f)
                    template = TemplateDefinition.from_dict(data)
                    templates.append(template)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load template {template_file}: {e}")

        return templates

    def list_template_versions(self, template_id: str) -> List[TemplateVersionInfo]:
        """
        List all versions of a template

        Args:
            template_id: ID of the template

        Returns:
            List of TemplateVersionInfo objects
        """
        versions_path = self._get_versions_path(template_id)
        if not versions_path.exists():
            return []

        version_files = list(versions_path.glob("*.json"))
        if not version_files:
            return []

        # Get git log information for this template
        template_path = self._get_template_path(template_id)
        returncode, output = self._run_git_command(
            "log",
            "--pretty=format:%H|%at|%an|%s",
            "--follow",
            "--",
            str(template_path)
        )

        if returncode != 0:
            logger.error(f"Failed to get git log for {template_id}")
            return []

        # Parse git log
        commit_info = {}
        for line in output.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                commit_hash, timestamp, author, message = parts
                commit_info[commit_hash] = {
                    "timestamp": int(timestamp),
                    "author": author,
                    "message": message
                }

        # Load each version
        versions = []
        for version_file in version_files:
            version = version_file.stem
            try:
                template = self.load_template_version(template_id, version)
                if template:
                    # Try to find the commit for this version
                    returncode, output = self._run_git_command(
                        "log",
                        "-1",
                        "--pretty=format:%H",
                        "--",
                        str(version_file)
                    )

                    commit_id = output.strip() if returncode == 0 else "unknown"
                    commit_data = commit_info.get(commit_id, {
                        "timestamp": template.metadata.updated_at,
                        "author": template.metadata.updated_by,
                        "message": "Update template"
                    })

                    version_info = TemplateVersionInfo(
                        template_id=template_id,
                        version=version,
                        commit_id=commit_id,
                        timestamp=commit_data["timestamp"],
                        author=commit_data["author"],
                        message=commit_data["message"],
                        compatibility_type=template.compatibility,
                        is_breaking_change=(template.compatibility == CompatibilityType.NONE)
                    )
                    versions.append(version_info)
            except Exception as e:
                logger.error(f"Failed to load template version {version_file}: {e}")

        # Sort by version (semver)
        versions.sort(
            key=lambda v: self._parse_semver(v.version),
            reverse=True
        )

        return versions

    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template

        Args:
            template_id: ID of the template to delete

        Returns:
            True if successful, False otherwise
        """
        template_path = self._get_template_path(template_id)
        versions_path = self._get_versions_path(template_id)

        if not template_path.exists():
            return False

        try:
            # Remove the files
            os.remove(template_path)
            if versions_path.exists():
                shutil.rmtree(versions_path)

            # Commit the changes
            self._run_git_command("add", str(template_path))
            if versions_path.exists():
                self._run_git_command("add", str(versions_path))

            self._run_git_command(
                "commit",
                "-m",
                f"Delete template {template_id}"
            )

            # Push if configured
            if self.push_on_update and self.remote:
                self._run_git_command("push", "origin", self.branch)

            return True
        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            return False

    def compare_template_versions(
        self,
        template_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a template

        Args:
            template_id: ID of the template
            version1: First version to compare
            version2: Second version to compare

        Returns:
            Dictionary with differences
        """
        template1 = self.load_template_version(template_id, version1)
        template2 = self.load_template_version(template_id, version2)

        if not template1 or not template2:
            return {"error": "One or both versions not found"}

        # Compare basic attributes
        differences = {
            "name": template1.name != template2.name,
            "description": template1.description != template2.description,
            "category": template1.category != template2.category,
            "compatibility": template1.compatibility != template2.compatibility,
            "sections": {},
            "metadata": {}
        }

        # Compare metadata
        for key in ["tags", "domain", "complexity"]:
            if getattr(template1.metadata, key) != getattr(template2.metadata, key):
                differences["metadata"][key] = {
                    "version1": getattr(template1.metadata, key),
                    "version2": getattr(template2.metadata, key)
                }

        # Compare sections
        sections1 = {s.name: s for s in template1.sections}
        sections2 = {s.name: s for s in template2.sections}

        # Find added and removed sections
        added_sections = set(sections2.keys()) - set(sections1.keys())
        removed_sections = set(sections1.keys()) - set(sections2.keys())
        common_sections = set(sections1.keys()) & set(sections2.keys())

        if added_sections:
            differences["sections"]["added"] = list(added_sections)

        if removed_sections:
            differences["sections"]["removed"] = list(removed_sections)

        # Compare common sections
        differences["sections"]["modified"] = {}
        for section_name in common_sections:
            section1 = sections1[section_name]
            section2 = sections2[section_name]

            # Check for description changes
            if section1.description != section2.description:
                differences["sections"]["modified"][section_name] = {
                    "description_changed": True
                }
            else:
                differences["sections"]["modified"][section_name] = {
                    "description_changed": False
                }

            # Compare fields
            fields1 = {f.name: f for f in section1.fields}
            fields2 = {f.name: f for f in section2.fields}

            added_fields = set(fields2.keys()) - set(fields1.keys())
            removed_fields = set(fields1.keys()) - set(fields2.keys())

            if added_fields:
                differences["sections"]["modified"][section_name]["added_fields"] = list(added_fields)

            if removed_fields:
                differences["sections"]["modified"][section_name]["removed_fields"] = list(removed_fields)

            # Compare common fields
            modified_fields = {}
            for field_name in set(fields1.keys()) & set(fields2.keys()):
                field1 = fields1[field_name]
                field2 = fields2[field_name]

                if (field1.type != field2.type or
                    field1.description != field2.description or
                    field1.required != field2.required or
                    field1.default_value != field2.default_value or
                    field1.options != field2.options):

                    modified_fields[field_name] = {
                        "type_changed": field1.type != field2.type,
                        "description_changed": field1.description != field2.description,
                        "required_changed": field1.required != field2.required,
                        "default_value_changed": field1.default_value != field2.default_value,
                        "options_changed": field1.options != field2.options
                    }

            if modified_fields:
                differences["sections"]["modified"][section_name]["modified_fields"] = modified_fields

        return differences
EOF

cat > ./template-registry/src/adapters/redis_cache.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Redis Cache Adapter

This module provides a Redis-based caching adapter for template registry
to improve performance and reduce load on the underlying storage.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Union

import redis

from models.template_models import (
    TemplateDefinition,
    TemplateVersionInfo,
    TemplateStats
)


logger = logging.getLogger(__name__)


class RedisCacheAdapter:
    """
    Redis-based caching adapter for template registry

    This adapter caches templates, version information, and stats in Redis
    to improve performance and reduce load on the primary storage.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Redis cache adapter

        Args:
            config: Configuration for the cache adapter
        """
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password")
        self.ttl = config.get("ttl", 3600)  # Default TTL: 1 hour
        self.prefix = config.get("prefix", "template-registry:")

        # Connect to Redis
        self.redis = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True
        )

        # Test connection
        try:
            self.redis.ping()
            logger.info("Connected to Redis cache")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _template_key(self, template_id: str) -> str:
        """
        Generate a key for a template

        Args:
            template_id: ID of the template

        Returns:
            Redis key for the template
        """
        return f"{self.prefix}template:{template_id}"

    def _template_version_key(self, template_id: str, version: str) -> str:
        """
        Generate a key for a template version

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            Redis key for the template version
        """
        return f"{self.prefix}template:{template_id}:version:{version}"

    def _template_versions_key(self, template_id: str) -> str:
        """
        Generate a key for template versions list

        Args:
            template_id: ID of the template

        Returns:
            Redis key for the template versions list
        """
        return f"{self.prefix}template:{template_id}:versions"

    def _templates_list_key(self) -> str:
        """
        Generate a key for the templates list

        Returns:
            Redis key for the templates list
        """
        return f"{self.prefix}templates"

    def _template_stats_key(self, template_id: str, version: str = None) -> str:
        """
        Generate a key for template stats

        Args:
            template_id: ID of the template
            version: Optional version of the template

        Returns:
            Redis key for the template stats
        """
        if version:
            return f"{self.prefix}stats:{template_id}:{version}"
        return f"{self.prefix}stats:{template_id}"

    def _categories_key(self) -> str:
        """
        Generate a key for the categories set

        Returns:
            Redis key for the categories set
        """
        return f"{self.prefix}categories"

    def _languages_key(self) -> str:
        """
        Generate a key for the languages set

        Returns:
            Redis key for the languages set
        """
        return f"{self.prefix}languages"

    def _domains_key(self) -> str:
        """
        Generate a key for the domains set

        Returns:
            Redis key for the domains set
        """
        return f"{self.prefix}domains"

    def _tags_key(self) -> str:
        """
        Generate a key for the tags set

        Returns:
            Redis key for the tags set
        """
        return f"{self.prefix}tags"

    def cache_template(self, template: TemplateDefinition) -> bool:
        """
        Cache a template

        Args:
            template: Template to cache

        Returns:
            True if successful, False otherwise
        """
        try:
            # Cache the template
            template_key = self._template_key(template.id)
            self.redis.set(
                template_key,
                template.to_json(),
                ex=self.ttl
            )

            # Cache as a version
            version_key = self._template_version_key(template.id, template.version)
            self.redis.set(
                version_key,
                template.to_json(),
                ex=self.ttl
            )

            # Add to templates list
            self.redis.sadd(self._templates_list_key(), template.id)

            # Add to versions list
            versions_key = self._template_versions_key(template.id)
            self.redis.zadd(
                versions_key,
                {template.version: time.time()},
                nx=True
            )
            self.redis.expire(versions_key, self.ttl)

            # Update category, languages, domain, and tags
            self.redis.sadd(self._categories_key(), template.category)
            for lang in template.languages:
                self.redis.sadd(self._languages_key(), lang)
            self.redis.sadd(self._domains_key(), template.metadata.domain)
            for tag in template.metadata.tags:
                self.redis.sadd(self._tags_key(), tag)

            return True
        except Exception as e:
            logger.error(f"Failed to cache template {template.id}: {e}")
            return False

    def get_cached_template(self, template_id: str) -> Optional[TemplateDefinition]:
        """
        Get a cached template

        Args:
            template_id: ID of the template

        Returns:
            TemplateDefinition object or None if not found
        """
        try:
            template_key = self._template_key(template_id)
            json_data = self.redis.get(template_key)

            if not json_data:
                return None

            return TemplateDefinition.from_json(json_data)
        except Exception as e:
            logger.error(f"Failed to get cached template {template_id}: {e}")
            return None

    def get_cached_template_version(
        self,
        template_id: str,
        version: str
    ) -> Optional[TemplateDefinition]:
        """
        Get a cached template version

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            TemplateDefinition object or None if not found
        """
        try:
            version_key = self._template_version_key(template_id, version)
            json_data = self.redis.get(version_key)

            if not json_data:
                return None

            return TemplateDefinition.from_json(json_data)
        except Exception as e:
            logger.error(f"Failed to get cached template version {template_id}/{version}: {e}")
            return None

    def cache_template_versions(
        self,
        template_id: str,
        versions: List[TemplateVersionInfo]
    ) -> bool:
        """
        Cache template versions

        Args:
            template_id: ID of the template
            versions: List of version info objects

        Returns:
            True if successful, False otherwise
        """
        try:
            versions_key = self._template_versions_key(template_id)

            # Store each version info
            pipe = self.redis.pipeline()

            for version_info in versions:
                # Store version info
                version_info_key = f"{self.prefix}template:{template_id}:version_info:{version_info.version}"
                pipe.set(
                    version_info_key,
                    json.dumps(version_info.to_dict()),
                    ex=self.ttl
                )

                # Add to versions sorted set
                pipe.zadd(
                    versions_key,
                    {version_info.version: version_info.timestamp},
                    nx=True
                )

            pipe.expire(versions_key, self.ttl)
            pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Failed to cache template versions for {template_id}: {e}")
            return False

    def get_cached_template_versions(
        self,
        template_id: str
    ) -> List[TemplateVersionInfo]:
        """
        Get cached template versions

        Args:
            template_id: ID of the template

        Returns:
            List of TemplateVersionInfo objects
        """
        try:
            versions_key = self._template_versions_key(template_id)
            versions = self.redis.zrange(
                versions_key,
                0,
                -1,
                desc=True,
                withscores=False
            )

            if not versions:
                return []

            # Get version info for each version
            result = []
            pipe = self.redis.pipeline()

            for version in versions:
                version_info_key = f"{self.prefix}template:{template_id}:version_info:{version}"
                pipe.get(version_info_key)

            version_info_data = pipe.execute()

            for version, info_data in zip(versions, version_info_data):
                if info_data:
                    try:
                        version_info = TemplateVersionInfo(**json.loads(info_data))
                        result.append(version_info)
                    except Exception as e:
                        logger.error(f"Failed to parse version info for {template_id}/{version}: {e}")

            return result
        except Exception as e:
            logger.error(f"Failed to get cached template versions for {template_id}: {e}")
            return []

    def cache_stats(
        self,
        template_id: str,
        version: str,
        stats: TemplateStats
    ) -> bool:
        """
        Cache template stats

        Args:
            template_id: ID of the template
            version: Version of the template
            stats: Stats to cache

        Returns:
            True if successful, False otherwise
        """
        try:
            stats_key = self._template_stats_key(template_id, version)
            self.redis.set(
                stats_key,
                json.dumps(stats.to_dict()),
                ex=self.ttl
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cache stats for {template_id}/{version}: {e}")
            return False

    def get_cached_stats(
        self,
        template_id: str,
        version: str
    ) -> Optional[TemplateStats]:
        """
        Get cached template stats

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            TemplateStats object or None if not found
        """
        try:
            stats_key = self._template_stats_key(template_id, version)
            json_data = self.redis.get(stats_key)

            if not json_data:
                return None

            return TemplateStats(**json.loads(json_data))
        except Exception as e:
            logger.error(f"Failed to get cached stats for {template_id}/{version}: {e}")
            return None

    def increment_usage_count(self, template_id: str, version: str) -> bool:
        """
        Increment usage count for a template

        Args:
            template_id: ID of the template
            version: Version of the template

        Returns:
            True if successful, False otherwise
        """
        try:
            stats_key = self._template_stats_key(template_id, version)
            usage_key = f"{stats_key}:usage_count"

            # Use HINCRBY for the usage count
            self.redis.hincrby(stats_key, "usage_count", 1)
            self.redis.hset(stats_key, "last_used", int(time.time()))
            self.redis.expire(stats_key, self.ttl)

            return True
        except Exception as e:
            logger.error(f"Failed to increment usage count for {template_id}/{version}: {e}")
            return False

    def record_completion_time(
        self,
        template_id: str,
        version: str,
        time_seconds: float,
        success: bool = True
    ) -> bool:
        """
        Record completion time for a template

        Args:
            template_id: ID of the template
            version: Version of the template
            time_seconds: Completion time in seconds
            success: Whether completion was successful

        Returns:
            True if successful, False otherwise
        """
        try:
            stats_key = self._template_stats_key(template_id, version)

            # Get current stats
            pipe = self.redis.pipeline()
            pipe.hget(stats_key, "completion_count")
            pipe.hget(stats_key, "success_count")
            pipe.hget(stats_key, "total_time")

            completion_count, success_count, total_time = pipe.execute()

            # Update stats
            completion_count = int(completion_count or 0) + 1
            success_count = int(success_count or 0) + (1 if success else 0)
            total_time = float(total_time or 0) + time_seconds

            # Calculate completion rate and average time
            completion_rate = success_count / completion_count if completion_count > 0 else 0
            avg_time = total_time / completion_count if completion_count > 0 else 0

            # Store updated stats
            pipe = self.redis.pipeline()
            pipe.hset(stats_key, "completion_count", completion_count)
            pipe.hset(stats_key, "success_count", success_count)
            pipe.hset(stats_key, "total_time", total_time)
            pipe.hset(stats_key, "completion_rate", completion_rate)
            pipe.hset(stats_key, "avg_completion_time", avg_time)
            pipe.expire(stats_key, self.ttl)
            pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Failed to record completion time for {template_id}/{version}: {e}")
            return False

    def clear_cache(self, template_id: str = None) -> bool:
        """
        Clear cache for a template or all templates

        Args:
            template_id: Optional ID of the template to clear cache for

        Returns:
            True if successful, False otherwise
        """
        try:
            if template_id:
                # Get all versions
                versions_key = self._template_versions_key(template_id)
                versions = self.redis.zrange(versions_key, 0, -1)

                # Delete template and all versions
                keys_to_delete = [
                    self._template_key(template_id),
                    versions_key
                ]

                for version in versions:
                    keys_to_delete.append(self._template_version_key(template_id, version))
                    keys_to_delete.append(self._template_stats_key(template_id, version))
                    keys_to_delete.append(f"{self.prefix}template:{template_id}:version_info:{version}")

                if keys_to_delete:
                    self.redis.delete(*keys_to_delete)
            else:
                # Clear all template cache
                pattern = f"{self.prefix}template:*"
                cursor = 0
                while True:
                    cursor, keys = self.redis.scan(cursor, pattern, 100)
                    if keys:
                        self.redis.delete(*keys)
                    if cursor == 0:
                        break

                # Clear all stats cache
                pattern = f"{self.prefix}stats:*"
                cursor = 0
                while True:
                    cursor, keys = self.redis.scan(cursor, pattern, 100)
                    if keys:
                        self.redis.delete(*keys)
                    if cursor == 0:
                        break

            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        try:
            # Get number of templates
            templates_count = self.redis.scard(self._templates_list_key())

            # Get memory usage
            memory_stats = self.redis.info("memory")

            # Get key count with prefix
            pattern = f"{self.prefix}*"
            key_count = 0
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, pattern, 100)
                key_count += len(keys)
                if cursor == 0:
                    break

            return {
                "templates_count": templates_count,
                "key_count": key_count,
                "memory_usage": memory_stats.get("used_memory_human", "unknown"),
                "peak_memory": memory_stats.get("used_memory_peak_human", "unknown"),
                "redis_version": self.redis.info().get("redis_version", "unknown")
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
EOF

cat > ./template-registry/src/adapters/pulsar_schema_registry.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pulsar Schema Registry Adapter

This module provides an adapter for Apache Pulsar Schema Registry.
It allows storing template schemas in Pulsar's schema registry for
validation and version management.
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple

import pulsar
from pulsar.schema import (
    JsonSchema,
    Schema,
    AvroSchema,
    KeyValueSchema,
    Record,
    String,
    Integer,
    Float,
    Boolean,
    Array
)
import requests

from models.template_models import (
    TemplateDefinition,
    CompatibilityType,
    TemplateCategory,
    TemplateLanguage
)


logger = logging.getLogger(__name__)


class PulsarSchemaRegistryAdapter:
    """
    Apache Pulsar Schema Registry adapter for template schemas

    This adapter stores template schemas in Pulsar's built-in schema registry
    for validation, version management, and compatibility checks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Pulsar schema registry adapter

        Args:
            config: Configuration for the schema registry adapter
        """
        self.service_url = config.get("serviceUrl", "pulsar://localhost:6650")
        self.admin_url = self.service_url.replace("pulsar://", "http://").replace("6650", "8080")
        self.tenant = config.get("tenant", "public")
        self.namespace = config.get("namespace", "template-registry")
        self.topic_prefix = config.get("topic", "schemas")

        # Create a Pulsar client
        self.client = pulsar.Client(self.service_url)

        # Ensure namespace exists
        self._ensure_namespace()

    def _ensure_namespace(self) -> None:
        """Ensure the namespace exists"""
        try:
            # Check if namespace exists
            response = requests.get(
                f"{self.admin_url}/admin/v2/namespaces/{self.tenant}/{self.namespace}"
            )

            if response.status_code == 404:
                # Create namespace
                requests.put(
                    f"{self.admin_url}/admin/v2/namespaces/{self.tenant}/{self.namespace}"
                )

                # Set schema compatibility policy
                requests.put(
                    f"{self.admin_url}/admin/v2/namespaces/{self.tenant}/{self.namespace}/schemaCompatibilityStrategy",
                    json="BACKWARD"
                )

                logger.info(f"Created namespace {self.tenant}/{self.namespace}")
        except Exception as e:
            logger.error(f"Failed to ensure namespace: {e}")

    def _get_topic_name(self, template_id: str) -> str:
        """
        Get the Pulsar topic name for a template

        Args:
            template_id: ID of the template

        Returns:
            Pulsar topic name
        """
        return f"persistent://{self.tenant}/{self.namespace}/{self.topic_prefix}-{template_id}"

    def _get_schema_info(self, topic: str) -> Dict[str, Any]:
        """
        Get schema info from Pulsar schema registry

        Args:
            topic: Pulsar topic name

        Returns:
            Schema info dictionary
        """
        try:
            response = requests.get(
                f"{self.admin_url}/admin/v2/schemas/{topic}/schema"
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get schema info: {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {}

    def _get_schema_versions(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get schema versions from Pulsar schema registry

        Args:
            topic: Pulsar topic name

        Returns:
            List of schema version dictionaries
        """
        try:
            response = requests.get(
                f"{self.admin_url}/admin/v2/schemas/{topic}/versions"
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get schema versions: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Failed to get schema versions: {e}")
            return []

    def _upload_schema(
        self,
        topic