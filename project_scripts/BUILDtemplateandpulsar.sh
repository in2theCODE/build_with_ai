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
create_model_modules()
{
    log "Creating model modules..."
}
    # Base template app
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

    # Template instance app
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
        }
EOF

    # Event app
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

    # Analytics app
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

    # Evolution analytics app
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
    def from_dict(cls, data: Dict[str, Any]) -> 'SectionEv

EOF