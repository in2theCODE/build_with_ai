#!/bin/bash

# Script to generate spec template library
# Creates folder structure and YAML files for spec templates

set -e  # Exit on error

echo "Generating Spec Template Library..."

# Root directory
TEMPLATE_ROOT="./spec-templates"
mkdir -p "$TEMPLATE_ROOT"

# Common templates directory - for reusable fields/sections
mkdir -p "$TEMPLATE_ROOT/common"

# Create folder hierarchy
directories=(
  # Architecture
  "architecture/microservices"
  "architecture/monolithic"
  "architecture/serverless"
  "architecture/event-driven"
  "architecture/design-patterns/creational"
  "architecture/design-patterns/structural"
  "architecture/design-patterns/behavioral"
  "architecture/design-patterns/concurrency"

  # Backend
  "backend/api/rest"
  "backend/api/graphql"
  "backend/api/grpc"
  "backend/api/websocket"
  "backend/database/relational"
  "backend/database/document"
  "backend/database/graph"
  "backend/database/key-value"
  "backend/database/time-series"
  "backend/database/vector"
  "backend/servers/nodejs"
  "backend/servers/python"
  "backend/servers/java"
  "backend/servers/go"
  "backend/servers/ruby"
  "backend/auth/oauth"
  "backend/auth/jwt"
  "backend/auth/session"

  # Database Providers
  "backend/database/providers/postgresql"
  "backend/database/providers/mysql"
  "backend/database/providers/mongodb"
  "backend/database/providers/redis"
  "backend/database/providers/supabase"
  "backend/database/providers/firebase"
  "backend/database/providers/dynamodb"
  "backend/database/providers/neo4j"
  "backend/database/providers/neon"
  "backend/database/providers/cockroachdb"
  "backend/database/providers/cassandra"
  "backend/database/providers/milvus"
  "backend/database/providers/pinecone"
  "backend/database/providers/qdrant"
  "backend/database/providers/weaviate"
  "backend/database/providers/pgvector"

  # Auth Providers
  "backend/auth/providers/auth0"
  "backend/auth/providers/firebase-auth"
  "backend/auth/providers/clerk"
  "backend/auth/providers/cognito"
  "backend/auth/providers/supabase-auth"
  "backend/auth/providers/keycloak"
  "backend/auth/providers/okta"

  # Frontend
  "frontend/web/components"
  "frontend/web/pages"
  "frontend/web/state-management"
  "frontend/web/routing"
  "frontend/mobile/native/ios"
  "frontend/mobile/native/android"
  "frontend/mobile/cross-platform/react-native"
  "frontend/mobile/cross-platform/flutter"
  "frontend/mobile/cross-platform/xamarin"
  "frontend/desktop/electron"
  "frontend/desktop/qt"
  "frontend/desktop/wpf"

  # Styling Frameworks
  "frontend/styling/tailwind"
  "frontend/styling/bootstrap"
  "frontend/styling/material-ui"
  "frontend/styling/styled-components"
  "frontend/styling/emotion"
  "frontend/styling/sass"
  "frontend/styling/css-modules"
  "frontend/styling/chakra-ui"

  # DevOps
  "devops/ci-cd"
  "devops/containers"
  "devops/configuration"
  "devops/monitoring"
  "devops/networking"
  "devops/infrastructure-as-code"
  "devops/load-balancing"
  "devops/service-mesh"

  # Cloud
  "cloud/aws"
  "cloud/azure"
  "cloud/gcp"
  "cloud/kubernetes"
  "cloud/digital-ocean"
  "cloud/heroku"
  "cloud/vercel"
  "cloud/netlify"
  "cloud/cloudflare"
  "cloud/linode"
  "cloud/railway"
  "cloud/render"
  "cloud/fly-io"

  # Managed Services
  "cloud/managed-services/databases"
  "cloud/managed-services/caching"
  "cloud/managed-services/messaging"
  "cloud/managed-services/storage"
  "cloud/managed-services/serverless"
  "cloud/managed-services/cdn"

  # Security
  "security/authentication"
  "security/authorization"
  "security/encryption"
  "security/auditing"
  "security/compliance"
  "security/penetration-testing"
  "security/secrets-management"

  # Data
  "data/storage"
  "data/processing"
  "data/analytics"
  "data/ml-ai"
  "data/etl"
  "data/data-warehousing"
  "data/data-lakes"
  "data/business-intelligence"

  # ML/AI
  "data/ml-ai/frameworks/tensorflow"
  "data/ml-ai/frameworks/pytorch"
  "data/ml-ai/frameworks/keras"
  "data/ml-ai/frameworks/scikit-learn"
  "data/ml-ai/frameworks/huggingface"
  "data/ml-ai/agent-frameworks"
  "data/ml-ai/llm-integration"
  "data/ml-ai/vector-embeddings"
  "data/ml-ai/rag-systems"

  # Testing
  "testing/unit"
  "testing/integration"
  "testing/e2e"
  "testing/performance"
  "testing/frameworks/jest"
  "testing/frameworks/pytest"
  "testing/frameworks/junit"
  "testing/frameworks/cypress"
  "testing/frameworks/selenium"
  "testing/frameworks/k6"
  "testing/frameworks/playwright"

  # Chat Templates
  "templates/chat/customer-support"
  "templates/chat/sales"
  "templates/chat/information-retrieval"
  "templates/chat/tutoring"
  "templates/chat/assistant"
  "templates/chat/agent"

  # API Integration
  "integration/payment-gateways"
  "integration/email-services"
  "integration/sms-services"
  "integration/mapping-services"
  "integration/analytics-services"
  "integration/crm-systems"
)

# Create all directories
for dir in "${directories[@]}"; do
  mkdir -p "$TEMPLATE_ROOT/$dir"
  echo "Created directory: $TEMPLATE_ROOT/$dir"
done

# Function to create a validation rule
create_validation_rule() {
  local rule_type="$1"
  local expression="$2"
  local error_message="$3"

  cat << EOF
      - rule_type: $rule_type
        expression: $expression
        error_message: $error_message
EOF
}

# Function to create a field definition
create_field() {
  local name="$1"
  local type="$2"
  local description="$3"
  local required="$4"
  local default_value="$5"
  local validation_rules="$6"

  cat << EOF
    - name: $name
      type: $type
      description: $description
      required: $required
      default_value: $default_value
      validation_rules:
$validation_rules
EOF
}

# Function to create a section
create_section() {
  local name="$1"
  local description="$2"
  local fields="$3"

  cat << EOF
  - name: $name
    description: $description
    fields:
$fields
EOF
}

# Function to create a template
create_template() {
  local id="$1"
  local name="$2"
  local description="$3"
  local version="$4"
  local category="$5"
  local sections="$6"

  cat << EOF
id: $id
name: $name
description: $description
version: $version
category: $category
sections:
$sections
EOF
}

# Create common templates
echo "Creating common templates..."

# Create common field templates
mkdir -p "$TEMPLATE_ROOT/common/fields"

# Common ID field
cat > "$TEMPLATE_ROOT/common/fields/id_field.yaml" << EOF
name: id
type: STRING
description: Unique identifier
required: true
default_value: ""
validation_rules:
  - rule_type: PATTERN
    expression: "^[a-zA-Z0-9_-]+$"
    error_message: "ID must contain only alphanumeric characters, underscores, and hyphens"
EOF

# Common name field
cat > "$TEMPLATE_ROOT/common/fields/name_field.yaml" << EOF
name: name
type: STRING
description: Display name
required: true
default_value: ""
validation_rules:
  - rule_type: MIN_LENGTH
    expression: "1"
    error_message: "Name cannot be empty"
  - rule_type: MAX_LENGTH
    expression: "100"
    error_message: "Name cannot exceed 100 characters"
EOF

# Common description field
cat > "$TEMPLATE_ROOT/common/fields/description_field.yaml" << EOF
name: description
type: STRING
description: Detailed description
required: false
default_value: ""
validation_rules:
  - rule_type: MAX_LENGTH
    expression: "1000"
    error_message: "Description cannot exceed 1000 characters"
EOF

# Create common section templates
mkdir -p "$TEMPLATE_ROOT/common/sections"

# Common metadata section
cat > "$TEMPLATE_ROOT/common/sections/metadata_section.yaml" << EOF
name: metadata
description: General information about the component
fields:
  - name: id
    type: STRING
    description: Unique identifier
    required: true
    default_value: ""
    validation_rules:
      - rule_type: PATTERN
        expression: "^[a-zA-Z0-9_-]+$"
        error_message: "ID must contain only alphanumeric characters, underscores, and hyphens"
  - name: name
    type: STRING
    description: Display name
    required: true
    default_value: ""
    validation_rules:
      - rule_type: MIN_LENGTH
        expression: "1"
        error_message: "Name cannot be empty"
  - name: description
    type: STRING
    description: Detailed description
    required: false
    default_value: ""
    validation_rules:
      - rule_type: MAX_LENGTH
        expression: "1000"
        error_message: "Description cannot exceed 1000 characters"
  - name: version
    type: STRING
    description: Version number
    required: true
    default_value: "1.0.0"
    validation_rules:
      - rule_type: PATTERN
        expression: "^\\d+\\.\\d+\\.\\d+$"
        error_message: "Version must follow semantic versioning (e.g., 1.0.0)"
  - name: author
    type: STRING
    description: Author name or organization
    required: false
    default_value: ""
    validation_rules: []
EOF

# Create example templates

# REST API Template
mkdir -p "$TEMPLATE_ROOT/backend/api/rest"
cat > "$TEMPLATE_ROOT/backend/api/rest/rest_api_template.yaml" << 'EOF'
id: rest_api_template
name: REST API Specification
description: Template for defining REST API endpoints
version: 1.0.0
category: backend/api
sections:
  - name: api_metadata
    description: General information about the API
    fields:
      - name: api_name
        type: STRING
        description: Name of the API
        required: true
        default_value: ""
        validation_rules:
          - rule_type: MIN_LENGTH
            expression: "1"
            error_message: "API name cannot be empty"
      - name: base_path
        type: STRING
        description: Base URL path for the API
        required: true
        default_value: "/api/v1"
        validation_rules:
          - rule_type: PATTERN
            expression: "^/[a-zA-Z0-9_/-]*$"
            error_message: "Base path must start with / and contain valid characters"
      - name: description
        type: STRING
        description: Detailed description of the API
        required: false
        default_value: ""
        validation_rules: []
      - name: version
        type: STRING
        description: API version
        required: true
        default_value: "v1"
        validation_rules:
          - rule_type: PATTERN
            expression: "^v\\d+$"
            error_message: "Version must be in format v1, v2, etc."
  - name: authentication
    description: Authentication settings for the API
    fields:
      - name: auth_type
        type: ENUM
        description: Type of authentication
        required: true
        default_value: "JWT"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "JWT,OAuth2,ApiKey,None"
            error_message: "Auth type must be one of JWT, OAuth2, ApiKey, or None"
      - name: auth_required
        type: BOOLEAN
        description: Whether authentication is required
        required: true
        default_value: "true"
        validation_rules: []
      - name: auth_header
        type: STRING
        description: Header name for authentication token
        required: false
        default_value: "Authorization"
        validation_rules: []
  - name: endpoints
    description: API endpoints
    fields:
      - name: endpoint_definitions
        type: ARRAY
        description: List of endpoint definitions
        required: true
        default_value: "[]"
        validation_rules: []
      - name: endpoint_template
        type: OBJECT
        description: Template for an endpoint (not to be filled out directly)
        required: false
        default_value: |
          {
            "path": "/resource",
            "method": "GET",
            "description": "Description of endpoint",
            "request_params": [
              {
                "name": "param1",
                "type": "string",
                "required": true,
                "description": "Description of param1"
              }
            ],
            "request_body": {
              "content_type": "application/json",
              "schema": {}
            },
            "responses": [
              {
                "status_code": 200,
                "description": "Success response",
                "content_type": "application/json",
                "schema": {}
              },
              {
                "status_code": 400,
                "description": "Bad request",
                "content_type": "application/json",
                "schema": {}
              }
            ],
            "auth_required": true
          }
        validation_rules: []
  - name: models
    description: Data models used by the API
    fields:
      - name: model_definitions
        type: ARRAY
        description: List of data models
        required: true
        default_value: "[]"
        validation_rules: []
      - name: model_template
        type: OBJECT
        description: Template for a data model (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "ModelName",
            "description": "Description of model",
            "properties": [
              {
                "name": "property1",
                "type": "string",
                "required": true,
                "description": "Description of property1"
              }
            ]
          }
        validation_rules: []
EOF

# Database Schema Template
mkdir -p "$TEMPLATE_ROOT/backend/database/relational"
cat > "$TEMPLATE_ROOT/backend/database/relational/relational_database_template.yaml" << 'EOF'
id: relational_database_template
name: Relational Database Schema
description: Template for defining relational database schemas
version: 1.0.0
category: backend/database
sections:
  - name: database_metadata
    description: General information about the database
    fields:
      - name: database_name
        type: STRING
        description: Name of the database
        required: true
        default_value: ""
        validation_rules:
          - rule_type: MIN_LENGTH
            expression: "1"
            error_message: "Database name cannot be empty"
          - rule_type: PATTERN
            expression: "^[a-zA-Z0-9_]+$"
            error_message: "Database name must contain only alphanumeric characters and underscores"
      - name: description
        type: STRING
        description: Detailed description of the database
        required: false
        default_value: ""
        validation_rules: []
      - name: database_type
        type: ENUM
        description: Type of relational database
        required: true
        default_value: "PostgreSQL"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "PostgreSQL,MySQL,SQLite,SQL Server,Oracle"
            error_message: "Database type must be one of PostgreSQL, MySQL, SQLite, SQL Server, or Oracle"
  - name: tables
    description: Database tables
    fields:
      - name: table_definitions
        type: ARRAY
        description: List of table definitions
        required: true
        default_value: "[]"
        validation_rules: []
      - name: table_template
        type: OBJECT
        description: Template for a table (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "table_name",
            "description": "Description of table",
            "columns": [
              {
                "name": "id",
                "type": "UUID",
                "primary_key": true,
                "nullable": false,
                "description": "Primary key"
              },
              {
                "name": "column_name",
                "type": "VARCHAR(255)",
                "primary_key": false,
                "nullable": true,
                "description": "Description of column",
                "default": null
              }
            ],
            "indexes": [
              {
                "name": "idx_table_column",
                "columns": ["column_name"],
                "unique": false
              }
            ]
          }
        validation_rules: []
  - name: relationships
    description: Relationships between tables
    fields:
      - name: relationship_definitions
        type: ARRAY
        description: List of relationships
        required: true
        default_value: "[]"
        validation_rules: []
      - name: relationship_template
        type: OBJECT
        description: Template for a relationship (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "relationship_name",
            "type": "ONE_TO_MANY",
            "source_table": "parent_table",
            "target_table": "child_table",
            "source_column": "id",
            "target_column": "parent_id",
            "on_delete": "CASCADE"
          }
        validation_rules: []
EOF

# React Component Template
mkdir -p "$TEMPLATE_ROOT/frontend/web/components"
cat > "$TEMPLATE_ROOT/frontend/web/components/react_component_template.yaml" << 'EOF'
id: react_component_template
name: React Component Specification
description: Template for defining React components
version: 1.0.0
category: frontend/web
sections:
  - name: component_metadata
    description: General information about the component
    fields:
      - name: component_name
        type: STRING
        description: Name of the component (PascalCase)
        required: true
        default_value: ""
        validation_rules:
          - rule_type: PATTERN
            expression: "^[A-Z][a-zA-Z0-9]*$"
            error_message: "Component name must be in PascalCase"
      - name: description
        type: STRING
        description: Detailed description of the component
        required: false
        default_value: ""
        validation_rules: []
      - name: component_type
        type: ENUM
        description: Type of React component
        required: true
        default_value: "Functional"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Functional,Class,HOC,Custom Hook"
            error_message: "Component type must be one of Functional, Class, HOC, or Custom Hook"
  - name: props
    description: Component props
    fields:
      - name: prop_definitions
        type: ARRAY
        description: List of prop definitions
        required: true
        default_value: "[]"
        validation_rules: []
      - name: prop_template
        type: OBJECT
        description: Template for a prop (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "propName",
            "type": "string",
            "required": false,
            "description": "Description of prop",
            "default_value": null
          }
        validation_rules: []
  - name: state
    description: Component state
    fields:
      - name: state_definitions
        type: ARRAY
        description: List of state definitions
        required: true
        default_value: "[]"
        validation_rules: []
      - name: state_template
        type: OBJECT
        description: Template for a state variable (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "stateName",
            "type": "string",
            "description": "Description of state",
            "initial_value": ""
          }
        validation_rules: []
  - name: lifecycle
    description: Component lifecycle methods or effects
    fields:
      - name: lifecycle_hooks
        type: ARRAY
        description: List of lifecycle hooks or effects
        required: true
        default_value: "[]"
        validation_rules: []
      - name: hook_template
        type: OBJECT
        description: Template for a lifecycle hook or effect (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "useEffect",
            "description": "Effect description",
            "dependencies": ["state1", "prop1"],
            "cleanup_required": false
          }
        validation_rules: []
  - name: rendering
    description: Component rendering details
    fields:
      - name: jsx_structure
        type: CODE
        description: Pseudo-code for the JSX structure
        required: false
        default_value: ""
        validation_rules: []
      - name: css_modules
        type: BOOLEAN
        description: Whether to use CSS modules
        required: true
        default_value: "true"
        validation_rules: []
      - name: styling_approach
        type: ENUM
        description: Approach to styling
        required: true
        default_value: "CSS Modules"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "CSS Modules,Styled Components,Emotion,Tailwind,Plain CSS,SASS/SCSS"
            error_message: "Styling approach must be one of the listed values"
EOF

# Microservice Architecture Template
mkdir -p "$TEMPLATE_ROOT/architecture/microservices"
cat > "$TEMPLATE_ROOT/architecture/microservices/microservice_template.yaml" << 'EOF'
id: microservice_template
name: Microservice Specification
description: Template for defining a microservice
version: 1.0.0
category: architecture/microservices
sections:
  - name: service_metadata
    description: General information about the microservice
    fields:
      - name: service_name
        type: STRING
        description: Name of the microservice
        required: true
        default_value: ""
        validation_rules:
          - rule_type: PATTERN
            expression: "^[a-z][a-z0-9-]*$"
            error_message: "Service name must be lowercase, start with a letter, and contain only letters, numbers, and hyphens"
      - name: description
        type: STRING
        description: Detailed description of the service
        required: false
        default_value: ""
        validation_rules: []
      - name: version
        type: STRING
        description: Service version
        required: true
        default_value: "1.0.0"
        validation_rules:
          - rule_type: PATTERN
            expression: "^\\d+\\.\\d+\\.\\d+$"
            error_message: "Version must follow semantic versioning (e.g., 1.0.0)"
      - name: owner
        type: STRING
        description: Team or individual responsible for the service
        required: true
        default_value: ""
        validation_rules: []
  - name: service_interface
    description: Service API and interface details
    fields:
      - name: communication_style
        type: ENUM
        description: Primary communication style
        required: true
        default_value: "REST"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "REST,GraphQL,gRPC,Event-Based,Message Queue"
            error_message: "Communication style must be one of the listed values"
      - name: port
        type: INTEGER
        description: Port number
        required: true
        default_value: "8080"
        validation_rules:
          - rule_type: RANGE
            expression: "1024,65535"
            error_message: "Port must be between 1024 and 65535"
      - name: health_check_endpoint
        type: STRING
        description: Health check endpoint
        required: true
        default_value: "/health"
        validation_rules: []
      - name: metrics_endpoint
        type: STRING
        description: Metrics endpoint
        required: true
        default_value: "/metrics"
        validation_rules: []
  - name: dependencies
    description: Service dependencies
    fields:
      - name: service_dependencies
        type: ARRAY
        description: List of other services this service depends on
        required: true
        default_value: "[]"
        validation_rules: []
      - name: external_dependencies
        type: ARRAY
        description: List of external dependencies
        required: true
        default_value: "[]"
        validation_rules: []
  - name: data_management
    description: Data management details
    fields:
      - name: database_type
        type: ENUM
        description: Type of database used
        required: true
        default_value: "None"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "None,PostgreSQL,MySQL,MongoDB,Redis,Elasticsearch,Other"
            error_message: "Database type must be one of the listed values"
      - name: database_name
        type: STRING
        description: Name of the database
        required: false
        default_value: ""
        validation_rules: []
      - name: data_ownership
        type: ENUM
        description: Approach to data ownership
        required: true
        default_value: "Service Owns Data"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Service Owns Data,Shared Database,External Data Source"
            error_message: "Data ownership must be one of the listed values"
  - name: deployment
    description: Deployment details
    fields:
      - name: deployment_platform
        type: ENUM
        description: Platform for deployment
        required: true
        default_value: "Kubernetes"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Kubernetes,Docker Swarm,AWS ECS,Cloud Foundry,Serverless,Other"
            error_message: "Deployment platform must be one of the listed values"
      - name: resource_requirements
        type: OBJECT
        description: Resource requirements
        required: true
        default_value: |
          {
            "cpu": "100m",
            "memory": "256Mi",
            "storage": "1Gi"
          }
        validation_rules: []
      - name: scaling_policy
        type: OBJECT
        description: Scaling policy
        required: true
        default_value: |
          {
            "min_replicas": 1,
            "max_replicas": 5,
            "cpu_threshold": 70
          }
        validation_rules: []
  - name: event_streaming
    description: Event streaming configuration
    fields:
      - name: uses_event_streaming
        type: BOOLEAN
        description: Whether the service uses event streaming
        required: true
        default_value: "false"
        validation_rules: []
      - name: event_broker
        type: ENUM
        description: Event broker used
        required: false
        default_value: "Kafka"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Kafka,RabbitMQ,Apache Pulsar,NATS,AWS Kinesis,Azure Event Hubs,Other"
            error_message: "Event broker must be one of the listed values"
      - name: published_events
        type: ARRAY
        description: Events published by this service
        required: false
        default_value: "[]"
        validation_rules: []
      - name: subscribed_events
        type: ARRAY
        description: Events subscribed to by this service
        required: false
        default_value: "[]"
        validation_rules: []
EOF

# Add event-driven template
mkdir -p "$TEMPLATE_ROOT/architecture/event-driven"
cat > "$TEMPLATE_ROOT/architecture/event-driven/event_template.yaml" << 'EOF'
id: event_template
name: Event Definition
description: Template for defining events in an event-driven architecture
version: 1.0.0
category: architecture/event-driven
sections:
  - name: event_metadata
    description: General information about the event
    fields:
      - name: event_name
        type: STRING
        description: Name of the event
        required: true
        default_value: ""
        validation_rules:
          - rule_type: PATTERN
            expression: "^[A-Z][a-zA-Z0-9]*$"
            error_message: "Event name must be in PascalCase"
      - name: description
        type: STRING
        description: Detailed description of the event
        required: true
        default_value: ""
        validation_rules: []
      - name: version
        type: STRING
        description: Event schema version
        required: true
        default_value: "1.0.0"
        validation_rules:
          - rule_type: PATTERN
            expression: "^\\d+\\.\\d+\\.\\d+$"
            error_message: "Version must follow semantic versioning (e.g., 1.0.0)"
      - name: domain
        type: STRING
        description: Business domain this event belongs to
        required: true
        default_value: ""
        validation_rules: []
  - name: event_schema
    description: Schema details for the event
    fields:
      - name: schema_format
        type: ENUM
        description: Format of the event schema
        required: true
        default_value: "JSON"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "JSON,Avro,Protobuf,XML,Other"
            error_message: "Schema format must be one of the listed values"
      - name: payload_schema
        type: CODE
        description: Schema definition for the event payload
        required: true
        default_value: "{}"
        validation_rules: []
      - name: examples
        type: ARRAY
        description: Example event payloads
        required: false
        default_value: "[]"
        validation_rules: []
  - name: event_flow
    description: Information about event flow
    fields:
      - name: producer_service
        type: STRING
        description: Service that produces this event
        required: true
        default_value: ""
        validation_rules: []
      - name: consumer_services
        type: ARRAY
        description: Services that consume this event
        required: true
        default_value: "[]"
        validation_rules: []
      - name: event_channel
        type: STRING
        description: Channel/topic/queue where the event is published
        required: true
        default_value: ""
        validation_rules: []
      - name: delivery_guarantees
        type: ENUM
        description: Delivery guarantee for this event
        required: true
        default_value: "AT_LEAST_ONCE"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "AT_LEAST_ONCE,EXACTLY_ONCE,AT_MOST_ONCE"
            error_message: "Delivery guarantee must be one of the listed values"
  - name: event_handling
    description: Details about event handling
    fields:
      - name: retry_policy
        type: OBJECT
        description: Retry policy for failed event processing
        required: false
        default_value: |
          {
            "max_retries": 3,
            "backoff_strategy": "exponential",
            "initial_delay_ms": 1000,
            "max_delay_ms": 60000
          }
        validation_rules: []
      - name: dead_letter_channel
        type: STRING
        description: Channel for unprocessable events
        required: false
        default_value: ""
        validation_rules: []
      - name: idempotency_strategy
        type: ENUM
        description: Strategy to ensure idempotent processing
        required: true
        default_value: "IDEMPOTENCY_KEY"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "IDEMPOTENCY_KEY,NATURAL_KEY,NONE"
            error_message: "Idempotency strategy must be one of the listed values"
EOF

# Create relationship manager template
mkdir -p "$TEMPLATE_ROOT/common/relationships"
cat > "$TEMPLATE_ROOT/common/relationships/template_relationship.yaml" << 'EOF'
id: template_relationship
name: Template Relationship
description: Defines relationships between templates
version: 1.0.0
category: common/relationships
sections:
  - name: relationship_metadata
    description: General information about the relationship
    fields:
      - name: relationship_id
        type: STRING
        description: Unique identifier for the relationship
        required: true
        default_value: ""
        validation_rules:
          - rule_type: PATTERN
            expression: "^[a-zA-Z0-9_-]+$"
            error_message: "Relationship ID must contain only alphanumeric characters, underscores, and hyphens"
      - name: relationship_type
        type: ENUM
        description: Type of relationship
        required: true
        default_value: "EXTENDS"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "EXTENDS,REFERENCES,REQUIRES,GROUPS"
            error_message: "Relationship type must be one of EXTENDS, REFERENCES, REQUIRES, or GROUPS"
  - name: relationship_details
    description: Details of the relationship
    fields:
      - name: source_template_id
        type: STRING
        description: ID of the source template
        required: true
        default_value: ""
        validation_rules: []
      - name: target_template_id
        type: STRING
        description: ID of the target template
        required: true
        default_value: ""
        validation_rules: []
      - name: bidirectional
        type: BOOLEAN
        description: Whether the relationship is bidirectional
        required: true
        default_value: "false"
        validation_rules: []
      - name: reference_path
        type: STRING
        description: Path to the referenced field/section (for REFERENCES relationship)
        required: false
        default_value: ""
        validation_rules: []
EOF

# Create template evolution manager
mkdir -p "$TEMPLATE_ROOT/evolution"
cat > "$TEMPLATE_ROOT/evolution/template_evolution_manager.yaml" << 'EOF'
id: template_evolution_manager
name: Template Evolution Manager
description: Configuration for template evolution tracking and suggestions
version: 1.0.0
category: evolution
sections:
  - name: evolution_config
    description: Configuration for evolution analysis
    fields:
      - name: analysis_frequency
        type: ENUM
        description: How frequently to analyze templates for evolution
        required: true
        default_value: "WEEKLY"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "DAILY,WEEKLY,MONTHLY,ON_DEMAND"
            error_message: "Analysis frequency must be one of the listed values"
      - name: min_instances_for_analysis
        type: INTEGER
        description: Minimum number of template instances required for meaningful analysis
        required: true
        default_value: "10"
        validation_rules:
          - rule_type: RANGE
            expression: "1,1000"
            error_message: "Value must be between 1 and 1000"
      - name: suggestion_confidence_threshold
        type: FLOAT
        description: Minimum confidence score required for suggestions
        required: true
        default_value: "0.7"
        validation_rules:
          - rule_type: RANGE
            expression: "0.0,1.0"
            error_message: "Confidence threshold must be between 0.0 and 1.0"
  - name: field_evolution_metrics
    description: Metrics tracked for field evolution
    fields:
      - name: track_usage_count
        type: BOOLEAN
        description: Track how many times a field is used
        required: true
        default_value: "true"
        validation_rules: []
      - name: track_completion_rate
        type: BOOLEAN
        description: Track the rate at which fields are completed
        required: true
        default_value: "true"
        validation_rules: []
      - name: track_error_rate
        type: BOOLEAN
        description: Track the rate of validation errors for fields
        required: true
        default_value: "true"
        validation_rules: []
      - name: track_fill_time
        type: BOOLEAN
        description: Track the average time taken to fill a field
        required: true
        default_value: "true"
        validation_rules: []
      - name: track_common_values
        type: BOOLEAN
        description: Track common values used for fields
        required: true
        default_value: "true"
        validation_rules: []
      - name: common_values_threshold
        type: INTEGER
        description: Number of common values to track
        required: true
        default_value: "10"
        validation_rules:
          - rule_type: RANGE
            expression: "1,100"
            error_message: "Value must be between 1 and 100"
  - name: suggestion_types
    description: Types of suggestions to generate
    fields:
      - name: suggest_add_field
        type: BOOLEAN
        description: Suggest adding new fields
        required: true
        default_value: "true"
        validation_rules: []
      - name: suggest_remove_field
        type: BOOLEAN
        description: Suggest removing rarely used fields
        required: true
        default_value: "true"
        validation_rules: []
      - name: suggest_modify_field
        type: BOOLEAN
        description: Suggest modifying field properties
        required: true
        default_value: "true"
        validation_rules: []
      - name: suggest_reorder_fields
        type: BOOLEAN
        description: Suggest reordering fields for better UX
        required: true
        default_value: "true"
        validation_rules: []
      - name: suggest_add_validation
        type: BOOLEAN
        description: Suggest adding validation rules
        required: true
        default_value: "true"
        validation_rules: []
      - name: suggest_split_template
        type: BOOLEAN
        description: Suggest splitting templates that are too large
        required: true
        default_value: "true"
        validation_rules: []
      - name: suggest_merge_templates
        type: BOOLEAN
        description: Suggest merging similar templates
        required: true
        default_value: "true"
        validation_rules: []
EOF

echo "Spec Template Library has been generated!"
echo "----------------------------------------"
echo "Directory structure:"
find "$TEMPLATE_ROOT" -type d | sort
echo "----------------------------------------"
echo "Template files created:"
find "$TEMPLATE_ROOT" -name "*.yaml" | sort

# Create a Tailwind template
mkdir -p "$TEMPLATE_ROOT/frontend/styling/tailwind"
cat > "$TEMPLATE_ROOT/frontend/styling/tailwind/tailwind_config_template.yaml" << 'EOF'
id: tailwind_config_template
name: Tailwind CSS Configuration
description: Template for Tailwind CSS configuration and theme customization
version: 1.0.0
category: frontend/styling
sections:
  - name: project_info
    description: Project information
    fields:
      - name: project_name
        type: STRING
        description: Name of the project
        required: true
        default_value: ""
        validation_rules:
          - rule_type: MIN_LENGTH
            expression: "1"
            error_message: "Project name cannot be empty"
      - name: tailwind_version
        type: STRING
        description: Tailwind CSS version
        required: true
        default_value: "3.3.3"
        validation_rules:
          - rule_type: PATTERN
            expression: "^\\d+\\.\\d+\\.\\d+$"
            error_message: "Version must follow semantic versioning"
  - name: customization
    description: Theme customization options
    fields:
      - name: custom_colors
        type: BOOLEAN
        description: Whether to customize the color palette
        required: true
        default_value: "true"
        validation_rules: []
      - name: color_palette
        type: OBJECT
        description: Custom color palette
        required: false
        default_value: |
          {
            "primary": {
              "50": "#f0f9ff",
              "100": "#e0f2fe",
              "200": "#bae6fd",
              "300": "#7dd3fc",
              "400": "#38bdf8",
              "500": "#0ea5e9",
              "600": "#0284c7",
              "700": "#0369a1",
              "800": "#075985",
              "900": "#0c4a6e",
              "950": "#082f49"
            },
            "secondary": {
              "50": "#f5f3ff",
              "100": "#ede9fe",
              "200": "#ddd6fe",
              "300": "#c4b5fd",
              "400": "#a78bfa",
              "500": "#8b5cf6",
              "600": "#7c3aed",
              "700": "#6d28d9",
              "800": "#5b21b6",
              "900": "#4c1d95",
              "950": "#2e1065"
            }
          }
        validation_rules: []
      - name: custom_fonts
        type: BOOLEAN
        description: Whether to customize the font families
        required: true
        default_value: "false"
        validation_rules: []
      - name: font_families
        type: OBJECT
        description: Custom font families
        required: false
        default_value: |
          {
            "sans": ["Inter", "ui-sans-serif", "system-ui"],
            "serif": ["Merriweather", "ui-serif", "Georgia"],
            "mono": ["JetBrains Mono", "ui-monospace", "monospace"]
          }
        validation_rules: []
      - name: custom_spacing
        type: BOOLEAN
        description: Whether to customize spacing scale
        required: true
        default_value: "false"
        validation_rules: []
      - name: custom_breakpoints
        type: BOOLEAN
        description: Whether to customize breakpoints
        required: true
        default_value: "false"
        validation_rules: []
  - name: plugins
    description: Tailwind plugins to use
    fields:
      - name: typography
        type: BOOLEAN
        description: Use @tailwindcss/typography plugin
        required: true
        default_value: "false"
        validation_rules: []
      - name: forms
        type: BOOLEAN
        description: Use @tailwindcss/forms plugin
        required: true
        default_value: "false"
        validation_rules: []
      - name: line_clamp
        type: BOOLEAN
        description: Use @tailwindcss/line-clamp plugin
        required: true
        default_value: "false"
        validation_rules: []
      - name: aspect_ratio
        type: BOOLEAN
        description: Use @tailwindcss/aspect-ratio plugin
        required: true
        default_value: "false"
        validation_rules: []
      - name: custom_plugins
        type: ARRAY
        description: Custom plugins to include
        required: false
        default_value: "[]"
        validation_rules: []
  - name: advanced
    description: Advanced configuration options
    fields:
      - name: jit_mode
        type: BOOLEAN
        description: Use JIT (Just-In-Time) mode
        required: true
        default_value: "true"
        validation_rules: []
      - name: purge_enabled
        type: BOOLEAN
        description: Enable content purging for production
        required: true
        default_value: "true"
        validation_rules: []
      - name: purge_content
        type: ARRAY
        description: Files to scan for class names
        required: false
        default_value: |
          [
            "./pages/**/*.{js,ts,jsx,tsx}",
            "./components/**/*.{js,ts,jsx,tsx}"
          ]
        validation_rules: []
      - name: dark_mode
        type: ENUM
        description: Dark mode strategy
        required: true
        default_value: "media"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "media,class,false"
            error_message: "Dark mode must be one of media, class, or false"
EOF

# Create a Supabase database provider template
mkdir -p "$TEMPLATE_ROOT/backend/database/providers/supabase"
cat > "$TEMPLATE_ROOT/backend/database/providers/supabase/supabase_template.yaml" << 'EOF'
id: supabase_template
name: Supabase Configuration
description: Template for setting up and configuring Supabase
version: 1.0.0
category: backend/database/providers
sections:
  - name: project_info
    description: Supabase project information
    fields:
      - name: project_name
        type: STRING
        description: Name of the Supabase project
        required: true
        default_value: ""
        validation_rules:
          - rule_type: MIN_LENGTH
            expression: "1"
            error_message: "Project name cannot be empty"
      - name: region
        type: ENUM
        description: Supabase hosting region
        required: true
        default_value: "us-east-1"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "us-east-1,us-west-1,eu-central-1,eu-west-1,ap-southeast-1,ap-northeast-1"
            error_message: "Must be a valid Supabase region"
      - name: pricing_tier
        type: ENUM
        description: Supabase pricing tier
        required: true
        default_value: "Free"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Free,Pro,Team,Enterprise"
            error_message: "Must be a valid Supabase pricing tier"
  - name: database
    description: Database configuration
    fields:
      - name: postgres_version
        type: ENUM
        description: PostgreSQL version
        required: true
        default_value: "14"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "14,15"
            error_message: "Must be a valid PostgreSQL version"
      - name: schemas
        type: ARRAY
        description: Database schemas to create
        required: true
        default_value: |
          [
            "public",
            "auth"
          ]
        validation_rules: []
      - name: tables
        type: ARRAY
        description: Tables to create
        required: true
        default_value: "[]"
        validation_rules: []
      - name: table_template
        type: OBJECT
        description: Template for a table definition (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "table_name",
            "schema": "public",
            "columns": [
              {
                "name": "id",
                "type": "uuid",
                "primary_key": true,
                "nullable": false,
                "default": "gen_random_uuid()"
              },
              {
                "name": "created_at",
                "type": "timestamp with time zone",
                "nullable": false,
                "default": "now()"
              }
            ],
            "rls_enabled": true,
            "policies": []
          }
        validation_rules: []
      - name: policy_template
        type: OBJECT
        description: Template for an RLS policy (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "policy_name",
            "operation": "SELECT",
            "definition": "auth.uid() = user_id",
            "roles": ["authenticated"]
          }
        validation_rules: []
      - name: enable_extensions
        type: ARRAY
        description: PostgreSQL extensions to enable
        required: false
        default_value: |
          [
            "uuid-ossp",
            "pgcrypto"
          ]
        validation_rules: []
  - name: authentication
    description: Authentication settings
    fields:
      - name: enable_email_auth
        type: BOOLEAN
        description: Enable email/password authentication
        required: true
        default_value: "true"
        validation_rules: []
      - name: enable_phone_auth
        type: BOOLEAN
        description: Enable phone authentication
        required: true
        default_value: "false"
        validation_rules: []
      - name: enable_oauth
        type: BOOLEAN
        description: Enable OAuth providers
        required: true
        default_value: "false"
        validation_rules: []
      - name: oauth_providers
        type: ARRAY
        description: OAuth providers to enable
        required: false
        default_value: |
          [
            "google",
            "github"
          ]
        validation_rules: []
      - name: jwt_expiry
        type: STRING
        description: JWT token expiry time
        required: true
        default_value: "3600"
        validation_rules:
          - rule_type: PATTERN
            expression: "^\\d+$"
            error_message: "Must be a valid number in seconds"
  - name: storage
    description: Storage configuration
    fields:
      - name: enable_storage
        type: BOOLEAN
        description: Enable Supabase Storage
        required: true
        default_value: "false"
        validation_rules: []
      - name: storage_buckets
        type: ARRAY
        description: Storage buckets to create
        required: false
        default_value: |
          [
            {
              "name": "public",
              "public": true
            },
            {
              "name": "private",
              "public": false
            }
          ]
        validation_rules: []
  - name: api
    description: API and serverless functions
    fields:
      - name: enable_edge_functions
        type: BOOLEAN
        description: Enable Edge Functions
        required: true
        default_value: "false"
        validation_rules: []
      - name: edge_functions
        type: ARRAY
        description: Edge Functions to create
        required: false
        default_value: "[]"
        validation_rules: []
EOF

# Create a Vector Database template
mkdir -p "$TEMPLATE_ROOT/backend/database/vector"
cat > "$TEMPLATE_ROOT/backend/database/vector/vector_database_template.yaml" << 'EOF'
id: vector_database_template
name: Vector Database Configuration
description: Template for setting up and configuring a vector database
version: 1.0.0
category: backend/database
sections:
  - name: database_info
    description: Vector database information
    fields:
      - name: database_type
        type: ENUM
        description: Type of vector database
        required: true
        default_value: "pgvector"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "pgvector,pinecone,milvus,weaviate,qdrant,redis-vectorstore,elastic-vector-search"
            error_message: "Must be a supported vector database type"
      - name: hosting_option
        type: ENUM
        description: How the database will be hosted
        required: true
        default_value: "managed"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "managed,self-hosted,embedded"
            error_message: "Must be a valid hosting option"
      - name: region
        type: STRING
        description: Region for managed service (if applicable)
        required: false
        default_value: "us-east-1"
        validation_rules: []
  - name: vector_configuration
    description: Vector configurations
    fields:
      - name: embedding_dimensions
        type: INTEGER
        description: Dimensionality of the vector embeddings
        required: true
        default_value: "1536"
        validation_rules:
          - rule_type: RANGE
            expression: "2,8192"
            error_message: "Dimensions must be between 2 and 8192"
      - name: distance_metric
        type: ENUM
        description: Distance metric for similarity search
        required: true
        default_value: "cosine"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "cosine,euclidean,dot,manhattan"
            error_message: "Must be a supported distance metric"
      - name: indexing_strategy
        type: ENUM
        description: Indexing strategy for vector search
        required: true
        default_value: "flat"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "flat,ivf,hnsw,pq,ivf_pq,ivf_hnsw"
            error_message: "Must be a supported indexing strategy"
  - name: collections
    description: Vector collections/tables
    fields:
      - name: collection_definitions
        type: ARRAY
        description: List of vector collections
        required: true
        default_value: "[]"
        validation_rules: []
      - name: collection_template
        type: OBJECT
        description: Template for a vector collection (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "embeddings",
            "description": "Main embeddings collection",
            "metadata_fields": [
              {
                "name": "text",
                "type": "text",
                "indexed": true
              },
              {
                "name": "source",
                "type": "text",
                "indexed": true
              },
              {
                "name": "created_at",
                "type": "timestamp",
                "indexed": false
              }
            ],
            "vector_field": "embedding"
          }
        validation_rules: []
  - name: integration
    description: Integration details
    fields:
      - name: embedding_model
        type: STRING
        description: Embedding model to use
        required: false
        default_value: "text-embedding-ada-002"
        validation_rules: []
      - name: api_provider
        type: ENUM
        description: Provider for embedding API
        required: false
        default_value: "openai"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "openai,cohere,huggingface,azure,local,custom"
            error_message: "Must be a supported API provider"
      - name: chunk_size
        type: INTEGER
        description: Text chunk size for embedding
        required: false
        default_value: "1000"
        validation_rules:
          - rule_type: RANGE
            expression: "100,8000"
            error_message: "Chunk size must be between 100 and 8000"
      - name: chunk_overlap
        type: INTEGER
        description: Overlap between chunks
        required: false
        default_value: "200"
        validation_rules:
          - rule_type: RANGE
            expression: "0,4000"
            error_message: "Chunk overlap must be between 0 and 4000"
EOF

# Create a Design Pattern template
mkdir -p "$TEMPLATE_ROOT/architecture/design-patterns/creational"
cat > "$TEMPLATE_ROOT/architecture/design-patterns/creational/factory_pattern_template.yaml" << 'EOF'
id: factory_pattern_template
name: Factory Design Pattern
description: Template for implementing the Factory design pattern
version: 1.0.0
category: architecture/design-patterns
sections:
  - name: pattern_info
    description: Basic information about the pattern implementation
    fields:
      - name: pattern_name
        type: STRING
        description: Name of the pattern (Factory, Abstract Factory, etc.)
        required: true
        default_value: "Factory Method"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Factory Method,Abstract Factory,Simple Factory"
            error_message: "Must be a valid factory pattern type"
      - name: language
        type: ENUM
        description: Programming language
        required: true
        default_value: "TypeScript"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "TypeScript,JavaScript,Python,Java,C#,Go,Rust,PHP,Ruby"
            error_message: "Must be a supported programming language"
      - name: purpose
        type: STRING
        description: Purpose of this factory implementation
        required: true
        default_value: ""
        validation_rules:
          - rule_type: MIN_LENGTH
            expression: "10"
            error_message: "Purpose description should be meaningful"
  - name: product_interface
    description: Definition of the product interface/class
    fields:
      - name: product_name
        type: STRING
        description: Name of the product interface or abstract class
        required: true
        default_value: ""
        validation_rules:
          - rule_type: PATTERN
            expression: "^[A-Z][a-zA-Z0-9]*$"
            error_message: "Product name should be in PascalCase"
      - name: methods
        type: ARRAY
        description: Methods that the product interface declares
        required: true
        default_value: "[]"
        validation_rules: []
      - name: method_template
        type: OBJECT
        description: Template for a method (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "operation",
            "return_type": "string",
            "parameters": [],
            "description": "Main operation that concrete products implement"
          }
        validation_rules: []
  - name: concrete_products
    description: Concrete product implementations
    fields:
      - name: product_implementations
        type: ARRAY
        description: List of concrete product implementations
        required: true
        default_value: "[]"
        validation_rules: []
      - name: product_template
        type: OBJECT
        description: Template for a concrete product (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "ConcreteProductA",
            "description": "First concrete implementation of the product",
            "method_implementations": [
              {
                "method": "operation",
                "implementation": "Returns result from ConcreteProductA"
              }
            ]
          }
        validation_rules: []
  - name: factory
    description: Factory implementation
    fields:
      - name: factory_name
        type: STRING
        description: Name of the factory class/interface
        required: true
        default_value: ""
        validation_rules:
          - rule_type: PATTERN
            expression: "^[A-Z][a-zA-Z0-9]*$"
            error_message: "Factory name should be in PascalCase"
      - name: factory_type
        type: ENUM
        description: Type of factory implementation
        required: true
        default_value: "Class"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Class,Interface,Function"
            error_message: "Must be a valid factory type"
      - name: creation_method
        type: STRING
        description: Name of the creation method
        required: true
        default_value: "createProduct"
        validation_rules: []
      - name: parameters
        type: ARRAY
        description: Parameters for the creation method
        required: false
        default_value: "[]"
        validation_rules: []
  - name: concrete_factories
    description: Concrete factory implementations (for abstract factory)
    fields:
      - name: has_concrete_factories
        type: BOOLEAN
        description: Whether this pattern uses concrete factories
        required: true
        default_value: "false"
        validation_rules: []
      - name: factory_implementations
        type: ARRAY
        description: List of concrete factory implementations
        required: false
        default_value: "[]"
        validation_rules: []
      - name: factory_template
        type: OBJECT
        description: Template for a concrete factory (not to be filled out directly)
        required: false
        default_value: |
          {
            "name": "ConcreteFactoryA",
            "description": "Factory that creates product variant A",
            "products_created": ["ConcreteProductA"]
          }
        validation_rules: []
  - name: client_code
    description: Example client code using the factory
    fields:
      - name: client_code_sample
        type: CODE
        description: Sample client code showing how to use the factory
        required: false
        default_value: ""
        validation_rules: []
      - name: dependencies
        type: ARRAY
        description: Dependencies required by the client code
        required: false
        default_value: "[]"
        validation_rules: []
EOF

# Create an ML/AI template
mkdir -p "$TEMPLATE_ROOT/data/ml-ai/rag-systems"
cat > "$TEMPLATE_ROOT/data/ml-ai/rag-systems/rag_system_template.yaml" << 'EOF'
id: rag_system_template
name: RAG System Configuration
description: Template for configuring a Retrieval-Augmented Generation system
version: 1.0.0
category: data/ml-ai
sections:
  - name: system_info
    description: General information about the RAG system
    fields:
      - name: system_name
        type: STRING
        description: Name of the RAG system
        required: true
        default_value: ""
        validation_rules:
          - rule_type: MIN_LENGTH
            expression: "1"
            error_message: "System name cannot be empty"
      - name: description
        type: STRING
        description: Description of the RAG system's purpose
        required: true
        default_value: ""
        validation_rules: []
      - name: architecture_type
        type: ENUM
        description: Architecture type of the RAG system
        required: true
        default_value: "Basic"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Basic,Hierarchical,Contextual,Multi-Query,Hybrid,Agent-based"
            error_message: "Must be a valid RAG architecture type"
  - name: llm_configuration
    description: Configuration for the Language Model
    fields:
      - name: model_provider
        type: ENUM
        description: Provider of the LLM
        required: true
        default_value: "openai"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "openai,anthropic,google,cohere,huggingface,local,azure,mistral,other"
            error_message: "Must be a supported model provider"
      - name: model_name
        type: STRING
        description: Name of the model
        required: true
        default_value: "gpt-4"
        validation_rules: []
      - name: temperature
        type: FLOAT
        description: Temperature setting for generation
        required: true
        default_value: "0.7"
        validation_rules:
          - rule_type: RANGE
            expression: "0.0,2.0"
            error_message: "Temperature must be between 0.0 and 2.0"
      - name: max_tokens
        type: INTEGER
        description: Maximum tokens in model response
        required: true
        default_value: "1024"
        validation_rules:
          - rule_type: RANGE
            expression: "1,32000"
            error_message: "Max tokens must be between 1 and 32000"
      - name: system_prompt_template
        type: STRING
        description: Template for the system prompt
        required: false
        default_value: "You are a helpful assistant that answers questions based on the provided context. If the answer isn't in the context, say you don't know."
        validation_rules: []
  - name: retriever_configuration
    description: Configuration for the retrieval component
    fields:
      - name: retriever_type
        type: ENUM
        description: Type of retrieval mechanism
        required: true
        default_value: "vector_search"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "vector_search,hybrid,keyword,semantic,reranking"
            error_message: "Must be a valid retriever type"
      - name: vector_database
        type: ENUM
        description: Vector database used for retrieval
        required: false
        default_value: "pinecone"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "pinecone,pgvector,milvus,weaviate,qdrant,elasticsearch,chroma,redis,none"
            error_message: "Must be a supported vector database"
      - name: embedding_model
        type: STRING
        description: Embedding model for vectorization
        required: false
        default_value: "text-embedding-ada-002"
        validation_rules: []
      - name: embedding_dimensions
        type: INTEGER
        description: Dimensionality of the embeddings
        required: false
        default_value: "1536"
        validation_rules:
          - rule_type: RANGE
            expression: "2,8192"
            error_message: "Dimensions must be between 2 and 8192"
      - name: similarity_metric
        type: ENUM
        description: Similarity metric for retrieval
        required: false
        default_value: "cosine"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "cosine,euclidean,dot,manhattan"
            error_message: "Must be a supported similarity metric"
      - name: top_k
        type: INTEGER
        description: Number of top documents to retrieve
        required: true
        default_value: "5"
        validation_rules:
          - rule_type: RANGE
            expression: "1,100"
            error_message: "Top-K must be between 1 and 100"
      - name: reranker_model
        type: STRING
        description: Model for reranking (if applicable)
        required: false
        default_value: ""
        validation_rules: []
  - name: data_sources
    description: Sources of data for the RAG system
    fields:
      - name: data_source_types
        type: ARRAY
        description: Types of data sources
        required: true
        default_value: |
          [
            "documents",
            "web"
          ]
        validation_rules: []
      - name: document_loaders
        type: ARRAY
        description: Document loaders for different file types
        required: false
        default_value: |
          [
            "pdf",
            "docx",
            "txt"
          ]
        validation_rules: []
      - name: text_splitter
        type: OBJECT
        description: Configuration for text splitting
        required: true
        default_value: |
          {
            "type": "recursive_character",
            "chunk_size": 1000,
            "chunk_overlap": 200
          }
        validation_rules: []
      - name: data_preprocessing
        type: ARRAY
        description: Preprocessing steps for data
        required: false
        default_value: |
          [
            "remove_extra_whitespace",
            "sanitize_html"
          ]
        validation_rules: []
  - name: system_features
    description: Additional features of the RAG system
    fields:
      - name: use_prompt_templates
        type: BOOLEAN
        description: Whether to use prompt templates
        required: true
        default_value: "true"
        validation_rules: []
      - name: use_agents
        type: BOOLEAN
        description: Whether to use agents for complex queries
        required: true
        default_value: "false"
        validation_rules: []
      - name: streaming_enabled
        type: BOOLEAN
        description: Whether to enable streaming responses
        required: true
        default_value: "true"
        validation_rules: []
      - name: enable_feedback_collection
        type: BOOLEAN
        description: Whether to enable feedback collection
        required: true
        default_value: "true"
        validation_rules: []
      - name: enable_caching
        type: BOOLEAN
        description: Whether to enable caching of results
        required: true
        default_value: "true"
        validation_rules: []
      - name: enable_search_fallback
        type: BOOLEAN
        description: Whether to enable search fallback when no results
        required: true
        default_value: "false"
        validation_rules: []
EOF

# Create a Chat Template
mkdir -p "$TEMPLATE_ROOT/templates/chat/customer-support"
cat > "$TEMPLATE_ROOT/templates/chat/customer-support/customer_support_template.yaml" << 'EOF'
id: customer_support_template
name: Customer Support Chatbot
description: Template for configuring a customer support chat assistant
version: 1.0.0
category: templates/chat
sections:
  - name: bot_info
    description: General information about the chat assistant
    fields:
      - name: bot_name
        type: STRING
        description: Name of the chat assistant
        required: true
        default_value: ""
        validation_rules:
          - rule_type: MIN_LENGTH
            expression: "1"
            error_message: "Bot name cannot be empty"
      - name: company_name
        type: STRING
        description: Name of the company
        required: true
        default_value: ""
        validation_rules:
          - rule_type: MIN_LENGTH
            expression: "1"
            error_message: "Company name cannot be empty"
      - name: description
        type: STRING
        description: Description of the chat assistant's purpose
        required: true
        default_value: ""
        validation_rules: []
      - name: avatar_url
        type: STRING
        description: URL for the chat assistant's avatar
        required: false
        default_value: ""
        validation_rules: []
  - name: personality
    description: Personality and tone settings
    fields:
      - name: tone
        type: ENUM
        description: Overall tone of the assistant
        required: true
        default_value: "Friendly"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Friendly,Professional,Casual,Formal,Helpful,Empathetic,Concise"
            error_message: "Must be a valid tone"
      - name: formality_level
        type: ENUM
        description: Level of formality
        required: true
        default_value: "Moderate"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Very Formal,Formal,Moderate,Casual,Very Casual"
            error_message: "Must be a valid formality level"
      - name: persona_description
        type: STRING
        description: Detailed description of the assistant's persona
        required: false
        default_value: "A friendly and helpful customer support representative who aims to resolve issues efficiently while maintaining a positive tone."
        validation_rules: []
      - name: conversation_style
        type: ENUM
        description: Conversational style
        required: true
        default_value: "Balanced"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Brief and Direct,Balanced,Detailed and Thorough"
            error_message: "Must be a valid conversation style"
  - name: knowledge_base
    description: Knowledge base configuration
    fields:
      - name: knowledge_source_type
        type: ENUM
        description: Type of knowledge source
        required: true
        default_value: "RAG"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "RAG,Fine-tuned Model,Hybrid,None"
            error_message: "Must be a valid knowledge source type"
      - name: product_documentation_url
        type: STRING
        description: URL to product documentation
        required: false
        default_value: ""
        validation_rules: []
      - name: faq_url
        type: STRING
        description: URL to FAQ page
        required: false
        default_value: ""
        validation_rules: []
      - name: use_web_search
        type: BOOLEAN
        description: Whether to enable web search
        required: true
        default_value: "false"
        validation_rules: []
      - name: knowledge_cutoff_date
        type: STRING
        description: Knowledge cutoff date for the model
        required: false
        default_value: ""
        validation_rules: []
  - name: capabilities
    description: Bot capabilities and features
    fields:
      - name: handle_account_issues
        type: BOOLEAN
        description: Handle account-related issues
        required: true
        default_value: "true"
        validation_rules: []
      - name: handle_product_inquiries
        type: BOOLEAN
        description: Handle product-related inquiries
        required: true
        default_value: "true"
        validation_rules: []
      - name: handle_billing_issues
        type: BOOLEAN
        description: Handle billing-related issues
        required: true
        default_value: "true"
        validation_rules: []
      - name: handle_technical_support
        type: BOOLEAN
        description: Handle technical support requests
        required: true
        default_value: "true"
        validation_rules: []
      - name: enable_human_handoff
        type: BOOLEAN
        description: Enable handoff to human agents
        required: true
        default_value: "true"
        validation_rules: []
      - name: collect_user_feedback
        type: BOOLEAN
        description: Collect feedback after conversations
        required: true
        default_value: "true"
        validation_rules: []
  - name: system_instructions
    description: System instructions for the assistant
    fields:
      - name: greeting_message
        type: STRING
        description: Initial greeting message
        required: true
        default_value: "Hello! I'm {bot_name}, the virtual assistant for {company_name}. How can I help you today?"
        validation_rules: []
      - name: system_prompt
        type: STRING
        description: System prompt for the assistant
        required: true
        default_value: "You are {bot_name}, a customer support assistant for {company_name}. Your goal is to help customers with their questions and issues in a {tone} and {formality_level} manner. Provide accurate information based on the company's knowledge base and policies. If you don't know the answer, acknowledge this and offer to connect the customer with a human agent."
        validation_rules: []
      - name: fallback_message
        type: STRING
        description: Message when the assistant can't help
        required: true
        default_value: "I apologize, but I don't have enough information to help with that. Would you like me to connect you with a human support agent?"
        validation_rules: []
      - name: human_handoff_message
        type: STRING
        description: Message when handing off to a human
        required: true
        default_value: "I'll connect you with a human support agent who can better assist you. Please wait a moment while I transfer your conversation."
        validation_rules: []
      - name: closing_message
        type: STRING
        description: Message at the end of a conversation
        required: true
        default_value: "Thank you for contacting {company_name} support. Is there anything else I can help you with today?"
        validation_rules: []
  - name: integration
    description: Integration settings
    fields:
      - name: platform
        type: ENUM
        description: Platform for deployment
        required: true
        default_value: "Website"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Website,Mobile App,WhatsApp,Facebook Messenger,Slack,Discord,SMS,Email,Intercom,Zendesk,Custom"
            error_message: "Must be a valid platform"
      - name: access_control
        type: ENUM
        description: Access control level
        required: true
        default_value: "Public"
        validation_rules:
          - rule_type: ENUM_VALUES
            expression: "Public,Authenticated Users Only,Specific User Groups"
            error_message: "Must be a valid access control level"
      - name: collect_user_info
        type: BOOLEAN
        description: Collect user information during chat
        required: true
        default_value: "true"
        validation_rules: []
      - name: required_user_fields
        type: ARRAY
        description: Required user information fields
        required: false
        default_value: |
          [
            "email",
            "name"
          ]
        validation_rules: []
      - name: analytics_enabled
        type: BOOLEAN
        description: Enable analytics collection
        required: true
        default_value: "true"
        validation_rules: []
EOF

echo "Done!"