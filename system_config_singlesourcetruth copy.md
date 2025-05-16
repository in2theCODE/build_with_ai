- Checksum verification for migrated data
   - Functional validation of queries and operations
   - Parallel testing in production-like environment

## Infrastructure Specifications

### Kubernetes Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Kubernetes | 1.26.x or higher | Production clusters must use official security-patched releases |
| CNI Plugin | Calico 3.25.x or higher | For network policy enforcement |
| CSI Driver | AWS EBS CSI 2.17.x / GCP PD CSI 1.9.x | Depending on cloud provider |
| Ingress Controller | Nginx Ingress Controller 1.5.x | With WAF capabilities |
| Service Mesh | Istio 1.18.x | For mTLS and advanced traffic management |
| Certificate Manager | cert-manager 1.11.x | For automatic TLS certificate management |
| Secrets Management | Vault 1.12.x | Accessible via the Vault Kubernetes integration |

### Node Specifications

| Node Pool | Instance Type (AWS) | Instance Type (GCP) | Min Nodes | Max Nodes | Node Labels | Node Taints |
|-----------|---------------------|---------------------|-----------|-----------|------------|------------|
| System | m5.2xlarge | n2-standard-8 | 3 | 5 | `role=system` | None |
| General | c5.4xlarge | c2-standard-16 | 5 | 20 | `role=general` | None |
| Memory-Optimized | r5.4xlarge | n2-highmem-16 | 3 | 15 | `role=memory` | None |
| ML-Inference | g4dn.xlarge | n1-standard-4-nvidia-tesla-t4 | 2 | 10 | `role=ml-inference` | `ml-gpu=true:NoSchedule` |
| Database | r5.2xlarge | n2-highmem-8 | 3 | 9 | `role=database` | `dedicated=database:NoSchedule` |

### Node Configuration Requirements

```yaml
# Operating System
os:
  distribution: "Ubuntu Server"
  version: "22.04 LTS"
  kernel_version: "5.15" or newer
  
# System Configuration
sysctl:
  vm.max_map_count: 262144
  vm.swappiness: 0
  fs.file-max: 1000000
  net.ipv4.ip_local_port_range: "1024 65535"
  net.ipv4.tcp_fin_timeout: 30
  net.core.somaxconn: 65535

# Storage Configuration
storage:
  tmp_size: "20Gi"
  docker_size: "100Gi"
  kubelet_directory: "/var/lib/kubelet"
  kubelet_size: "100Gi"
  
# Security Requirements
security:
  seccomp: "runtime/default"
  selinux: Enforcing
  apparmor: Enabled
  cis_benchmark: Level 1 Compliant
```

### Cloud Provider Specific Configuration

#### AWS-Specific Configuration

```yaml
aws:
  region: us-east-1
  secondary_region: us-west-2  # For disaster recovery
  vpc:
    cidr: "10.0.0.0/16"
    private_subnets:
      - "10.0.1.0/24"
      - "10.0.2.0/24"
      - "10.0.3.0/24"
    public_subnets:
      - "10.0.101.0/24"
      - "10.0.102.0/24"
      - "10.0.103.0/24"
  eks:
    version: "1.26"
    control_plane_logging:
      - "api"
      - "audit"
      - "authenticator"
      - "controllerManager"
      - "scheduler"
  rds:
    instance_class: "db.r6g.2xlarge"
    multi_az: true
    backup_retention_period: 30
  elasticache:
    node_type: "cache.r6g.xlarge"
    num_cache_nodes: 3
    multi_az: true
  s3:
    backup_bucket: "edrs-backups-{environment}"
    log_bucket: "edrs-logs-{environment}"
    lifecycle_rules:
      transition_to_ia: 30  # days
      transition_to_glacier: 90  # days
      expiration: 365  # days
```

#### GCP-Specific Configuration

```yaml
gcp:
  region: us-central1
  secondary_region: us-west1  # For disaster recovery
  network:
    name: "edrs-network"
    subnets:
      - name: "edrs-subnet-1"
        cidr: "10.0.0.0/20"
      - name: "edrs-subnet-2"
        cidr: "10.0.16.0/20"
      - name: "edrs-subnet-3"
        cidr: "10.0.32.0/20"
  gke:
    version: "1.26"
    logging_components:
      - "SYSTEM_COMPONENTS"
      - "WORKLOADS"
    monitoring_components:
      - "SYSTEM_COMPONENTS"
      - "WORKLOADS"
  cloud_sql:
    tier: "db-custom-16-61440"
    high_availability: true
    backup_retention_period: 30
  memorystore:
    tier: "standard"
    memory_size_gb: 32
    read_replicas_mode: "READ_REPLICAS_ENABLED"
  gcs:
    backup_bucket: "edrs-backups-{environment}"
    log_bucket: "edrs-logs-{environment}"
    lifecycle_rules:
      nearline_age: 30  # days
      coldline_age: 90  # days
      archive_age: 365  # days
```

### Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: edrs-api-gateway
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "X-XSS-Protection: 1; mode=block";
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.edrs.example.com
    secretName: edrs-api-tls
  rules:
  - host: api.edrs.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nginx-gateway
            port:
              number: 80
```

### Resource Quotas and Limits

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: edrs-resource-quota
spec:
  hard:
    # Compute Resources
    requests.cpu: "500"
    requests.memory: 1000Gi
    limits.cpu: "750"
    limits.memory: 1500Gi
    
    # Storage Resources
    requests.storage: 10Ti
    persistentvolumeclaims: 500
    
    # Object Count Limits
    pods: 1000
    services: 200
    secrets: 300
    configmaps: 300
    services.loadbalancers: 10
```

## CI/CD and Development Workflow

### CI/CD Pipeline Specifications

```yaml
ci_cd_pipeline:
  tools:
    # Core CI/CD Platform
    platform: "GitLab CI/CD"  # Alternative: "GitHub Actions", "Jenkins"
    version: "15.x or higher"
    
    # Container Registry
    container_registry: "AWS ECR"  # Alternative: "GCR", "Harbor"
    
    # Artifact Repository
    artifact_repository: "Nexus Repository OSS"
    version: "3.x or higher"
    
    # IaC Tool
    infrastructure_as_code: "Terraform"
    version: "1.4.x or higher"
    
    # Secret Management
    secret_management: "HashiCorp Vault"
    version: "1.12.x or higher"
    
  stages:
    - name: "code-quality"
      tools: ["SonarQube", "ESLint", "Black"]
      sla: "< 10 minutes"
      
    - name: "security-scan"
      tools: ["Snyk", "Trivy", "OWASP Dependency Check"]
      sla: "< 15 minutes"
      
    - name: "build"
      tools: ["Docker", "Buildah"]
      sla: "< 20 minutes"
      
    - name: "unit-test"
      tools: ["Jest", "PyTest", "JUnit"]
      sla: "< 15 minutes"
      
    - name: "integration-test"
      tools: ["Postman", "Cypress"]
      sla: "< 30 minutes"
      
    - name: "deploy-dev"
      tools: ["Helm", "ArgoCD"]
      sla: "< 15 minutes"
      auto_approve: true
      
    - name: "deploy-staging"
      tools: ["Helm", "ArgoCD"]
      sla: "< 15 minutes"
      auto_approve: false
      manual_approval: true
      
    - name: "performance-test"
      tools: ["JMeter", "k6"]
      sla: "< 45 minutes"
      
    - name: "deploy-prod"
      tools: ["Helm", "ArgoCD"]
      sla: "< 30 minutes"
      auto_approve: false
      manual_approval: true
      rollback_plan: true
```

### Build Pipeline Details

| Service Type | Build Steps | Test Steps | Artifact Type | Build Time SLA |
|--------------|-------------|------------|--------------|----------------|
| Node.js Services | npm install → lint → test → build | Jest, ESLint | Docker Image | < 10 minutes |
| Python Services | pip install → black → pytest → build | Pytest, Black | Docker Image | < 10 minutes |
| Java Services | gradle build | JUnit, SpotBugs | Docker Image | < 15 minutes |
| Database Migrations | validate → test | Schema validation | SQL Scripts | < 5 minutes |
| Infrastructure | terraform validate → plan | Terratest | Terraform Plan | < 10 minutes |

### Development Environment Specifications

```yaml
development_environment:
  # Local Development
  local:
    requirements:
      cpu: "8+ cores"
      memory: "32GB+ RAM"
      storage: "100GB+ SSD"
      docker_version: "20.10.x or higher"
      kubernetes: "minikube v1.28.x or higher" or "kind v0.19.x or higher"
    
    local_services:
      databases:
        - "PostgreSQL 14.x (Docker)"
        - "MongoDB 6.x (Docker)"
        - "Redis 7.x (Docker)"
      message_brokers:
        - "Apache Pulsar 3.x (Docker)"
      
    mocks:
      external_services: "Wiremock"
      ml_models: "Pre-trained lightweight app for local testing"
  
  # Development Environment
  dev:
    deployment_strategy: "Continuous Deployment"
    data_persistence: "Ephemeral (cleared weekly)"
    resource_limits: "50% of production"
    data_subset: "Anonymized sample data (5% of production)"
  
  # Testing Environment
  test:
    deployment_strategy: "On-demand deployment"
    data_persistence: "Reset after each test run"
    resource_limits: "75% of production"
    data_subset: "Synthetic test data"
    
  # Staging Environment
  staging:
    deployment_strategy: "Manual deployment after UAT"
    data_persistence: "Persistent (synchronized with production weekly)"
    resource_limits: "100% of production"
    data_subset: "Anonymized production data (30% volume)"
```

### Version Control and Branching Strategy

```yaml
version_control:
  platform: "Git"
  repository_hosting: "GitLab" # or "GitHub"
  
  branching_strategy:
    model: "GitFlow"
    branches:
      main:
        description: "Production-ready code"
        protection: true
        require_reviews: true
        require_ci_pass: true
      develop:
        description: "Integration branch for next release"
        protection: true
        require_reviews: true
        require_ci_pass: true
      feature:
        naming: "feature/{ticket-id}-{short-description}"
        branched_from: "develop"
        merged_to: "develop"
        require_reviews: true
      release:
        naming: "release/v{major}.{minor}.{patch}"
        branched_from: "develop"
        merged_to: ["main", "develop"]
        require_reviews: true
      hotfix:
        naming: "hotfix/{ticket-id}-{short-description}"
        branched_from: "main"
        merged_to: ["main", "develop"]
        require_reviews: true
```

### Testing Requirements

```yaml
testing:
  unit_testing:
    coverage_threshold: 80%
    frameworks:
      node: "Jest"
      python: "pytest"
      java: "JUnit5"
  
  integration_testing:
    coverage_threshold: 70%
    frameworks:
      api: "Postman + Newman"
      service: "Custom test harness + Docker Compose"
  
  performance_testing:
    requirements:
      throughput:
        api_gateway: "1000 requests/second"
        neural_interpreter: "500 requests/second"
        llm_inference: "50 requests/second"
      latency:
        p95:
          api_gateway: "< 200ms"
          neural_interpreter: "< 500ms"
          llm_inference: "< 2000ms"
      endurance:
        duration: "24 hours"
        degradation_threshold: "< 10%"
    tools: ["k6", "JMeter"]
  
  security_testing:
    requirements:
      sast: "SonarQube, Snyk Code"
      dast: "OWASP ZAP"
      dependency_scanning: "Snyk, OWASP Dependency Check"
      container_scanning: "Trivy, Clair"
      cadence: "Every build + Weekly full scan"
```

### Artifact Management

```yaml
artifacts:
  container_images:
    registry: "AWS ECR" # or "GCR", "Harbor"
    naming: "edrs/{service-name}:{git-tag}"
    retention:
      development: "7 days"
      staging: "30 days"
      production: "180 days"
      
  helm_charts:
    repository: "Nexus Repository" # or "ChartMuseum", "AWS ECR"
    naming: "edrs-{service-name}-{chart-version}"
    
  documentation:
    repository: "GitLab Wiki" # or "Confluence"
    api_docs: "Swagger UI + OpenAPI 3.0 Specs"
    
  infrastructure:
    terraform_state:
      storage: "AWS S3 + DynamoDB" # or "GCS + Datastore"
      locking: true
      versioning: true
```

## Disaster Recovery and Business Continuity

### Disaster Recovery Strategy

```yaml
disaster_recovery:
  # Recovery Objectives
  recovery_time_objective:
    tier1_services: "4 hours"  # Critical path services
    tier2_services: "8 hours"  # Supporting services
    tier3_services: "24 hours" # Non-critical services
  
  recovery_point_objective:
    tier1_data: "15 minutes"   # Critical data (user sessions, active workloads)
    tier2_data: "1 hour"       # Important operational data
    tier3_data: "24 hours"     # Historical/analytics data
  
  # Service Tier Classifications
  service_tiers:
    tier1:
      - "nginx-gateway"
      - "neural-interpreter"
      - "identity-manager"
      - "working-memory-service"
    tier2:
      - "llm-inference"
      - "embedding-service"
      - "knowledge-synthesizer"
      - "episodic-memory-service"
    tier3:
      - "feedback-processor"
      - "meta-learner"
      - "information-retrieval"
  
  # Recovery Strategy
  strategy:
    approach: "Multi-region active-passive"
    primary_region: "us-east-1" # or "us-central1" on GCP
    dr_region: "us-west-2" # or "us-west1" on GCP
    data_replication:
      databases: "Continuous replication"
      object_storage: "Cross-region replication"
      kafka_topics: "MirrorMaker 2"
    dr_activation:
      decision_authority: "Head of Operations + CTO"
      estimated_activation_time: "< 30 minutes"
```

### Business Continuity Planning

```yaml
business_continuity:
  # Critical Business Functions
  critical_functions:
    - function: "User authentication and authorization"
      systems: ["identity-manager", "auth-sidecar"]
      recovery_priority: 1
      
    - function: "Core reasoning capabilities"
      systems: ["neural-interpreter", "working-memory-service", "llm-inference"]
      recovery_priority: 1
      
    - function: "API availability"
      systems: ["nginx-gateway", "api-connector-framework"]
      recovery_priority: 1
      
    - function: "Knowledge access"
      systems: ["domain-knowledge-base", "vector-store", "information-retrieval"]
      recovery_priority: 2
  
  # Operational Procedures
  procedures:
    incident_response_plan: "docs/incident-response.md"
    dr_activation_plan: "docs/dr-activation.md"
    dr_testing_plan: "docs/dr-testing.md"
    communication_plan: "docs/crisis-communication.md"
    
  # Testing Schedule
  testing:
    full_dr_test: "Semi-annually"
    component_recovery_test: "Quarterly"
    tabletop_exercises: "Quarterly"
    chaos_engineering: "Monthly"
```

### Backup and Recovery Procedures

```yaml
backup_procedures:
  # Database Backups
  databases:
    postgresql:
      method: "WAL archiving + daily full backups"
      retention: "30 days"
      encryption: "AES-256"
      validation: "Weekly recovery validation"
      
    mongodb:
      method: "Daily snapshots + oplog archiving"
      retention: "30 days"
      encryption: "AES-256"
      validation: "Weekly recovery validation"
      
    vector_databases:
      method: "Daily snapshots"
      retention: "14 days"
      encryption: "AES-256"
      validation: "Weekly recovery validation"
  
  # Object Storage
  object_storage:
    method: "Cross-region replication + versioning"
    retention: "90 days for deleted objects"
    lifecycle: "Transition to cold storage after 30 days"
  
  # Application Configuration
  configuration:
    method: "GitOps with version control"
    recovery: "Infrastructure as Code deployment"
```

### Multi-Region Architecture

```yaml
multi_region:
  # Primary Region Components
  primary_region:
    active_components: "All services"
    ingress: "Primary DNS entry points"
    data_storage: "Primary databases and storage"
    
  # DR Region Components
  dr_region:
    active_components: "Monitoring, read replicas, standby services"
    ingress: "Secondary DNS entry points (inactive until failover)"
    data_storage: "Read replicas, backup storage"
    
  # Traffic Management
  traffic_management:
    dns_provider: "AWS Route53" # or "Google Cloud DNS"
    failover_mechanism: "Health-based automated DNS failover"
    health_check_endpoints:
      - "https://api.edrs.example.com/health"
      - "https://admin.edrs.example.com/health"
```

### Backup Validation Procedures

```yaml
backup_validation:
  # Validation Process
  process:
    frequency: "Weekly for critical systems, Monthly for others"
    method: "Automated recovery to isolated environment"
    verification: "Automated data integrity checks + functional tests"
    
  # Documentation Requirements
  documentation:
    evidence_retention: "13 months"
    validation_checklist: "docs/backup-validation-checklist.md"
    defect_tracking: "JIRA project: EDRS-BACKUP"
    
  # Validation Metrics
  metrics:
    recovery_time: "Time to recover from backup"
    data_completeness: "Percentage of data successfully validated"
    success_rate: "Percentage of successful validations"
```

## API Gateway and Documentation

### API Versioning Strategy

```yaml
api_versioning:
  # API Versioning Approach
  strategy: "URL path versioning"
  url_pattern: "/api/v{major_version}/{resource}"
  
  # Version Lifecycle
  lifecycle:
    deprecation_period: "6 months after next version release"
    sunset_notification: "3 months before end-of-life"
    maximum_supported_versions: 2
    
  # Version Compatibility
  compatibility:
    breaking_changes: "Major version increment only"
    backward_compatibility: "Required within same major version"
    feature_toggle: "Used for phased rollouts within minor versions"
```

### API Documentation Requirements

```yaml
api_documentation:
  # Documentation Standards
  standard: "OpenAPI 3.1"
  repository: "Git repository: edrs-api-specs"
  
  # Documentation Requirements
  requirements:
    - "Complete endpoint descriptions with method, URL, parameters"
    - "Request and response schemas with examples"
    - "Authentication requirements"
    - "Rate limiting information"
    - "Error responses and codes"
    - "Pagination patterns where applicable"
    
  # Documentation Delivery
  delivery:
    interactive: "Swagger UI at /api/docs"
    static: "Redoc at /api/redoc"
    downloadable: "OpenAPI JSON/YAML at /api/openapi.json"
    
  # Documentation CI/CD
  ci_cd:
    validation: "OpenAPI linting in CI pipeline"
    preview: "Generated for each PR"
    publication: "Automatic on merge to main branch"
```

### API Schema Validation

```yaml
api_schema_validation:
  # Validation Approach
  approach: "Runtime validation against OpenAPI schema"
  implementation: "API Gateway middleware"
  
  # Validation Policies
  policies:
    request_validation: "Strict validation on all endpoints"
    response_validation: "Log-only validation for responses"
    error_handling: "Standardized 400 response for validation errors"
    
  # Testing Requirements
  testing:
    contract_testing: "Required for all API changes"
    negative_testing: "Required for all input validations"
    fuzz_testing: "Weekly automated fuzzing of public endpoints"
```

### API Governance

```yaml
api_governance:
  # Design Guidelines
  design_principles:
    - "REST-based design with resource-oriented URLs"
    - "Consistent naming conventions (plural nouns for resources)"
    - "HTTP methods for CRUD operations"
    - "Query parameters for filtering, sorting, pagination"
    - "Standardized error formats"
    
  # Style Guide
  style_guide:
    url_format: "lowercase with hyphens (kebab-case)"
    query_params: "camelCase"
    json_properties: "camelCase"
    header_names: "Hyphenated-Pascal-Case"
    
  # Review Process
  review_process:
    required_reviewers: ["API architect", "Security team member"]
    review_checklist: "docs/api-review-checklist.md"
    approval_workflow: "Documented in JIRA workflow"
```

### External API Integration

```yaml
external_api_integration:
  # Integration Standards
  standards:
    authentication: "OAuth 2.0 or API Key based on provider requirements"
    retry_policy: "Exponential backoff with jitter"
    circuit_breaker: "Required for all external dependencies"
    
  # API Management
  management:
    catalog: "Internal API catalog in Developer Portal"
    credential_storage: "HashiCorp Vault"
    monitoring: "Prometheus metrics for all external calls"
    
  # Integration Testing
  testing:
    mocking: "Wiremock for development and testing"
    contract_testing: "Pact for provider contract verification"
    integration_testing: "Automated tests against test environments"
    performance_testing: "Load testing with realistic usage patterns"
```

## Service Level Objectives

### SLO Definitions

```yaml
service_level_objectives:
  # API Gateway SLOs
  api_gateway:
    availability: "99.9% monthly uptime"
    latency: "p95 < 250ms, p99 < 500ms"
    error_rate: "< 0.1% 5xx errors"
    throughput: "Support 1000 requests/second"
    
  # Core Processing SLOs
  neural_interpreter:
    availability: "99.9% monthly uptime"
    latency: "p95 < 500ms, p99 < 1000ms"
    error_rate: "< 0.5% 5xx errors"
    throughput: "Support 500 requests/second"
    
  # ML Inference SLOs
  llm_inference:
    availability: "99.5% monthly uptime"
    latency: "p95 < 2000ms, p99 < 5000ms"
    error_rate: "< 1% 5xx errors"
    throughput: "Support 50 requests/second"
    
  # Database SLOs
  databases:
    availability: "99.99% monthly uptime"
    latency: "p95 < 100ms, p99 < 200ms"
    error_rate: "< 0.01% errors"
    replication_lag: "< 10 seconds"
    
  # Event System SLOs
  event_system:
    availability: "99.9% monthly uptime"
    latency: "p95 < 500ms end-to-end delivery"
    error_rate: "< 0.01% message loss"
    throughput: "Support 5000 events/second"
```

### Error Budget Policy

```yaml
error_budget:
  # Budget Allocation
  allocation:
    monthly_budget: "Complement of SLO (e.g., 0.1% for 99.9% SLO)"
    consumption_tracking: "Rolling 30-day window"
    
  # Budget Consumption Rules
  consumption_rules:
    planned_maintenance: "Counted against budget if customer-impacting"
    emergency_fixes: "Counted against budget"
    third_party_outages: "Counted against budget if within our SLO boundary"
    
  # Budget Enforcement
  enforcement:
    0_25_percent_consumed: "No restrictions"
    25_50_percent_consumed: "Heightened monitoring"
    50_75_percent_consumed: "Non-essential deployments require approval"
    75_100_percent_consumed: "Feature freeze, only emergency changes"
    budget_exhausted: "Complete freeze, emergency changes require CTO approval"
```

### Monitoring and Alerting Thresholds

```yaml
monitoring_thresholds:
  # API Gateway Thresholds
  api_gateway:
    latency:
      warning: "p95 > 200ms for 5 minutes"
      critical: "p95 > 400ms for 5 minutes"
    error_rate:
      warning: "Error rate > 0.05% for 5 minutes"
      critical: "Error rate > 0.1% for 5 minutes"
    saturation:
      warning: "CPU > 70% for 15 minutes"
      critical: "CPU > 85% for 5 minutes"
      
  # Neural Interpreter Thresholds
  neural_interpreter:
    latency:
      warning: "p95 > 400ms for 5 minutes"
      critical: "p95 > 800ms for 5 minutes"
    error_rate:
      warning: "Error rate > 0.2% for 5 minutes"
      critical: "Error rate > 0.5% for 5 minutes"
    saturation:
      warning: "Memory usage > 75% for 15 minutes"
      critical: "Memory usage > 90% for 5 minutes"
      
  # LLM Inference Thresholds
  llm_inference:
    latency:
      warning: "p95 > 1800ms for 5 minutes"
      critical: "p95 > 4000ms for 5 minutes"
    error_rate:
      warning: "Error rate > 0.5% for 5 minutes"
      critical: "Error rate > 1% for 5 minutes"
    gpu_utilization:
      warning: "GPU utilization > 85% for 15 minutes"
      critical: "GPU memory usage > 95% for 5 minutes"
      
  # Database Thresholds
  databases:
    latency:
      warning: "p95 > 80ms for 5 minutes"
      critical: "p95 > 150ms for 5 minutes"
    connection_usage:
      warning: "Connection pool usage > 70% for 15 minutes"
      critical: "Connection pool usage > 90% for 5 minutes"
    replication_lag:
      warning: "Replication lag > 5 seconds for 5 minutes"
      critical: "Replication lag > 30 seconds for 5 minutes"
```

### Dashboard Requirements

```yaml
dashboards:
  # Business Dashboards
  business:
    user_activity:
      metrics: ["Active users", "Sessions", "User retention"]
      refresh_rate: "5 minutes"
      time_ranges: ["24 hours", "7 days", "30 days"]
      
    system_health:
      metrics: ["Service availability", "Error rates", "Latency trends"]
      refresh_rate: "1 minute"
      time_ranges: ["1 hour", "24 hours", "7 days"]
      
  # Operational Dashboards
  operational:
    service_health:
      metrics: ["Request rate", "Error rate", "Latency", "Saturation"]
      refresh_rate: "30 seconds"
      time_ranges: ["1 hour", "6 hours", "24 hours"]
      
    resource_utilization:
      metrics: ["CPU", "Memory", "Disk", "Network", "GPU"]
      refresh_rate: "30 seconds"
      time_ranges: ["1 hour", "6 hours", "24 hours"]
      
    database_performance:
      metrics: ["Query latency", "Connection count", "Transaction rate", "Replication lag"]
      refresh_rate: "30 seconds"
      time_ranges: ["1 hour", "6 hours", "24 hours"]
      
  # Technical Requirements
  requirements:
    platform: "Grafana 9.x"
    data_sources: ["Prometheus", "Loki", "Tempo"]
    access_control: "Role-based with SSO integration"
    export_formats: ["PDF", "PNG", "CSV"]
    embedding: "Support for embedding in internal portals"
```

## Compliance and Security Controls

### Security Controls

```yaml
security_controls:
  # Access Controls
  access_control:
    authentication: "OAuth 2.0 with OpenID Connect"
    authorization: "Role-Based Access Control (RBAC)"
    multi_factor: "Required for all administrative access"
    session_management:
      timeout: "12 hours maximum"
      refresh_tokens: "30 day maximum lifetime"
      inactivity_timeout: "30 minutes"
      
  # Data Protection
  data_protection:
    encryption_at_rest: "AES-256"
    encryption_in_transit: "TLS 1.2+"
    key_management: "AWS KMS or GCP KMS with automatic rotation"
    data_classification:
      - level: "Public"
        controls: "No special controls"
      - level: "Internal"
        controls: "Authenticated access only"
      - level: "Confidential"
        controls: "Encryption, access logging, restricted access"
      - level: "Restricted"
        controls: "Strong encryption, strict access controls, DLP monitoring"
        
  # Network Security
  network_security:
    ingress_filtering: "WAF with OWASP Top 10 protection"
    egress_filtering: "Default deny with explicit allow list"
    segmentation: "Network policies between all services"
    ddos_protection: "Cloud provider DDoS protection + rate limiting"
    
  # Vulnerability Management
  vulnerability_management:
    scanning:
      frequency: "Daily automated scans"
      coverage: "All containers, code, dependencies"
    patching:
      sla:
        critical: "24 hours"
        high: "1 week"
        medium: "1 month"
        low: "Next release cycle"
    penetration_testing:
      frequency: "Annual third-party penetration test"
      methodology: "OWASP Testing Guide"
```

### Compliance Requirements

```yaml
compliance:
  # Regulatory Frameworks
  frameworks:
    - name: "SOC 2 Type II"
      controls: "docs/controls/soc2-controls.md"
      attestation_frequency: "Annual"
      
    - name: "GDPR"
      controls: "docs/controls/gdpr-controls.md"
      assessment_frequency: "Annual"
      
    - name: "HIPAA"
      controls: "docs/controls/hipaa-controls.md"
      assessment_frequency: "Annual"
      
  # Audit Requirements
  audit:
    log_retention: "Minimum 1 year for all security events"
    access_reviews: "Quarterly review of all privileged access"
    change_control: "All production changes must follow change management process"
    evidence_collection: "Automated collection and retention of compliance artifacts"
    
  # Privacy Controls
  privacy:
    data_minimization: "Collect only necessary data"
    purpose_limitation: "Process data only for specified purposes"
    storage_limitation: "Define retention periods for all data types"
    data_subject_rights: "Support access, rectification, erasure, portability"
    privacy_by_design: "Privacy impact assessment for all new features"
```

### Security Monitoring

```yaml
security_monitoring:
  # Event Collection
  event_collection:
    sources:
      - "Application logs"
      - "Container logs"
      - "Network flow logs"
      - "API Gateway logs"
      - "Database audit logs"
      - "Identity provider logs"
    centralization: "ELK Stack or Cloud-native SIEM"
    
  # Detection Rules
  detection:
    categories:
      - category: "Access Anomalies"
        rules: ["Unusual login times", "Geographical anomalies", "Privilege escalation"]
      - category: "Data Exfiltration"
        rules: ["Unusual data access patterns", "Large data transfers", "Access from unusual locations"]
      - category: "Application Attacks"
        rules: ["SQL injection attempts", "XSS attempts", "CSRF attempts", "SSRF attempts"]
      
  # Incident Response
  incident_response:
    sla:
      critical: "15 minute response, 4 hour containment"
      high: "1 hour response, 8 hour containment"
      medium: "4 hour response, 24 hour containment"
      low: "24 hour response, 72 hour containment"
    playbooks: "docs/security/incident-playbooks/"
    tabletop_exercises: "Quarterly"
```

### Certificate Management

```yaml
certificate_management:
  # TLS Certificates
  tls_certificates:
    provider: "Let's Encrypt" # Alternative: "Commercial CA"
    automation: "cert-manager in Kubernetes"
    renewal_threshold: "30 days before expiration"
    certificate_transparency: "Required for all certificates"
    
  # Internal PKI
  internal_pki:
    provider: "Vault PKI"
    root_ca_validity: "10 years"
    intermediate_ca_validity: "5 years"
    leaf_certificate_validity: "90 days"
    automatic_rotation: "Required for all internal certificates"
    
  # Certificate Standards
  standards:
    key_type: "RSA 4096-bit or ECDSA P-384"
    signature_algorithm: "SHA-256 or better"
    minimum_tls_version: "TLS 1.2"
    preferred_cipher_suites: 
      - "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
      - "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384"
    
  # Certificate Monitoring
  monitoring:
    expiration_alerts:
      warning: "30 days before expiration"
      critical: "7 days before expiration"
    revocation_checking: "OCSP stapling required"
```

### Dependency Management

```yaml
dependency_management:
  # Management Strategy
  strategy:
    dependency_scanning: "Automated in CI/CD pipeline"
    vulnerability_monitoring: "Continuous with automatic alerts"
    license_compliance: "Automated checking against approved licenses"
    
  # Update Policy
  update_policy:
    security_updates:
      critical: "Same day review and deployment"
      high: "Within 1 week"
      medium: "Within 1 month"
      low: "Next regular release cycle"
    feature_updates:
      major_versions: "Planned migration with testing plan"
      minor_versions: "Regular update cycles (monthly)"
      patch_versions: "Automatic updates after tests"
      
  # Approved Licenses
  approved_licenses:
    - "MIT"
    - "Apache 2.0"
    - "BSD (2-clause and 3-clause)"
    - "ISC"
    
  # Internal Libraries
  internal_libraries:
    repository: "Internal artifact repository"
    versioning: "Semantic versioning"
    deprecation_policy: "6 month notice before EOL"
    documentation: "Required API docs and usage examples"
```

### Logging Requirements

```yaml
logging_requirements:
  # Log Levels and Content
  content:
    required_fields:
      - "timestamp (ISO 8601 with timezone)"
      - "service name"
      - "log level"
      - "request ID or correlation ID"
      - "user ID or service account (where applicable)"
      - "event message"
    sensitive_data:
      - "No passwords or authentication tokens"
      - "No complete PII (masked or tokenized only)"
      - "No encryption keys or secrets"
      
  # Log Storage and Retention
  storage:
    centralization: "ELK Stack or cloud-native logging service"
    retention:
      security_events: "Minimum 1 year"
      application_logs: "Minimum a month"
      debug_logs: "Maximum 7 days"
    archiving:
      method: "Cold storage (S3 Glacier or GCS Coldline)"
      format: "Compressed and encrypted"
      retention: "7 years for security and audit logs"
      
  # Log Access Controls
  access_controls:
    read_access: "Role-based with least privilege"
    write_access: "Service accounts only"
    admin_access: "Security team only"
    audit: "All log access is logged and reviewed"
```

## Migration Path

The migration from the previous architecture to the optimized database and cache architecture should follow these phases:

### Phase 1: Foundation and Preparation (Weeks 1-2)

1. **Assessment & Planning**
   - Document current data models and access patterns
   - Define migration sequence and dependencies
   - Establish migration success criteria and rollback plans

2. **Environment Setup**
   - Deploy new database instances (PostgreSQL, MongoDB, Milvus, Qdrant, Weaviate, etc.)
   - Configure replication and sharding
   - Set up monitoring and observability

### Phase 2: Vector Database Migration (Weeks 3-5)

1. **Vector Store Services**
   - Deploy Qdrant, Weaviate alongside existing Milvus
   - Implement dual-write for Embedding Service (MongoDB + Qdrant)
   - Validate query performance and functionality
   - Complete migration and switch read traffic

2. **Pattern Matching and Knowledge Services**
   - Implement Weaviate for Pattern Matcher and Knowledge Synthesizer
   - Build data transformation pipelines
   - Migrate historical data in batches
   - Validate and cut over

### Phase 3: Specialized Database Adoption (Weeks 6-9)

1. **TimescaleDB for Time-Series Data**
   - Deploy TimescaleDB instances
   - Migrate Working Memory Service to TimescaleDB
   - Implement continuous aggregates and retention policies

2. **ScyllaDB for High-Throughput Services**
   - Deploy ScyllaDB cluster
   - Migrate Episodic Memory and Event logs
   - Validate performance at scale

### Phase 4: Cache Optimization (Weeks 10-12)

1. **Redis Stack Deployment**
   - Deploy Redis Stack instances for vector and specialized operations
   - Configure Vector modules for embedding caches
   - Implement service-specific eviction policies

2. **Redis Cluster Migration**
   - Deploy Redis Cluster for high-throughput services
   - Configure sharding and replication
   - Migrate from standalone Redis instances

### Phase 5: Validation and Optimization (Weeks 13-14)

1. **Performance Testing**
   - Conduct load tests against new architecture
   - Measure query performance, latency, and throughput
   - Compare with baseline metrics

2. **Fine-Tuning**
   - Optimize indexes and query patterns
   - Adjust cache sizes and eviction policies
   - Refine sharding strategies based on observed patterns

3. **Documentation and Knowledge Transfer**
   - Update system documentation
   - Conduct training for operational teams
   - Establish new monitoring dashboards and alerts

### Risk Mitigation

1. **Progressive Migration Strategy**
   - Service by service migration rather than big bang
   - Dual-write approach where possible
   - Blue/green deployments for critical services

2. **Rollback Capability**
   - Maintain original databases until validation complete
   - Script automated rollback procedures
   - Practice recovery scenarios

3. **Data Validation**
   - Checksum verification for migrated data
   - Functional validation of queries and operations
   - Parallel testing in production-like environment# Enterprise Deep Reasoning System - Single Source of Truth Configuration

This document serves as the definitive reference for all configuration parameters across the Enterprise Deep Reasoning System (EDRS). It provides a comprehensive, system-wide specification that should be used as the master reference for all development, deployment, and operational activities.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Container Registry](#container-registry)
3. [Container Specifications](#container-specifications)
4. [Database Strategy](#database-strategy)
5. [Database Specifications](#database-specifications)
6. [Vector Database Implementation](#vector-database-implementation)
7. [Cache Specifications](#cache-specifications)
8. [Volume Specifications](#volume-specifications)
9. [Network Configuration](#network-configuration)
10. [Resource Requirements](#resource-requirements)
11. [Health Checks](#health-checks)
12. [Security Configuration](#security-configuration)
13. [Observability](#observability)
14. [Rate Limiting](#rate-limiting)
15. [Scaling Configuration](#scaling-configuration)
16. [Environment Variables](#environment-variables)
17. [Load Balancing and High Availability](#load-balancing-and-high-availability)
18. [Connection Pooling Configuration](#connection-pooling-configuration)
19. [Data Sharding and Partitioning Strategy](#data-sharding-and-partitioning-strategy)
20. [Database Backup and Replication](#database-backup-and-replication)
21. [Updates and Rollout Strategy](#updates-and-rollout-strategy)
22. [Client Isolation](#client-isolation)
23. [Infrastructure Specifications](#infrastructure-specifications)
24. [CI/CD and Development Workflow](#cicd-and-development-workflow)
25. [Disaster Recovery and Business Continuity](#disaster-recovery-and-business-continuity)
26. [API Gateway and Documentation](#api-gateway-and-documentation)
27. [Service Level Objectives](#service-level-objectives)
28. [Compliance and Security Controls](#compliance-and-security-controls)
29. [Migration Path](#migration-path)
30. [Events and Event Processing](#events-and-event-processing)

## System Architecture

The EDRS is composed of a set of specialized PODs, each containing multiple containers that perform specific functions:

### POD Structure
1. **Workforce AI Persona POD**
   - Identity Manager
   - Persona Synthesizer
   - TTS/STT Services
   - Feedback Processor
   - Performance Monitoring
   - Human Interface
   - Embedding Service

2. **Memory POD**
   - Working Memory Service
   - Episodic Memory Service
   - Context Protocol
   - Casual Trace Networks
   - Memory Retrieval/Consolidation
   - Attention Mechanism
   - Quantum-Inspired Memory

3. **Knowledge POD**
   - Domain Knowledge Base
   - Vector Store
   - Knowledge Synthesis
   - Factual Knowledge
   - Knowledge Graph
   - Knowledge Validation
   - Knowledge Update
   - Feedback Optimizer
   

4. **Reasoning POD**
   - DeepSeek ServiceR1-8b or Moe-33b
   - Abstract Reasoner
   - Logical Engine
   - Causal Engine
   - Decision Tree Service
   - Hypothesis Generator
   - Verification Service

5. **AI Services POD**
   - Embedding Service
   - Multi-Modal Content-Creation Services
   - Vision Analysis

6. **Task Management POD**
   - Workflow Analyzer
   - Task Decomposer
   - Workflow Builder
   - Workflow Executor
   - Workflow Registry

7. **Research POD**
   - Information Retrieval
   - Smart Web Scraper
   - Source Validation
   - Knowledge Synthesis
   - Research Planning
   - Citation Manager
   - Learning Gap Identifier
   - Domain Exploration[events-processor](shared-pod/events-processor)

8. **Shared POD**
    -Event Bus Manager (Pulsar administration)
    -Service Discovery
    -Distributed Tracing
    -Global Configuration
    -Circuit Breaker Dashboard
    -Health Monitoring
    -Metrics Aggregator
    -Security Token Service
    -embedding service

9. **API Gateway**
   - Nginx Gateway
   - Auth Sidecar
   - 

10. **Integration POD**
- API Connector Framework
- Enterprise System Connectors
- Tool Integration
- Data Import/Export
- Webhook Service
- Authentication Proxy
- Format Conversion


## Container Registry

```
Registry URL: registry.edrs.example.com
Authentication: OAuth2/Bearer token or username/password
Image naming convention: edrs/<container-name>:<version>[-<environment>]
```

## Container Specifications

| Container Name | Base Image | Build Context | Image Tag Format | Non-root User | Exposed Ports | Languages/Runtimes |
|----------------|------------|---------------|------------------|---------------|---------------|-------------------|
| identity-manager | node:18-alpine | ./services/identity-manager | edrs/identity-manager:1.2.3 | appuser (1000:1000) | 8080, 8081 | Node.js |
| persona-synthesizer | python:3.10-slim | ./services/persona-synthesizer | edrs/persona-synthesizer:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| tts-service | python:3.10-slim | ./services/tts-service | edrs/tts-service:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| stt-service | python:3.10-slim | ./services/stt-service | edrs/stt-service:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| feedback-processor | node:18-alpine | ./services/feedback-processor | edrs/feedback-processor:1.2.3 | appuser (1000:1000) | 8080, 8081 | Node.js |
| working-memory-service | python:3.10-slim | ./services/working-memory-service | edrs/working-memory-service:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| episodic-memory-service | python:3.10-slim | ./services/episodic-memory-service | edrs/episodic-memory-service:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| context-manager | node:18-alpine | ./services/context-manager | edrs/context-manager:1.2.3 | appuser (1000:1000) | 8080, 8081 | Node.js |
| domain-knowledge-base | python:3.10-slim | ./services/domain-knowledge-base | edrs/domain-knowledge-base:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| vector-store | python:3.10-slim | ./services/vector-store | edrs/vector-store:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| knowledge-synthesizer | python:3.10-slim | ./services/knowledge-synthesizer | edrs/knowledge-synthesizer:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| neural-interpreter | node:18-alpine | ./services/neural-interpreter | edrs/neural-interpreter:1.2.3 | appuser (1000:1000) | 8080, 8081 | Node.js |
| deepseek-service | python:3.10-slim | ./services/deepseek-service | edrs/deepseek-service:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| pattern-matcher | python:3.10-slim | ./services/pattern-matcher | edrs/pattern-matcher:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| llm-inference | python:3.10-slim | ./services/llm-inference | edrs/llm-inference:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| embedding-service | python:3.10-slim | ./services/embedding-service | edrs/embedding-service:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| task-decomposer | python:3.10-slim | ./services/task-decomposer | edrs/task-decomposer:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| workflow-builder | node:18-alpine | ./services/workflow-builder | edrs/workflow-builder:1.2.3 | appuser (1000:1000) | 8080, 8081 | Node.js |
| workflow-executor | node:18-alpine | ./services/workflow-executor | edrs/workflow-executor:1.2.3 | appuser (1000:1000) | 8080, 8081 | Node.js |
| workflow-registry | node:18-alpine | ./services/workflow-registry | edrs/workflow-registry:1.2.3 | appuser (1000:1000) | 8080, 8081 | Node.js |
| information-retrieval | python:3.10-slim | ./services/information-retrieval | edrs/information-retrieval:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| source-validation | python:3.10-slim | ./services/source-validation | edrs/source-validation:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| event-system-core | java:17-slim | ./services/event-system-core | edrs/event-system-core:1.2.3 | appuser (1000:1000) | 8080, 8081 | Java |
| response-aggregator | node:18-alpine | ./services/response-aggregator | edrs/response-aggregator:1.2.3 | appuser (1000:1000) | 8080, 8081 | Node.js |
| meta-learner | python:3.10-slim | ./services/meta-learner | edrs/meta-learner:1.2.3 | appuser (1000:1000) | 8080, 8081 | Python |
| nginx-gateway | nginx:1.23-alpine | ./services/nginx-gateway | edrs/nginx-gateway:1.2.3 | nginx (101:101) | 80, 443 | Nginx |
| auth-sidecar | node:18-alpine | ./services/auth-sidecar | edrs/auth-sidecar:1.2.3 | appuser (1000:1000) | 8080, 8081 | Node.js |
| api-connector-framework | node:18-alpine | ./services/api-connector-framework | edrs/api-connector-framework:1.2.3 | appuser (1000:1000) | 8080, 8081 | Node.js |

### Container Build Details

Each container should follow these build guidelines:

1. **Multi-stage builds** to reduce image size
2. **Non-root user** for security (UID 1000, GID 1000)
3. **Read-only filesystem** where possible
4. **Dropped capabilities** (using `--cap-drop=ALL`)
5. **Health check** endpoints at `/health`
6. **Metrics** exposed at port `8081` path `/metrics`

## Database Strategy

### Database Technology Matrix

| Database Type | Specialized For | Primary Use Cases | EDRS Services |
|---------------|-----------------|-------------------|---------------|
| PostgreSQL | Relational data, ACID transactions, complex querying | Identity, workflow management, structured business logic | Identity Manager, Workflow Registry, Neural Interpreter |
| MongoDB | Document storage, flexible schema, high throughput | Session records, logs, semi-structured content | Feedback Processor, Context Manager, Event System Core |
| Qdrant | Vector similarity search, metadata filtering | Embeddings, semantic search | Embedding Service, Pattern Matcher, Research Services |
| Weaviate | Contextual search, semantic classification | Classification, multi-modal search | Knowledge Synthesizer, Source Validation |
| Milvus | High-dimensional vector search, ANN algorithms | Large-scale vector operations | Vector Store, Domain Knowledge Base |
| TimescaleDB | Time-series data, temporal analytics | Performance metrics, behavioral tracking | Meta Learner, Working Memory Service |
| Redis | In-memory operations, pub/sub, leaderboards | Caching, real-time features | All services (as cache) |
| ScyllaDB | High throughput, low-latency, large datasets | Event streaming, high-volume logging | Episodic Memory Service, Information Retrieval |

### Strategic Database Selection Principles

1. **Match data characteristics to database strengths**
   - Structured relational data → PostgreSQL or TimescaleDB
   - Document-oriented flexible data → MongoDB
   - Vector embeddings → Qdrant, Weaviate, or Milvus
   - Time-series data → TimescaleDB
   - High-throughput event data → ScyllaDB

2. **Consider access patterns**
   - Read-heavy analytical workloads → Column stores or OLAP databases
   - Write-heavy operational workloads → Row stores or specialized write-optimized DBs
   - Mixed workloads → Multi-model databases with appropriate indexing

3. **Feature requirements**
   - Full-text search → Databases with integrated search capabilities
   - Graph operations → Databases with graph capabilities or dedicated graph DBs
   - Geospatial → Databases with geospatial extensions
   - Vector operations → Vector-optimized databases

## Database Specifications

| Container | Database Type | Database Name | Port | Cache | Connection String |
|-----------|---------------|--------------|------|-------|------------------ |
| identity-manager | PostgreSQL | Identity DB | 5432 | Identity Cache | `postgresql://user:password@identity-db:5432/identity` |
| persona-synthesizer | PostgreSQL | Persona DB | 5433 | Persona Cache | `postgresql://user:password@persona-db:5433/persona` |
| tts-service | MongoDB | TTS DB | 27023 | TTS Cache | `mongodb://tts-db:27023/tts-service` |
| stt-service | MongoDB | STT DB | 27022 | STT Cache | `mongodb://stt-db:27022/stt-service` |
| feedback-processor | MongoDB | Feedback Processor DB | 27027 | Feedback Processor Cache | `mongodb://feedback-processor-db:27027/feedback-processor` |
| working-memory-service | TimescaleDB | Working Memory DB | 5453 | Working Memory Cache | `postgresql://user:password@working-memory-db:5453/working-memory` |
| episodic-memory-service | ScyllaDB | Episodic Memory DB | 9042 | Episodic Memory Cache | `scylla://episodic-memory-db:9042/episodic-memory` |
| context-manager | MongoDB | Context Manager DB | 27028 | Context Manager Cache | `mongodb://context-manager-db:27028/context-manager` |
| domain-knowledge-base | Milvus + MongoDB | Domain Knowledge DB | 19531, 27019 | Domain Knowledge Cache | `milvus://domain-knowledge-db:19531`, `mongodb://domain-knowledge-db:27019/domain-knowledge` |
| vector-store | Milvus | Vector Store DB | 19530 | Vector Store Cache | `milvus://vector-store-db:19530` |
| knowledge-synthesizer | Weaviate | Knowledge Synthesizer DB | 8080 | Knowledge Synthesizer Cache | `http://knowledge-synthesizer-db:8080` |
| neural-interpreter | PostgreSQL | Neural Interpreter DB | 5434 | Neural Interpreter Cache | `postgresql://user:password@neural-interpreter-db:5434/neural-interpreter` |
| deepseek-service | PostgreSQL | DeepSeek DB | 5435 | DeepSeek Cache | `postgresql://user:password@deepseek-db:5435/deepseek` |
| pattern-matcher | Qdrant | Pattern Matcher DB | 6333 | Pattern Matcher Cache | `http://pattern-matcher-db:6333` |
| llm-inference | PostgreSQL | LLM Inference DB | 5436 | LLM Inference Cache | `postgresql://user:password@llm-inference-db:5436/llm-inference` |
| embedding-service | Qdrant | Embedding DB | 6334 | Embedding Cache | `http://embedding-db:6334` |
| task-decomposer | PostgreSQL | Task Decomposer DB | 5437 | Task Decomposer Cache | `postgresql://user:password@task-decomposer-db:5437/task-decomposer` |
| workflow-builder | PostgreSQL | Workflow Builder DB | 5438 | Workflow Builder Cache | `postgresql://user:password@workflow-builder-db:5438/workflow-builder` |
| workflow-executor | PostgreSQL | Workflow Executor DB | 5439 | Workflow Executor Cache | `postgresql://user:password@workflow-executor-db:5439/workflow-executor` |
| workflow-registry | PostgreSQL | Workflow Registry DB | 5450 | Workflow Registry Cache | `postgresql://user:password@workflow-registry-db:5450/workflow-registry` |
| information-retrieval | Qdrant | Info Retrieval DB | 6335 | Info Retrieval Cache | `http://info-retrieval-db:6335` |
| source-validation | Weaviate | Source Validation DB | 8081 | Source Validation Cache | `http://source-validation-db:8081` |
| event-system-core | MongoDB | Event System DB | 27029 | Event System Cache | `mongodb://event-system-db:27029/event-system` |
| response-aggregator | MongoDB | Response Aggregator DB | 27030 | Response Aggregator Cache | `mongodb://response-aggregator-db:27030/response-aggregator` |
| meta-learner | TimescaleDB + MongoDB | Meta Learner DB | 5454, 27025 | Meta Learner Cache | `postgresql://user:password@meta-learner-tsdb:5454/meta-learner`, `mongodb://meta-learner-db:27025/meta-learner` |
| nginx-gateway | PostgreSQL | Nginx DB | 5444 | Nginx Cache | `postgresql://user:password@nginx-db:5444/nginx` |
| auth-sidecar | PostgreSQL | Auth DB | 5443 | Auth Cache | `postgresql://user:password@auth-db:5443/auth` |
| api-connector-framework | PostgreSQL | API Connector DB | 5446 | API Connector Cache | `postgresql://user:password@api-connector-db:5446/api-connector` |

## Vector Database Implementation

### Vector Database Comparison and Selection

| Vector DB | Throughput | Query Speed | Features | Best For |
|-----------|------------|-------------|----------|----------|
| Milvus | Very High | Fast | Hybrid search, load balancing, auto scaling | Large-scale vector operations |
| Qdrant | High | Very Fast | Payload filtering, rich queries, transactions | Filtered vector search with metadata |
| Weaviate | Moderate | Fast | Contextual search, multi-modal, GraphQL | Semantic classification and search |
| pgvector | Moderate | Moderate | SQL integration, ACID | Simple vector operations within PostgreSQL |

### Vector Database Service Assignments

| Service | Database | Vector Dimension | Index Type | Justification |
|---------|----------|-----------------|------------|---------------|
| Vector Store | Milvus | 1536 | HNSW | Large-scale vector operations |
| Embedding Service | Qdrant | 768, 1536 | HNSW | Storage and retrieval of embeddings with metadata |
| Pattern Matcher | Qdrant | 1536 | HNSW | Fast pattern matching with filtering |
| Knowledge Synthesizer | Weaviate | 1536 | HNSW | Knowledge graph concepts with semantic relationships |
| Domain Knowledge Base | Milvus + MongoDB | 1536 | HNSW | Document storage in MongoDB, vectors in Milvus |
| Information Retrieval | Qdrant | 768 | HNSW | Fast search with filtering for relevant information |
| Source Validation | Weaviate | 768 | HNSW | Semantic classification for source credibility |

### Vector Database Configuration Details

#### Milvus Configuration

```yaml
milvus:
  cluster:
    enabled: true
    replicas: 3
  storage:
    primary:
      size: 200Gi
      storageClassName: fast-ssd
    logs:
      size: 50Gi
      storageClassName: standard
  resources:
    limits:
      cpu: "8"
      memory: "32Gi"
    requests:
      cpu: "4"
      memory: "16Gi"
  indexing:
    type: "HNSW"
    parameters:
      M: 16
      efConstruction: 200
  search:
    parameters:
      ef: 100
      nprobe: 16
```

#### Qdrant Configuration

```yaml
qdrant:
  cluster:
    enabled: true
    replicas: 3
  storage:
    path: /qdrant/storage
    size: 100Gi
    storageClassName: fast-ssd
  resources:
    limits:
      cpu: "4"
      memory: "16Gi"
    requests:
      cpu: "2"
      memory: "8Gi"
  indexing:
    max_vectors_per_segment: 50000
    memmap_threshold: 10000
    indexing_threshold: 20000
    optimizer_poll_interval: 60 # seconds
  vector_params:
    size: 1536
    distance: Cosine
```

#### Weaviate Configuration

```yaml
weaviate:
  cluster:
    enabled: true
    replicas: 3
  storage:
    size: 100Gi
    storageClassName: fast-ssd
    backupEnabled: true
  resources:
    limits:
      cpu: "4"
      memory: "16Gi"
    requests:
      cpu: "2"
      memory: "8Gi"
  vector_index:
    type: "hnsw"
    maxConnections: 64
    efConstruction: 128
    ef: 64
  contextionary:
    enabled: true
    url: "contextionary:9999"
  transformers:
    enabled: true
    inferenceUrl: "transformers:8080"
```

## Cache Specifications

| Cache Name | Type | Port | Memory Limit | Eviction Policy | Persistence |
|------------|------|------|--------------|-----------------|-------------|
| Identity Cache | Redis | 6379 | 2Gi | allkeys-lru | AOF |
| Persona Cache | Redis | 6380 | 2Gi | allkeys-lru | None |
| TTS Cache | Redis | 6402 | 1Gi | allkeys-lru | None |
| STT Cache | Redis | 6401 | 1Gi | allkeys-lru | None |
| Feedback Processor Cache | Redis | 6409 | 1Gi | allkeys-lru | None |
| Working Memory Cache | Redis Cluster | 6381-6386 | 4Gi | volatile-ttl | AOF |
| Episodic Memory Cache | Redis Cluster | 6421-6426 | 6Gi | allkeys-lru | RDB daily |
| Context Manager Cache | Redis | 6411 | 2Gi | allkeys-lru | None |
| Domain Knowledge Cache | Redis Stack (Vector) | 6383 | 4Gi | volatile-lfu | RDB hourly |
| Vector Store Cache | Redis Stack (Vector) | 6384 | 8Gi | volatile-lfu | RDB hourly |
| Knowledge Synthesizer Cache | Redis Stack | 6410 | 4Gi | volatile-lfu | RDB hourly |
| Neural Interpreter Cache | Redis Stack | 6385 | 4Gi | volatile-lfu | AOF |
| DeepSeek Cache | Redis | 6386 | 2Gi | allkeys-lru | None |
| Pattern Matcher Cache | Redis Stack (Vector) | 6403 | 4Gi | volatile-lfu | RDB hourly |
| LLM Inference Cache | Redis Stack | 6387 | 8Gi | volatile-lfu | RDB hourly |
| Embedding Cache | Redis Stack (Vector) | 6388 | 6Gi | volatile-lfu | RDB hourly |
| Task Decomposer Cache | Redis | 6389 | 2Gi | volatile-lru | AOF |
| Workflow Builder Cache | Redis | 6390 | 2Gi | allkeys-lru | None |
| Workflow Executor Cache | Redis | 6391 | 2Gi | volatile-lru | AOF |
| Workflow Registry Cache | Redis | 6407 | 2Gi | allkeys-lru | None |
| Info Retrieval Cache | Redis Stack (Vector) | 6392 | 4Gi | volatile-lfu | RDB hourly |
| Source Validation Cache | Redis Stack | 6405 | 2Gi | volatile-lfu | RDB hourly |
| Event System Cache | Redis Cluster | 6431-6436 | 4Gi | volatile-ttl | None |
| Response Aggregator Cache | Redis | 6399 | 2Gi | allkeys-lru | None |
| Meta Learner Cache | Redis Cluster | 6441-6446 | 4Gi | volatile-lru | RDB daily |
| Nginx Cache | Redis Cluster | 6451-6456 | 2Gi | allkeys-lru | None |
| Auth Cache | Redis | 6397 | 2Gi | allkeys-lru | AOF |
| API Connector Cache | Redis | 6400 | 2Gi | allkeys-lru | None |

### Redis Cluster Configuration

```yaml
redis-cluster:
  nodes: 6  # 3 masters, 3 replicas
  persistence:
    enabled: true
    strategy: "rdb"  # or "aof" for services needing transactional integrity
    schedule: "*/60 * * * *"  # Every hour
  resources:
    limits:
      cpu: "2"
      memory: "4Gi"
    requests:
      cpu: "1"
      memory: "2Gi"
  maxmemory: "3Gi"  # 75% of available memory
  maxmemory-policy: "volatile-lfu"  # Default policy, overridden per service
  cluster:
    slaveReplicas: 1
    failover: true
```

### Redis Stack (Vector) Configuration

```yaml
redis-stack:
  enabled: true
  modules:
    - search
    - json
    - timeseries
    - graph
    - bloom
    - vector  # Vector similarity module
  resources:
    limits:
      cpu: "4"
      memory: "8Gi"
    requests:
      cpu: "2"
      memory: "4Gi"
  maxmemory: "6Gi"  # 75% of available memory
  maxmemory-policy: "volatile-lfu"
  vector:
    dimension: 1536
    similarity-algorithm: "cosine"
    index-type: "HNSW"
```

## Volume Specifications

| Container | Volume Mount Path | Purpose | Size | Storage Class | Access Mode | Backup Requirement |
|-----------|-------------------|---------|------|---------------|-------------|-------------------|
| Identity DB | `/var/lib/postgresql/data` | Database files | 10Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Persona DB | `/var/lib/postgresql/data` | Database files | 10Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| TTS DB | `/data/db` | Database files | 20Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| TTS Service | `/data/voice-profiles` | Voice profile storage | 10Gi | standard | ReadWriteOnce | Yes - Weekly |
| STT DB | `/data/db` | Database files | 20Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Feedback Processor DB | `/data/db` | Database files | 20Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Working Memory DB | `/var/lib/postgresql/timescaledb` | Database files | 50Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Episodic Memory DB | `/var/lib/scylla` | Database files | 100Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Context Manager DB | `/data/db` | Database files | 20Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Domain Knowledge DB (MongoDB) | `/data/db` | Document storage | 100Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Domain Knowledge DB (Milvus) | `/milvus/data` | Vector data | 100Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Vector Store DB | `/milvus/data` | Vector data | 100Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Knowledge Synthesizer DB | `/var/lib/weaviate` | Database files | 50Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Knowledge Synthesizer | `/data/knowledge-graph` | Knowledge graph data | 20Gi | standard | ReadWriteOnce | Yes - Daily |
| Neural Interpreter DB | `/var/lib/postgresql/data` | Database files | 10Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| DeepSeek DB | `/var/lib/postgresql/data` | Database files | 20Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| DeepSeek Service | `/models` | LLM models | 50Gi | fast-ssd | ReadOnlyMany | Yes - On update |
| Pattern Matcher DB | `/qdrant/storage` | Vector database files | 50Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| LLM Inference DB | `/var/lib/postgresql/data` | Database files | 20Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| LLM Inference | `/models` | LLM models | 50Gi | fast-ssd | ReadOnlyMany | Yes - On update |
| Embedding DB | `/qdrant/storage` | Vector database files | 30Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Embedding Service | `/models` | Embedding models | 10Gi | fast-ssd | ReadOnlyMany | Yes - On update |
| Task Decomposer DB | `/var/lib/postgresql/data` | Database files | 10Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Workflow Builder DB | `/var/lib/postgresql/data` | Database files | 10Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Workflow Executor DB | `/var/lib/postgresql/data` | Database files | 20Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Workflow Registry DB | `/var/lib/postgresql/data` | Database files | 10Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Workflow Registry | `/data/workflow-templates` | Workflow templates | 1Gi | standard | ReadWriteOnce | Yes - On update |
| Info Retrieval DB | `/qdrant/storage` | Vector database files | 50Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Source Validation DB | `/var/lib/weaviate` | Database files | 20Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Source Validation | `/data/blacklisted-sources.json` | Blacklisted sources | 10Mi | standard | ReadWriteOnce | Yes - On update |
| Event System DB | `/data/db` | Database files | 30Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Response Aggregator DB | `/data/db` | Database files | 20Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Meta Learner DB (TimescaleDB) | `/var/lib/postgresql/timescaledb` | Time-series data | 30Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Meta Learner DB (MongoDB) | `/data/db` | Learning patterns | 20Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Nginx DB | `/var/lib/postgresql/data` | Database files | 10Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| Nginx Gateway | `/var/log/nginx` | Access and error logs | 5Gi | standard | ReadWriteOnce | Yes - Weekly |
| Auth DB | `/var/lib/postgresql/data` | Database files | 10Gi | fast-ssd | ReadWriteOnce | Yes - Daily |
| API Connector DB | `/var/lib/postgresql/data` | Database files | 10Gi | fast-ssd | ReadWriteOnce | Yes - Daily |

## Network Configuration

### Port Assignments

| Service | Internal Port | External Port | Protocol | Purpose |
|---------|--------------|--------------|----------|---------|
| Nginx Gateway | 8080 | 443 | HTTPS | External API access, TLS termination, auth |
| Neural Interpreter | 8080 | N/A | HTTP | Primary query processing entry point |
| Pulsar Proxy | 6650 | N/A | TCP | Event system access point |
| Pulsar Broker | 6651 | N/A | TCP | Event broker (internal only) |
| Pulsar Admin | 8080 | N/A | HTTP | Pulsar administration |
| Zookeeper | 2181 | N/A | TCP | Pulsar metadata |
| Bookkeeper | 3181 | N/A | TCP | Pulsar storage |
| Identity DB | 5432 | N/A | TCP | Identity Manager database |
| Persona DB | 5433 | N/A | TCP | Persona Synthesizer database |
| Working Memory DB | 5453 | N/A | TCP | Working Memory Service database (TimescaleDB) |
| Episodic Memory DB | 9042 | N/A | TCP | Episodic Memory Service database (ScyllaDB) |
| Domain Knowledge DB (MongoDB) | 27019 | N/A | TCP | Domain Knowledge Base document storage |
| Domain Knowledge DB (Milvus) | 19531 | N/A | gRPC | Domain Knowledge Base vector storage |
| Vector Store DB | 19530 | N/A | gRPC | Vector Store database |
| Knowledge Synthesizer DB | 8080 | N/A | HTTP | Knowledge Synthesizer database (Weaviate) |
| Neural Interpreter DB | 5434 | N/A | TCP | Neural Interpreter database |
| DeepSeek DB | 5435 | N/A | TCP | DeepSeek Service database |
| Pattern Matcher DB | 6333 | N/A | HTTP | Pattern Matcher database (Qdrant) |
| LLM Inference DB | 5436 | N/A | TCP | LLM Inference database |
| Embedding DB | 6334 | N/A | HTTP | Embedding Service database (Qdrant) |
| Task Decomposer DB | 5437 | N/A | TCP | Task Decomposer database |
| Workflow Builder DB | 5438 | N/A | TCP | Workflow Builder database |
| Workflow Executor DB | 5439 | N/A | TCP | Workflow Executor database |
| Workflow Registry DB | 5450 | N/A | TCP | Workflow Registry database |
| Info Retrieval DB | 6335 | N/A | HTTP | Information Retrieval database (Qdrant) |
| Source Validation DB | 8081 | N/A | HTTP | Source Validation database (Weaviate) |
| Event System DB | 27029 | N/A | TCP | Event System Core database (MongoDB) |
| Response Aggregator DB | 27030 | N/A | TCP | Response Aggregator database (MongoDB) |
| Meta Learner DB (TimescaleDB) | 5454 | N/A | TCP | Meta Learner time-series database |
| Meta Learner DB (MongoDB) | 27025 | N/A | TCP | Meta Learner pattern database |
| TTS DB | 27023 | N/A | TCP | TTS Service database |
| STT DB | 27022 | N/A | TCP | STT Service database |
| Feedback Processor DB | 27027 | N/A | TCP | Feedback Processor database (MongoDB) |
| Context Manager DB | 27028 | N/A | TCP | Context Manager database (MongoDB) |
| Nginx DB | 5444 | N/A | TCP | Nginx database |
| Auth DB | 5443 | N/A | TCP | Auth Sidecar database |
| API Connector DB | 5446 | N/A | TCP | API Connector Framework database |
| Identity Cache | 6379 | N/A | TCP | Identity Manager cache |
| Persona Cache | 6380 | N/A | TCP | Persona Synthesizer cache |
| Working Memory Cache | 6381-6386 | N/A | TCP | Working Memory Service cache (Redis Cluster) |
| Episodic Memory Cache | 6421-6426 | N/A | TCP | Episodic Memory Service cache (Redis Cluster) |
| Domain Knowledge Cache | 6383 | N/A | TCP | Domain Knowledge Base cache (Redis Stack) |
| Vector Store Cache | 6384 | N/A | TCP | Vector Store cache (Redis Stack) |
| Neural Interpreter Cache | 6385 | N/A | TCP | Neural Interpreter cache (Redis Stack) |
| DeepSeek Cache | 6386 | N/A | TCP | DeepSeek Service cache |
| LLM Inference Cache | 6387 | N/A | TCP | LLM Inference cache (Redis Stack) |
| Embedding Cache | 6388 | N/A | TCP | Embedding Service cache (Redis Stack) |
| Task Decomposer Cache | 6389 | N/A | TCP | Task Decomposer cache |
| Workflow Builder Cache | 6390 | N/A | TCP | Workflow Builder cache |
| Workflow Executor Cache | 6391 | N/A | TCP | Workflow Executor cache |
| Info Retrieval Cache | 6392 | N/A | TCP | Information Retrieval cache (Redis Stack) |
| Pattern Matcher Cache | 6403 | N/A | TCP | Pattern Matcher cache (Redis Stack) |
| Source Validation Cache | 6405 | N/A | TCP | Source Validation cache (Redis Stack) |
| Event System Cache | 6431-6436 | N/A | TCP | Event System Core cache (Redis Cluster) |
| Workflow Registry Cache | 6407 | N/A | TCP | Workflow Registry cache |
| Meta Learner Cache | 6441-6446 | N/A | TCP | Meta Learner cache (Redis Cluster) |
| Feedback Processor Cache | 6409 | N/A | TCP | Feedback Processor cache |
| Knowledge Synthesizer Cache | 6410 | N/A | TCP | Knowledge Synthesizer cache (Redis Stack) |
| Context Manager Cache | 6411 | N/A | TCP | Context Manager cache |
| TTS Cache | 6402 | N/A | TCP | TTS Service cache |
| STT Cache | 6401 | N/A | TCP | STT Service cache |
| Response Aggregator Cache | 6399 | N/A | TCP | Response Aggregator cache |
| Nginx Cache | 6451-6456 | N/A | TCP | Nginx Cache (Redis Cluster) |
| Auth Cache | 6397 | N/A | TCP | Auth Sidecar cache |
| API Connector Cache | 6400 | N/A | TCP | API Connector Framework cache |
| Prometheus | 9090 | N/A | HTTP | Metrics collection |
| Grafana | 3000 | 3000 | HTTP | Metrics visualization |

### Network Policies

Each service should have a dedicated network policy:

1. **Default deny all ingress/egress traffic**
2. **Allow only specific paths to specific services**
3. **Container-to-database communication strictly limited**
4. **Event bus communication through Pulsar Proxy only**

Example policy pattern:
```
Container -> Database: Direct access to own database only
Container -> Cache: Direct access to own cache only
Container -> Container: Communication via event bus only
Container -> External Services: Via egress controller only
```

## Resource Requirements

| Container | CPU Request | CPU Limit | Memory Request | Memory Limit | GPU |
|-----------|-------------|-----------|----------------|--------------|-----|
| identity-manager | 0.5 | 1 | 1Gi | 2Gi | No |
| persona-synthesizer | 2 | 4 | 4Gi | 8Gi | No |
| tts-service | 2 | 4 | 4Gi | 8Gi | Yes |
| stt-service | 2 | 4 | 4Gi | 8Gi | Yes |
| feedback-processor | 0.5 | 1 | 1Gi | 2Gi | No |
| working-memory-service | 2 | 4 | 8Gi | 16Gi | No |
| episodic-memory-service | 2 | 4 | 8Gi | 16Gi | No |
| context-manager | 2 | 4 | 4Gi | 8Gi | No |
| domain-knowledge-base | 2 | 4 | 8Gi | 16Gi | No |
| vector-store | 2 | 4 | 8Gi | 16Gi | No |
| knowledge-synthesizer | 2 | 4 | 4Gi | 8Gi | No |
| neural-interpreter | 2 | 4 | 8Gi | 16Gi | No |
| deepseek-service | 4 | 8 | 16Gi | 32Gi | Yes |
| pattern-matcher | 1 | 2 | 2Gi | 4Gi | No |
| llm-inference | 4 | 8 | 16Gi | 32Gi | Yes |
| embedding-service | 2 | 4 | 8Gi | 16Gi | Yes |
| task-decomposer | 2 | 4 | 4Gi | 8Gi | No |
| workflow-builder | 1 | 2 | 2Gi | 4Gi | No |
| workflow-executor | 1 | 2 | 2Gi | 4Gi | No |
| workflow-registry | 1 | 2 | 2Gi | 4Gi | No |
| information-retrieval | 2 | 4 | 8Gi | 16Gi | No |
| source-validation | 1 | 2 | 4Gi | 8Gi | No |
| event-system-core | 2 | 4 | 4Gi | 8Gi | No |
| response-aggregator | 1 | 2 | 4Gi | 8Gi | No |
| meta-learner | 2 | 4 | 8Gi | 16Gi | No |
| nginx-gateway | 1 | 2 | 2Gi | 4Gi | No |
| auth-sidecar | 0.5 | 1 | 1Gi | 2Gi | No |
| api-connector-framework | 1 | 2 | 2Gi | 4Gi | No |

### Database Resource Requirements

| Database Container | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| identity-db | 1 | 2 | 2Gi | 4Gi |
| persona-db | 1 | 2 | 2Gi | 4Gi |
| tts-db | 1 | 2 | 2Gi | 4Gi |
| stt-db | 1 | 2 | 2Gi | 4Gi |
| feedback-processor-db | 1 | 2 | 2Gi | 4Gi |
| working-memory-db | 2 | 4 | 4Gi | 8Gi |
| episodic-memory-db | 4 | 8 | 8Gi | 16Gi |
| context-manager-db | 1 | 2 | 2Gi | 4Gi |
| domain-knowledge-db-mongodb | 2 | 4 | 4Gi | 8Gi |
| domain-knowledge-db-milvus | 4 | 8 | 8Gi | 16Gi |
| vector-store-db | 4 | 8 | 8Gi | 16Gi |
| knowledge-synthesizer-db | 2 | 4 | 4Gi | 8Gi |
| neural-interpreter-db | 1 | 2 | 2Gi | 4Gi |
| deepseek-db | 1 | 2 | 2Gi | 4Gi |
| pattern-matcher-db | 2 | 4 | 4Gi | 8Gi |
| llm-inference-db | 1 | 2 | 2Gi | 4Gi |
| embedding-db | 2 | 4 | 4Gi | 8Gi |
| task-decomposer-db | 1 | 2 | 2Gi | 4Gi |
| workflow-builder-db | 1 | 2 | 2Gi | 4Gi |
| workflow-executor-db | 1 | 2 | 2Gi | 4Gi |
| workflow-registry-db | 1 | 2 | 2Gi | 4Gi |
| info-retrieval-db | 2 | 4 | 4Gi | 8Gi |
| source-validation-db | 2 | 4 | 4Gi | 8Gi |
| event-system-db | 2 | 4 | 4Gi | 8Gi |
| response-aggregator-db | 1 | 2 | 2Gi | 4Gi |
| meta-learner-db-mongodb | 1 | 2 | 2Gi | 4Gi |
| meta-learner-db-timescaledb | 2 | 4 | 4Gi | 8Gi |
| nginx-db | 1 | 2 | 2Gi | 4Gi |
| auth-db | 1 | 2 | 2Gi | 4Gi |
| api-connector-db | 1 | 2 | 2Gi | 4Gi |

### Cache Resource Requirements

| Cache Container | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------------|-------------|-----------|----------------|--------------|
| identity-cache | 0.5 | 1 | 2Gi | 3Gi |
| persona-cache | 0.5 | 1 | 2Gi | 3Gi |
| tts-cache | 0.5 | 1 | 1Gi | 2Gi |
| stt-cache | 0.5 | 1 | 1Gi | 2Gi |
| feedback-processor-cache | 0.5 | 1 | 1Gi | 2Gi |
| working-memory-cache | 1 | 2 | 4Gi | 6Gi |
| episodic-memory-cache | 1 | 2 | 6Gi | 8Gi |
| context-manager-cache | 0.5 | 1 | 2Gi | 3Gi |
| domain-knowledge-cache | 1 | 2 | 4Gi | 6Gi |
| vector-store-cache | 2 | 4 | 8Gi | 10Gi |
| knowledge-synthesizer-cache | 1 | 2 | 4Gi | 6Gi |
| neural-interpreter-cache | 1 | 2 | 4Gi | 6Gi |
| deepseek-cache | 0.5 | 1 | 2Gi | 4Gi |
| pattern-matcher-cache | 1 | 2 | 4Gi | 6Gi |
| llm-inference-cache | 2 | 4 | 8Gi | 10Gi |
| embedding-cache | 1.5 | 3 | 6Gi | 8Gi |
| task-decomposer-cache | 0.5 | 1 | 2Gi | 3Gi |
| workflow-builder-cache | 0.5 | 1 | 2Gi | 3Gi |
| workflow-executor-cache | 0.5 | 1 | 2Gi | 3Gi |
| workflow-registry-cache | 0.5 | 1 | 2Gi | 3Gi |
| info-retrieval-cache | 1 | 2 | 4Gi | 6Gi |
| source-validation-cache | 0.5 | 1 | 2Gi | 3Gi |
| event-system-cache | 1 | 2 | 4Gi | 6Gi |
| response-aggregator-cache | 0.5 | 1 | 2Gi | 3Gi |
| meta-learner-cache | 1 | 2 | 4Gi | 6Gi |
| nginx-cache | 1 | 2 | 2Gi | 3Gi |
| auth-cache | 0.5 | 1 | 2Gi | 3Gi |
| api-connector-cache | 0.5 | 1 | 2Gi | 3Gi |

## Health Checks

Each container must implement the following health check endpoints:

| Endpoint | Purpose | Status Codes | Response Time SLA |
|----------|---------|--------------|-------------------|
| `/health/liveness` | Container is alive | 200 OK, 500 Internal Server Error | < 1s |
| `/health/readiness` | Container is ready to serve requests | 200 OK, 503 Service Unavailable | < 2s |
| `/health/startup` | Container has completed startup | 200 OK, 504 Gateway Timeout | < 5s |

All health checks must adhere to the following principles:
1. **Fast response time** (under 1 second for liveness)
2. **Minimal resource usage**
3. **No external dependencies** for liveness checks
4. **Comprehensive system checks** for readiness (includes database connectivity)
5. **Cached results** to prevent overwhelming underlying systems

### Health Check Configuration (Kubernetes)

```yaml
livenessProbe:
  httpGet:
    path: /health/liveness
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 15
  timeoutSeconds: 5
  failureThreshold: 3
readinessProbe:
  httpGet:
    path: /health/readiness
    port: 8080
  initialDelaySeconds: 15
  periodSeconds: 10
  timeoutSeconds: 3
  successThreshold: 1
  failureThreshold: 3
startupProbe:
  httpGet:
    path: /health/startup
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 12
```

## Security Configuration

### Container Security Context

All containers must run with the following security context:

```yaml
securityContext:
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
```

### Network Security

1. **TLS Everywhere**: All communications must use TLS 1.2+
2. **Certificate Rotation**: Automatic rotation every 90 days
3. **mTLS for Service-to-Service**: Mutual TLS authentication between services
4. **JWT for User Authentication**: Short-lived JWTs (1 hour) with refresh tokens

### Authentication Methods

| Interface | Auth Method | Token Lifetime | Refresh Allowed |
|-----------|-------------|----------------|-----------------|
| External API | JWT | 1 hour | Yes |
| Internal Services | mTLS | N/A | N/A |
| Admin Portal | OAuth2 | 4 hours | Yes |
| Event System | Client Certificate | N/A | N/A |
| Database Access | Database-specific auth | N/A | N/A |

### Secret Management

1. **No hardcoded secrets** in code or config files
2. **Vault integration** for secret retrieval
3. **Secret rotation** scheduled automatically
4. **Least privilege access** to secrets

## Observability

### Logging

All containers must implement structured logging with the following specifications:

```json
{
  "timestamp": "2023-09-08T12:34:56.789Z",
  "level": "info",
  "service": "neural-interpreter",
  "trace_id": "abcdef123456",
  "span_id": "fedcba654321",
  "client_id": "client-123",
  "message": "Request processed successfully",
  "request_id": "req-789",
  "duration_ms": 42,
  "additional_context": {}
}
```

Log levels must adhere to the following guidelines:
- **ERROR**: System errors that require immediate attention
- **WARN**: Potential issues that don't cause system failure
- **INFO**: Normal operational events
- **DEBUG**: Detailed information for troubleshooting (development only)
- **TRACE**: Very detailed debugging information (development only)

### Metrics

All containers must expose Prometheus metrics on port 8081 at path `/metrics` with the following:

1. **Request rate**: Requests per second
2. **Error rate**: Errors per second
3. **Latency**: P50, P95, P99 latencies
4. **System resources**: CPU, memory, disk, network usage
5. **Business metrics**: Domain-specific metrics (e.g., task completion rate)

Example metrics:
```
# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="post",code="200",handler="/api/query"} 1027
# HELP http_request_duration_seconds HTTP request latency in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.05",method="post",handler="/api/query"} 24054
```

### Tracing

All containers must implement distributed tracing using OpenTelemetry with the following:

1. **Trace propagation** via W3C Trace Context headers
2. **Sampling rate**: 10% in production, 100% in development/testing
3. **Span naming convention**: `{service}.{operation}`
4. **Critical path tracking**: Flagging spans in critical path
5. **Error annotation**: Explicit error tagging

### Alerting

Alert rules must be defined for:

1. **Service availability**: < 99.9% over 5 minutes
2. **Error rate**: > 1% over 5 minutes
3. **Latency**: P95 > 500ms over 5 minutes
4. **Resource usage**: > 80% CPU or memory for 10 minutes
5. **Business metrics**: Domain-specific thresholds

## Rate Limiting

All services must implement rate limiting to protect against abuse and ensure fair resource allocation:

### External API Rate Limits

| Client Tier | Requests per Minute | Burst Capacity | Concurrent Requests |
|-------------|---------------------|----------------|---------------------|
| Standard | 60 | 10 | 5 |
| Professional | 300 | 30 | 15 |
| Enterprise | 1200 | 100 | 50 |

### Internal Service Rate Limits

| Service | Requests per Second | Burst Capacity | Concurrent Requests |
|---------|---------------------|----------------|---------------------|
| Neural Interpreter | 100 | 50 | 20 |
| LLM Inference | 50 | 20 | 10 |
| Embedding Service | 200 | 100 | 30 |
| Knowledge Services | 150 | 75 | 25 |
| Event System Core | 500 | 250 | 100 |

### Rate Limit Implementation

1. **Token bucket algorithm** for rate limiting
2. **Client identification** via API key or client ID
3. **Response headers** indicating rate limit status
4. **Graceful degradation** when approaching limits
5. **Circuit breaking** for overloaded services

Example rate limit headers:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 56
X-RateLimit-Reset: 1678984265
```

## Scaling Configuration

### Autoscaling Parameters

| Container | Min Replicas | Max Replicas | Scale Up Threshold | Scale Down Threshold | Scale Up Rate | Scale Down Rate |
|-----------|--------------|--------------|--------------------|--------------------|--------------|----------------|
| neural-interpreter | 2 | 10 | CPU > 75% | CPU < 30% | 2 pods / minute | 1 pod / 5 minutes |
| deepseek-service | 1 | 5 | CPU > 80% | CPU < 40% | 1 pod / 2 minutes | 1 pod / 10 minutes |
| llm-inference | 1 | 5 | GPU > 80% | GPU < 40% | 1 pod / 2 minutes | 1 pod / 10 minutes |
| embedding-service | 1 | 5 | CPU > 75% | CPU < 35% | 1 pod / minute | 1 pod / 5 minutes |
| working-memory-service | 2 | 8 | CPU > 70% | CPU < 30% | 2 pods / minute | 1 pod / 5 minutes |
| episodic-memory-service | 2 | 8 | CPU > 70% | CPU < 30% | 2 pods / minute | 1 pod / 5 minutes |
| nginx-gateway | 2 | 10 | CPU > 60% | CPU < 20% | 2 pods / minute | 1 pod / 5 minutes |
| task-decomposer | 2 | 8 | CPU > 75% | CPU < 30% | 2 pods / minute | 1 pod / 5 minutes |

### Pod Disruption Budget

To ensure service availability during updates and node maintenance:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: neural-interpreter-pdb
spec:
  minAvailable: 1  # At least 1 pod must be available
  selector:
    matchLabels:
      app: neural-interpreter
```

### Topology Spread Constraints

To ensure pods are distributed across nodes:

```yaml
topologySpreadConstraints:
- maxSkew: 1
  topologyKey: kubernetes.io/hostname
  whenUnsatisfiable: DoNotSchedule
  labelSelector:
    matchLabels:
      app: neural-interpreter
```

## Environment Variables

### Common Environment Variables (All Containers)

```
CLIENT_ID="{client_id}"
ENVIRONMENT="{environment}"  # development, staging, production
LOG_LEVEL="INFO"  # ERROR, WARN, INFO, DEBUG, TRACE
METRICS_ENABLED="true"
PULSAR_SERVICE_URL="pulsar://pulsar-proxy:6650"
OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"
OTEL_SERVICE_NAME="{service_name}"
OTEL_RESOURCE_ATTRIBUTES="deployment.environment={environment},service.version={version}"
```

### Container-Specific Environment Variables

#### Identity Manager

```
IDENTITY_DB_URL="postgresql://user:password@identity-db:5432/identity"
IDENTITY_CACHE_URL="redis://identity-cache:6379/0"
AUTH_SERVICE_URL="http://auth-sidecar:8080"
PROFILE_STORAGE_PATH="/data/profiles"
TOKEN_SECRET_KEY="{secret_key}"
TOKEN_EXPIRY_SECONDS="3600"
```

#### Neural Interpreter

```
NEURAL_INTERPRETER_DB_URL="postgresql://user:password@neural-interpreter-db:5434/neural-interpreter"
NEURAL_INTERPRETER_CACHE_URL="redis://neural-interpreter-cache:6385/0"
PATTERN_MATCHING_ENABLED="true"
INTENT_ANALYSIS_THRESHOLD="0.7"
MAX_COMPLEXITY_LEVEL="1.0"
REACTIVE_COMPLEXITY_THRESHOLD="0.3"
STANDARD_COMPLEXITY_THRESHOLD="0.7"
```

#### DeepSeek Service

```
DEEPSEEK_DB_URL="postgresql://user:password@deepseek-db:5435/deepseek"
DEEPSEEK_CACHE_URL="redis://deepseek-cache:6386/0"
MODEL_PATH="/models/deepseek-v1"
INFERENCE_PRECISION="float16"  # float32, float16, int8
MAX_BATCH_SIZE="4"
USE_GPU="true"
CUDA_VISIBLE_DEVICES="0,1"
MAX_REQUEST_TOKENS="2048"
MAX_RESPONSE_TOKENS="1024"
```

#### LLM Inference

```
LLM_INFERENCE_DB_URL="postgresql://user:password@llm-inference-db:5436/llm-inference"
LLM_INFERENCE_CACHE_URL="redis://llm-inference-cache:6387/0"
MODEL_PATH="/models/llm-v1"
MODEL_TYPE="completion"  # completion, chat, embedding
MAX_BATCH_SIZE="8"
USE_GPU="true"
CUDA_VISIBLE_DEVICES="0,1"
INFERENCE_TIMEOUT_SECONDS="30"
```

#### Working Memory Service

```
WORKING_MEMORY_DB_URL="postgresql://user:password@working-memory-db:5453/working-memory"
WORKING_MEMORY_CACHE_URL="redis://working-memory-cache:6381/0"
MEMORY_TTL_SECONDS="3600"
MAX_MEMORY_ITEMS="1000"
PRIORITY_LEVELS="5"
TIMESCALE_ENABLED="true"
RETENTION_POLICY_DAYS="30"
```

#### Episodic Memory Service

```
EPISODIC_MEMORY_DB_URL="scylla://episodic-memory-db:9042/episodic-memory"
EPISODIC_MEMORY_CACHE_URL="redis://episodic-memory-cache:6421/0"
MEMORY_RETENTION_DAYS="90"
MEMORY_INDEXING_ENABLED="true"
CONSISTENCY_LEVEL="LOCAL_QUORUM"
READ_CONSISTENCY="LOCAL_ONE"
```

#### Vector Store

```
VECTOR_STORE_DB_URL="milvus://vector-store-db:19530"
VECTOR_STORE_CACHE_URL="redis://vector-store-cache:6384/0"
VECTOR_DIMENSION="1536"
INDEX_TYPE="HNSW"
METRIC_TYPE="IP"  # IP for inner product, L2 for Euclidean
```

#### Domain Knowledge Base

```
DOMAIN_KNOWLEDGE_DB_URL="mongodb://domain-knowledge-db-mongodb:27019/domain-knowledge"
DOMAIN_KNOWLEDGE_VECTOR_URL="milvus://domain-knowledge-db-milvus:19531"
DOMAIN_KNOWLEDGE_CACHE_URL="redis://domain-knowledge-cache:6383/0"
KNOWLEDGE_REFRESH_INTERVAL_SECONDS="3600"
VECTOR_DIMENSION="1536"
INDEX_TYPE="HNSW"
```

#### Embedding Service

```
EMBEDDING_DB_URL="http://embedding-db:6334"
EMBEDDING_CACHE_URL="redis://embedding-cache:6388/0"
MODEL_PATH="/models/embedding-model"
VECTOR_DIMENSION="1536"
EMBEDDING_BATCH_SIZE="32"
NORMALIZE_VECTORS="true"
USE_GPU="true"
CUDA_VISIBLE_DEVICES="0"
```

#### Pattern Matcher

```
PATTERN_MATCHER_DB_URL="http://pattern-matcher-db:6333"
PATTERN_MATCHER_CACHE_URL="redis://pattern-matcher-cache:6403/0"
SIMILARITY_THRESHOLD="0.75"
MAX_SEARCH_RESULTS="50"
USE_EXACT_MATCH="false"
INDEX_TYPE="HNSW"
```

#### Knowledge Synthesizer

```
KNOWLEDGE_SYNTHESIZER_DB_URL="http://knowledge-synthesizer-db:8080"
KNOWLEDGE_SYNTHESIZER_CACHE_URL="redis://knowledge-synthesizer-cache:6410/0"
GRAPH_STORAGE_PATH="/data/knowledge-graph"
CONFIDENCE_THRESHOLD="0.65"
CONTEXTIONARY_URL="contextionary:9999"
```

#### Task Decomposer

```
TASK_DECOMPOSER_DB_URL="postgresql://user:password@task-decomposer-db:5437/task-decomposer"
TASK_DECOMPOSER_CACHE_URL="redis://task-decomposer-cache:6389/0"
MAX_SUBTASKS="10"
MIN_DECOMPOSITION_COMPLEXITY="0.5"
TASK_TEMPLATE_PATH="/data/task-templates"
```

#### Workflow Registry

```
WORKFLOW_REGISTRY_DB_URL="postgresql://user:password@workflow-registry-db:5450/workflow-registry"
WORKFLOW_REGISTRY_CACHE_URL="redis://workflow-registry-cache:6407/0"
TEMPLATE_STORAGE_PATH="/data/workflow-templates"
TEMPLATE_VALIDATION_ENABLED="true"
```

#### Nginx Gateway

```
NGINX_WORKER_PROCESSES="auto"
NGINX_WORKER_CONNECTIONS="1024"
NEURAL_INTERPRETER_URL="http://neural-interpreter:8080"
SESSION_STORE_URL="redis://nginx-cache:6451/0"
SSL_CERTIFICATE_PATH="/etc/nginx/ssl/tls.crt"
SSL_KEY_PATH="/etc/nginx/ssl/tls.key"
CLIENT_MAX_BODY_SIZE="10m"
KEEPALIVE_TIMEOUT="65"
```

#### Event System Core

```
EVENT_SYSTEM_DB_URL="mongodb://event-system-db:27029/event-system"
EVENT_SYSTEM_CACHE_URL="redis://event-system-cache:6431/0"
PULSAR_BROKER_URL="pulsar://pulsar-main:6651"
PULSAR_WEBSOCKET_URL="ws://pulsar-proxy:8080"
PULSAR_FUNCTIONS_WORKER_URL="pulsar://pulsar-functions:6650"
EVENT_RETENTION_DAYS="7"
MAX_PENDING_EVENTS="10000"
```

## Load Balancing and High Availability

### Load Balancing Configuration

1. **Service Mesh**: All containers operate within a service mesh (e.g., Istio, Linkerd)
2. **Load Balancing Algorithm**: Round-robin with session affinity
3. **Health-Based Routing**: Traffic directed only to healthy instances
4. **Circuit Breaking**: Automatic circuit breaking when error threshold exceeded
5. **Retry Policy**: Automatic retry with exponential backoff

```yaml
# Service mesh configuration pattern
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: neural-interpreter
spec:
  host: neural-interpreter
  trafficPolicy:
    loadBalancer:
      simple: ROUND_ROBIN
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 5s
      http:
        http1MaxPendingRequests: 100
        maxRequestsPerConnection: 10
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
```

### High Availability Requirements

| Component | Minimum Replicas | Availability Target | Recovery Time Objective | Recovery Point Objective |
|-----------|-----------------|---------------------|-------------------------|--------------------------|
| Nginx Gateway | 2 | 99.95% | < 1 minute | 0 (stateless) |
| Neural Interpreter | 2 | 99.9% | < 2 minutes | 0 (stateless) |
| Database Services | 3 | 99.99% | < 5 minutes | < 1 minute |
| Event Bus | 3 | 99.99% | < 3 minutes | 0 (event replay) |
| LLM Services | 2 | 99.5% | < 5 minutes | 0 (stateless) |
| Core Services | 2 | 99.9% | < 3 minutes | 0 (stateless) |

## Connection Pooling Configuration

### Database Connection Pooling

```yaml
connection_pool:
  default:  # Default settings applied to all services
    min_connections: 5
    max_connections: 30
    idle_timeout: 300s  # 5 minutes
    max_lifetime: 1800s  # 30 minutes
    validation_query_timeout: 5s
    connection_timeout: 10s
    
  high_throughput:  # For services with high request rates
    min_connections: 20
    max_connections: 100
    idle_timeout: 120s  # 2 minutes
    max_lifetime: 3600s  # 1 hour
    validation_query_timeout: 3s
    connection_timeout: 5s
    
  low_latency:  # For latency-sensitive services
    min_connections: 10
    max_connections: 50
    idle_timeout: 60s  # 1 minute
    max_lifetime: 1800s  # 30 minutes
    validation_query_timeout: 2s
    connection_timeout: 3s
```

### Service-Specific Connection Pooling

| Service | Pool Type | Max Connections | Notes |
|---------|-----------|-----------------|-------|
| Neural Interpreter | high_throughput | 100 | Primary entry point, needs high concurrency |
| LLM Inference | high_throughput | 80 | Model inference, high request volume |
| Identity Manager | high_throughput | 100 | Authentication service, high concurrency |
| Working Memory | low_latency | 50 | Fast access required for memory operations |
| Vector Store | low_latency | 80 | Fast similarity search |
| Episodic Memory | default | 40 | Standard throughput requirements |
| Knowledge Services | default | 40 | Knowledge retrieval, moderate load |
| Task Management | default | 30 | Task operations, standard concurrency |

## Data Sharding and Partitioning Strategy

### Sharding Approaches

1. **Client-Based Sharding**
   - Used for: Identity Management, Working Memory, Episodic Memory
   - Implementation: Shard by client_id (hash-based)
   - Benefits: Tenant isolation, parallel processing

2. **Time-Based Partitioning**
   - Used for: Episodic Memory, Event System, Logs
   - Implementation: Partition by time periods (day, week, month)
   - Benefits: Efficient pruning, faster recent queries

3. **Feature-Based Sharding**
   - Used for: Domain Knowledge, Vector Store
   - Implementation: Shard by domain/feature category
   - Benefits: Domain-specific optimization, query isolation

4. **Workload-Based Sharding**
   - Used for: LLM Inference, Embedding Service
   - Implementation: Shard by model or embedding type
   - Benefits: Specialized resources per workload type

### Sharding Configuration Examples

#### MongoDB Sharding

```yaml
mongodb:
  sharding:
    enabled: true
    shards: 3
    configServers: 3
    mongos: 2
    shardKey:
      episodic_memory: { client_id: "hashed", timestamp: 1 }
      working_memory: { client_id: "hashed" }
      domain_knowledge: { domain: "hashed" }
  resources:
    shards:
      limits:
        cpu: "4"
        memory: "16Gi"
      requests:
        cpu: "2"
        memory: "8Gi"
```

#### PostgreSQL Partitioning

```yaml
postgresql:
  partitioning:
    enabled: true
    strategies:
      events:
        type: "RANGE"
        column: "created_at"
        interval: "1 week"
        retention: "6 months"
      tasks:
        type: "LIST"
        column: "status"
        values: ["pending", "active", "completed", "failed"]
      clients:
        type: "HASH"
        column: "client_id"
        partitions: 16
```

#### Redis Cluster Sharding

```yaml
redis-cluster:
  sharding:
    enabled: true
    slots: 16384  # Default Redis Cluster slots
    distribution:
      embedding_cache: { key_prefix: "emb:", slots_percentage: 30 }
      llm_cache: { key_prefix: "llm:", slots_percentage: 40 }
      user_session: { key_prefix: "sess:", slots_percentage: 20 }
      misc: { slots_percentage: 10 }  # Remaining slots
```

#### Vector Database Sharding

```yaml
milvus:
  sharding:
    enabled: true
    shards: 3
    replication_factor: 2
    strategy:
      partition_key: "domain"
      collections:
        embeddings: { partitions: 8, key: "embedding_type" }
        knowledge: { partitions: 16, key: "knowledge_domain" }
  
qdrant:
  sharding:
    enabled: true
    shards: 3
    replication_factor: 2
    strategy:
      collections:
        pattern_matches: { key: "pattern_category" }
        embeddings: { key: "model_type" }
```

## Database Backup and Replication

### Backup Strategy

| Database Type | Backup Method | Frequency | Retention | RPO | RTO |
|---------------|---------------|-----------|-----------|-----|-----|
| PostgreSQL | WAL Archiving + Full Backup | Daily full, continuous WAL | 30 days | < 5 min | < 30 min |
| MongoDB | Cluster Snapshots | 6 hours | 14 days | < 15 min | < 1 hour |
| Milvus | Metadata + Vector Snapshots | 12 hours | 7 days | < 1 hour | < 2 hours |
| Qdrant | Snapshot Backup | 12 hours | 7 days | < 1 hour | < 2 hours |
| Weaviate | Backup API | 12 hours | 7 days | < 1 hour | < 2 hours |
| Redis | RDB + AOF | RDB: 1 hour, AOF: continuous | 7 days | < 1 sec (AOF), < 1 hour (RDB) | < 15 min |
| ScyllaDB | Incremental Backups | 6 hours | 14 days | < 15 min | < 1 hour |
| TimescaleDB | Continuous Archiving | Daily full, continuous WAL | 30 days | < 5 min | < 30 min |

### Replication Configuration

#### PostgreSQL Replication

```yaml
postgresql:
  replication:
    mode: "streaming"  # Synchronous streaming replication
    synchronous_commit: "on"
    replicas: 2
    standby_mode: "hot_standby"
    max_wal_senders: 10
    wal_keep_segments: 64
    primary_slot_name: "replication_slot"
```

#### MongoDB Replication

```yaml
mongodb:
  replication:
    enabled: true
    replicas: 3
    arbiter: 1
    readPreference: "primaryPreferred"
    writeConcern: "majority"
    readConcern: "majority"
    enableMajorityReadConcern: true
```

#### Vector Database Replication

```yaml
milvus:
  replication:
    enabled: true
    replicas: 3
    readPreference: "nearest"
    
qdrant:
  replication:
    enabled: true
    replicas: 3
    readPreference: "nearest"
    
weaviate:
  replication:
    enabled: true
    replicas: 3
```

#### ScyllaDB Replication

```yaml
scylladb:
  replication:
    enabled: true
    strategy: "NetworkTopologyStrategy"
    datacenters:
      - name: "dc1"
        replicas: 3
      - name: "dc2"
        replicas: 2
    read_consistency: "LOCAL_QUORUM"
    write_consistency: "LOCAL_QUORUM"
```

#### TimescaleDB Replication

```yaml
timescaledb:
  replication:
    enabled: true
    replicas: 2
    synchronous_commit: "on"
    streaming_replication: true
    backup_retention_policy: "30 days"
```

## Updates and Rollout Strategy

### Update Strategy

1. **Blue-Green Deployment**: For major updates
2. **Canary Releases**: For risky changes (10% → 50% → 100%)
3. **Rolling Updates**: For routine updates
4. **Feature Flags**: For conditional feature enablement

```yaml
# Kubernetes Deployment Strategy
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

### Release Cadence

1. **Critical security updates**: Immediate deployment
2. **Bug fixes**: Weekly releases
3. **Feature updates**: Bi-weekly releases
4. **Major upgrades**: Monthly with advance notice

### Rollback Plan

1. **Automatic rollback**: Based on error rate thresholds
2. **Manual rollback**: Available through admin console
3. **Rollback testing**: Required for each deployment
4. **State reconciliation**: Automated for database schema changes

## Client Isolation

Each client receives dedicated resources in an isolated environment:

```yaml
namespace: client-{client_id}
resources:
  cpu_min: 16 cores
  memory_min: 64GB
  storage: 500GB SSD
network:
  isolated: true
  dns: client-{client_id}.edrs.local
schedule:
  start: "07:45" # 15 minutes before business hours
  stop: "17:15"  # 15 minutes after business hours
  timezone: "America/New_York"
  active_days: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
  maintenance_window: "00:00-04:00"
```

## Migration Path

The migration from the previous architecture to the optimized database and cache architecture should follow these phases:

### Phase 1: Foundation and Preparation (Weeks 1-2)

1. **Assessment & Planning**
   - Document current data models and access patterns
   - Define migration sequence and dependencies
   - Establish migration success criteria and rollback plans

2. **Environment Setup**
   - Deploy new database instances (PostgreSQL, MongoDB, Milvus, Qdrant, Weaviate, etc.)
   - Configure replication and sharding
   - Set up monitoring and observability

### Phase 2: Vector Database Migration (Weeks 3-5)

1. **Vector Store Services**
   - Deploy Qdrant, Weaviate alongside existing Milvus
   - Implement dual-write for Embedding Service (MongoDB + Qdrant)
   - Validate query performance and functionality
   - Complete migration and switch read traffic

2. **Pattern Matching and Knowledge Services**
   - Implement Weaviate for Pattern Matcher and Knowledge Synthesizer
   - Build data transformation pipelines
   - Migrate historical data in batches
   - Validate and cut over

### Phase 3: Specialized Database Adoption (Weeks 6-9)

1. **TimescaleDB for Time-Series Data**
   - Deploy TimescaleDB instances
   - Migrate Working Memory Service to TimescaleDB
   - Implement continuous aggregates and retention policies

2. **ScyllaDB for High-Throughput Services**
   - Deploy ScyllaDB cluster
   - Migrate Episodic Memory and Event logs
   - Validate performance at scale

### Phase 4: Cache Optimization (Weeks 10-12)

1. **Redis Stack Deployment**
   - Deploy Redis Stack instances for vector and specialized operations
   - Configure Vector modules for embedding caches
   - Implement service-specific eviction policies

2. **Redis Cluster Migration**
   - Deploy Redis Cluster for high-throughput services
   - Configure sharding and replication
   - Migrate from standalone Redis instances

### Phase 5: Validation and Optimization (Weeks 13-14)

1. **Performance Testing**
   - Conduct load tests against new architecture
   - Measure query performance, latency, and throughput
   - Compare with baseline metrics

2. **Fine-Tuning**
   - Optimize indexes and query patterns
   - Adjust cache sizes and eviction policies
   - Refine sharding strategies based on observed patterns

3. **Documentation and Knowledge Transfer**
   - Update system documentation
   - Conduct training for operational teams
   - Establish new monitoring dashboards and alerts

### Risk Mitigation

1. **Progressive Migration Strategy**
   - Service by service migration rather than big bang
   - Dual-write approach where possible
   - Blue/green deployments for critical services

2. **Rollback Capability**
   - Maintain original databases until validation complete
   - Script automated rollback procedures
   - Practice recovery scenarios

3. **Data Validation**
   - Checksum verification for migrated data
   - Functional validation of queries and operations
   - Parallel testing in production-like environment
     etc/nginx/ssl` | SSL certificates | 100Mi | standard | ReadWriteOnce | Yes - On update |
| Nginx Gateway | `/etc/nginx/conf.d` | Configuration files | 100Mi | standard | ReadWriteOnce | Yes - On update |
| Nginx Gateway | `/


## Events and Event Processing

### Event Schema Standards

All containers must follow the standardized event schema system based on the BaseEvent class:

```python
@dataclass
class BaseEvent:
    """Base event class for all system events"""
    
    event_id: uuid7
    source_container: str
    event_type: str
    priority: Priority
    payload: Dict[str, Any]
    metadata: Dict[str, Any]

```

Event Naming Convention
Event topics must follow the hierarchical naming pattern:

Primary domain/service
Followed by subtopic or action
Using dot notation for separation

For example:

knowledge.knowledge_updated
reasoning.abstract_reasoning_requested
gateway.request_received
research.web_research_completed

Event Priority Levels
All events must use the standardized Priority enum:
pythonCopyclass Priority(Enum):
"""Priority levels for event processing"""
LOW = "low"
NORMAL = "normal"
HIGH = "high"
CRITICAL = "critical"

# Persona info 
	1	Centralized Cognitive Layer - The core AI system operates as a unified distributed service rather than separate instances
	2	Dynamic Persona Management - Instead of hardcoded employee instances, you create flexible "personas" that:
	◦     That give the perception to the user that they are dealing with a single entity rather than a distrubuted 
                       Task decomposition, and aggregator service even though their queries are handled by many diferent areas and parts of a bigger system
               -        Are responsible for maintinaning the state and persistence of that personas responsibilities, available skills, user requests, user personality, likes and dislikes,  pretty much all the things you would expect one single employee to remember.  Including the interactions of co-workers how to deal with diferent co-workers. Stuff like that.
	3	Event Aggregation - Rather than having separate employees listening for specific events:
	◦	Events are published to a common Pulsar topic structure
	◦	Aggregator services route and prioritize events based on current workload
	◦	The system dynamically assigns appropriate personas to handle events
	4	Resource Optimization - This approach:
	◦	Eliminates idle AI capacity waiting for specific events
	◦	Allows dynamic scaling based on overall system load
	◦	Shares computational resources across all tasks
	◦	Prevents redundant processing across instances
	5	Knowledge Unification - Learning from all experiences can be unified while still maintaining persona-specific context where needed
This distributed system with dynamic persona assignment is much more scalable and efficient than creating separate "instances" of employees. The Apache Pulsar event system works even better in this model, as it can intelligently route messages based on load, priority, and other factors rather than predetermined employee assignments.