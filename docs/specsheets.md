Specification Sheets
1. Spec Registry Service
yamlservice_name: spec_registry
version: 0.1.0
description: Manages specification templates and instances for projects
tenant_aware: true

database:
  type: supabase
  tables:
    - name: spec_templates
      columns:
        - name: id
          type: uuid
          primary_key: true
        - name: tenant_id
          type: uuid
          nullable: false
        - name: name
          type: text
          nullable: false
        - name: version
          type: text
          nullable: false
        - name: description
          type: text
        - name: schema
          type: jsonb
          nullable: false
        - name: created_at
          type: timestamp
          nullable: false
        - name: updated_at
          type: timestamp
          nullable: false
      constraints:
        - type: unique
          columns: [tenant_id, name, version]
    
    - name: spec_instances
      columns:
        - name: id
          type: uuid
          primary_key: true
        - name: template_id
          type: uuid
          nullable: false
        - name: project_id
          type: uuid
          nullable: false
        - name: user_id
          type: uuid
          nullable: false
        - name: tenant_id
          type: uuid
          nullable: false
        - name: values
          type: jsonb
          nullable: false
        - name: created_at
          type: timestamp
          nullable: false
        - name: updated_at
          type: timestamp
          nullable: false
      constraints:
        - type: foreign_key
          columns: [template_id]
          references: spec_templates(id)

api_endpoints:
  - path: /templates
    methods: [GET, POST]
    description: List and create templates
  - path: /templates/{template_id}
    methods: [GET, PUT, DELETE]
    description: Retrieve, update, and delete templates
  - path: /instances
    methods: [GET, POST]
    description: List and create spec instances
  - path: /instances/{instance_id}
    methods: [GET, PUT, DELETE]
    description: Retrieve, update, and delete spec instances
  - path: /projects/{project_id}/specs
    methods: [GET]
    description: Get all specs for a project

events:
  produces:
    - name: template.created
      schema: spec_template_event
    - name: template.updated
      schema: spec_template_event
    - name: instance.created
      schema: spec_instance_event
    - name: instance.updated
      schema: spec_instance_event
  consumes:
    - name: project.created
      schema: project_event

container:
  resources:
    cpu: 0.5
    memory: 512Mi
  ports:
    - 8000:8000
  health_check:
    path: /health
    port: 8000
2. Neural Code Generator Service
yamlservice_name: neural_code_generator
version: 0.1.0
description: Generates code based on specifications using advanced neural techniques
tenant_aware: true

components:
  - name: generator_engine
    description: Core code generation logic
    techniques:
      - multi_head_attention
      - retrieval_augmentation
      - tree_transformers
      - hierarchical_generation
      - syntax_aware_beam_search
      - hybrid_grammar_neural
  
  - name: model_manager
    description: Manages LLM models (local and API)
    supported_models:
      - deepseek_8b
      - openai_api
      - anthropic_api
      - local_llama

api_endpoints:
  - path: /generate
    methods: [POST]
    description: Generate code from specifications
  - path: /techniques
    methods: [GET]
    description: List available generation techniques
  - path: /models
    methods: [GET]
    description: List available models

events:
  produces:
    - name: code.generated
      schema: code_generation_result
  consumes:
    - name: spec.complete
      schema: complete_spec_event

container:
  resources:
    cpu: 2.0
    memory: 8Gi
    gpu: 1
  ports:
    - 8001:8001
  health_check:
    path: /health
    port: 8001
  volumes:
    - name: models
      path: /app/models
3. Knowledge Base Service
yamlservice_name: knowledge_base
version: 0.1.0
description: Vector database for code snippets and project knowledge
tenant_aware: true

components:
  - name: vector_store
    type: qdrant
    description: Stores code embeddings for retrieval
  
  - name: embedder
    description: Creates embeddings from code snippets
    models:
      - sentence_transformers
      - code_bert

database:
  type: supabase
  tables:
    - name: code_snippets
      columns:
        - name: id
          type: uuid
          primary_key: true
        - name: tenant_id
          type: uuid
          nullable: false
        - name: user_id
          type: uuid
          nullable: false
        - name: language
          type: text
          nullable: false
        - name: content
          type: text
          nullable: false
        - name: description
          type: text
        - name: tags
          type: jsonb
        - name: created_at
          type: timestamp
          nullable: false
    
    - name: project_knowledge
      columns:
        - name: id
          type: uuid
          primary_key: true
        - name: project_id
          type: uuid
          nullable: false
        - name: tenant_id
          type: uuid
          nullable: false
        - name: key
          type: text
          nullable: false
        - name: value
          type: jsonb
          nullable: false
        - name: created_at
          type: timestamp
          nullable: false
        - name: updated_at
          type: timestamp
          nullable: false

api_endpoints:
  - path: /snippets
    methods: [GET, POST]
    description: List and create code snippets
  - path: /snippets/{snippet_id}
    methods: [GET, PUT, DELETE]
    description: Retrieve, update, and delete code snippets
  - path: /knowledge/{project_id}
    methods: [GET, POST]
    description: List and create project knowledge entries
  - path: /similar
    methods: [POST]
    description: Find similar code snippets to query

events:
  produces:
    - name: knowledge.updated
      schema: knowledge_update_event
  consumes:
    - name: code.generated
      schema: code_generation_result

container:
  resources:
    cpu: 1.0
    memory: 2Gi
  ports:
    - 8002:8002
  health_check:
    path: /health
    port: 8002
  volumes:
    - name: vector_db
      path: /app/data
4. Workflow Orchestrator Service
yamlservice_name: workflow_orchestrator
version: 0.1.0
description: Coordinates the overall workflow and maintains project state
tenant_aware: true

components:
  - name: state_machine
    description: Manages workflow state transitions
    states:
      - initialization
      - spec_definition
      - code_generation
      - integration
      - debugging
  
  - name: task_scheduler
    description: Schedules and coordinates tasks across services

database:
  type: supabase
  tables:
    - name: projects
      columns:
        - name: id
          type: uuid
          primary_key: true
        - name: tenant_id
          type: uuid
          nullable: false
        - name: name
          type: text
          nullable: false
        - name: description
          type: text
        - name: state
          type: text
          nullable: false
        - name: user_id
          type: uuid
          nullable: false
        - name: created_at
          type: timestamp
          nullable: false
        - name: updated_at
          type: timestamp
          nullable: false
    
    - name: workflow_tasks
      columns:
        - name: id
          type: uuid
          primary_key: true
        - name: project_id
          type: uuid
          nullable: false
        - name: tenant_id
          type: uuid
          nullable: false
        - name: service
          type: text
          nullable: false
        - name: task_type
          type: text
          nullable: false
        - name: status
          type: text
          nullable: false
        - name: created_at
          type: timestamp
          nullable: false
        - name: updated_at
          type: timestamp
          nullable: false

api_endpoints:
  - path: /projects
    methods: [GET, POST]
    description: List and create projects
  - path: /projects/{project_id}
    methods: [GET, PUT, DELETE]
    description: Retrieve, update, and delete projects
  - path: /projects/{project_id}/state
    methods: [GET, PUT]
    description: Get or update project state
  - path: /tasks
    methods: [GET, POST]
    description: List and create workflow tasks
  - path: /tasks/{task_id}
    methods: [GET, PUT]
    description: Retrieve and update workflow tasks

events:
  produces:
    - name: project.created
      schema: project_event
    - name: project.state_changed
      schema: project_state_event
    - name: task.created
      schema: task_event
    - name: task.completed
      schema: task_event
  consumes:
    - name: spec.complete
      schema: complete_spec_event
    - name: code.generated
      schema: code_generation_result

container:
  resources:
    cpu: 0.5
    memory: 512Mi
  ports:
    - 8003:8003
  health_check:
    path: /health
    port: 8003
5. API Gateway Service
yamlservice_name: api_gateway
version: 0.1.0
description: Entry point for client requests, handles routing, authentication, and rate limiting
tenant_aware: true

components:
  - name: router
    description: Routes requests to appropriate services
  - name: auth_middleware
    description: Handles authentication and authorization
  - name: rate_limiter
    description: Implements rate limiting policies

api_endpoints:
  - path: /api/*
    methods: [GET, POST, PUT, DELETE]
    description: All API paths, proxied to appropriate services
  - path: /auth/*
    methods: [GET, POST]
    description: Authentication endpoints

security:
  authentication:
    - jwt
    - api_key
  authorization:
    - rbac
    - tenant_isolation

container:
  resources:
    cpu: 1.0
    memory: 1Gi
  ports:
    - 80:80
    - 443:443
  health_check:
    path: /health
    port: 80
