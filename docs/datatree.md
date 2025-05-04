```
program_synthesis_system/
├── bin/
│   ├── synthesize                      # Command-line script entry point
│   └── synthesize.py                   # Main CLI implementation
│
├── components/
│   ├── neural_code_generator/          # Core neural code generation component
│   │   ├── enhanced_neural_code_generator.py  # Advanced implementation with all techniques
│   │   ├── neural_code_generator.py    # Base implementation 
│   │   ├── service.py                  # Service module for event-driven architecture
│   │   └── standalone.py               # Standalone execution module
│   │
│   ├── knowledge_base/                # Knowledge base for code retrieval
│   │   ├── vector_knowledge_base.py   # Vector-based storage implementation
│   │   └── initializer.py             # Knowledge base initialization
│   │
│   ├── ast_code_generator/            # AST to code conversion
│   │   └── ast_code_generator.py      # Code generation from ASTs
│   │
│   ├── interactive_synthesis/          # Interactive synthesis components 
│   │   ├── interactive_synthesizer.md  # Interactive synthesis controller
│   │   ├── persona_synthesizer.md      # Persona-based interaction
│   │   └── event_adapter.md           # Event adapter for integration
│   │
│   ├── incremental_synthesis/         # Support for complex specifications
│   │   └── incremental_synthesis.py   # Divide-and-conquer strategies
│   │
│   └── constraint_relaxer/            # Constraint relaxation
│       └── constraint_relaxer.py      # For when synthesis fails
│
├── utils/
│   ├── health_check.py                # Health check endpoints
│   └── neural_integration.py          # Integration with broader system
│
├── configs/                           # Configuration files
│   ├── system_config.yaml             # System-wide configuration
│   ├── deepseek_model_config.yaml     # DeepSeek model configuration
│   └── neural_code_technique.md       # Technique documentation
│
├── kubernetes/                        # Kubernetes deployment
│   ├── neural-code-generator-deployment.yaml  # Deployment configuration
│   ├── neural-code-generator-service.yaml     # Service configuration 
│   ├── neural-pvc.yaml                        # Persistent volume claims
│   └── knowledge-base-initializer-job.yaml    # KB initialization job
│
├── docker/                            # Container configuration
│   ├── Dockerfile                     # Container definition
│   ├── neural_entrypoint.sh           # Container entrypoint
│   ├── requirements-neural.txt        # Python dependencies
│   └── docker-compose-neural.yml      # Local development setup
│
└── src/                               # Core system components
    ├── system.py                      # Main system implementation
    ├── synthesis_engine.py            # Orchestrates synthesis process
    ├── component_factory.py           # Factory for creating components
    └── statistical_verifier.py        # Verification implementation
```