# Neural Code Generator

Advanced neural code generation component for the Program Synthesis System. This component uses cutting-edge techniques and the DeepSeek 6.7B model to generate high-quality code from specifications.

## Features

- **Multi-head Attention with Extended Context**: Specialized attention mechanisms with 8K-16K token context window
- **Retrieval-Augmented Generation (RAG)**: Enhances code generation by retrieving relevant code examples
- **Tree-based Transformers**: Processes abstract syntax tree (AST) structures directly
- **Hierarchical Code Generation**: Two-phase approach with outline generation before implementation
- **Syntax-Aware Beam Search**: Beam search algorithm that incorporates syntax constraints
- **Hybrid Grammar-Neural Models**: Combines grammar-based approaches with neural prediction

## Requirements

- NVIDIA GPU with at least 16GB VRAM (recommended: 24GB+)
- CUDA 11.8+
- Python 3.10+
- Docker with NVIDIA Container Toolkit
- Kubernetes with GPU support (for production deployment)
- Apache Pulsar (for event-driven microservice architecture)

## Quick Start

### Local Development

1. **Build the Docker image**:
   ```bash
   docker build -f neural_code_generator.Dockerfile -t program-synthesis/neural-code-generator:dev .
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose -f docker-compose-neural.yml up
   ```

3. **Test the standalone mode**:
   ```bash
   docker run --gpus all -it --rm \
     -v $(pwd)/app:/app/app \
     -v $(pwd)/knowledge_base:/app/knowledge_base \
     program-synthesis/neural-code-generator:dev \
     python -m program_synthesis_system.services.neural_code_generator.standalone \
       --spec "Generate a function to sort a list of dictionaries by a given key" \
       --language python \
       --technique all
   ```

### Production Deployment

1. **Configure environment variables**:
   ```bash
   export REGISTRY=your-registry.example.com
   export TAG=latest
   export NAMESPACE=program-synthesis
   ```

2. **Build and push the Docker image**:
   ```bash
   docker build -f neural_code_generator.Dockerfile -t ${REGISTRY}/program-synthesis/neural-code-generator:${TAG} .
   docker push ${REGISTRY}/program-synthesis/neural-code-generator:${TAG}
   ```

3. **Deploy to Kubernetes**:
   ```bash
   ./deploy-neural-generator.sh
   ```

## Component Structure

```
program_synthesis_system/
└── components/
    └── neural_code_generator/
        ├── enhanced_neural_code_generator.py  # Core implementation
        ├── service.py                         # Service module
        └── standalone.py                      # Standalone execution module
    └── knowledge_base/
        └── initializer.py                     # Knowledge base initialization
└── utils/
    └── health_check.py                        # Health check API
```

## Usage

### As a Service

The neural code generator integrates with the event-driven architecture using Apache Pulsar:

1. Send a request to the `code-generation-requests` topic:
   ```json
   {
     "request_id": "req-123",
     "function_name": "sort_dictionaries",
     "language": "python",
     "ast": {
       "name": "sort_dictionaries",
       "parameters": [
         {"name": "items", "type": "List[Dict]"},
         {"name": "key", "type": "str"}
       ],
       "return_type": "List[Dict]"
     },
            "constraints": [
       "Items should maintain original order if key values are equal",
       "Should handle missing keys gracefully",
       "Should work with any comparable type for the key value"
     ],
     "examples": [
       {
         "input": {
           "items": [{"id": 3, "name": "C"}, {"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
           "key": "id"
         },
         "output": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}, {"id": 3, "name": "C"}]
       }
     ]
   }
   ```

2. Receive the result from the `code-generation-results` topic:
   ```json
   {
     "request_id": "req-123",
     "status": "success",
     "result": {
       "program_ast": {
         "type": "function",
         "name": "sort_dictionaries",
         "parameters": [
           {"name": "items", "type": "List[Dict]"},
           {"name": "key", "type": "str"}
         ],
         "return_type": "List[Dict]",
         "code": "def sort_dictionaries(items: List[Dict], key: str) -> List[Dict]:\n    \"\"\"Sort a list of dictionaries by the specified key.\n\n    Args:\n        items: List of dictionaries to sort\n        key: Dictionary key to sort by\n\n    Returns:\n        Sorted list of dictionaries\n\n    Examples:\n        >>> sort_dictionaries([{\"id\": 3, \"name\": \"C\"}, {\"id\": 1, \"name\": \"A\"}, {\"id\": 2, \"name\": \"B\"}], \"id\")\n        [{\"id\": 1, \"name\": \"A\"}, {\"id\": 2, \"name\": \"B\"}, {\"id\": 3, \"name\": \"C\"}]\n    \"\"\"\n    # Create a function to get the sort key, handling missing keys\n    def get_key(item):\n        return item.get(key, None)\n    \n    # Sort the list using a stable sort to maintain original order for equal values\n    return sorted(items, key=get_key)"
       },
       "confidence_score": 0.92,
       "time_taken": 2.43,
       "strategy": "hierarchical"
     }
   }
   ```

### As a Standalone Tool

For direct usage without Pulsar:

```bash
python -m program_synthesis_system.services.neural_code_generator.standalone \
  --spec path/to/specification.json \
  --language python \
  --technique hierarchical \
  --output generated_code.py
```

## Configuration

See `configs/deepseek_model_config.yaml` for detailed model configuration options.

Environment variables for the container:

| Variable | Description | Default |
|----------|-------------|---------|
| PULSAR_SERVICE_URL | Apache Pulsar service URL | pulsar://pulsar:6650 |
| INPUT_TOPIC | Topic for incoming requests | code-generation-requests |
| OUTPUT_TOPIC | Topic for outgoing results | code-generation-results |
| MODEL_PATH | Path to the DeepSeek model | /app/models/deepseek-coder-8b-instruct |
| QUANTIZATION | Model quantization level | int8 |
| USE_FLASH_ATTENTION | Enable Flash Attention | true |
| BATCH_SIZE | Batch size for processing | 1 |
| MAX_CONTEXT_LENGTH | Maximum context length | 8192 |
| ENABLE_HEALTH_CHECK | Enable health check API | true |
| HEALTH_CHECK_PORT | Port for health check API | 8000 |

## File Placement

Here's where each file should be placed in the project structure:

### Python Modules
- `program_synthesis_system/components/neural_code_generator/enhanced_neural_code_generator.py` (new file)
- `program_synthesis_system/components/neural_code_generator/service.py` (new file)
- `program_synthesis_system/components/neural_code_generator/standalone.py` (new file)
- `program_synthesis_system/components/knowledge_base/initializer.py` (new file)
- `program_synthesis_system/shared/health_check.py` (new file)

### Configuration Files
- `configs/deepseek_model_config.yaml` (new file)

### Deployment Files
- `neural_code_generator.Dockerfile` (new file at project root)
- `docker-compose-neural.yml` (new file at project root)
- `requirements-neural.txt` (new file at project root)
- `neural_entrypoint.sh` (new file at project root)

### Kubernetes Files
- `kubernetes/neural-code-generator-deployment.yaml` (new directory and file)
- `kubernetes/neural-code-generator-service.yaml` (new file)
- `kubernetes/neural-pvc.yaml` (new file)
- `kubernetes/knowledge-base-initializer-job.yaml` (new file)
- `deploy-neural-generator.sh` (new file at project root)

### Documentation
- `docs/neural_code_generator/README.md` (this file)
- `docs/neural_code_generator/ssot.yaml` (SSOT configuration snippet)

## Contributing

When adding new features to the neural code generator:

1. **Techniques**: Add new neural generation techniques to `enhanced_neural_code_generator.py`
2. **Models**: Add model configuration to `configs/deepseek_model_config.yaml`
3. **Testing**: Add tests in `tests/components/neural_code_generator/`

## License

Same as the main Program Synthesis System.