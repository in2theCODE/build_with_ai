Shared Models
Show Image
Show Image
A collection of Pydantic v2 models and schemas used across microservices in the Neural Code Generation System. This package provides standardized data models for events, messages, domain objects, and more.
Features

Event models for asynchronous communication
Message schemas for synchronous interactions
Base models with Avro schema support
Domain-specific models for code generation
Serialization utilities for Avro/JSON compatibility
Consistent schema versioning

Installation
From Private Repository
bash# Using UV (recommended)
uv pip install --index-url http://your-repository-url/simple/ shared-models

# Using pip
pip install --index-url http://your-repository-url/simple/ shared-models
For Development
bash# Clone the repository
git clone https://your-git-repo/shared-models.git
cd shared-models

# Install in development mode with dev dependencies
uv pip install -e ".[dev]"
Usage
Base Models
pythonfrom shared_models.base import AvroBaseModel, BaseEvent

# Create a custom event model
class MyCustomEvent(BaseEvent):
    event_type: str = "custom.event"
    
    @classmethod
    def create(cls, source_container: str, payload: dict):
        return cls(
            source_container=source_container,
            payload=payload
        )
Enums and Constants
pythonfrom shared_models.enums import EventType, EventPriority

# Use predefined event types
event_type = EventType.CODE_GENERATION_REQUESTED
priority = EventPriority.HIGH
Serialization and Schema Support
pythonfrom shared_models.base import AvroBaseModel

class MyModel(AvroBaseModel):
    name: str
    value: int
    
# Serialize to Avro binary format
instance = MyModel(name="test", value=42)
binary_data = instance.serialize()

# Deserialize from Avro binary format
restored = MyModel.deserialize(binary_data)
Development
Setup
bash# Install development dependencies
uv pip install -e ".[dev]"
Testing
bash# Run tests
pytest

# Test with coverage
pytest --cov=shared_models
Linting and Formatting
bash# Format code
black src

# Lint code
ruff check src

# Type checking
mypy
Package Structure
shared_models/
├── __init__.py          # Package exports and version
├── base.py              # Base models and abstractions
├── enums.py             # Enumerations and constants
├── events.py            # Event models
├── messages.py          # Message schemas
├── domain.py            # Domain-specific models
└── schema_registry.py   # Schema registry integration
Contributing

Fork the repository
Create a feature branch: git checkout -b feature-name
Commit your changes: git commit -am 'Add feature'
Push to the branch: git push origin feature-name
Submit a pull request

Versioning
We use Semantic Versioning. For the versions available, see the tags on this repository.
License
This project is licensed under the MIT License - see the LICENSE file for details.