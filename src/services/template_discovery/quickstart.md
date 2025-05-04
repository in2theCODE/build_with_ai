# Quick Start Guide: Template Registry System

This guide will help you quickly set up and test the Template Registry System for code generation.

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/template-registry-system.git
cd template-registry-system
```

2. **Install dependencies**

```bash
pip install pulsar-client dataclasses threading logging typing
```

3. **Set up Apache Pulsar** (if you don't have it already)

```bash
# Option 1: Using Docker
docker run -it -p 6650:6650 -p 8080:8080 --mount source=pulsardata,target=/pulsar/data --mount source=pulsarconf,target=/pulsar/conf apachepulsar/pulsar:latest bin/pulsar standalone

# Option 2: Local installation
wget https://archive.apache.org/dist/pulsar/pulsar-2.10.0/apache-pulsar-2.10.0-bin.tar.gz
tar xvfz apache-pulsar-2.10.0-bin.tar.gz
cd apache-pulsar-2.10.0
bin/pulsar standalone
```

## Basic Usage

Let's create a simple script to test the template system:

1. **Create a test script** (`test_template_system.py`)

```python
from template_registry_system import (
    TemplateRegistry, TemplateManager, TemplateOrganizer,
    TemplateType, Language, Placeholder
)

# Initialize the registry
registry = TemplateRegistry(
    base_dir="./template_registry_data",
    broker_url="pulsar://localhost:6650",
    enable_events=True
)

# Create manager and organizer
manager = TemplateManager(registry)
organizer = TemplateOrganizer(registry)

# Create a simple Python function template
def create_python_function_template():
    code = """def {{ function_name }}({{ parameters }}):
    \"\"\"
    {{ description }}
    \"\"\"
    {{ implementation }}"""
    
    placeholders = [
        Placeholder(
            name="function_name",
            type="string",
            description="Name of the function",
            required=True
        ),
        Placeholder(
            name="parameters",
            type="string",
            description="Function parameters",
            required=False,
            default_value=""
        ),
        Placeholder(
            name="description",
            type="string",
            description="Function description",
            required=True
        ),
        Placeholder(
            name="implementation",
            type="code",
            description="Function implementation",
            required=False,
            default_value="pass"
        )
    ]
    
    template = manager.create_template(
        name="Python Function",
        description="Template for Python functions",
        code=code,
        template_type=TemplateType.CODE_PATTERN,
        languages=[Language.PYTHON],
        placeholders=placeholders,
        tags=["function", "python"]
    )
    
    return template.id

# Create a simple Next.js component template
def create_nextjs_component_template():
    code = """import React from 'react';

const {{ component_name }} = ({ {{ props }} }) => {
    return (
        <div>
            {{ content }}
        </div>
    );
};

export default {{ component_name }};"""
    
    placeholders = [
        Placeholder(
            name="component_name",
            type="string",
            description="Name of the component",
            required=True
        ),
        Placeholder(
            name="props",
            type="string",
            description="Component props",
            required=False,
            default_value=""
        ),
        Placeholder(
            name="content",
            type="code",
            description="Component content",
            required=True
        )
    ]
    
    template = manager.create_template(
        name="Next.js Component",
        description="Template for Next.js functional services",
        code=code,
        template_type=TemplateType.CODE_PATTERN,
        languages=[Language.JAVASCRIPT],
        placeholders=placeholders,
        tags=["react", "nextjs", "component"]
    )
    
    return template.id

# Test the template system
def test_template_system():
    # Create templates
    python_template_id = create_python_function_template()
    nextjs_template_id = create_nextjs_component_template()
    
    print(f"Created Python template with ID: {python_template_id}")
    print(f"Created Next.js template with ID: {nextjs_template_id}")
    
    # Instantiate Python template
    python_values = {
        "function_name": "calculate_factorial",
        "parameters": "n: int",
        "description": "Calculate the factorial of a number",
        "implementation": """if n <= 1:
        return 1
    else:
        return n * calculate_factorial(n - 1)"""
    }
    
    python_code = registry.instantiate_template(python_template_id, python_values)
    print("\nGenerated Python code:")
    print("=====================")
    print(python_code)
    
    # Instantiate Next.js template
    nextjs_values = {
        "component_name": "UserProfile",
        "props": "user, onUpdate",
        "content": """<div className="user-profile">
            <h1>{user.name}</h1>
            <p>{user.email}</p>
            <button onClick={onUpdate}>Update Profile</button>
        </div>"""
    }
    
    nextjs_code = registry.instantiate_template(nextjs_template_id, nextjs_values)
    print("\nGenerated Next.js code:")
    print("=====================")
    print(nextjs_code)
    
    # List templates
    templates = registry.list_templates()
    print("\nAll templates:")
    print("=============")
    for template in templates:
        print(f"- {template['name']} ({template['id']})")
    
    # Clean up
    registry.close()

if __name__ == "__main__":
    test_template_system()
```

2. **Run the script**

```bash
python test_template_system.py
```

You should see the created templates, the generated code, and a list of all templates in the registry.

## Testing Template Evolution

Let's create another script to test the template evolution system:

1. **Create a test script** (`test_template_evolution.py`)

```python
from template_registry_system import (
    TemplateRegistry, TemplateManager, TemplateEvolutionSystem
)
import time

# Initialize the registry (use the same base_dir as before)
registry = TemplateRegistry(
    base_dir="./template_registry_data",
    broker_url="pulsar://localhost:6650",
    enable_events=True
)

# Get all templates
templates = registry.list_templates()
if not templates:
    print("No templates found. Please run test_template_system.py first.")
    exit(1)

# Select the first template for testing
template_id = templates[0]['id']
template = registry.get_template(template_id)
print(f"Testing evolution for template: {template.metadata.name}")

# Create evolution system
evolution = TemplateEvolutionSystem(registry)

# Collect some feedback
print("\nCollecting feedback...")
for i in range(5):
    success = i % 2 == 0  # Alternate between success and failure
    
    feedback_data = {
        "success": success,
        "generation_time": 0.5,
        "comment": "Good template" if success else "Error in generated code"
    }
    
    evolution.collect_feedback(template_id, feedback_data)
    print(f"  Added {'positive' if success else 'negative'} feedback")
    time.sleep(0.5)

# Analyze feedback
print("\nAnalyzing feedback...")
analysis = evolution.analyze_feedback(template_id)

print("\nFeedback Analysis:")
print("=================")
print(f"Template: {analysis['template_name']}")
print(f"Feedback count: {analysis['feedback_count']}")
print(f"Success rate: {analysis['success_rate']:.2f}")

if analysis['common_issues']:
    print("\nCommon issues:")
    for issue, count in analysis['common_issues']:
        print(f"- {issue}: {count} occurrences")

if analysis['improvement_suggestions']:
    print("\nImprovement suggestions:")
    for suggestion in analysis['improvement_suggestions']:
        print(f"- {suggestion}")

# Clean up
registry.close()
```

2. **Run the script**

```bash
python test_template_evolution.py
```

This will collect feedback for a template and analyze it to generate improvement suggestions.

## Testing with Apache Pulsar Events

Let's create a script to test the event-driven features:

1. **Create a test script** (`test_event_system.py`)

```python
from template_registry_system import (
    TemplateRegistry, PulsarTemplateEventManager
)
import time
import threading

# Initialize the registry
registry = TemplateRegistry(
    base_dir="./template_registry_data",
    broker_url="pulsar://localhost:6650",
    enable_events=True
)

# Create event manager
event_manager = PulsarTemplateEventManager(
    registry_path="./template_registry_data",
    broker_url="pulsar://localhost:6650"
)
event_manager.initialize()
event_manager.start_processing()

# Get all templates
templates = registry.list_templates()
if not templates:
    print("No templates found. Please run test_template_system.py first.")
    exit(1)

# Select the first template for testing
template_id = templates[0]['id']
template = registry.get_template(template_id)
print(f"Testing events for template: {template.metadata.name}")

# Function to handle events
def event_publisher():
    for i in range(5):
        event_type = "template.updated" if i % 2 == 0 else "template.used"
        
        event_data = {
            "template_id": template_id,
            "user_id": "test_user",
            "details": f"Test event {i+1}"
        }
        
        print(f"Publishing {event_type} event...")
        event_manager.publish_event(event_type, event_data)
        time.sleep(1)

# Start event publisher in a separate thread
publisher_thread = threading.Thread(target=event_publisher)
publisher_thread.start()

# Wait for events to be processed
print("Waiting for events to be processed...")
publisher_thread.join()
time.sleep(2)  # Additional time to process events

# Clean up
event_manager.stop_processing()
event_manager.close()
registry.close()

print("Event test completed")
```

2. **Run the script**

```bash
python test_event_system.py
```

This will publish events and process them using the Pulsar event system.

## Next Steps

Now that you've tested the basic functionality of the Template Registry System, you can:

1. **Create more templates** for your specific domain
2. **Integrate** the system with your existing code generation project
3. **Extend** the system with custom features
4. **Deploy** it as part of your development workflow

See the [Implementation Guide](implementation-guide.md) for detailed integration instructions, and the [README](README.md) for a complete overview of the system's capabilities.

## Troubleshooting

If you encounter issues:

- **Connection errors**: Make sure Apache Pulsar is running and accessible
- **Registry errors**: Check if the base directory is writable
- **Template errors**: Verify template syntax and placeholder usage
- **Event errors**: Check Pulsar topic configuration

For detailed logging, add this to your scripts:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```