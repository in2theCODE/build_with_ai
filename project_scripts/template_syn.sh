#!/bin/bash
# template_synthesizer.sh - Create a lightweight template generator

# Set up environment
TEMPLATE_DIR="./templates"
SYNTHESIS_DIR="./synthesis_engine"

# Create the synthesis directory
mkdir -p $SYNTHESIS_DIR

# Create the template generator
cat > $SYNTHESIS_DIR/generate_template.py << 'EOF'
#!/usr/bin/env python3
"""
Template Generator
Creates specialized templates based on requirements
"""
import os
import json
import uuid
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

class TemplateGenerator:
    """Generate specialized templates based on requirements"""

    def __init__(self, template_dir: str):
        self.template_dir = template_dir

    def create_template(self,
                       category: str,
                       name: str,
                       description: str,
                       components: List[Dict[str, Any]],
                       audience: str = "system") -> Dict[str, Any]:
        """Create a new template based on components"""
        # Create unique ID with category prefix
        template_id = f"{category}-{name.replace(' ', '-').lower()}-{uuid.uuid4().hex[:8]}"

        # Create metadata
        metadata = {
            "name": name,
            "description": description,
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat().replace("+00:00","z"),
            "updated_at": datetime.utcnow().isoformat().replace("+00:00","z"),
            "category": category,
            "subcategory": "",
            "audience": audience,
            "tags": [category, "synthesized"],
            "complexity": self._calculate_complexity(components)
        }

        # Create template
        template = {
            "id": template_id,
            "metadata": metadata,
            "source_type": "synthesized",
            "source_location": "",
            "relationships": [],
            "components": components,
            "variables": [],
            "is_cached": False,
            "cache_path": ""
        }

        # Save template to appropriate directory
        self._save_template(template, category)
        return template

    def _calculate_complexity(self, components: List[Dict[str, Any]]) -> int:
        """Calculate template complexity based on components"""
        # Simple complexity calculation based on component count and required status
        base_complexity = min(len(components), 10)
        required_components = sum(1 for comp in components if comp.get("required", False))

        # Scale complexity between 1-10
        return max(1, min(10, base_complexity + (required_components // 2)))

    def _save_template(self, template: Dict[str, Any], category: str) -> None:
        """Save template to appropriate directory"""
        category_dir = os.path.join(self.template_dir, "categories", category)
        os.makedirs(category_dir, exist_ok=True)

        template_path = os.path.join(category_dir, f"{template['id']}.json")
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)

        print(f"Template saved to: {template_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate a template")
    parser.add_argument("--template-dir", default="./templates", help="Template directory")
    parser.add_argument("--category", required=True, help="Template category")
    parser.add_argument("--name", required=True, help="Template name")
    parser.add_argument("--description", required=True, help="Template description")
    parser.add_argument("--components-file", required=True, help="JSON file with component definitions")
    parser.add_argument("--audience", default="system", help="Target audience")

    args = parser.parse_args()

    # Load components
    try:
        with open(args.components_file, 'r') as f:
            components = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading components: {e}")
        return 1

    # Create template
    generator = TemplateGenerator(args.template_dir)
    template = generator.create_template(
        category=args.category,
        name=args.name,
        description=args.description,
        components=components,
        audience=args.audience
    )

    print(f"Template generated: {template['id']}")
    print(json.dumps(template, indent=2))
    return 0

if __name__ == "__main__":
    exit(main())
EOF

chmod +x $SYNTHESIS_DIR/generate_template.py

# Create a sample component definition
cat > $SYNTHESIS_DIR/sample_components.json << 'EOF'
[
  {
    "name": "task_analysis",
    "description": "Analyze and understand task requirements",
    "execution_order": 1,
    "required": true
  },
  {
    "name": "subtask_generation",
    "description": "Generate logical subtasks for the main task",
    "execution_order": 2,
    "required": true
  },
  {
    "name": "dependency_resolution",
    "description": "Identify dependencies between subtasks",
    "execution_order": 3,
    "required": true
  },
  {
    "name": "execution_planning",
    "description": "Create an efficient execution plan",
    "execution_order": 4,
    "required": true
  },
  {
    "name": "resource_allocation",
    "description": "Allocate resources for execution",
    "execution_order": 5,
    "required": false
  }
]
EOF

echo "Template Synthesis Engine created successfully"