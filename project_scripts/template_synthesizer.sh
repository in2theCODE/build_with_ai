-#!/bin/bash
# template_synthesizer.sh - Add this script to create templates dynamically

# Set up environment
source ./common_utils.sh
TEMPLATE_DIR="./templates"
SYNTHESIS_ENGINE="./src/synthesis_engine"

# Create the synthesis engine directory structure
mkdir -p $SYNTHESIS_ENGINE/models
mkdir -p $SYNTHESIS_ENGINE/adapters
mkdir -p $SYNTHESIS_ENGINE/generators

# Create the template synthesis model adapter
cat > $SYNTHESIS_ENGINE/adapters/llm_adapter.py << 'EOF'
"""
LLM Adapter for Template Synthesis
Connects to multiple LLM backends for template generation
"""
import os
import asyncio
import json
from typing import Dict, Any, List, Optional

class LLMAdapter:
    """Interface with various LLM engines"""

    def __init__(self, config_path: str = "config/llm_config.json"):
        self.config = self._load_config(config_path)
        self.available_models = self._initialize_models()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load LLM configuration"""
        if not os.path.exists(config_path):
            return {"default_model": "local", "models": {}}

        with open(config_path, 'r') as f:
            return json.load(f)

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize connection to available models"""
        # Implementation for connecting to different model backends
        return {name: self._create_model_instance(cfg)
                for name, cfg in self.config.get("models", {}).items()}

    def _create_model_instance(self, model_config: Dict[str, Any]) -> Any:
        """Create appropriate model instance based on type"""
        model_type = model_config.get("type", "unknown")
        # Model instantiation logic here
        return {"type": model_type, "config": model_config}

    async def generate_template(self,
                               template_spec: Dict[str, Any],
                               model_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate template based on specification"""
        model = self._get_model(model_name)
        # Template generation logic
        # This would connect to the actual LLM API
        return {"status": "success", "template": template_spec}

    def _get_model(self, model_name: Optional[str] = None) -> Any:
        """Get requested model or default"""
        if model_name and model_name in self.available_models:
            return self.available_models[model_name]
        return self.available_models.get(
            self.config.get("default_model", next(iter(self.available_models), None))
        )
EOF

# Create the template generator
cat > $SYNTHESIS_ENGINE/generators/template_generator.py << 'EOF'
"""
Template Generator
Creates specialized templates based on requirements
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..adapters.llm_adapter import LLMAdapter

class TemplateGenerator:
    """Generate specialized templates based on requirements"""

    def __init__(self, template_dir: str, llm_adapter: Optional[LLMAdapter] = None):
        self.template_dir = template_dir
        self.llm_adapter = llm_adapter or LLMAdapter()

    async def create_template(self,
                             category: str,
                             name: str,
                             description: str,
                             requirements: List[str],
                             audience: str = "system") -> Dict[str, Any]:
        """Create a new template based on requirements"""
        template_id = f"{category}-{name.replace(' ', '-')}-{uuid.uuid4().hex[:8]}"

        # Prepare template specification
        template_spec = {
            "metadata": {
                "name": name,
                "description": description,
                "version": "1.0.0",
                "category": category,
                "subcategory": "",
                "audience": audience,
                "tags": [category, "synthesized"],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            },
            "requirements": requirements,
            "id": template_id
        }

        # Generate template using LLM
        generated_template = await self.llm_adapter.generate_template(template_spec)

        # Save the template
        if "template" in generated_template:
            self._save_template(template_id, category, generated_template["template"])

        return generated_template.get("template", {})

    def _save_template(self, template_id: str, category: str, template: Dict[str, Any]) -> None:
        """Save template to appropriate directory"""
        category_dir = os.path.join(self.template_dir, "categories", category)
        os.makedirs(category_dir, exist_ok=True)

        template_path = os.path.join(category_dir, f"{template_id}.json")
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
EOF

# Create a simple CLI for template synthesis
cat > $SYNTHESIS_ENGINE/synthesize.py << 'EOF'
#!/usr/bin/env python3
"""
Template Synthesis CLI
Command-line interface for synthesizing new templates
"""
import argparse
import asyncio
import json
import os
import sys

from generators.template_generator import TemplateGenerator

async def main():
    parser = argparse.ArgumentParser(description="Synthesize new templates")
    parser.add_argument("--category", required=True, help="Template category")
    parser.add_argument("--name", required=True, help="Template name")
    parser.add_argument("--description", required=True, help="Template description")
    parser.add_argument("--requirements", required=True, help="Requirements file path")
    parser.add_argument("--audience", default="system", help="Target audience")
    parser.add_argument("--output-dir", default="./templates", help="Output directory")

    args = parser.parse_args()

    # Load requirements
    try:
        with open(args.requirements, 'r') as f:
            requirements = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading requirements: {e}")
        sys.exit(1)

    # Create template generator
    generator = TemplateGenerator(args.output_dir)

    # Generate template
    template = await generator.create_template(
        category=args.category,
        name=args.name,
        description=args.description,
        requirements=requirements,
        audience=args.audience
    )

    print(f"Template generated: {template.get('id', 'unknown')}")
    print(json.dumps(template, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x $SYNTHESIS_ENGINE/synthesize.py

echo "Template Synthesis Engine created successfully"