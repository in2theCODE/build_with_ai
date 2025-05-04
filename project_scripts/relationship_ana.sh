#!/bin/bash
# relationship_analyzer.sh - Create a relationship analyzer for templates

# Set up directories
TOOLS_DIR="./tools"
REL_ANALYZER_DIR="$TOOLS_DIR/relationship_analyzer"

mkdir -p $REL_ANALYZER_DIR

# Create the relationship analyzer
cat > $REL_ANALYZER_DIR/analyze_relationships.py << 'EOF'
#!/usr/bin/env python3
"""
Template Relationship Analyzer
Discovers and manages relationships between templates
"""
import os
import json
import argparse
from typing import Dict, List, Any, Set
from collections import defaultdict

class RelationshipAnalyzer:
    """Analyze and manage relationships between templates"""

    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all templates from the directory"""
        templates = {}

        for root, _, files in os.walk(self.template_dir):
            for file in files:
                if file.endswith('.json'):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            template = json.load(f)
                            template_id = template.get('id', os.path.splitext(file)[0])
                            templates[template_id] = {
                                'path': file_path,
                                'data': template
                            }
                    except Exception as e:
                        print(f"Error loading template {file}: {e}")

        print(f"Loaded {len(templates)} templates")
        return templates

    def discover_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover relationships between templates"""
        relationships = defaultdict(list)

        # Find complementary templates in same category
        categories = defaultdict(list)
        for template_id, template_info in self.templates.items():
            template = template_info['data']
            category = template.get('metadata', {}).get('category', '')
            if category:
                categories[category].append(template_id)

        # Connect templates in same category
        for category, template_ids in categories.items():
            for i, template_id1 in enumerate(template_ids):
                for template_id2 in template_ids[i+1:]:
                    # Check if templates have complementary components
                    if self._are_complementary(
                        self.templates[template_id1]['data'],
                        self.templates[template_id2]['data']
                    ):
                        # Add relationship both ways
                        relationships[template_id1].append({
                            'related_id': template_id2,
                            'relationship_type': 'complements',
                            'description': f"Complements {template_id2}"
                        })

                        relationships[template_id2].append({
                            'related_id': template_id1,
                            'relationship_type': 'complements',
                            'description': f"Complements {template_id1}"
                        })

        # Find dependency relationships
        for template_id, template_info in self.templates.items():
            template = template_info['data']
            category = template.get('metadata', {}).get('category', '')

            # For certain categories, templates often depend on others
            if category in ['adaptive-workflows', 'task-automation']:
                # Look for templates it might depend on
                for other_id, other_info in self.templates.items():
                    if template_id == other_id:
                        continue

                    other = other_info['data']
                    other_category = other.get('metadata', {}).get('category', '')

                    # Task automation often depends on decision-making
                    if category == 'task-automation' and other_category == 'decision-making':
                        relationships[template_id].append({
                            'related_id': other_id,
                            'relationship_type': 'depends-on',
                            'description': f"Depends on decision-making template {other_id}"
                        })

                    # Adaptive workflows often extend base workflows
                    if category == 'adaptive-workflows' and other_category == 'task-automation':
                        relationships[template_id].append({
                            'related_id': other_id,
                            'relationship_type': 'extends',
                            'description': f"Extends task automation template {other_id}"
                        })

        return dict(relationships)

    def _are_complementary(self, template1: Dict[str, Any], template2: Dict[str, Any]) -> bool:
        """Check if two templates are complementary"""
        # Templates are complementary if they have different components
        # that could work together

        # Get component names
        components1 = {comp.get('name') for comp in template1.get('components', [])}
        components2 = {comp.get('name') for comp in template2.get('components', [])}

        # If no overlap but in same category, likely complementary
        return len(components1.intersection(components2)) == 0 and len(components1) > 0 and len(components2) > 0

    def apply_relationships(self, relationships: Dict[str, List[Dict[str, Any]]]) -> int:
        """Apply discovered relationships to templates"""
        updated_count = 0

        for template_id, related_templates in relationships.items():
            if template_id not in self.templates:
                continue

            template_info = self.templates[template_id]
            template = template_info['data']
            path = template_info['path']

            # Get existing relationships
            existing_relationships = template.get('relationships', [])
            existing_related_ids = {rel.get('related_id') for rel in existing_relationships}

            # Add new relationships
            added = False
            for rel in related_templates:
                related_id = rel.get('related_id')
                if related_id and related_id not in existing_related_ids:
                    existing_relationships.append(rel)
                    existing_related_ids.add(related_id)
                    added = True

            if added:
                # Update template
                template['relationships'] = existing_relationships

                # Save updated template
                with open(path, 'w') as f:
                    json.dump(template, f, indent=2)

                updated_count += 1

        return updated_count

def main():
    parser = argparse.ArgumentParser(description="Analyze and manage template relationships")
    parser.add_argument("--template-dir", required=True, help="Template directory")
    parser.add_argument("--output", help="Output file for relationships")
    parser.add_argument("--apply", action="store_true", help="Apply discovered relationships to templates")

    args = parser.parse_args()

    analyzer = RelationshipAnalyzer(args.template_dir)
    relationships = analyzer.discover_relationships()

    # Print summary
    total_relationships = sum(len(rels) for rels in relationships.values())
    print(f"Discovered {total_relationships} potential relationships between {len(relationships)} templates")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(relationships, f, indent=2)
        print(f"Relationships saved to {args.output}")

    # Apply relationships if requested
    if args.apply:
        updated = analyzer.apply_relationships(relationships)
        print(f"Updated {updated} templates with new relationships")

if __name__ == "__main__":
    main()
EOF

chmod +x $REL_ANALYZER_DIR/analyze_relationships.py

echo "Template Relationship Analyzer created successfully"