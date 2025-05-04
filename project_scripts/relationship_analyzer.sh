#!/bin/bash
# relationship_analyzer.sh

# Create the relationship analyzer directories
mkdir -p tools/relationship_analyzer

# Create the relationship analyzer
cat > tools/relationship_analyzer/analyzer.py << 'EOF'
#!/usr/bin/env python3
"""
Template Relationship Analyzer
Discovers relationships between templates based on content and functionality
"""
import os
import json
import argparse
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import numpy as np

class RelationshipAnalyzer:
    """Analyze and discover relationships between templates"""

    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.templates = self._load_templates()
        self.relationships = defaultdict(list)

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all templates from the directory"""
        templates = {}

        for root, _, files in os.walk(self.template_dir):
            for file in files:
                if file.endswith(('.json', '.yaml', '.yml')):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            if file.endswith('.json'):
                                template = json.load(f)
                            else:
                                # Would use PyYAML here
                                continue

                            template_id = template.get('id', os.path.splitext(file)[0])
                            templates[template_id] = template
                    except Exception as e:
                        print(f"Error loading template {file}: {e}")

        return templates

    def discover_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover relationships between templates"""
        # Find explicit relationships first
        self._find_explicit_relationships()

        # Find implicit relationships
        self._find_implicit_relationships()

        return dict(self.relationships)

    def _find_explicit_relationships(self) -> None:
        """Find explicitly defined relationships"""
        for template_id, template in self.templates.items():
            relationships = template.get('relationships', [])
            for rel in relationships:
                related_id = rel.get('related_id')
                rel_type = rel.get('relationship_type')

                if related_id and rel_type:
                    # Record the relationship
                    self.relationships[template_id].append({
                        'related_id': related_id,
                        'relationship_type': rel_type,
                        'source': 'explicit',
                        'confidence': 1.0
                    })

    def _find_implicit_relationships(self) -> None:
        """Find implicit relationships based on content similarity"""
        # This would use more sophisticated techniques like:
        # - Component similarity
        # - Functionality overlap
        # - Complementary capabilities

        # Simplified implementation for demonstration
        categories = defaultdict(list)

        # Group by category
        for template_id, template in self.templates.items():
            category = template.get('metadata', {}).get('category', '')
            if category:
                categories[category].append(template_id)

        # Templates in same category might be related
        for category, template_ids in categories.items():
            for i, template_id1 in enumerate(template_ids):
                for template_id2 in template_ids[i+1:]:
                    # Simple relationship: same category
                    similarity = self._calculate_similarity(
                        self.templates[template_id1],
                        self.templates[template_id2]
                    )

                    if similarity > 0.5:
                        rel_type = "complements" if similarity > 0.7 else "similar-to"

                        self.relationships[template_id1].append({
                            'related_id': template_id2,
                            'relationship_type': rel_type,
                            'source': 'implicit',
                            'confidence': similarity
                        })

                        self.relationships[template_id2].append({
                            'related_id': template_id1,
                            'relationship_type': rel_type,
                            'source': 'implicit',
                            'confidence': similarity
                        })

    def _calculate_similarity(self, template1: Dict[str, Any], template2: Dict[str, Any]) -> float:
        """Calculate similarity between two templates"""
        # This would use more sophisticated techniques in practice
        # Simplified scoring for demonstration
        score = 0.0

        # Same category bonus
        if (template1.get('metadata', {}).get('category') ==
            template2.get('metadata', {}).get('category')):
            score += 0.3

        # Same subcategory bonus
        if (template1.get('metadata', {}).get('subcategory') ==
            template2.get('metadata', {}).get('subcategory')):
            score += 0.2

        # Tag overlap
        tags1 = set(template1.get('metadata', {}).get('tags', []))
        tags2 = set(template2.get('metadata', {}).get('tags', []))
        if tags1 and tags2:
            overlap = len(tags1.intersection(tags2)) / max(len(tags1), len(tags2))
            score += 0.5 * overlap

        return min(score, 1.0)

    def save_relationships(self, output_file: str) -> None:
        """Save discovered relationships to file"""
        with open(output_file, 'w') as f:
            json.dump(dict(self.relationships), f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Analyze template relationships")
    parser.add_argument("--template-dir", required=True, help="Template directory")
    parser.add_argument("--output", required=True, help="Output file for relationships")

    args = parser.parse_args()

    analyzer = RelationshipAnalyzer(args.template_dir)
    relationships = analyzer.discover_relationships()
    analyzer.save_relationships(args.output)

    print(f"Discovered {sum(len(rels) for rels in relationships.values())} relationships")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
EOF

chmod +x tools/relationship_analyzer/analyzer.py

echo "Relationship Analyzer created successfully"