#!/bin/bash
# template_tester.sh - Create a testing framework for templates

# Create test framework directories
mkdir -p testing/template_tester
mkdir -p testing/fixtures

# Create the template tester
cat > testing/template_tester/tester.py << 'EOF'
#!/usr/bin/env python3
"""
Template Testing Framework
Validates templates against requirements and performance criteria
"""
import os
import json
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime

class TemplateTester:
    """Test and validate templates against requirements"""

    def __init__(self, template_path: str, test_specs_path: Optional[str] = None):
        self.template_path = template_path
        self.test_specs_path = test_specs_path
        self.template = self._load_template()
        self.test_specs = self._load_test_specs()
        self.results = {
            "template_id": self.template.get("id", "unknown"),
            "template_name": self.template.get("metadata", {}).get("name", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }

    def _load_template(self) -> Dict[str, Any]:
        """Load template from file"""
        try:
            with open(self.template_path, 'r') as f:
                if self.template_path.endswith('.json'):
                    return json.load(f)
                # Add YAML support here if needed
                return {}
        except Exception as e:
            print(f"Error loading template: {e}")
            return {}

    def _load_test_specs(self) -> List[Dict[str, Any]]:
        """Load test specifications"""
        if not self.test_specs_path or not os.path.exists(self.test_specs_path):
            # Default tests if no specs provided
            return [
                {
                    "name": "metadata_validation",
                    "type": "structure",
                    "required_fields": ["name", "description", "version", "category"]
                },
                {
                    "name": "components_check",
                    "type": "components",
                    "min_components": 1
                }
            ]

        try:
            with open(self.test_specs_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading test specs: {e}")
            return []

    def run_tests(self) -> Dict[str, Any]:
        """Run all tests against the template"""
        for test_spec in self.test_specs:
            test_type = test_spec.get("type", "unknown")
            test_name = test_spec.get("name", "unnamed_test")

            test_result = {
                "name": test_name,
                "type": test_type,
                "status": "unknown",
                "details": ""
            }

            try:
                if test_type == "structure":
                    self._run_structure_test(test_spec, test_result)
                elif test_type == "components":
                    self._run_components_test(test_spec, test_result)
                elif test_type == "relationships":
                    self._run_relationships_test(test_spec, test_result)
                else:
                    test_result["status"] = "skipped"
                    test_result["details"] = f"Unknown test type: {test_type}"
            except Exception as e:
                test_result["status"] = "error"
                test_result["details"] = f"Test execution error: {str(e)}"

            self.results["tests"].append(test_result)
            self.results["summary"]["total"] += 1

            if test_result["status"] == "passed":
                self.results["summary"]["passed"] += 1
            elif test_result["status"] == "failed":
                self.results["summary"]["failed"] += 1
            elif test_result["status"] == "warning":
                self.results["summary"]["warnings"] += 1

        return self.results

    def _run_structure_test(self, test_spec: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Test template structure"""
        required_fields = test_spec.get("required_fields", [])
        metadata = self.template.get("metadata", {})

        missing_fields = [field for field in required_fields if field not in metadata]

        if not missing_fields:
            result["status"] = "passed"
            result["details"] = "All required metadata fields present"
        else:
            result["status"] = "failed"
            result["details"] = f"Missing required metadata fields: {', '.join(missing_fields)}"

    def _run_components_test(self, test_spec: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Test template components"""
        min_components = test_spec.get("min_components", 1)
        components = self.template.get("components", [])

        if len(components) >= min_components:
            result["status"] = "passed"
            result["details"] = f"Template has {len(components)} components (min: {min_components})"
        else:
            result["status"] = "failed"
            result["details"] = f"Template has only {len(components)} components, minimum is {min_components}"

    def _run_relationships_test(self, test_spec: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Test template relationships"""
        min_relationships = test_spec.get("min_relationships", 0)
        required_relationships = test_spec.get("required_relationships", [])
        relationships = self.template.get("relationships", [])

        # Check minimum count
        if len(relationships) < min_relationships:
            result["status"] = "failed"
            result["details"] = f"Template has only {len(relationships)} relationships, minimum is {min_relationships}"
            return

        # Check for required relationships
        if required_relationships:
            found_required = []
            for required in required_relationships:
                req_type = required.get("type")
                if any(rel.get("relationship_type") == req_type for rel in relationships):
                    found_required.append(req_type)

            if len(found_required) == len(required_relationships):
                result["status"] = "passed"
                result["details"] = "All required relationship types found"
            else:
                missing = [r.get("type") for r in required_relationships
                          if r.get("type") not in found_required]
                result["status"] = "failed"
                result["details"] = f"Missing required relationship types: {', '.join(missing)}"
        else:
            result["status"] = "passed"
            result["details"] = f"Template has {len(relationships)} relationships"

    def save_results(self, output_path: str) -> None:
        """Save test results to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Test results saved to {output_path}")
        print(f"Summary: {self.results['summary']['passed']} passed, "
              f"{self.results['summary']['failed']} failed, "
              f"{self.results['summary']['warnings']} warnings")

def main():
    parser = argparse.ArgumentParser(description="Test templates against requirements")
    parser.add_argument("--template", required=True, help="Template file path")
    parser.add_argument("--test-specs", help="Test specifications file")
    parser.add_argument("--output", required=True, help="Output file for test results")

    args = parser.parse_args()

    tester = TemplateTester(args.template, args.test_specs)
    tester.run_tests()
    tester.save_results(args.output)

if __name__ == "__main__":
    main()
EOF

chmod +x testing/template_tester/tester.py

# Create a simple test specification
cat > testing/fixtures/default_test_specs.json << 'EOF'
[
  {
    "name": "metadata_validation",
    "type": "structure",
    "required_fields": ["name", "description", "version", "category", "audience"]
  },
  {
    "name": "components_check",
    "type": "components",
    "min_components": 2
  },
  {
    "name": "relationships_check",
    "type": "relationships",
    "min_relationships": 1,
    "required_relationships": [
      {"type": "depends-on"}
    ]
  }
]
EOF

echo "Template Testing Framework created successfully"