#!/usr/bin/env python3
"""
Application Architecture Mapper

This script analyzes your codebase to generate a comprehensive map of your application's
architecture, showing relationships between components, data flows, and dependencies.
"""

import os
import re
import ast
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app_mapper")


@dataclass
class Component:
    """Represents a component in the application architecture."""

    name: str
    file_path: str
    type: str  # class, function, module
    parent: Optional[str] = None
    docstring: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    methods: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    event_handlers: List[str] = field(default_factory=list)


@dataclass
class EventFlow:
    """Represents an event flow in the application."""

    name: str
    source: str
    targets: List[str] = field(default_factory=list)
    handlers: List[str] = field(default_factory=list)
    schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Template:
    """Represents a template in the application."""

    path: str
    used_by: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)


class ApplicationMapper:
    """Maps the application architecture by analyzing the codebase."""

    def __init__(self, root_path: str, exclude_patterns: List[str] = None):
        self.root_path = Path(root_path)
        self.exclude_patterns = exclude_patterns or []

        # Data structures for application mapping
        self.components: Dict[str, Component] = {}
        self.event_flows: List[EventFlow] = []
        self.templates: List[Template] = []
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.roles_and_responsibilities: Dict[str, List[str]] = defaultdict(list)

        # Patterns for detecting specific aspects
        self.event_patterns = [
            r"handle_(\w+)",
            r"on_(\w+)_event",
            r"when_(\w+)",
        ]
        self.template_patterns = [
            r"\.j2$",
            r"\.jinja$",
            r"\.jinja2$",
            r"\.tmpl$",
            r"\.template$",
        ]
        self.event_bus_patterns = [
            "EventBus",
            "EventEmitter",
            "PubSub",
            "MessageBroker",
        ]
        self.pulsar_patterns = [
            "pulsar.Client",
            "subscribe",
            "publish",
            "producer",
            "consumer",
        ]

    def find_files(self, extension: str = ".py") -> List[Path]:
        """Find all files with the given extension in the root path."""
        files = []
        for file_path in self.root_path.glob(f"**/*{extension}"):
            # Skip excluded paths
            if any(re.search(pattern, str(file_path)) for pattern in self.exclude_patterns):
                continue
            files.append(file_path)
        return files

    def analyze_file(self, file_path: Path) -> None:
        """Analyze a single file to extract components and relationships."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            module_name = self._get_module_name(file_path)

            # Add the module as a component
            self.components[module_name] = Component(
                name=module_name,
                file_path=str(file_path),
                type="module",
                docstring=self._extract_docstring(content),
            )

            # Parse the file with AST
            tree = ast.parse(content, filename=str(file_path))

            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                        self.dependency_graph[module_name].append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        self.dependency_graph[module_name].append(node.module)

            self.components[module_name].imports = imports

            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = f"{module_name}.{node.name}"

                    # Analyze class docstring
                    class_docstring = ast.get_docstring(node)

                    # Extract methods and detect event handlers
                    methods = []
                    event_handlers = []

                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)

                            # Check for event handler patterns
                            for pattern in self.event_patterns:
                                if re.match(pattern, item.name):
                                    event_name = re.match(pattern, item.name).group(1)
                                    event_handlers.append(event_name)

                    # Extract base classes
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                            self.dependency_graph[class_name].append(base.id)
                        elif isinstance(base, ast.Attribute):
                            base_name = self._get_attribute_name(base)
                            bases.append(base_name)
                            self.dependency_graph[class_name].append(base_name)

                    # Create component for the class
                    self.components[class_name] = Component(
                        name=class_name,
                        file_path=str(file_path),
                        type="class",
                        parent=module_name,
                        docstring=class_docstring,
                        methods=methods,
                        event_handlers=event_handlers,
                        attributes={"bases": bases},
                    )

                    # Also register the relationship with the parent module
                    self.dependency_graph[module_name].append(class_name)

                    # Check for interesting roles
                    self._detect_component_role(class_name, node, class_docstring)

                # Extract top-level functions
                elif isinstance(node, ast.FunctionDef) and node.parent_field is None:
                    func_name = f"{module_name}.{node.name}"
                    func_docstring = ast.get_docstring(node)

                    # Check for event handler patterns
                    is_event_handler = False
                    event_name = None
                    for pattern in self.event_patterns:
                        match = re.match(pattern, node.name)
                        if match:
                            is_event_handler = True
                            event_name = match.group(1)
                            break

                    # Create component for the function
                    self.components[func_name] = Component(
                        name=func_name,
                        file_path=str(file_path),
                        type="function",
                        parent=module_name,
                        docstring=func_docstring,
                        event_handlers=[event_name] if is_event_handler else [],
                    )

                    # Register relationship with parent module
                    self.dependency_graph[module_name].append(func_name)

            # Look for template references
            self._detect_templates(file_path, content, module_name)

            # Look for event flows
            self._detect_event_flows(file_path, content, module_name)

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")

    def _get_module_name(self, file_path: Path) -> str:
        """Get the module name from a file path."""
        rel_path = file_path.relative_to(self.root_path)
        module_name = str(rel_path).replace("/", ".").replace("\\", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        return module_name

    def _extract_docstring(self, content: str) -> Optional[str]:
        """Extract docstring from module content."""
        try:
            tree = ast.parse(content)
            return ast.get_docstring(tree)
        except:
            # Try with regex as a fallback
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                return docstring_match.group(1).strip()
            return None

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get the full attribute name."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        return f"<unknown>.{node.attr}"

    def _detect_component_role(
        self, component_name: str, node: ast.AST, docstring: Optional[str]
    ) -> None:
        """Detect the role of a component based on its content and docstring."""
        if not docstring:
            return

        # Check docstring for role indicators
        docstring_lower = docstring.lower()

        roles = []
        if any(pattern in docstring_lower for pattern in self.event_bus_patterns):
            roles.append("event_bus")

        if any(pattern in docstring_lower for pattern in self.pulsar_patterns):
            roles.append("message_queue")

        if "template" in docstring_lower and (
            "render" in docstring_lower or "jinja" in docstring_lower
        ):
            roles.append("template_engine")

        if "transform" in docstring_lower and "model" in docstring_lower:
            roles.append("model_transformer")

        if "emit" in docstring_lower and "code" in docstring_lower:
            roles.append("code_emitter")

        if "parse" in docstring_lower and "schema" in docstring_lower:
            roles.append("schema_parser")

        if "validate" in docstring_lower:
            roles.append("validator")

        if "observer" in docstring_lower or "monitor" in docstring_lower:
            roles.append("observer")

        if "agent" in docstring_lower:
            roles.append("agent")

        # Add detected roles
        for role in roles:
            self.roles_and_responsibilities[role].append(component_name)

    def _detect_templates(self, file_path: Path, content: str, module_name: str) -> None:
        """Detect templates and their usage."""
        # Look for template file patterns
        if any(re.search(pattern, str(file_path)) for pattern in self.template_patterns):
            template = Template(
                path=str(file_path), used_by=[], variables=self._extract_template_variables(content)
            )
            self.templates.append(template)

        # Look for template usage in code
        template_refs = re.findall(
            r'[\'"]([^\'\"]*\.(?:j2|jinja2|jinja|tmpl|template))[\'"]', content
        )
        for ref in template_refs:
            # Check if this template exists
            template_path = self.root_path / ref
            if not template_path.exists():
                # Try relative to file
                template_path = file_path.parent / ref

            if template_path.exists():
                # Find or create template
                template = next((t for t in self.templates if t.path == str(template_path)), None)
                if not template:
                    template = Template(
                        path=str(template_path),
                        used_by=[module_name],
                        variables=self._extract_template_variables(
                            template_path.read_text(encoding="utf-8", errors="ignore")
                        ),
                    )
                    self.templates.append(template)
                else:
                    template.used_by.append(module_name)

    def _extract_template_variables(self, content: str) -> List[str]:
        """Extract variables from a template."""
        variables = []

        # Jinja2 variables {{ var }}
        for match in re.finditer(r"{{\s*([a-zA-Z0-9_\.]+)\s*}}", content):
            var_name = match.group(1)
            if var_name not in variables:
                variables.append(var_name)

        # String template variables ${var}
        for match in re.finditer(r"\$\{([a-zA-Z0-9_\.]+)\}", content):
            var_name = match.group(1)
            if var_name not in variables:
                variables.append(var_name)

        return variables

    def _detect_event_flows(self, file_path: Path, content: str, module_name: str) -> None:
        """Detect event flows between components."""
        # Look for event handler definitions
        event_handlers = []
        for pattern in self.event_patterns:
            handlers = re.findall(pattern, content)
            event_handlers.extend(handlers)

        # Look for event publish/emit calls
        emit_patterns = [
            r'emit\([\'"](\w+)[\'"]',
            r'publish\([\'"](\w+)[\'"]',
            r'produce\([\'"](\w+)[\'"]',
            r'send\([\'"](\w+)[\'"]',
        ]

        emitted_events = []
        for pattern in emit_patterns:
            events = re.findall(pattern, content)
            emitted_events.extend(events)

        # Create event flows
        for event in set(emitted_events):
            # Check if we already have this event flow
            event_flow = next((ef for ef in self.event_flows if ef.name == event), None)
            if not event_flow:
                event_flow = EventFlow(name=event, source=module_name, targets=[], handlers=[])
                self.event_flows.append(event_flow)
            else:
                if event_flow.source != module_name:
                    # This is weird, we have multiple sources for the same event
                    # Let's track both
                    event_flow.source = f"{event_flow.source},{module_name}"

        # Link handlers to events
        for handler in set(event_handlers):
            # Find matching event flow
            event_flow = next((ef for ef in self.event_flows if ef.name == handler), None)
            if event_flow:
                event_flow.targets.append(module_name)
                event_flow.handlers.append(f"{module_name}.handle_{handler}")

    def analyze_codebase(self) -> None:
        """Analyze the entire codebase."""
        logger.info(f"Analyzing codebase at {self.root_path}")

        # Find Python files
        python_files = self.find_files(".py")
        logger.info(f"Found {len(python_files)} Python files to analyze")

        # Find template files
        template_files = []
        for pattern in self.template_patterns:
            for path in self.root_path.glob(f"**/*{pattern[:-1]}"):
                if not any(re.search(ex, str(path)) for ex in self.exclude_patterns):
                    template_files.append(path)

        logger.info(f"Found {len(template_files)} template files")

        # Analyze each file
        for file_path in python_files:
            self.analyze_file(file_path)

        # Process relationships
        self._process_relationships()

        # Log results
        logger.info(f"Analysis complete:")
        logger.info(f"  - Components: {len(self.components)}")
        logger.info(f"  - Event flows: {len(self.event_flows)}")
        logger.info(f"  - Templates: {len(self.templates)}")
        logger.info(f"  - Roles identified: {len(self.roles_and_responsibilities)}")

    def _process_relationships(self) -> None:
        """Process and complete relationships between components."""
        # Fill in dependents based on dependencies
        for source, targets in self.dependency_graph.items():
            for target in targets:
                if target in self.components:
                    self.components[target].dependents.append(source)

        # Ensure event handlers are properly linked
        for event_flow in self.event_flows:
            for handler in event_flow.handlers:
                if handler in self.components:
                    if event_flow.name not in self.components[handler].event_handlers:
                        self.components[handler].event_handlers.append(event_flow.name)

    def generate_application_map(self) -> Dict[str, Any]:
        """Generate a comprehensive application map."""
        app_map = {
            "components": {},
            "event_flows": [],
            "templates": [],
            "roles": {},
            "dependencies": {},
        }

        # Add components
        for name, component in self.components.items():
            app_map["components"][name] = {
                "name": component.name,
                "type": component.type,
                "file_path": component.file_path,
                "parent": component.parent,
                "dependencies": component.dependencies,
                "dependents": component.dependents,
                "methods": component.methods,
                "event_handlers": component.event_handlers,
                "description": component.docstring.split("\n")[0] if component.docstring else None,
            }

        # Add event flows
        for event_flow in self.event_flows:
            app_map["event_flows"].append(
                {
                    "name": event_flow.name,
                    "source": event_flow.source,
                    "targets": event_flow.targets,
                    "handlers": event_flow.handlers,
                }
            )

        # Add templates
        for template in self.templates:
            app_map["templates"].append(
                {
                    "path": template.path,
                    "used_by": template.used_by,
                    "variables": template.variables,
                }
            )

        # Add roles
        app_map["roles"] = self.roles_and_responsibilities

        # Add dependency graph
        app_map["dependencies"] = self.dependency_graph

        return app_map

    def save_application_map(self, output_path: str = "app_map.json") -> None:
        """Save the application map to a JSON file."""
        app_map = self.generate_application_map()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(app_map, f, indent=2)

        logger.info(f"Application map saved to {output_path}")

    def generate_markdown_report(self, output_path: str = "app_map.md") -> None:
        """Generate a Markdown report of the application map."""
        app_map = self.generate_application_map()

        #
