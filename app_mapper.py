#!/usr/bin/env python3
"""
Improved Application Architecture Mapper

This script analyzes your microservice codebase to generate a comprehensive map of your application's
architecture, showing relationships between components, data flows, and dependencies.
It groups results by container/service and supports multiple output formats.
"""

import re
import ast
import json
import yaml
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
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
    service: str  # The service/container this component belongs to
    parent: Optional[str] = None
    docstring: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    methods: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    event_handlers: List[str] = field(default_factory=list)
    event_publishers: List[str] = field(default_factory=list)


@dataclass
class EventFlow:
    """Represents an event flow in the application."""

    name: str
    source: str
    service: str  # The service/container this event belongs to
    targets: List[str] = field(default_factory=list)
    handlers: List[str] = field(default_factory=list)
    schema: Dict[str, Any] = field(default_factory=dict)
    event_type: str = "generic"  # generic, pulsar, kafka, etc.


@dataclass
class Template:
    """Represents a template in the application."""

    path: str
    service: str  # The service/container this template belongs to
    used_by: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)


@dataclass
class InfraFile:
    """Represents an infrastructure-related file."""

    path: str
    type: str  # Dockerfile, compose, k8s, env, etc.
    service: str  # The service/container this file belongs to
    references: List[str] = field(default_factory=list)


class ImprovedApplicationMapper:
    """Maps the application architecture by analyzing the codebase with improved service grouping."""

    def __init__(
        self,
        root_path: str,
        exclude_patterns: List[str] = None,
        output_format: str = "json",
    ):
        self.root_path = Path(root_path)
        self.exclude_patterns = exclude_patterns or []
        self.output_format = output_format

        # Data structures for application mapping
        self.components: Dict[str, Component] = {}
        self.event_flows: List[EventFlow] = []
        self.templates: List[Template] = []
        self.infra_files: List[InfraFile] = []
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.roles_and_responsibilities: Dict[str, List[str]] = defaultdict(list)

        # Service mapping
        self.services: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "components": [],
                "event_flows": [],
                "templates": [],
                "infra_files": [],
            }
        )

        # Patterns for detecting specific aspects
        self.event_patterns = [
            r"handle_(\w+)",
            r"on_(\w+)_event",
            r"when_(\w+)",
            r"subscribe_to_(\w+)",
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
            "pulsar.Client",
        ]
        self.pulsar_patterns = [
            "pulsar.Client",
            "subscribe",
            "publish",
            "producer",
            "consumer",
            ".send\\(",
        ]
        self.infra_file_patterns = {
            "dockerfile": [r"Dockerfile", r"\.dockerfile$", r"Dockerfile\.\w+$"],
            "requirements": [r"requirements.*\.txt$"],
            "compose": [r"docker-compose.*\.yml$", r"docker-compose.*\.yaml$"],
            "kubernetes": [r"deployment\.yaml$", r"service\.yaml$", r".*\.k8s\.ya?ml$"],
            "terraform": [r"\.tf$", r"\.tfvars$"],
            "helm": [r"Chart\.yaml$", r"values\.yaml$"],
            "env": [r"\.env", r"\.env\.\w+$"],
            "config": [r"config\.ya?ml$", r"settings\.ya?ml$"],
            "schema": [r"\.avsc$", r"\.proto$", r"\.schema\.json$"],
        }

    def get_service_name(self, file_path: Path) -> str:
        """Determine the service name based on the file path."""
        rel_path = file_path.relative_to(self.root_path)
        parts = rel_path.parts

        # If file is in a direct subdirectory, use that as the service name
        if len(parts) > 0:
            return parts[0]

        # Fallback to "common" if not in a specific service directory
        return "common"

    def find_files(self, extension: str = ".py") -> List[Tuple[Path, str]]:
        """Find all files with the given extension in the root path along with their service names."""
        files = []
        for file_path in self.root_path.glob(f"**/*{extension}"):
            # Skip excluded paths
            if any(
                re.search(pattern, str(file_path)) for pattern in self.exclude_patterns
            ):
                continue

            service_name = self.get_service_name(file_path)
            files.append((file_path, service_name))

        return files

    def find_infra_files(self) -> List[Tuple[Path, str, str]]:
        """Find all infrastructure-related files in the root path."""
        files = []
        for file_type, patterns in self.infra_file_patterns.items():
            for pattern in patterns:
                for file_path in self.root_path.glob("**/*"):
                    if re.search(pattern, str(file_path.name), re.IGNORECASE):
                        # Skip excluded paths
                        if any(
                            re.search(ex_pattern, str(file_path))
                            for ex_pattern in self.exclude_patterns
                        ):
                            continue

                        service_name = self.get_service_name(file_path)
                        files.append((file_path, service_name, file_type))

        return files

    def analyze_file(self, file_path: Path, service_name: str) -> None:
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
                service=service_name,
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
                    event_publishers = []

                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)

                            # Check for event handler patterns
                            for pattern in self.event_patterns:
                                if re.match(pattern, item.name):
                                    event_name = re.match(pattern, item.name).group(1)
                                    event_handlers.append(event_name)

                            # Check for Pulsar event publishing
                            method_source = ast.unparse(item)
                            if re.search(r"\.send\(", method_source) or re.search(
                                r"\.publish\(", method_source
                            ):
                                # Look for send/publish with event class
                                publisher_matches = re.findall(
                                    r"(?:send|publish)\(\s*(\w+?Event)", method_source
                                )
                                if publisher_matches:
                                    event_publishers.extend(publisher_matches)
                                else:
                                    # Look for string-based event names
                                    string_matches = re.findall(
                                        r'(?:send|publish)\(\s*[\'"](\w+)[\'"]',
                                        method_source,
                                    )
                                    event_publishers.extend(string_matches)

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

                    # Check for event-related classes
                    is_event_class = False
                    if "Event" in node.name or any(
                        "event" in base.lower() for base in bases
                    ):
                        is_event_class = True
                        # Register this as an event
                        self._register_event_class(node.name, module_name, service_name)

                    # Create component for the class
                    self.components[class_name] = Component(
                        name=class_name,
                        file_path=str(file_path),
                        type="class",
                        service=service_name,
                        parent=module_name,
                        docstring=class_docstring,
                        methods=methods,
                        event_handlers=event_handlers,
                        event_publishers=event_publishers,
                        attributes={"bases": bases, "is_event_class": is_event_class},
                    )

                    # Also register the relationship with the parent module
                    self.dependency_graph[module_name].append(class_name)

                    # Check for interesting roles
                    self._detect_component_role(class_name, node, class_docstring)

                # Extract top-level functions
                elif isinstance(node, ast.FunctionDef):
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

                    # Check for Pulsar event publishing in function
                    event_publishers = []
                    func_source = ast.unparse(node)
                    if re.search(r"\.send\(", func_source) or re.search(
                        r"\.publish\(", func_source
                    ):
                        # Look for send/publish with event class
                        publisher_matches = re.findall(
                            r"(?:send|publish)\(\s*(\w+?Event)", func_source
                        )
                        if publisher_matches:
                            event_publishers.extend(publisher_matches)
                        else:
                            # Look for string-based event names
                            string_matches = re.findall(
                                r'(?:send|publish)\(\s*[\'"](\w+)[\'"]', func_source
                            )
                            event_publishers.extend(string_matches)

                    # Create component for the function
                    self.components[func_name] = Component(
                        name=func_name,
                        file_path=str(file_path),
                        type="function",
                        service=service_name,
                        parent=module_name,
                        docstring=func_docstring,
                        event_handlers=[event_name] if is_event_handler else [],
                        event_publishers=event_publishers,
                    )

                    # Register relationship with parent module
                    self.dependency_graph[module_name].append(func_name)

            # Look for template references
            self._detect_templates(file_path, content, module_name, service_name)

            # Look for event flows
            self._detect_event_flows(file_path, content, module_name, service_name)

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")

    def _register_event_class(
        self, class_name: str, module_name: str, service_name: str
    ) -> None:
        """Register an event class as a potential event flow."""
        event_flow = EventFlow(
            name=class_name,
            source=module_name,
            service=service_name,
            targets=[],
            handlers=[],
            event_type="class_based",
        )
        self.event_flows.append(event_flow)

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

        # Convert the node to source code for pattern matching
        try:
            node_source = ast.unparse(node)
        except:
            node_source = ""

        source_lower = node_source.lower()

        # Check for event bus patterns in docstring or code
        if any(
            pattern in docstring_lower for pattern in self.event_bus_patterns
        ) or any(
            pattern.lower() in source_lower for pattern in self.event_bus_patterns
        ):
            roles.append("event_bus")

        # Check for Pulsar patterns in docstring or code
        if any(pattern in docstring_lower for pattern in self.pulsar_patterns) or any(
            re.search(pattern.lower(), source_lower) for pattern in self.pulsar_patterns
        ):
            roles.append("message_queue")
            roles.append("pulsar_client")

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

    def _detect_templates(
        self, file_path: Path, content: str, module_name: str, service_name: str
    ) -> None:
        """Detect templates and their usage."""
        # Look for template file patterns
        if any(
            re.search(pattern, str(file_path)) for pattern in self.template_patterns
        ):
            template = Template(
                path=str(file_path),
                service=service_name,
                used_by=[],
                variables=self._extract_template_variables(content),
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
                template = next(
                    (t for t in self.templates if t.path == str(template_path)), None
                )
                if not template:
                    template = Template(
                        path=str(template_path),
                        service=service_name,
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

    def _detect_event_flows(
        self, file_path: Path, content: str, module_name: str, service_name: str
    ) -> None:
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

        # Look for class-based event instantiation
        class_event_patterns = [
            r"emit\(\s*(\w+?Event)\(",
            r"publish\(\s*(\w+?Event)\(",
            r"produce\(\s*(\w+?Event)\(",
            r"send\(\s*(\w+?Event)\(",
        ]

        emitted_events = []
        for pattern in emit_patterns:
            events = re.findall(pattern, content)
            emitted_events.extend(events)

        # Add class-based events
        for pattern in class_event_patterns:
            events = re.findall(pattern, content)
            emitted_events.extend(events)

        # Create event flows
        for event in set(emitted_events):
            # Check if we already have this event flow
            event_flow = next((ef for ef in self.event_flows if ef.name == event), None)
            if not event_flow:
                # Determine if this is a Pulsar event
                event_type = "generic"
                if any(pattern in content for pattern in self.pulsar_patterns):
                    event_type = "pulsar"

                event_flow = EventFlow(
                    name=event,
                    source=module_name,
                    service=service_name,
                    targets=[],
                    handlers=[],
                    event_type=event_type,
                )
                self.event_flows.append(event_flow)
            else:
                if event_flow.source != module_name:
                    # This is weird, we have multiple sources for the same event
                    # Let's track both
                    event_flow.source = f"{event_flow.source},{module_name}"

        # Link handlers to events
        for handler in set(event_handlers):
            # Find matching event flow
            event_flow = next(
                (ef for ef in self.event_flows if ef.name == handler), None
            )
            if event_flow:
                event_flow.targets.append(module_name)
                event_flow.handlers.append(f"{module_name}.handle_{handler}")

        # Look for Pulsar-specific patterns
        pulsar_subscribe_pattern = r'subscribe\(\s*[\'"]([\w\.]+)[\'"]'
        subscriptions = re.findall(pulsar_subscribe_pattern, content)
        for topic in subscriptions:
            # Create an event flow for this Pulsar topic if it doesn't exist
            event_flow = next((ef for ef in self.event_flows if ef.name == topic), None)
            if not event_flow:
                event_flow = EventFlow(
                    name=topic,
                    source="unknown",  # We don't know the source yet
                    service=service_name,
                    targets=[module_name],
                    handlers=[],
                    event_type="pulsar",
                )
                self.event_flows.append(event_flow)
            else:
                if module_name not in event_flow.targets:
                    event_flow.targets.append(module_name)

    def analyze_infra_file(
        self, file_path: Path, service_name: str, file_type: str
    ) -> None:
        """Analyze an infrastructure file to extract relevant information."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            references = []

            # Look for container/service references
            if file_type in ["dockerfile", "compose", "kubernetes", "helm"]:
                # Look for FROM references in Dockerfiles
                if file_type == "dockerfile":
                    from_refs = re.findall(
                        r"FROM\s+([^\s:]+)(?::[^\s]+)?", content, re.IGNORECASE
                    )
                    references.extend(from_refs)

                # Look for image references in compose or k8s files
                if file_type in ["compose", "kubernetes", "helm"]:
                    # Try to parse as YAML
                    try:
                        data = yaml.safe_load(content)
                        if data and isinstance(data, dict):
                            if file_type == "compose" and "services" in data:
                                for service, config in data["services"].items():
                                    if "image" in config:
                                        references.append(config["image"])
                            elif file_type in ["kubernetes", "helm"]:
                                # Look for container images in k8s manifests
                                if "spec" in data and "containers" in data["spec"]:
                                    for container in data["spec"]["containers"]:
                                        if "image" in container:
                                            references.append(container["image"])
                    except Exception as e:
                        logger.debug(f"Failed to parse YAML in {file_path}: {e}")

            # Create InfraFile object
            infra_file = InfraFile(
                path=str(file_path),
                type=file_type,
                service=service_name,
                references=references,
            )
            self.infra_files.append(infra_file)

        except Exception as e:
            logger.error(f"Error analyzing infra file {file_path}: {e}")

    def analyze_codebase(self) -> None:
        """Analyze the entire codebase."""
        logger.info(f"Analyzing codebase at {self.root_path}")

        # Find Python files
        python_files = self.find_files(".py")
        logger.info(f"Found {len(python_files)} Python files to analyze")

        # Find infrastructure files
        infra_files = self.find_infra_files()
        logger.info(f"Found {len(infra_files)} infrastructure files")

        # Find template files
        template_files = []
        for pattern in self.template_patterns:
            pattern_str = pattern[:-1] if pattern.endswith("$") else pattern
            for file_tuple in self.find_files(pattern_str):
                if not any(
                    re.search(ex, str(file_tuple[0])) for ex in self.exclude_patterns
                ):
                    template_files.append(file_tuple)

        logger.info(f"Found {len(template_files)} template files")

        # Analyze each Python file
        for file_path, service_name in python_files:
            self.analyze_file(file_path, service_name)

        # Analyze each infrastructure file
        for file_path, service_name, file_type in infra_files:
            self.analyze_infra_file(file_path, service_name, file_type)

        # Process relationships
        self._process_relationships()

        # Organize by service
        self._organize_by_service()

        # Log results
        logger.info("Analysis complete:")
        logger.info(f"  - Components: {len(self.components)}")
        logger.info(f"  - Event flows: {len(self.event_flows)}")
        logger.info(f"  - Templates: {len(self.templates)}")
        logger.info(f"  - Infrastructure files: {len(self.infra_files)}")
        logger.info(f"  - Roles identified: {len(self.roles_and_responsibilities)}")
        logger.info(f"  - Services: {len(self.services)}")

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

    def _organize_by_service(self) -> None:
        """Organize components, event_flows, templates, and infra files by service."""
        # Group components by service
        for name, component in self.components.items():
            self.services[component.service]["components"].append(name)

        # Group event flows by service
        for event_flow in self.event_flows:
            self.services[event_flow.service]["event_flows"].append(event_flow.name)

        # Group templates by service
        for template in self.templates:
            self.services[template.service]["templates"].append(template.path)

        # Group infra files by service
        for infra_file in self.infra_files:
            self.services[infra_file.service]["infra_files"].append(infra_file.path)

    def generate_application_map(self) -> Dict[str, Any]:
        """Generate a comprehensive application map."""
        app_map = {
            "services": {},
            "components": {},
            "event_flows": [],
            "templates": [],
            "infra_files": [],
            "roles": {},
            "dependencies": {},
        }

        # Add services structure
        for service_name, service_data in self.services.items():
            app_map["services"][service_name] = {
                "components": service_data["components"],
                "event_flows": service_data["event_flows"],
                "templates": service_data["templates"],
                "infra_files": service_data["infra_files"],
            }

        # Add components
        for name, component in self.components.items():
            app_map["components"][name] = {
                "name": component.name,
                "type": component.type,
                "service": component.service,
                "file_path": component.file_path,
                "parent": component.parent,
                "dependencies": component.dependencies,
                "dependents": component.dependents,
                "methods": component.methods,
                "event_handlers": component.event_handlers,
                "event_publishers": component.event_publishers,
                "description": component.docstring.split("\n")[0]
                if component.docstring
                else None,
            }

        # Add event flows
        for event_flow in self.event_flows:
            app_map["event_flows"].append(
                {
                    "name": event_flow.name,
                    "source": event_flow.source,
                    "service": event_flow.service,
                    "targets": event_flow.targets,
                    "handlers": event_flow.handlers,
                    "event_type": event_flow.event_type,
                }
            )

        # Add templates
        for template in self.templates:
            app_map["templates"].append(
                {
                    "path": template.path,
                    "service": template.service,
                    "used_by": template.used_by,
                    "variables": template.variables,
                }
            )

        # Add infra files
        for infra_file in self.infra_files:
            app_map["infra_files"].append(
                {
                    "path": infra_file.path,
                    "type": infra_file.type,
                    "service": infra_file.service,
                    "references": infra_file.references,
                }
            )

        # Add roles
        app_map["roles"] = self.roles_and_responsibilities

        # Add dependency graph
        app_map["dependencies"] = self.dependency_graph

        return app_map

    def save_output(self, output_path: str = None) -> None:
        """Save the application map to the specified format."""
        app_map = self.generate_application_map()

        # Determine output path if not provided
        if output_path is None:
            output_path = f"app_map.{self.output_format}"

        # Save in the appropriate format
        if self.output_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(app_map, f, indent=2)
        elif self.output_format == "yaml":
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(app_map, f, default_flow_style=False)
        elif self.output_format == "markdown":
            self._save_markdown(app_map, output_path)
        elif self.output_format == "graphml":
            self._save_graphml(app_map, output_path)

        logger.info(f"Application map saved to {output_path}")

    def _save_markdown(self, app_map: Dict[str, Any], output_path: str) -> None:
        """Save the application map as a Markdown report."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Application Architecture Map\n\n")

            # Services overview
            f.write("## Services Overview\n\n")
            for service_name, service_data in app_map["services"].items():
                f.write(f"### {service_name}\n\n")
                f.write(f"- Components: {len(service_data['components'])}\n")
                f.write(f"- Event Flows: {len(service_data['event_flows'])}\n")
                f.write(f"- Templates: {len(service_data['templates'])}\n")
                f.write(
                    f"- Infrastructure Files: {len(service_data['infra_files'])}\n\n"
                )

            # Components
            f.write("## Components\n\n")

            # Group components by service
            service_components = defaultdict(list)
            for name, component in app_map["components"].items():
                service_components[component["service"]].append(component)

            for service_name, components in service_components.items():
                f.write(f"### {service_name} Components\n\n")

                for component in sorted(components, key=lambda c: c["name"]):
                    f.write(f"#### {component['name']} ({component['type']})\n\n")

                    if component.get("description"):
                        f.write(f"{component['description']}\n\n")

                    f.write(f"- File: `{component['file_path']}`\n")

                    if component.get("methods") and len(component["methods"]) > 0:
                        f.write("- Methods:\n")
                        for method in sorted(component["methods"]):
                            f.write(f"  - `{method}`\n")

                    if (
                        component.get("event_handlers")
                        and len(component["event_handlers"]) > 0
                    ):
                        f.write("- Event Handlers:\n")
                        for handler in sorted(component["event_handlers"]):
                            f.write(f"  - `{handler}`\n")

                    if (
                        component.get("event_publishers")
                        and len(component["event_publishers"]) > 0
                    ):
                        f.write("- Event Publishers:\n")
                        for publisher in sorted(component["event_publishers"]):
                            f.write(f"  - `{publisher}`\n")

                    f.write("\n")

            # Event Flows
            f.write("## Event Flows\n\n")

            # Group event flows by service
            service_events = defaultdict(list)
            for event in app_map["event_flows"]:
                service_events[event["service"]].append(event)

            for service_name, events in service_events.items():
                f.write(f"### {service_name} Events\n\n")

                for event in sorted(events, key=lambda e: e["name"]):
                    f.write(f"#### {event['name']} ({event['event_type']})\n\n")
                    f.write(f"- Source: {event['source']}\n")

                    if event.get("targets") and len(event["targets"]) > 0:
                        f.write("- Targets:\n")
                        for target in sorted(event["targets"]):
                            f.write(f"  - {target}\n")

                    if event.get("handlers") and len(event["handlers"]) > 0:
                        f.write("- Handlers:\n")
                        for handler in sorted(event["handlers"]):
                            f.write(f"  - `{handler}`\n")

                    f.write("\n")

            # Infrastructure Files
            f.write("## Infrastructure Files\n\n")

            # Group infra files by service
            service_infra = defaultdict(list)
            for infra_file in app_map["infra_files"]:
                service_infra[infra_file["service"]].append(infra_file)

            for service_name, infra_files in service_infra.items():
                f.write(f"### {service_name} Infrastructure\n\n")

                # Group by type
                type_files = defaultdict(list)
                for infra_file in infra_files:
                    type_files[infra_file["type"]].append(infra_file)

                for file_type, files in type_files.items():
                    f.write(f"#### {file_type.capitalize()} Files\n\n")

                    for file in sorted(files, key=lambda f: f["path"]):
                        f.write(f"- `{file['path']}`\n")

                        if file.get("references") and len(file["references"]) > 0:
                            f.write("  - References:\n")
                            for ref in sorted(file["references"]):
                                f.write(f"    - {ref}\n")

                    f.write("\n")

            # Roles and Responsibilities
            f.write("## Roles and Responsibilities\n\n")

            for role, components in sorted(app_map["roles"].items()):
                f.write(f"### {role}\n\n")

                for component in sorted(components):
                    f.write(f"- {component}\n")

                f.write("\n")

    def _save_graphml(self, app_map: Dict[str, Any], output_path: str) -> None:
        """Save the application map as a GraphML file for network visualization tools."""
        try:
            import networkx as nx

            # Create a directed graph
            G = nx.DiGraph()

            # Add components as nodes
            for name, component in app_map["components"].items():
                G.add_node(
                    name,
                    type=component["type"],
                    service=component["service"],
                    description=component.get("description", ""),
                )

            # Add dependencies as edges
            for source, targets in app_map["dependencies"].items():
                for target in targets:
                    if target in app_map["components"]:  # Only add if target exists
                        G.add_edge(source, target, relationship="dependency")

            # Add event flows as nodes and edges
            for event in app_map["event_flows"]:
                event_node = f"Event:{event['name']}"
                G.add_node(
                    event_node,
                    type="event",
                    service=event["service"],
                    event_type=event["event_type"],
                )

                # Add edge from source to event
                if event["source"] in app_map["components"]:
                    G.add_edge(event["source"], event_node, relationship="publishes")

                # Add edges from event to targets
                for target in event["targets"]:
                    if target in app_map["components"]:
                        G.add_edge(event_node, target, relationship="consumed_by")

            # Write to GraphML
            nx.write_graphml(G, output_path)

        except ImportError:
            logger.error(
                "networkx library is required for GraphML output. Please install with 'pip install networkx'"
            )
            # Fallback to JSON
            with open(
                output_path.replace(".graphml", ".json"), "w", encoding="utf-8"
            ) as f:
                json.dump(app_map, f, indent=2)
            logger.info(
                f"Falling back to JSON output: {output_path.replace('.graphml', '.json')}"
            )


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Application Architecture Mapper")
    parser.add_argument("root_path", help="Root directory of the application")
    parser.add_argument(
        "--exclude",
        "-e",
        help="Patterns to exclude (comma-separated)",
        default="venv,node_modules,__pycache__,\\.git",
    )
    parser.add_argument("--output", "-o", help="Output file path", default=None)
    parser.add_argument(
        "--format",
        "-f",
        help="Output format",
        choices=["json", "yaml", "markdown", "graphml"],
        default="json",
    )
    parser.add_argument(
        "--verbose", "-v", help="Enable verbose logging", action="store_true"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create exclude patterns
    exclude_patterns = [p.strip() for p in args.exclude.split(",") if p.strip()]

    # Create mapper
    mapper = ImprovedApplicationMapper(
        root_path=args.root_path,
        exclude_patterns=exclude_patterns,
        output_format=args.format,
    )

    # Analyze codebase
    mapper.analyze_codebase()

    # Save output
    mapper.save_output(args.output)


if __name__ == "__main__":
    main()
