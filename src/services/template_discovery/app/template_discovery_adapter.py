#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Discovery Adapter - Bridge between SpecRegistry and TemplateDiscoveryService

This module adapts the TemplateDiscoveryService to work with our SpecRegistry,
providing intelligent template discovery capabilities to find the most appropriate
template based on project requirements.
"""

import os
import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from models.project import Project, ProjectAnalysisResult
from models.spec_sheet import SpecSheetTemplate
from core.spec_registry import SpecRegistry
from template_discovery import TemplateDiscoveryService, TemplateRequest, TemplateResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateDiscoveryAdapter:
    """
    Adapter that integrates the TemplateDiscoveryService with our SpecRegistry.

    This adapter provides methods to find templates based on various criteria,
    leveraging the TemplateDiscoveryService's matching algorithm.
    """

    def __init__(self, spec_registry: SpecRegistry, template_root: str = "./spec-templates"):
        """
        Initialize the template discovery adapter.

        Args:
            spec_registry: Registry for managing templates
            template_root: Root directory for templates
        """
        self.spec_registry = spec_registry
        self.template_discovery_service = TemplateDiscoveryService(
            template_root=template_root,
            # Use local or in-process discovery by default (no Pulsar)
            pulsar_url=os.environ.get("PULSAR_URL", ""),
            request_topic=os.environ.get("TEMPLATE_REQUEST_TOPIC", ""),
            response_topic=os.environ.get("TEMPLATE_RESPONSE_TOPIC", ""),
            metrics_topic=os.environ.get("METRICS_TOPIC", "")
        )

    async def initialize(self):
        """Initialize the adapter and underlying services."""
        # Initialize the template discovery service
        # Only connect to Pulsar if URLs are provided
        if self._is_pulsar_configured():
            await self.template_discovery_service.initialize()

        # Make sure templates are scanned
        self.template_discovery_service.scan_templates()

        logger.info("Template Discovery Adapter initialized")

    def _is_pulsar_configured(self) -> bool:
        """Check if Pulsar is configured."""
        return bool(os.environ.get("PULSAR_URL", ""))

    async def close(self):
        """Close any connections and resources."""
        if self._is_pulsar_configured():
            await self.template_discovery_service.close()

    async def find_template_for_project(self, project: Project) -> Tuple[Optional[SpecSheetTemplate], float]:
        """
        Find the best matching template for a project.

        Args:
            project: Project to find a template for

        Returns:
            Tuple of (template, confidence)
        """
        # Extract criteria from project
        criteria = self._extract_criteria_from_project(project)

        # Create a template request
        request = TemplateRequest(
            domain=criteria.get("domain"),
            language=criteria.get("language"),
            framework=criteria.get("framework"),
            component=criteria.get("component"),
            pattern=criteria.get("pattern"),
            generator_type="project",
            formal_spec=criteria.get("formal_spec", {}),
            additional_criteria=criteria.get("additional_criteria", {})
        )

        # Find matching template
        response = self.template_discovery_service.find_template(request)

        if not response.template_id:
            logger.warning(f"No template found for project {project.id}")
            return None, 0.0

        # Load template from registry
        template_id = response.template_id.replace(".", "/").replace("/yaml", "")
        template = await self.spec_registry.get_template(template_id)

        return template, response.match_confidence

    async def find_templates_for_analysis(self, analysis: ProjectAnalysisResult) -> Dict[
        str, Tuple[Optional[SpecSheetTemplate], float]]:
        """
        Find templates based on project analysis.

        Args:
            analysis: Result of project requirement analysis

        Returns:
            Dictionary mapping requirement types to (template, confidence)
        """
        results = {}

        for req in analysis.spec_sheet_requirements:
            # Create criteria for this specific requirement
            criteria = {
                "spec_sheet_type": req.spec_sheet_type,
                "reason": req.reason,
                "related_requirements": req.related_requirements
            }

            # Map spec_sheet_type to discovery criteria
            parts = req.spec_sheet_type.split("/")
            request_params = {
                "generator_type": "spec_sheet",
                "formal_spec": {"spec_sheet_type": req.spec_sheet_type},
                "additional_criteria": criteria
            }

            # Add domain/language/framework/component based on path
            if len(parts) > 0:
                if parts[0] in ["frontend", "backend", "architecture", "data"]:
                    request_params["domain"] = parts[0]

                # Look for common frameworks, languages, services
                for part in parts:
                    if part in ["react", "angular", "vue", "tailwind", "bootstrap"]:
                        request_params["framework"] = part
                    elif part in ["python", "typescript", "javascript", "java", "go"]:
                        request_params["language"] = part
                    elif part in ["services", "pages", "api", "database", "auth"]:
                        request_params["component"] = part
                    elif part in ["microservices", "event-driven", "factory", "observer"]:
                        request_params["pattern"] = part

            # Create a request
            request = TemplateRequest(**request_params)

            # Find matching template
            response = self.template_discovery_service.find_template(request)

            if not response.template_id:
                logger.warning(f"No template found for requirement {req.spec_sheet_type}")
                results[req.spec_sheet_type] = (None, 0.0)
                continue

            # Load template from registry
            template_id = response.template_id.replace(".", "/").replace("/yaml", "")
            template = await self.spec_registry.get_template(template_id)

            results[req.spec_sheet_type] = (template, response.match_confidence)

        return results

    def _extract_criteria_from_project(self, project: Project) -> Dict[str, Any]:
        """
        Extract discovery criteria from a project.

        Args:
            project: Project to extract criteria from

        Returns:
            Dictionary of criteria
        """
        criteria = {
            "domain": None,
            "language": None,
            "framework": None,
            "component": None,
            "pattern": None,
            "formal_spec": {},
            "additional_criteria": {
                "project_type": project.project_type,
                "tags": []
            }
        }

        # Set domain based on project type
        if project.project_type:
            if project.project_type == "WEB_APP":
                criteria["domain"] = "frontend"
            elif project.project_type == "API_SERVICE":
                criteria["domain"] = "backend"

        # Extract from technology stack
        if project.technology_stack:
            # Get primary language
            if project.technology_stack.languages and len(project.technology_stack.languages) > 0:
                criteria["language"] = project.technology_stack.languages[0].lower()

            # Get primary framework
            if project.technology_stack.frameworks and len(project.technology_stack.frameworks) > 0:
                criteria["framework"] = project.technology_stack.frameworks[0].lower()

            # Add all as tags
            criteria["additional_criteria"]["tags"].extend(
                [lang.lower() for lang in project.technology_stack.languages]
            )
            criteria["additional_criteria"]["tags"].extend(
                [framework.lower() for framework in project.technology_stack.frameworks]
            )

        # Extract from requirements
        if project.requirements:
            # Look for common services in requirements
            components = ["auth", "database", "api", "ui", "frontend", "backend"]
            patterns = ["microservice", "event-driven", "serverless", "mvc"]

            for req in project.requirements:
                desc = req.description.lower()

                # Check for services
                for comp in components:
                    if comp in desc and not criteria["component"]:
                        criteria["component"] = comp

                # Check for patterns
                for pattern in patterns:
                    if pattern in desc and not criteria["pattern"]:
                        criteria["pattern"] = pattern

                # Add as tags
                keywords = desc.split()
                criteria["additional_criteria"]["tags"].extend(
                    [word for word in keywords if len(word) > 4]
                )

        # Additional formal spec info
        criteria["formal_spec"]["project_name"] = project.name
        criteria["formal_spec"]["project_description"] = project.description

        # Clean up tags (unique, lowercase)
        criteria["additional_criteria"]["tags"] = list(set(
            [tag.lower() for tag in criteria["additional_criteria"]["tags"]]
        ))

        return criteria

    async def suggest_templates(self, query: str, limit: int = 5) -> List[Tuple[SpecSheetTemplate, float]]:
        """
        Suggest templates based on a text query.

        Args:
            query: Text query to match templates
            limit: Maximum number of templates to return

        Returns:
            List of (template, confidence) tuples
        """
        # Extract keywords from query
        keywords = query.lower().split()

        # Create criteria based on keywords
        criteria = {
            "domain": None,
            "language": None,
            "framework": None,
            "component": None,
            "pattern": None,
            "generator_type": "suggestion",
            "formal_spec": {"query": query},
            "additional_criteria": {"tags": keywords}
        }

        # Check for common domains
        domains = ["frontend", "backend", "architecture", "data", "cloud", "security"]
        for domain in domains:
            if domain in keywords:
                criteria["domain"] = domain
                break

        # Check for common languages
        languages = ["python", "javascript", "typescript", "java", "go", "ruby"]
        for lang in languages:
            if lang in keywords:
                criteria["language"] = lang
                break

        # Check for common frameworks
        frameworks = ["react", "angular", "vue", "express", "django", "spring", "tailwind"]
        for framework in frameworks:
            if framework in keywords:
                criteria["framework"] = framework
                break

        # Check for common services
        components = ["api", "database", "auth", "ui", "component", "page"]
        for component in components:
            if component in keywords:
                criteria["component"] = component
                break

        # Check for common patterns
        patterns = ["microservice", "event-driven", "serverless", "mvc", "factory"]
        for pattern in patterns:
            if pattern in keywords:
                criteria["pattern"] = pattern
                break

        # Create a request
        request = TemplateRequest(**criteria)

        # Find all matching templates
        matches = []

        # Score all templates against the request
        for template_id, template_path in self.template_discovery_service.templates.items():
            metadata = self.template_discovery_service.template_metadata[template_id]
            score = _score_template_match(request, metadata)

            if score > 0:
                # Get template from registry
                template_id_clean = template_id.replace(".", "/").replace("/yaml", "")
                template = await self.spec_registry.get_template(template_id_clean)

                if template:
                    matches.append((template, score / 100.0))

        # Sort by score and limit
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches[:limit]