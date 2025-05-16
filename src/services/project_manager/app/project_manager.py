#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project Manager - Manages projects and their associated spec sheets

This module implements the ProjectManager class which is responsible for
creating and managing projects, analyzing requirements, and generating
the appropriate spec sheets based on project needs.
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid


from infra.storage.repository import StorageRepository
from models.project import Project
from models.project import ProjectAnalysisResult
from models.project import ProjectStatus
from models.project import ProjectType
from models.project import Requirement
from models.project import SpecSheetRequirement
from models.project import TechnologyStack
from models.spec_sheet import FieldValue
from models.spec_sheet import SectionValues
from models.spec_sheet import SpecSheet
from models.spec_sheet import SpecSheetTemplate


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectManager:
    """
    Manages projects and their associated spec sheets.

    The ProjectManager is responsible for creating and managing projects,
    analyzing project requirements, and generating the appropriate spec sheets
    based on project needs.
    """

    def __init__(self, spec_registry: SpecRegistry, storage_repository: StorageRepository):
        """
        Initialize the project manager.

        Args:
            spec_registry: Registry for managing templates
            storage_repository: Repository for storing and retrieving data
        """
        self.spec_registry = spec_registry
        self.storage_repository = storage_repository

    async def create_project(self, project_data: Dict[str, Any]) -> Project:
        """
        Create a new project.

        Args:
            project_data: Project data containing name, description, etc.

        Returns:
            Project: Created project
        """
        logger.info(f"Creating project: {project_data.get('name', 'Unnamed project')}")

        # Generate a unique ID if none provided
        if "id" not in project_data:
            project_data["id"] = str(uuid.uuid4())

        # Set default values
        if "status" not in project_data:
            project_data["status"] = ProjectStatus.INITIALIZING

        if "created_at" not in project_data:
            project_data["created_at"] = datetime.now().isoformat()

        if "updated_at" not in project_data:
            project_data["updated_at"] = project_data["created_at"]

        if "spec_sheet_ids" not in project_data:
            project_data["spec_sheet_ids"] = []

        if "code_generation_ids" not in project_data:
            project_data["code_generation_ids"] = []

        # Create project object
        project = Project(
            id=project_data["id"],
            name=project_data["name"],
            description=project_data.get("description", ""),
            project_type=project_data.get("project_type", ProjectType.WEB_APP),
            status=project_data["status"],
            technology_stack=TechnologyStack(
                **project_data.get("technology_stack", {"languages": [], "frameworks": []})
            ),
            requirements=[Requirement(**r) for r in project_data.get("requirements", [])],
            spec_sheet_ids=project_data["spec_sheet_ids"],
            code_generation_ids=project_data["code_generation_ids"],
            created_at=project_data["created_at"],
            updated_at=project_data["updated_at"],
        )

        # Store in repository
        await self.storage_repository.store_project(project.to_dict())

        return project

    async def get_project(self, project_id: str) -> Optional[Project]:
        """
        Get a project by ID.

        Args:
            project_id: ID of the project to get

        Returns:
            Optional[Project]: Retrieved project or None if not found
        """
        project_data = await self.storage_repository.get_project(project_id)

        if not project_data:
            return None

        return Project(
            id=project_data["id"],
            name=project_data["name"],
            description=project_data.get("description", ""),
            project_type=project_data.get("project_type", ProjectType.WEB_APP),
            status=project_data["status"],
            technology_stack=TechnologyStack(
                **project_data.get("technology_stack", {"languages": [], "frameworks": []})
            ),
            requirements=[Requirement(**r) for r in project_data.get("requirements", [])],
            spec_sheet_ids=project_data.get("spec_sheet_ids", []),
            code_generation_ids=project_data.get("code_generation_ids", []),
            created_at=project_data.get("created_at", datetime.now().isoformat()),
            updated_at=project_data.get("updated_at", datetime.now().isoformat()),
        )

    async def update_project(self, project: Project) -> bool:
        """
        Update a project.

        Args:
            project: Updated project

        Returns:
            bool: True if update was successful
        """
        # Update the update timestamp
        project.updated_at = datetime.now().isoformat()

        # Store in repository
        return await self.storage_repository.store_project(project.to_dict())

    async def delete_project(self, project_id: str) -> bool:
        """
        Delete a project and its associated spec sheets.

        Args:
            project_id: ID of the project to delete

        Returns:
            bool: True if deletion was successful
        """
        try:
            # Get project first
            project = await self.get_project(project_id)

            if not project:
                logger.error(f"Cannot delete non-existent project: {project_id}")
                return False

            # Delete associated spec sheets
            for spec_sheet_id in project.spec_sheet_ids:
                await self.storage_repository.delete_spec_sheet(spec_sheet_id)

            # Delete associated code generations
            for code_gen_id in project.code_generation_ids:
                await self.storage_repository.delete_code_generation(code_gen_id)

            # Delete the project itself
            return await self.storage_repository.delete_project(project_id)

        except Exception as e:
            logger.error(f"Failed to delete project: {str(e)}")
            return False

    async def list_projects(self) -> List[Project]:
        """
        List all projects.

        Returns:
            List[Project]: List of projects
        """
        projects = []

        try:
            project_data_list = await self.storage_repository.list_all_projects()

            for project_data in project_data_list:
                project = Project(
                    id=project_data["id"],
                    name=project_data["name"],
                    description=project_data.get("description", ""),
                    project_type=project_data.get("project_type", ProjectType.WEB_APP),
                    status=project_data["status"],
                    technology_stack=TechnologyStack(
                        **project_data.get("technology_stack", {"languages": [], "frameworks": []})
                    ),
                    requirements=[Requirement(**r) for r in project_data.get("requirements", [])],
                    spec_sheet_ids=project_data.get("spec_sheet_ids", []),
                    code_generation_ids=project_data.get("code_generation_ids", []),
                    created_at=project_data.get("created_at", datetime.now().isoformat()),
                    updated_at=project_data.get("updated_at", datetime.now().isoformat()),
                )

                projects.append(project)

            return projects

        except Exception as e:
            logger.error(f"Failed to list projects: {str(e)}")
            return []

    async def analyze_project_requirements(self, project_id: str) -> ProjectAnalysisResult:
        """
        Analyze project requirements and determine needed spec sheets.

        Args:
            project_id: ID of the project to analyze

        Returns:
            ProjectAnalysisResult: Analysis result with recommended spec sheets
        """
        project = await self.get_project(project_id)

        if not project:
            raise ValueError(f"Project not found: {project_id}")

        logger.info(f"Analyzing requirements for project: {project.name}")

        # Analyze requirements to determine needed spec sheets
        spec_sheet_requirements = []

        # Architecture requirements
        if project.project_type == ProjectType.WEB_APP:
            if self._has_api_requirements(project):
                if self._has_microservice_requirements(project):
                    # Microservice architecture
                    spec_sheet_requirements.append(
                        SpecSheetRequirement(
                            spec_sheet_type="architecture/microservices",
                            count=1,
                            reason="Project requires a microservice architecture",
                            related_requirements=self._get_requirement_ids_by_keyword(
                                project, ["microservice", "distributed"]
                            ),
                        )
                    )
                else:
                    # Monolithic architecture
                    spec_sheet_requirements.append(
                        SpecSheetRequirement(
                            spec_sheet_type="architecture/monolithic",
                            count=1,
                            reason="Project requires a monolithic architecture",
                            related_requirements=self._get_requirement_ids_by_keyword(
                                project, ["monolithic", "single"]
                            ),
                        )
                    )

            # Check for event-driven requirements
            if self._has_event_driven_requirements(project):
                spec_sheet_requirements.append(
                    SpecSheetRequirement(
                        spec_sheet_type="architecture/event-driven",
                        count=1,
                        reason="Project requires event-driven architecture",
                        related_requirements=self._get_requirement_ids_by_keyword(
                            project, ["event", "message", "queue", "pubsub"]
                        ),
                    )
                )

                # Estimate the number of events
                event_count = self._estimate_event_count(project)
                if event_count > 0:
                    spec_sheet_requirements.append(
                        SpecSheetRequirement(
                            spec_sheet_type="architecture/event-driven/event_template",
                            count=event_count,
                            reason=f"Project requires {event_count} event definitions",
                            related_requirements=self._get_requirement_ids_by_keyword(project, ["event", "message"]),
                        )
                    )

        # Backend requirements
        if self._has_api_requirements(project):
            # REST API
            api_endpoint_count = self._estimate_api_endpoint_count(project)

            spec_sheet_requirements.append(
                SpecSheetRequirement(
                    spec_sheet_type="backend/api/rest",
                    count=1,
                    reason="Project requires a REST API",
                    related_requirements=self._get_requirement_ids_by_keyword(project, ["api", "rest", "endpoint"]),
                )
            )

            # Database
            if self._has_database_requirements(project):
                db_type = self._determine_database_type(project)
                model_count = self._estimate_model_count(project)

                spec_sheet_requirements.append(
                    SpecSheetRequirement(
                        spec_sheet_type=f"backend/database/{db_type}",
                        count=1,
                        reason=f"Project requires a {db_type} database",
                        related_requirements=self._get_requirement_ids_by_keyword(
                            project, ["database", "data", "storage", db_type]
                        ),
                    )
                )

                # Check if a specific database provider is mentioned
                db_provider = self._determine_database_provider(project)
                if db_provider:
                    spec_sheet_requirements.append(
                        SpecSheetRequirement(
                            spec_sheet_type=f"backend/database/providers/{db_provider}",
                            count=1,
                            reason=f"Project requires {db_provider} as database provider",
                            related_requirements=self._get_requirement_ids_by_keyword(project, [db_provider]),
                        )
                    )

            # Authentication
            if self._has_auth_requirements(project):
                auth_type = self._determine_auth_type(project)

                spec_sheet_requirements.append(
                    SpecSheetRequirement(
                        spec_sheet_type=f"backend/auth/{auth_type}",
                        count=1,
                        reason=f"Project requires {auth_type} authentication",
                        related_requirements=self._get_requirement_ids_by_keyword(
                            project, ["auth", "authentication", "login", auth_type]
                        ),
                    )
                )

                # Check if a specific auth provider is mentioned
                auth_provider = self._determine_auth_provider(project)
                if auth_provider:
                    spec_sheet_requirements.append(
                        SpecSheetRequirement(
                            spec_sheet_type=f"backend/auth/providers/{auth_provider}",
                            count=1,
                            reason=f"Project requires {auth_provider} as auth provider",
                            related_requirements=self._get_requirement_ids_by_keyword(project, [auth_provider]),
                        )
                    )

        # Frontend requirements
        if project.project_type in [ProjectType.WEB_APP, ProjectType.MOBILE_APP]:
            frontend_type = "web" if project.project_type == ProjectType.WEB_APP else "mobile"

            # Pages/screens
            page_count = self._estimate_page_count(project)

            if page_count > 0:
                spec_sheet_requirements.append(
                    SpecSheetRequirement(
                        spec_sheet_type=f"frontend/{frontend_type}/pages",
                        count=page_count,
                        reason=f"Project requires {page_count} {frontend_type} pages/screens",
                        related_requirements=self._get_requirement_ids_by_keyword(
                            project, ["page", "screen", "view", "ui"]
                        ),
                    )
                )

            # Components
            component_count = self._estimate_component_count(project)

            if component_count > 0:
                spec_sheet_requirements.append(
                    SpecSheetRequirement(
                        spec_sheet_type=f"frontend/{frontend_type}/services",
                        count=component_count,
                        reason=f"Project requires {component_count} reusable services",
                        related_requirements=self._get_requirement_ids_by_keyword(
                            project, ["component", "widget", "reusable"]
                        ),
                    )
                )

            # Styling
            styling_framework = self._determine_styling_framework(project)
            if styling_framework:
                spec_sheet_requirements.append(
                    SpecSheetRequirement(
                        spec_sheet_type=f"frontend/styling/{styling_framework}",
                        count=1,
                        reason=f"Project requires {styling_framework} for styling",
                        related_requirements=self._get_requirement_ids_by_keyword(
                            project, [styling_framework, "styling", "css"]
                        ),
                    )
                )

        # Cloud/DevOps requirements
        cloud_provider = self._determine_cloud_provider(project)
        if cloud_provider:
            spec_sheet_requirements.append(
                SpecSheetRequirement(
                    spec_sheet_type=f"cloud/{cloud_provider}",
                    count=1,
                    reason=f"Project requires {cloud_provider} for hosting",
                    related_requirements=self._get_requirement_ids_by_keyword(
                        project, [cloud_provider, "cloud", "hosting"]
                    ),
                )
            )

        # Recommend a technology stack
        recommended_stack = self._recommend_technology_stack(project, spec_sheet_requirements)

        # Create the analysis result
        result = ProjectAnalysisResult(
            project_id=project_id,
            spec_sheet_requirements=spec_sheet_requirements,
            recommended_technology_stack=recommended_stack,
            notes=f"Analysis based on {len(project.requirements)} requirements",
        )

        return result

    def _has_api_requirements(self, project: Project) -> bool:
        """Check if the project has API requirements."""
        keywords = ["api", "endpoint", "rest", "http", "service"]
        return self._has_requirements_with_keywords(project, keywords)

    def _has_database_requirements(self, project: Project) -> bool:
        """Check if the project has database requirements."""
        keywords = ["database", "db", "data", "storage", "sql", "nosql", "persist"]
        return self._has_requirements_with_keywords(project, keywords)

    def _has_auth_requirements(self, project: Project) -> bool:
        """Check if the project has authentication requirements."""
        keywords = ["auth", "authentication", "login", "user", "access", "permission"]
        return self._has_requirements_with_keywords(project, keywords)

    def _has_microservice_requirements(self, project: Project) -> bool:
        """Check if the project has microservice requirements."""
        keywords = ["microservice", "micro-service", "distributed", "service-oriented"]
        return self._has_requirements_with_keywords(project, keywords)

    def _has_event_driven_requirements(self, project: Project) -> bool:
        """Check if the project has event-driven requirements."""
        keywords = [
            "event",
            "message",
            "queue",
            "pubsub",
            "pub-sub",
            "event-driven",
            "kafka",
            "rabbitmq",
            "pulsar",
        ]
        return self._has_requirements_with_keywords(project, keywords)

    def _has_requirements_with_keywords(self, project: Project, keywords: List[str]) -> bool:
        """Check if project has requirements matching any of the keywords."""
        for req in project.requirements:
            desc = req.description.lower()
            for keyword in keywords:
                if keyword.lower() in desc:
                    return True
        return False

    def _get_requirement_ids_by_keyword(self, project: Project, keywords: List[str]) -> List[str]:
        """Get IDs of requirements matching any of the keywords."""
        matching_ids = []

        for req in project.requirements:
            desc = req.description.lower()
            for keyword in keywords:
                if keyword.lower() in desc:
                    matching_ids.append(req.id)
                    break

        return matching_ids

    def _estimate_api_endpoint_count(self, project: Project) -> int:
        """Estimate the number of API endpoints needed."""
        # This is a simplistic estimate - in a real implementation,
        # we'd use more sophisticated NLP or ML techniques

        # Count requirements related to API endpoints
        endpoint_keywords = ["endpoint", "api", "route", "controller", "service"]
        endpoint_count = 0

        for req in project.requirements:
            desc = req.description.lower()
            if any(keyword.lower() in desc for keyword in endpoint_keywords):
                # Increment by 1 for each matching requirement
                endpoint_count += 1

        # Ensure at least a minimum number
        return max(endpoint_count, 5)

    def _estimate_model_count(self, project: Project) -> int:
        """Estimate the number of database app needed."""
        # Count requirements that mention entities or app
        model_keywords = ["model", "entity", "table", "schema", "data"]
        model_count = 0

        for req in project.requirements:
            desc = req.description.lower()
            if any(keyword.lower() in desc for keyword in model_keywords):
                model_count += 1

        # Ensure at least a minimum number
        return max(model_count, 3)

    def _estimate_page_count(self, project: Project) -> int:
        """Estimate the number of UI pages needed."""
        # Count requirements related to pages or screens
        page_keywords = ["page", "screen", "view", "route", "ui"]
        page_count = 0

        for req in project.requirements:
            desc = req.description.lower()
            if any(keyword.lower() in desc for keyword in page_keywords):
                page_count += 1

        # Ensure at least a minimum number for common pages
        return max(page_count, 4)  # At least homepage, login, profile, etc.

    def _estimate_component_count(self, project: Project) -> int:
        """Estimate the number of UI services needed."""
        # Roughly estimate 2-3 services per page
        page_count = self._estimate_page_count(project)
        return page_count * 3

    def _estimate_event_count(self, project: Project) -> int:
        """Estimate the number of events needed."""
        # Count requirements related to events
        event_keywords = ["event", "message", "notification", "alert", "trigger"]
        event_count = 0

        for req in project.requirements:
            desc = req.description.lower()
            if any(keyword.lower() in desc for keyword in event_keywords):
                event_count += 1

        return event_count

    def _determine_database_type(self, project: Project) -> str:
        """Determine the type of database needed."""
        # Check for specific database types in requirements
        nosql_keywords = ["nosql", "document", "mongodb", "firestore", "dynamodb"]
        graph_keywords = ["graph", "neo4j", "relationship"]
        vector_keywords = [
            "vector",
            "embedding",
            "similarity",
            "semantic",
            "pinecone",
            "milvus",
        ]

        for req in project.requirements:
            desc = req.description.lower()

            if any(keyword.lower() in desc for keyword in vector_keywords):
                return "vector"

            if any(keyword.lower() in desc for keyword in graph_keywords):
                return "graph"

            if any(keyword.lower() in desc for keyword in nosql_keywords):
                return "document"

        # Default to relational
        return "relational"

    def _determine_database_provider(self, project: Project) -> Optional[str]:
        """Determine the database provider if specified."""
        # Known database providers
        providers = {
            "postgresql": ["postgresql", "postgres"],
            "mysql": ["mysql", "mariadb"],
            "mongodb": ["mongodb", "mongo"],
            "supabase": ["supabase"],
            "firebase": ["firestore", "firebase"],
            "dynamodb": ["dynamodb", "dynamo"],
            "neo4j": ["neo4j"],
            "neon": ["neon"],
            "cockroachdb": ["cockroachdb", "cockroach"],
            "pinecone": ["pinecone"],
            "milvus": ["milvus"],
            "qdrant": ["qdrant"],
            "pgvector": ["pgvector"],
        }

        for req in project.requirements:
            desc = req.description.lower()

            for provider, keywords in providers.items():
                if any(keyword.lower() in desc for keyword in keywords):
                    return provider

        return None

    def _determine_auth_type(self, project: Project) -> str:
        """Determine the authentication type needed."""
        # Check for specific auth types in requirements
        oauth_keywords = ["oauth", "social login", "google auth", "facebook auth"]
        jwt_keywords = ["jwt", "token", "bearer"]

        for req in project.requirements:
            desc = req.description.lower()

            if any(keyword.lower() in desc for keyword in oauth_keywords):
                return "oauth"

            if any(keyword.lower() in desc for keyword in jwt_keywords):
                return "jwt"

        # Default to session-based
        return "session"

    def _determine_auth_provider(self, project: Project) -> Optional[str]:
        """Determine the auth provider if specified."""
        # Known auth providers
        providers = {
            "auth0": ["auth0"],
            "firebase-auth": ["firebase auth", "firebase authentication"],
            "clerk": ["clerk"],
            "cognito": ["cognito", "aws auth"],
            "supabase-auth": ["supabase auth"],
            "okta": ["okta"],
        }

        for req in project.requirements:
            desc = req.description.lower()

            for provider, keywords in providers.items():
                if any(keyword.lower() in desc for keyword in keywords):
                    return provider

        return None

    def _determine_styling_framework(self, project: Project) -> Optional[str]:
        """Determine the styling framework if specified."""
        # Known styling frameworks
        frameworks = {
            "tailwind": ["tailwind", "tailwindcss", "utility-first"],
            "bootstrap": ["bootstrap"],
            "material-ui": ["material-ui", "material design", "mui"],
            "styled-services": ["styled-services", "styled services"],
            "sass": ["sass", "scss"],
            "chakra-ui": ["chakra", "chakra-ui"],
        }

        for req in project.requirements:
            desc = req.description.lower()

            for framework, keywords in frameworks.items():
                if any(keyword.lower() in desc for keyword in keywords):
                    return framework

        return None

    def _determine_cloud_provider(self, project: Project) -> Optional[str]:
        """Determine the cloud provider if specified."""
        # Known cloud providers
        providers = {
            "aws": ["aws", "amazon", "amazon web services"],
            "azure": ["azure", "microsoft cloud"],
            "gcp": ["gcp", "google cloud", "google cloud platform"],
            "digital-ocean": ["digital ocean", "digitalocean"],
            "heroku": ["heroku"],
            "vercel": ["vercel", "zeit"],
            "netlify": ["netlify"],
        }

        for req in project.requirements:
            desc = req.description.lower()

            for provider, keywords in providers.items():
                if any(keyword.lower() in desc for keyword in keywords):
                    return provider

        return None

    def _recommend_technology_stack(
        self, project: Project, spec_sheet_requirements: List[SpecSheetRequirement]
    ) -> TechnologyStack:
        """Recommend a technology stack based on requirements."""
        # Extract spec sheet types
        spec_types = [req.spec_sheet_type for req in spec_sheet_requirements]

        # Initialize stack with empty lists
        stack = TechnologyStack(
            languages=[],
            frameworks=[],
            databases=[],
            frontend=[],
            backend=[],
            infrastructure=[],
        )

        # Determine languages based on requirements
        if self._has_api_requirements(project):
            if any("backend/servers/nodejs" in s for s in spec_types):
                stack.languages.append("JavaScript")
                stack.languages.append("TypeScript")
                stack.backend.append("Node.js")
                stack.frameworks.append("Express.js")
            elif any("backend/servers/python" in s for s in spec_types):
                stack.languages.append("Python")
                stack.backend.append("Python")
                stack.frameworks.append("FastAPI")
            elif any("backend/servers/java" in s for s in spec_types):
                stack.languages.append("Java")
                stack.backend.append("Java")
                stack.frameworks.append("Spring Boot")
            elif any("backend/servers/go" in s for s in spec_types):
                stack.languages.append("Go")
                stack.backend.append("Go")
                stack.frameworks.append("Gin")
            elif any("backend/servers/ruby" in s for s in spec_types):
                stack.languages.append("Ruby")
                stack.backend.append("Ruby")
                stack.frameworks.append("Ruby on Rails")
            else:
                # Default to Node.js if not specified
                stack.languages.append("JavaScript")
                stack.languages.append("TypeScript")
                stack.backend.append("Node.js")
                stack.frameworks.append("Express.js")

        # Determine database
        if self._has_database_requirements(project):
            db_type = self._determine_database_type(project)
            db_provider = self._determine_database_provider(project)

            if db_type == "relational":
                if db_provider == "postgresql":
                    stack.databases.append("PostgreSQL")
                elif db_provider == "mysql":
                    stack.databases.append("MySQL")
                else:
                    stack.databases.append("PostgreSQL")  # Default
            elif db_type == "document":
                if db_provider == "mongodb":
                    stack.databases.append("MongoDB")
                elif db_provider == "firebase":
                    stack.databases.append("Firestore")
                else:
                    stack.databases.append("MongoDB")  # Default
            elif db_type == "vector":
                if db_provider:
                    stack.databases.append(db_provider.capitalize())
                else:
                    stack.databases.append("Pinecone")  # Default

        # Determine frontend stack
        if project.project_type == ProjectType.WEB_APP:
            stack.frontend.append("React")
            stack.languages.append("JavaScript")
            stack.languages.append("TypeScript")

            # Add styling
            styling = self._determine_styling_framework(project)
            if styling:
                if styling == "tailwind":
                    stack.frontend.append("Tailwind CSS")
                elif styling == "bootstrap":
                    stack.frontend.append("Bootstrap")
                elif styling == "material-ui":
                    stack.frontend.append("Material UI")
                elif styling == "chakra-ui":
                    stack.frontend.append("Chakra UI")
            else:
                stack.frontend.append("Tailwind CSS")  # Default
        elif project.project_type == ProjectType.MOBILE_APP:
            stack.frontend.append("React Native")
            stack.languages.append("JavaScript")
            stack.languages.append("TypeScript")

        # Determine infrastructure
        cloud_provider = self._determine_cloud_provider(project)
        if cloud_provider:
            if cloud_provider == "aws":
                stack.infrastructure.append("AWS")
            elif cloud_provider == "azure":
                stack.infrastructure.append("Azure")
            elif cloud_provider == "gcp":
                stack.infrastructure.append("Google Cloud")
            elif cloud_provider == "digital-ocean":
                stack.infrastructure.append("Digital Ocean")
            elif cloud_provider == "vercel":
                stack.infrastructure.append("Vercel")
            elif cloud_provider == "netlify":
                stack.infrastructure.append("Netlify")
            else:
                stack.infrastructure.append(cloud_provider.capitalize())
        else:
            # Default to cloud provider
            stack.infrastructure.append("AWS")

        # Add Docker/Kubernetes if microservices architecture
        if self._has_microservice_requirements(project):
            stack.infrastructure.append("Docker")
            stack.infrastructure.append("Kubernetes")

        # Add event bus if event-driven architecture
        if self._has_event_driven_requirements(project):
            stack.infrastructure.append("Apache Pulsar")

        # Remove duplicates and ensure lists have unique elements
        stack.languages = list(set(stack.languages))
        stack.frameworks = list(set(stack.frameworks))
        stack.databases = list(set(stack.databases))
        stack.frontend = list(set(stack.frontend))
        stack.backend = list(set(stack.backend))
        stack.infrastructure = list(set(stack.infrastructure))

        return stack

    async def generate_spec_sheets(self, project_id: str) -> Tuple[List[SpecSheet], List[str]]:
        """
        Generate spec sheets for a project based on requirements analysis.

        Args:
            project_id: ID of the project to generate spec sheets for

        Returns:
            Tuple[List[SpecSheet], List[str]]: Generated spec sheets and error messages
        """
        project = await self.get_project(project_id)

        if not project:
            raise ValueError(f"Project not found: {project_id}")

        logger.info(f"Generating spec sheets for project: {project.name}")

        # Analyze project requirements
        analysis = await self.analyze_project_requirements(project_id)

        # Create spec sheets for each requirement
        spec_sheets = []
        errors = []

        for req in analysis.spec_sheet_requirements:
            # Get template for this requirement
            try:
                # Determine template category from spec_sheet_type
                template_category = (
                    req.spec_sheet_type.split("/")[0] if "/" in req.spec_sheet_type else req.spec_sheet_type
                )

                # List templates in this category
                templates = await self.spec_registry.list_templates(category=template_category)

                if not templates:
                    logger.warning(f"No templates found for category: {template_category}")
                    errors.append(f"No templates found for requirement: {req.spec_sheet_type}")
                    continue

                # For each required instance
                for i in range(req.count):
                    # Use first template as default
                    template = templates[0]

                    # Try to find a more specific template that matches the requirement type
                    for t in templates:
                        if req.spec_sheet_type in t.id or req.spec_sheet_type in t.category:
                            template = t
                            break

                    # Create spec sheet name based on requirement
                    sheet_name = f"{template.name} - {i + 1}" if req.count > 1 else template.name

                    # Create blank spec sheet
                    spec_sheet = await self._create_blank_spec_sheet(
                        project_id=project_id, template=template, name=sheet_name
                    )

                    if spec_sheet:
                        spec_sheets.append(spec_sheet)
                    else:
                        errors.append(f"Failed to create spec sheet for template: {template.id}")

            except Exception as e:
                logger.error(f"Error creating spec sheet for requirement {req.spec_sheet_type}: {str(e)}")
                errors.append(f"Error creating spec sheet for requirement {req.spec_sheet_type}: {str(e)}")

        # Update project with spec sheet IDs
        if spec_sheets:
            project.spec_sheet_ids = [sheet.id for sheet in spec_sheets]
            project.status = ProjectStatus.SPEC_SHEETS_GENERATED
            await self.update_project(project)

        return spec_sheets, errors

    async def _create_blank_spec_sheet(
        self, project_id: str, template: SpecSheetTemplate, name: str
    ) -> Optional[SpecSheet]:
        """
        Create a blank spec sheet based on a template.

        Args:
            project_id: ID of the project
            template: Template to use
            name: Name for the spec sheet

        Returns:
            Optional[SpecSheet]: Created spec sheet or None if creation failed
        """
        try:
            logger.info(f"Creating blank spec sheet from template: {template.id}")

            # Create sections with empty field values
            sections = []

            for template_section in template.sections:
                fields = []

                for template_field in template_field.fields:
                    fields.append(FieldValue(name=template_field.name, value=template_field.default_value))

                sections.append(SectionValues(name=template_section.name, fields=fields))

            # Create spec sheet
            spec_sheet = SpecSheet(
                id=str(uuid.uuid4()),
                template_id=template.id,
                project_id=project_id,
                name=name,
                sections=sections,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                completed=False,
                validated=False,
            )

            # Store in repository
            await self.storage_repository.store_spec_sheet(spec_sheet.to_dict())

            return spec_sheet

        except Exception as e:
            logger.error(f"Failed to create blank spec sheet: {str(e)}")
            return None
