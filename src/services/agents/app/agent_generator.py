# agent_template_service/services/agent_generator.py
import os
import logging
import jinja2
import uuid
from typing import Dict, List, Optional, Any

from agent_template_service.models.agent_blocks import AgentBlock, AgentTemplate, AgentInstance
from agent_template_service.services.block_registry import BlockRegistry
from agent_template_service.services.template_service import TemplateService
from agent_template_service.services.event_service import AgentEventService
from agent_template_service.models.events import (
    AgentInstanceCreatedEvent,
    AgentInstanceStartedEvent,
    AgentInstanceCompletedEvent,
    AgentInstanceFailedEvent,
)

logger = logging.getLogger(__name__)


class AgentGenerator:
    """Service for generating agent implementations from templates."""

    def __init__(
        self,
        output_dir: str = "./output/agents",
        template_service: TemplateService = None,
        block_registry: BlockRegistry = None,
        event_service: Optional[AgentEventService] = None,
    ):
        self.output_dir = output_dir
        self.template_service = template_service
        self.block_registry = block_registry
        self.event_service = event_service
        self.instances: Dict[str, AgentInstance] = {}

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set up Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("./templates"),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

    async def generate_agent(
        self, template_id: str, name: str, description: str, configuration: Dict[str, Any] = {}
    ) -> Optional[AgentInstance]:
        """Generate an agent implementation from a template."""
        # Get the template
        template = self.template_service.get_template(template_id)
        if not template:
            logger.error(f"Template {template_id} not found")
            return None

        # Create agent instance
        instance_id = str(uuid.uuid4())
        instance = AgentInstance(
            id=instance_id,
            template_id=template_id,
            name=name,
            description=description,
            configuration=configuration,
            status="created",
        )

        # Store instance
        self.instances[instance_id] = instance

        # Emit event
        if self.event_service:
            event = AgentInstanceCreatedEvent(
                source_container="agent_template_service",
                instance_id=instance_id,
                template_id=template_id,
                instance_name=name,
            )
            await self.event_service.emit_event(event)

        # Generate implementation asynchronously
        try:
            # Update status
            instance.status = "generating"

            # Emit event
            if self.event_service:
                event = AgentInstanceStartedEvent(
                    source_container="agent_template_service",
                    instance_id=instance_id,
                    template_id=template_id,
                )
                await self.event_service.emit_event(event)

            # Generate the agent
            implementation = await self._generate_implementation(template, instance)

            # Create output directory
            agent_dir = os.path.join(self.output_dir, instance_id)
            os.makedirs(agent_dir, exist_ok=True)

            # Write implementation files
            for filename, content in implementation.items():
                with open(os.path.join(agent_dir, filename), "w") as f:
                    f.write(content)

            # Update status
            instance.status = "completed"

            # Emit event
            if self.event_service:
                event = AgentInstanceCompletedEvent(
                    source_container="agent_template_service",
                    instance_id=instance_id,
                    template_id=template_id,
                    results={"output_dir": agent_dir},
                )
                await self.event_service.emit_event(event)

            logger.info(f"Generated agent {instance_id} from template {template_id}")
            return instance

        except Exception as e:
            # Update status
            instance.status = "failed"

            # Emit event
            if self.event_service:
                event = AgentInstanceFailedEvent(
                    source_container="agent_template_service",
                    instance_id=instance_id,
                    template_id=template_id,
                    error=str(e),
                )
                await self.event_service.emit_event(event)

            logger.error(f"Failed to generate agent {instance_id}: {e}")
            return instance

    async def _generate_implementation(
        self, template: AgentTemplate, instance: AgentInstance
    ) -> Dict[str, str]:
        """Generate the implementation files for an agent."""
        # Get blocks
        blocks = {}
        for block_id in template.blocks:
            block = self.block_registry.get_block(block_id)
            if not block:
                raise ValueError(f"Block {block_id} not found")
            blocks[block_id] = block

        # Prepare context for templates
        context = {
            "agent_name": instance.name,
            "agent_description": instance.description,
            "template": template,
            "blocks": blocks,
            "configuration": instance.configuration,
        }

        # Generate files
        implementation = {}

        # Generate main.py
        main_template = self.jinja_env.get_template("agent_main.py.j2")
        implementation["main.py"] = main_template.render(context)

        # Generate agent.json
        agent_json_template = self.jinja_env.get_template("agent.json.j2")
        implementation["agent.json"] = agent_json_template.render(context)

        # Generate block implementations
        for block_id, block in blocks.items():
            block_filename = f"blocks/{block.metadata.name.lower().replace(' ', '_')}.py"
            block_template = self.jinja_env.get_template("block_implementation.py.j2")
            block_context = {**context, "block": block}
            implementation[block_filename] = block_template.render(block_context)

        # Generate Docker files
        dockerfile_template = self.jinja_env.get_template("Dockerfile.j2")
        implementation["Dockerfile"] = dockerfile_template.render(context)

        docker_compose_template = self.jinja_env.get_template("docker-compose.yml.j2")
        implementation["docker-compose.yml"] = docker_compose_template.render(context)

        # Generate README
        readme_template = self.jinja_env.get_template("README.md.j2")
        implementation["README.md"] = readme_template.render(context)

        return implementation

    def get_instance(self, instance_id: str) -> Optional[AgentInstance]:
        """Get an agent instance by ID."""
        return self.instances.get(instance_id)

    def list_instances(self, template_id: Optional[str] = None) -> List[AgentInstance]:
        """List agent instances, optionally filtered by template ID."""
        instances = list(self.instances.values())

        # Filter by template ID if specified
        if template_id:
            instances = [i for i in instances if i.template_id == template_id]

        return instances
