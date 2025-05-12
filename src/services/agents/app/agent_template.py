# agent_template_service/services/template_service.py
import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from agent_template_service.models.agent_blocks import AgentTemplate
from agent_template_service.services.event_service import AgentEventService
from agent_template_service.services.block_registry import BlockRegistry
from agent_template_service.models.events import (
    AgentTemplateCreatedEvent,
    AgentTemplateUpdatedEvent,
)

logger = logging.getLogger(__name__)


class TemplateService:
    """Service for managing agent templates."""

    def __init__(
        self,
        storage_dir: str = "./data/agent_templates",
        event_service: Optional[AgentEventService] = None,
        block_registry: Optional[BlockRegistry] = None,
    ):
        self.storage_dir = storage_dir
        self.event_service = event_service
        self.block_registry = block_registry
        self.templates: Dict[str, AgentTemplate] = {}

        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)

        # Load existing templates
        self._load_templates()

    def _load_templates(self):
        """Load templates from storage."""
        try:
            # Load each template file
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(self.storage_dir, filename), "r") as f:
                        template_data = json.load(f)
                        template = AgentTemplate(**template_data)
                        self.templates[template.id] = template

            logger.info(f"Loaded {len(self.templates)} agent templates from storage")
        except Exception as e:
            logger.error(f"Failed to load agent templates: {e}")

    def _save_template(self, template: AgentTemplate):
        """Save a template to storage."""
        try:
            filename = f"{template.id}.json"
            with open(os.path.join(self.storage_dir, filename), "w") as f:
                f.write(template.model_dump_json())
        except Exception as e:
            logger.error(f"Failed to save agent template {template.id}: {e}")

    async def create_template(
        self,
        name: str,
        description: str,
        blocks: List[str],
        connections: List[Dict[str, str]],
        metadata: Dict[str, Any] = {},
        version: str = "1.0.0",
    ) -> AgentTemplate:
        """Create a new agent template."""
        # Validate that all blocks exist
        if self.block_registry:
            for block_id in blocks:
                if not self.block_registry.get_block(block_id):
                    raise ValueError(f"Block {block_id} not found")

        # Generate a new ID
        template_id = str(uuid.uuid4())

        # Create the template
        template = AgentTemplate(
            id=template_id,
            name=name,
            description=description,
            blocks=blocks,
            connections=connections,
            metadata=metadata,
            version=version,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Store the template
        self.templates[template_id] = template
        self._save_template(template)

        # Emit event if event service is available
        if self.event_service:
            event = AgentTemplateCreatedEvent(
                source_container="agent_template_service",
                template_id=template_id,
                template_name=name,
                blocks=blocks,
            )
            await self.event_service.emit_event(event)

        logger.info(f"Created agent template {template_id} with {len(blocks)} blocks")
        return template

    async def update_template(self, template_id: str, **updates) -> Optional[AgentTemplate]:
        """Update an existing agent template."""
        if template_id not in self.templates:
            logger.warning(f"Template {template_id} not found")
            return None

        # Get the existing template
        template = self.templates[template_id]

        # Apply updates
        template_data = template.model_dump()
        template_data.update(updates)
        template_data["updated_at"] = datetime.utcnow()

        # Validate blocks if they're being updated
        if "blocks" in updates and self.block_registry:
            for block_id in updates["blocks"]:
                if not self.block_registry.get_block(block_id):
                    raise ValueError(f"Block {block_id} not found")

        # Create updated template
        updated_template = AgentTemplate(**template_data)

        # Store the updated template
        self.templates[template_id] = updated_template
        self._save_template(updated_template)

        # Emit event if event service is available
        if self.event_service:
            event = AgentTemplateUpdatedEvent(
                source_container="agent_template_service",
                template_id=template_id,
                template_name=updated_template.name,
                blocks=updated_template.blocks,
            )
            await self.event_service.emit_event(event)

        logger.info(f"Updated agent template {template_id}")
        return updated_template

    def get_template(self, template_id: str) -> Optional[AgentTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)

    def list_templates(self) -> List[AgentTemplate]:
        """List all templates."""
        return list(self.templates.values())

    def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        if template_id not in self.templates:
            return False

        # Remove from memory
        del self.templates[template_id]

        # Remove from storage
        try:
            os.remove(os.path.join(self.storage_dir, f"{template_id}.json"))
            logger.info(f"Deleted agent template {template_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete agent template {template_id}: {e}")
            return False
