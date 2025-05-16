# agent_template_service/services/block_registry.py
import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from agent_template_service.models.agent_blocks import (
    AgentBlock,
    BlockMetadata,
    BlockType,
)
from agent_template_service.services.event_service import AgentEventService
from agent_template_service.models.events import (
    AgentBlockCreatedEvent,
    AgentBlockUpdatedEvent,
)

logger = logging.getLogger(__name__)


class BlockRegistry:
    """Registry service for storing and retrieving agent blocks."""

    def __init__(
        self,
        storage_dir: str = "./data/agent_blocks",
        event_service: Optional[AgentEventService] = None,
    ):
        self.storage_dir = storage_dir
        self.event_service = event_service
        self.blocks: Dict[str, AgentBlock] = {}

        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)

        # Load existing blocks
        self._load_blocks()

    def _load_blocks(self):
        """Load blocks from storage."""
        try:
            # Load each block file
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(self.storage_dir, filename), "r") as f:
                        block_data = json.load(f)
                        block = AgentBlock(**block_data)
                        self.blocks[block.id] = block

            logger.info(f"Loaded {len(self.blocks)} agent blocks from storage")
        except Exception as e:
            logger.error(f"Failed to load agent blocks: {e}")

    def _save_block(self, block: AgentBlock):
        """Save a block to storage."""
        try:
            filename = f"{block.id}.json"
            with open(os.path.join(self.storage_dir, filename), "w") as f:
                f.write(block.model_dump_json())
        except Exception as e:
            logger.error(f"Failed to save agent block {block.id}: {e}")

    async def create_block(
        self,
        type: BlockType,
        name: str,
        description: str,
        version: str,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        implementation: str,
        tags: List[str] = [],
        language: str = "python",
        dependencies: List[str] = [],
        requirements: List[str] = [],
        events_emitted: List[str] = [],
        events_handled: List[str] = [],
        author: Optional[str] = None,
    ) -> AgentBlock:
        """Create a new agent block."""
        # Generate a new ID
        block_id = str(uuid.uuid4())

        # Create metadata
        metadata = BlockMetadata(
            name=name,
            description=description,
            version=version,
            author=author,
            tags=tags,
            language=language,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Create the block
        block = AgentBlock(
            id=block_id,
            type=type,
            metadata=metadata,
            inputs=inputs,
            outputs=outputs,
            implementation=implementation,
            dependencies=dependencies,
            requirements=requirements,
            events_emitted=events_emitted,
            events_handled=events_handled,
        )

        # Store the block
        self.blocks[block_id] = block
        self._save_block(block)

        # Emit event if event service is available
        if self.event_service:
            event = AgentBlockCreatedEvent(
                source_container="agent_template_service",
                block_id=block_id,
                block_type=type,
                block_name=name,
            )
            await self.event_service.emit_event(event)

        logger.info(f"Created agent block {block_id} of type {type}")
        return block

    async def update_block(self, block_id: str, **updates) -> Optional[AgentBlock]:
        """Update an existing agent block."""
        if block_id not in self.blocks:
            logger.warning(f"Block {block_id} not found")
            return None

        # Get the existing block
        block = self.blocks[block_id]

        # Apply updates
        block_data = block.model_dump()

        # Update metadata fields
        if "name" in updates or "description" in updates or "version" in updates or "tags" in updates:
            metadata = block_data["metadata"]
            if "name" in updates:
                metadata["name"] = updates["name"]
            if "description" in updates:
                metadata["description"] = updates["description"]
            if "version" in updates:
                metadata["version"] = updates["version"]
            if "tags" in updates:
                metadata["tags"] = updates["tags"]
            metadata["updated_at"] = datetime.utcnow().isoformat()
            updates["metadata"] = metadata

        # Remove metadata fields from direct updates
        for field in ["name", "description", "version", "tags"]:
            if field in updates:
                del updates[field]

        # Apply remaining updates directly
        block_data.update(updates)

        # Create updated block
        updated_block = AgentBlock(**block_data)

        # Store the updated block
        self.blocks[block_id] = updated_block
        self._save_block(updated_block)

        # Emit event if event service is available
        if self.event_service:
            event = AgentBlockUpdatedEvent(
                source_container="agent_template_service",
                block_id=block_id,
                block_type=updated_block.type,
                block_name=updated_block.metadata.name,
            )
            await self.event_service.emit_event(event)

        logger.info(f"Updated agent block {block_id}")
        return updated_block

    def get_block(self, block_id: str) -> Optional[AgentBlock]:
        """Get a block by ID."""
        return self.blocks.get(block_id)

    def list_blocks(self, block_type: Optional[BlockType] = None, tags: Optional[List[str]] = None) -> List[AgentBlock]:
        """List blocks, optionally filtered by type and tags."""
        blocks = list(self.blocks.values())

        # Filter by type if specified
        if block_type:
            blocks = [b for b in blocks if b.type == block_type]

        # Filter by tags if specified
        if tags:
            blocks = [b for b in blocks if all(tag in b.metadata.tags for tag in tags)]

        return blocks

    def delete_block(self, block_id: str) -> bool:
        """Delete a block."""
        if block_id not in self.blocks:
            return False

        # Remove from memory
        del self.blocks[block_id]

        # Remove from storage
        try:
            os.remove(os.path.join(self.storage_dir, f"{block_id}.json"))
            logger.info(f"Deleted agent block {block_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete agent block {block_id}: {e}")
            return False
