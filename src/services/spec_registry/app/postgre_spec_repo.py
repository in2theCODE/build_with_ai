# src/services/spec_registry/app/postgresql_spec_repository.py
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json

import asyncpg
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PostgreSQLSpecRepository:
    """PostgreSQL/Supabase implementation of the storage repository for spec sheets."""

    def __init__(self, connection_string: str, schema: str = "public", max_connections: int = 10):
        """Initialize the PostgreSQL repository."""
        self.connection_string = connection_string
        self.schema = schema
        self.max_connections = max_connections
        self.pool = None
        self.logger = logger

    async def initialize(self):
        """Initialize the database connection pool and create tables if they don't exist."""
        self.pool = await asyncpg.create_pool(
            self.connection_string, min_size=2, max_size=self.max_connections
        )

        # Create tables if they don't exist
        await self._create_tables()

    async def _create_tables(self):
        """Create necessary tables if they don't exist."""
        async with self.pool.acquire() as connection:
            # Create specs table
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.specs (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    project_id TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    fields JSONB NOT NULL DEFAULT 
                    status TEXT NOT NULL DEFAULT 'empty',
                    validation_errors JSONB DEFAULT '[]',
                    metadata JSONB DEFAULT 
                )
            """
            )

            # Create spec relations table
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.spec_relations (
                    id SERIAL PRIMARY KEY,
                    spec_id TEXT NOT NULL REFERENCES {self.schema}.specs(id) ON DELETE CASCADE,
                    related_spec_id TEXT NOT NULL REFERENCES {self.schema}.specs(id) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    UNIQUE(spec_id, related_spec_id)
                )
            """
            )

            # Create templates table
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.templates (
                    id SERIAL PRIMARY KEY,
                    type TEXT NOT NULL,
                    version TEXT NOT NULL DEFAULT '1.0',
                    fields JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    metadata JSONB DEFAULT 
                    UNIQUE(type, version)
                )
            """
            )

            # Create indexes for better performance
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_specs_project_id ON {self.schema}.specs(project_id);
                CREATE INDEX IF NOT EXISTS idx_specs_type ON {self.schema}.specs(type);
                CREATE INDEX IF NOT EXISTS idx_specs_status ON {self.schema}.specs(status);
                CREATE INDEX IF NOT EXISTS idx_templates_type ON {self.schema}.templates(type);
                CREATE INDEX IF NOT EXISTS idx_templates_active ON {self.schema}.templates(is_active);
            """
            )

    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()

    def _format_record(self, record) -> Dict[str, Any]:
        """Format a database record to a dictionary."""
        if record is None:
            return None

        result = dict(record)  # Fixed: Changed 'return=' to 'result ='

        # Convert datetime objects to ISO format strings for serialization
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()

        return result

    async def store_spec(self, spec: Dict[str, Any]) -> bool:
        """Store a spec sheet."""
        try:
            # Ensure timestamp fields are properly formatted
            if isinstance(spec.get("created_at"), datetime):
                spec["created_at"] = spec["created_at"].isoformat()
            elif not isinstance(spec.get("created_at"), str):
                spec["created_at"] = datetime.now().isoformat()

            if isinstance(spec.get("updated_at"), datetime):
                spec["updated_at"] = spec["updated_at"].isoformat()
            else:
                spec["updated_at"] = datetime.now().isoformat()

            # Extract fields that go directly into columns
            spec_id = spec["id"]
            spec_type = spec["type"]
            project_id = spec.get("project_id")
            created_at = spec["created_at"]
            updated_at = spec["updated_at"]
            fields = json.dumps(spec.get("fields", {}))
            status = spec.get("status", "empty")
            validation_errors = json.dumps(spec.get("validation_errors", []))

            # Put any other fields into metadata
            metadata = {
                k: v
                for k, v in spec.items()
                if k
                not in [
                    "id",
                    "type",
                    "project_id",
                    "created_at",
                    "updated_at",
                    "fields",
                    "status",
                    "validation_errors",
                ]
            }

            async with self.pool.acquire() as connection:
                await connection.execute(
                    f"""
                    INSERT INTO {self.schema}.specs 
                    (id, type, project_id, created_at, updated_at, fields, status, validation_errors, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (id) 
                    DO UPDATE SET
                        type = $2,
                        project_id = $3,
                        updated_at = $5,
                        fields = $6,
                        status = $7,
                        validation_errors = $8,
                        metadata = $9
                """,
                    spec_id,
                    spec_type,
                    project_id,
                    created_at,
                    updated_at,
                    fields,
                    status,
                    validation_errors,
                    json.dumps(metadata),
                )

            return True
        except Exception as e:
            self.logger.error(f"Error storing spec {spec.get('id')}: {e}")
            return False

    async def update_spec(self, spec: Dict[str, Any]) -> bool:
        """Update a spec sheet."""
        try:
            # Just use store_spec since it handles both insert and update
            return await self.store_spec(spec)
        except Exception as e:
            self.logger.error(f"Error updating spec {spec.get('id')}: {e}")
            return False

    async def get_spec(self, spec_id: str) -> Optional[Dict[str, Any]]:
        """Get a spec by ID."""
        try:
            async with self.pool.acquire() as connection:
                record = await connection.fetchrow(
                    f"""
                    SELECT * FROM {self.schema}.specs WHERE id = $1
                """,
                    spec_id,
                )

                if not record:
                    return None

                result = self._format_record(record)

                # Convert JSON string fields back to Python objects
                result["fields"] = json.loads(result["fields"]) if result["fields"] else {}
                result["validation_errors"] = (
                    json.loads(result["validation_errors"]) if result["validation_errors"] else []
                )
                result["metadata"] = json.loads(result["metadata"]) if result["metadata"] else {}

                # Merge metadata back into the result for backward compatibility
                result.update(result["metadata"])
                del result["metadata"]

                return result
        except Exception as e:
            self.logger.error(f"Error getting spec {spec_id}: {e}")
            return None

    async def delete_spec(self, spec_id: str) -> bool:
        """Delete a spec sheet."""
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(
                    f"""
                    DELETE FROM {self.schema}.specs WHERE id = $1
                """,
                    spec_id,
                )

            return True
        except Exception as e:
            self.logger.error(f"Error deleting spec {spec_id}: {e}")
            return False

    async def list_specs(
        self,
        project_id: Optional[str] = None,
        spec_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List specs, optionally filtered by project ID, type or status."""
        try:
            query = f"SELECT * FROM {self.schema}.specs WHERE 1=1"
            params = []

            if project_id:
                params.append(project_id)
                query += f" AND project_id = ${len(params)}"

            if spec_type:
                params.append(spec_type)
                query += f" AND type = ${len(params)}"

            if status:
                params.append(status)
                query += f" AND status = ${len(params)}"

            query += " ORDER BY updated_at DESC"

            async with self.pool.acquire() as connection:
                records = await connection.fetch(query, *params)

                results = []
                for record in records:
                    result = self._format_record(record)

                    # Convert JSON string fields back to Python objects
                    result["fields"] = json.loads(result["fields"]) if result["fields"] else {}
                    result["validation_errors"] = (
                        json.loads(result["validation_errors"])
                        if result["validation_errors"]
                        else []
                    )
                    result["metadata"] = (
                        json.loads(result["metadata"]) if result["metadata"] else {}
                    )

                    # Merge metadata back into the result for backward compatibility
                    result.update(result["metadata"])
                    del result["metadata"]

                    results.append(result)

                return results
        except Exception as e:
            self.logger.error(f"Error listing specs: {e}")
            return []

    async def store_spec_relation(
        self, spec_id: str, related_spec_id: str, relation_type: str
    ) -> bool:
        """Store a relation between two spec sheets."""
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(
                    f"""
                    INSERT INTO {self.schema}.spec_relations 
                    (spec_id, related_spec_id, relation_type)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (spec_id, related_spec_id) 
                    DO UPDATE SET relation_type = $3
                """,
                    spec_id,
                    related_spec_id,
                    relation_type,
                )

            return True
        except Exception as e:
            self.logger.error(
                f"Error storing relation between {spec_id} and {related_spec_id}: {e}"
            )
            return False

    async def get_related_specs(self, spec_id: str) -> List[Dict[str, Any]]:
        """Get specs related to the given spec."""
        try:
            async with self.pool.acquire() as connection:
                records = await connection.fetch(
                    f"""
                    SELECT r.relation_type, s.* 
                    FROM {self.schema}.spec_relations r
                    JOIN {self.schema}.specs s ON r.related_spec_id = s.id
                    WHERE r.spec_id = $1
                """,
                    spec_id,
                )

                results = []
                for record in records:
                    result = self._format_record(record)
                    relation_type = result.pop("relation_type")

                    # Convert JSON string fields back to Python objects
                    result["fields"] = json.loads(result["fields"]) if result["fields"] else {}
                    result["validation_errors"] = (
                        json.loads(result["validation_errors"])
                        if result["validation_errors"]
                        else []
                    )
                    result["metadata"] = (
                        json.loads(result["metadata"]) if result["metadata"] else {}
                    )

                    # Merge metadata back into the result for backward compatibility
                    result.update(result["metadata"])
                    del result["metadata"]

                    results.append({"spec": result, "relation_type": relation_type})

                return results
        except Exception as e:
            self.logger.error(f"Error getting related specs for {spec_id}: {e}")
            return []

    async def store_template(
        self,
        template_type: str,
        fields: Dict[str, Any],
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a spec template."""
        try:
            metadata = metadata or {}

            async with self.pool.acquire() as connection:
                await connection.execute(
                    f"""
                    INSERT INTO {self.schema}.templates 
                    (type, version, fields, updated_at, metadata)
                    VALUES ($1, $2, $3, NOW(), $4)
                    ON CONFLICT (type, version) 
                    DO UPDATE SET
                        fields = $3,
                        updated_at = NOW(),
                        metadata = $4,
                        is_active = TRUE
                """,
                    template_type,
                    version,
                    json.dumps(fields),
                    json.dumps(metadata),
                )

            return True
        except Exception as e:
            self.logger.error(f"Error storing template {template_type} v{version}: {e}")
            return False

    async def get_template(
        self, template_type: str, version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a template by type and optional version."""
        try:
            query = f"""
                SELECT * FROM {self.schema}.templates 
                WHERE type = $1 AND is_active = TRUE
            """
            params = [template_type]

            if version:
                query += " AND version = $2"
                params.append(version)
            else:
                # Get the latest version based on semantic versioning
                query += " ORDER BY version DESC LIMIT 1"

            async with self.pool.acquire() as connection:
                record = await connection.fetchrow(query, *params)

                if not record:
                    return None

                result = self._format_record(record)

                # Convert JSON string fields back to Python objects
                result["fields"] = json.loads(result["fields"]) if result["fields"] else {}
                result["metadata"] = json.loads(result["metadata"]) if result["metadata"] else {}

                # Merge metadata back into the result for backward compatibility
                result.update(result["metadata"])
                del result["metadata"]

                return result
        except Exception as e:
            self.logger.error(f"Error getting template {template_type}: {e}")
            return None

    async def list_templates(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """List all available templates."""
        try:
            query = f"SELECT * FROM {self.schema}.templates"
            if not include_inactive:
                query += " WHERE is_active = TRUE"
            query += " ORDER BY type, version DESC"

            async with self.pool.acquire() as connection:
                records = await connection.fetch(query)

                results = []
                for record in records:
                    result = self._format_record(record)

                    # Convert JSON string fields back to Python objects
                    result["fields"] = json.loads(result["fields"]) if result["fields"] else {}
                    result["metadata"] = (
                        json.loads(result["metadata"]) if result["metadata"] else {}
                    )

                    # Merge metadata back into the result for backward compatibility
                    result.update(result["metadata"])
                    del result["metadata"]

                    results.append(result)

                return results
        except Exception as e:
            self.logger.error(f"Error listing templates: {e}")
            return []

    async def delete_template(
        self, template_type: str, version: Optional[str] = None, hard_delete: bool = False
    ) -> bool:
        """Delete a template or mark it as inactive."""
        try:
            async with self.pool.acquire() as connection:
                if hard_delete:
                    # Hard delete
                    query = f"DELETE FROM {self.schema}.templates WHERE type = $1"
                    params = [template_type]

                    if version:
                        query += " AND version = $2"
                        params.append(version)

                    await connection.execute(query, *params)
                else:
                    # Soft delete (mark as inactive)
                    query = f"UPDATE {self.schema}.templates SET is_active = FALSE WHERE type = $1"
                    params = [template_type]

                    if version:
                        query += " AND version = $2"
                        params.append(version)

                    await connection.execute(query, *params)

            return True
        except Exception as e:
            self.logger.error(f"Error deleting template {template_type}: {e}")
            return False
