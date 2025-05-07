# src/services/spec_registry/app/spec_registry.py
import asyncio
from datetime import datetime
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Union

from src.services.shared.logging.logger import get_logger
from src.services.spec_registry.app.models import (
    FieldType,
    SpecStatus,
    SpecTemplate,
    ValidationResult,
    FieldDefinition,
    FieldConstraint,
)

logger = get_logger(__name__)


class SpecRegistry:
    """
    Registry for managing specification sheets.

    Stores, retrieves, and manages spec sheets that define requirements
    for code generation. Implements constraint validation.
    """

    def __init__(self, storage_repository=None):
        """Initialize the spec registry."""
        self._specs = {}  # spec_id -> spec
        self._spec_relations = {}  # spec_id -> related_spec_ids
        self._storage_repository = storage_repository
        self._lock = asyncio.Lock()
        self.logger = logger

        # Initialize the type and constraint validators
        self._init_validators()

    def _init_validators(self):
        """Initialize the validators for spec sheets."""
        # Map of field types to validation functions
        self._type_validators = {
            FieldType.STRING: self._validate_string,
            FieldType.TEXT: self._validate_string,
            FieldType.INT: self._validate_int,
            FieldType.FLOAT: self._validate_float,
            FieldType.BOOLEAN: self._validate_boolean,
            FieldType.LIST: self._validate_list,
            FieldType.JSON: self._validate_json,
            FieldType.CODE: self._validate_code,
            FieldType.DATETIME: self._validate_datetime,
            FieldType.REFERENCE: self._validate_reference,
        }

        # Constraint patterns with validators
        self._constraint_validators = [
            (re.compile(r"required"), self._validate_required),
            (re.compile(r"min_length\((\d+)\)"), self._validate_min_length),
            (re.compile(r"max_length\((\d+)\)"), self._validate_max_length),
            (re.compile(r"min_value\(([^)]+)\)"), self._validate_min_value),
            (re.compile(r"max_value\(([^)]+)\)"), self._validate_max_value),
            (re.compile(r"pattern\(([^)]+)\)"), self._validate_pattern),
        ]

    async def create_empty_spec(
        self,
        spec_type: str,
        project_id: Optional[str] = None,
        template_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create an empty spec sheet of the specified type.

        Args:
            spec_type: Type of spec to create
            project_id: Optional project ID
            template_version: Optional template version to use

        Returns:
            Created spec sheet
        """
        spec_id = str(uuid.uuid4())

        try:
            # Try to get fields from a template in the repository
            fields = None
            if self._storage_repository:
                template = await self._storage_repository.get_template(spec_type, template_version)
                if template:
                    fields = template.get("fields", {})
                    template_version = template.get("version")

            # If no template found, use hardcoded defaults
            if not fields:
                fields = self.get_fields_for_spec_type(spec_type)

            # Create spec object
            spec = {
                "id": spec_id,
                "type": spec_type,
                "project_id": project_id,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "fields": fields,
                "status": SpecStatus.EMPTY,
                "validation_errors": [],
                "template_version": template_version,
            }

            # Store spec
            async with self._lock:
                self._specs[spec_id] = spec
                if self._storage_repository:
                    await self._storage_repository.store_spec(spec)

            return spec
        except Exception as e:
            self.logger.error(f"Error creating empty spec: {e}", exc_info=True)
            raise

    def get_fields_for_spec_type(self, spec_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get the field definitions for a spec type.

        Args:
            spec_type: Type of spec

        Returns:
            Field definitions for the spec type
        """
        return self._get_fields_for_spec_type(spec_type)

    def _get_fields_for_spec_type(self, spec_type: str) -> Dict[str, Dict[str, Any]]:
        """Get the field definitions for a spec type."""
        # Field definitions for different spec types with enhanced validation
        field_definitions = {
            "container": self._get_container_field_definitions(),
            "api": self._get_api_field_definitions(),
            "database": self._get_database_field_definitions(),
        }

        # If spec_type not found, return empty field definitions
        if spec_type not in field_definitions:
            self.logger.warning(f"No field definitions found for spec type: {spec_type}")
            return {}

        # Convert to the format expected by the spec system
        fields_dict = {}
        for field in field_definitions.get(spec_type, []):
            constraints = []
            for constraint in field.constraints:
                if constraint.type == "required":
                    constraints.append("required")
                elif constraint.type == "min_length":
                    constraints.append(f"min_length({constraint.parameters['value']})")
                elif constraint.type == "max_length":
                    constraints.append(f"max_length({constraint.parameters['value']})")
                elif constraint.type == "pattern":
                    constraints.append(f"pattern({constraint.parameters['pattern']})")
                elif constraint.type == "min_value":
                    constraints.append(f"min_value({constraint.parameters['value']})")
                elif constraint.type == "max_value":
                    constraints.append(f"max_value({constraint.parameters['value']})")
                else:
                    constraints.append(f"{constraint.type}({constraint.parameters})")

            fields_dict[field.name] = {
                "value": field.default_value,
                "type": field.type.value if isinstance(field.type, FieldType) else field.type,
                "required": field.required,
                "constraints": constraints,
                "label": field.label or field.name,
                "description": field.description or "",
                "placeholder": field.placeholder,
                "help_text": field.help_text,
                "hidden": field.hidden,
                "order": field.order,
                "group": field.group,
                "conditional": field.conditional,
                "metadata": field.metadata,
            }

        return fields_dict

    def _get_container_field_definitions(self) -> List[FieldDefinition]:
        """Get field definitions for container spec type."""
        return [
            FieldDefinition(
                name="container_name",
                type=FieldType.STRING,
                label="Container Name",
                description="Name of the container",
                required=True,
                constraints=[
                    FieldConstraint(type="min_length", parameters={"value": 3}),
                    FieldConstraint(type="max_length", parameters={"value": 50}),
                    FieldConstraint(type="pattern", parameters={"pattern": "^[a-zA-Z0-9_-]+$"}),
                ],
                placeholder="my-container",
                help_text="Enter a unique name for this container (alphanumeric, dashes and underscores only)",
            ),
            FieldDefinition(
                name="description",
                type=FieldType.TEXT,
                label="Description",
                description="Detailed description of the container's purpose",
                required=True,
                constraints=[FieldConstraint(type="min_length", parameters={"value": 10})],
                placeholder="This container handles...",
            ),
            FieldDefinition(
                name="dependencies",
                type=FieldType.LIST,
                label="Dependencies",
                description="List of dependencies required by this container",
                required=False,
            ),
            FieldDefinition(
                name="event_handlers",
                type=FieldType.LIST,
                label="Event Handlers",
                description="Events this container responds to",
                required=True,
            ),
            FieldDefinition(
                name="event_bus_config",
                type=FieldType.JSON,
                label="Event Bus Configuration",
                description="Configuration for connecting to the event bus",
                required=True,
            ),
            FieldDefinition(
                name="main_logic",
                type=FieldType.CODE,
                label="Main Business Logic",
                description="Main logic for the container implementation",
                required=True,
                constraints=[FieldConstraint(type="min_length", parameters={"value": 5})],
            ),
        ]

    def _get_api_field_definitions(self) -> List[FieldDefinition]:
        """Get field definitions for API spec type."""
        return [
            FieldDefinition(
                name="api_name",
                type=FieldType.STRING,
                label="API Name",
                description="Name of the API",
                required=True,
                constraints=[
                    FieldConstraint(type="min_length", parameters={"value": 3}),
                    FieldConstraint(type="max_length", parameters={"value": 50}),
                ],
                placeholder="users-api",
            ),
            FieldDefinition(
                name="endpoints",
                type=FieldType.LIST,
                label="Endpoints",
                description="List of API endpoints",
                required=True,
            ),
            FieldDefinition(
                name="auth_method",
                type=FieldType.STRING,
                label="Authentication Method",
                description="Method used for authentication",
                required=False,
                placeholder="JWT",
            ),
            FieldDefinition(
                name="documentation",
                type=FieldType.TEXT,
                label="Documentation",
                description="API documentation",
                required=True,
                constraints=[FieldConstraint(type="min_length", parameters={"value": 10})],
            ),
        ]

    def _get_database_field_definitions(self) -> List[FieldDefinition]:
        """Get field definitions for database spec type."""
        return [
            FieldDefinition(
                name="db_name",
                type=FieldType.STRING,
                label="Database Name",
                description="Name of the database",
                required=True,
                constraints=[
                    FieldConstraint(type="min_length", parameters={"value": 3}),
                    FieldConstraint(type="max_length", parameters={"value": 50}),
                ],
                placeholder="users_db",
            ),
            FieldDefinition(
                name="db_type",
                type=FieldType.STRING,
                label="Database Type",
                description="Type of database (PostgreSQL, MongoDB, etc.)",
                required=True,
                placeholder="PostgreSQL",
            ),
            FieldDefinition(
                name="tables",
                type=FieldType.JSON,
                label="Tables",
                description="Database tables and their schemas",
                required=True,
            ),
            FieldDefinition(
                name="relationships",
                type=FieldType.JSON,
                label="Relationships",
                description="Relationships between tables",
                required=False,
            ),
        ]

    async def update_spec(self, spec_id: str, field_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update fields in a spec sheet.

        Args:
            spec_id: ID of the spec to update
            field_updates: Field updates to apply

        Returns:
            Updated spec sheet
        """
        try:
            async with self._lock:
                spec = await self.get_spec(spec_id)
                if not spec:
                    raise ValueError(f"Spec with ID {spec_id} not found")

                for field_name, value in field_updates.items():
                    if field_name in spec["fields"]:
                        spec["fields"][field_name]["value"] = value
                    else:
                        # If the field doesn't exist, add it as a custom field
                        spec["fields"][field_name] = {
                            "value": value,
                            "type": "string",  # Default type
                            "required": False,
                            "constraints": [],
                        }

                spec["updated_at"] = datetime.now()
                spec["status"] = SpecStatus.UPDATED

                # Validate the spec after updates
                validation_errors = self._validate_spec(spec)
                spec["validation_errors"] = validation_errors

                if not validation_errors:
                    spec["status"] = SpecStatus.VALIDATED

                self._specs[spec_id] = spec

                if self._storage_repository:
                    await self._storage_repository.update_spec(spec)

                return spec
        except Exception as e:
            self.logger.error(f"Error updating spec {spec_id}: {e}", exc_info=True)
            raise

    def _validate_spec(self, spec: Dict[str, Any]) -> List[str]:
        """
        Validate all fields in a spec sheet.

        Args:
            spec: Spec sheet to validate

        Returns:
            List of validation errors
        """
        validation_errors = []

        for field_name, field_data in spec["fields"].items():
            # Skip validation for empty fields that aren't required
            if field_data.get("value") is None or field_data.get("value") == "":
                if field_data.get("required", False):
                    validation_errors.append(f"Field '{field_name}' is required but missing")
                continue

            # Validate type
            field_type = field_data.get("type", "string")
            # Handle both enum and string type values
            if isinstance(field_type, str) and field_type in [e.value for e in FieldType]:
                field_type = FieldType(field_type)

            type_validator = self._type_validators.get(field_type)
            if type_validator:
                error = type_validator(field_name, field_data["value"])
                if error:
                    validation_errors.append(error)
            else:
                validation_errors.append(
                    f"Unknown field type '{field_type}' for field '{field_name}'"
                )

            # Validate constraints
            for constraint in field_data.get("constraints", []):
                for pattern, validator in self._constraint_validators:
                    match = pattern.match(constraint) if isinstance(constraint, str) else None
                    if match:
                        error = validator(field_name, field_data["value"], match)
                        if error:
                            validation_errors.append(error)
                        break

        return validation_errors

    # Validation functions
    def _validate_required(self, field_name: str, value: Any, match) -> Optional[str]:
        """Validate that a required field has a value."""
        if value is None or value == "":
            return f"Field '{field_name}' is required"
        return None

    def _validate_min_length(self, field_name: str, value: Any, match) -> Optional[str]:
        """Validate minimum length."""
        min_length = int(match.group(1))
        if isinstance(value, str) and len(value) < min_length:
            return f"Field '{field_name}' must be at least {min_length} characters long"
        if isinstance(value, list) and len(value) < min_length:
            return f"Field '{field_name}' must have at least {min_length} items"
        return None

    def _validate_max_length(self, field_name: str, value: Any, match) -> Optional[str]:
        """Validate maximum length."""
        max_length = int(match.group(1))
        if isinstance(value, str) and len(value) > max_length:
            return f"Field '{field_name}' must not exceed {max_length} characters"
        if isinstance(value, list) and len(value) > max_length:
            return f"Field '{field_name}' must not have more than {max_length} items"
        return None

    def _validate_min_value(self, field_name: str, value: Any, match) -> Optional[str]:
        """Validate minimum value."""
        min_value_str = match.group(1)
        try:
            min_value = int(min_value_str)
        except ValueError:
            min_value = float(min_value_str)

        if isinstance(value, (int, float)) and value < min_value:
            return f"Field '{field_name}' must be at least {min_value}"
        return None

    def _validate_max_value(self, field_name: str, value: Any, match) -> Optional[str]:
        """Validate maximum value."""
        max_value_str = match.group(1)
        try:
            max_value = int(max_value_str)
        except ValueError:
            max_value = float(max_value_str)

        if isinstance(value, (int, float)) and value > max_value:
            return f"Field '{field_name}' must not exceed {max_value}"
        return None

    def _validate_pattern(self, field_name: str, value: Any, match) -> Optional[str]:
        """Validate against a regex pattern."""
        pattern = match.group(1)
        if isinstance(value, str) and not re.match(pattern, value):
            return f"Field '{field_name}' does not match the required pattern"
        return None

    # Type validators
    def _validate_string(self, field_name: str, value: Any) -> Optional[str]:
        """Validate string type."""
        if not isinstance(value, str):
            return f"Field '{field_name}' must be a string"
        return None

    def _validate_int(self, field_name: str, value: Any) -> Optional[str]:
        """Validate integer type."""
        if not isinstance(value, int) or isinstance(value, bool):
            return f"Field '{field_name}' must be an integer"
        return None

    def _validate_float(self, field_name: str, value: Any) -> Optional[str]:
        """Validate float type."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"Field '{field_name}' must be a number"
        return None

    def _validate_boolean(self, field_name: str, value: Any) -> Optional[str]:
        """Validate boolean type."""
        if not isinstance(value, bool):
            return f"Field '{field_name}' must be a boolean"
        return None

    def _validate_list(self, field_name: str, value: Any) -> Optional[str]:
        """Validate list type."""
        if not isinstance(value, list):
            return f"Field '{field_name}' must be a list"
        return None

    def _validate_json(self, field_name: str, value: Any) -> Optional[str]:
        """Validate JSON."""
        if isinstance(value, str):
            try:
                json.loads(value)
                return None
            except json.JSONDecodeError:
                return f"Field '{field_name}' must be valid JSON"
        elif not isinstance(value, (dict, list)):
            return f"Field '{field_name}' must be valid JSON"
        return None

    def _validate_code(self, field_name: str, value: Any) -> Optional[str]:
        """Validate code."""
        if not isinstance(value, str):
            return f"Field '{field_name}' must be a string containing code"
        return None

    def _validate_datetime(self, field_name: str, value: Any) -> Optional[str]:
        """Validate datetime."""
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
                return None
            except ValueError:
                return f"Field '{field_name}' must be a valid ISO 8601 datetime string"
        elif not isinstance(value, datetime):
            return f"Field '{field_name}' must be a datetime"
        return None

    def _validate_reference(self, field_name: str, value: Any) -> Optional[str]:
        """Validate reference to another spec."""
        if not isinstance(value, str):
            return f"Field '{field_name}' must be a string reference to another spec"
        # Additional validation could be added to check if the referenced spec exists
        return None

    async def get_spec(self, spec_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a spec by ID.

        Args:
            spec_id: ID of the spec to get

        Returns:
            Spec sheet or None if not found
        """
        try:
            # First check in-memory cache
            if spec_id in self._specs:
                return self._specs[spec_id]

            # Then check storage
            if self._storage_repository:
                spec = await self._storage_repository.get_spec(spec_id)
                if spec:
                    # Cache the spec in memory
                    self._specs[spec_id] = spec
                    return spec

            return None
        except Exception as e:
            self.logger.error(f"Error retrieving spec {spec_id}: {e}", exc_info=True)
            return None

    async def list_specs(
        self,
        project_id: Optional[str] = None,
        spec_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List specs, optionally filtered by project ID, type or status.

        Args:
            project_id: Optional project ID to filter by
            spec_type: Optional spec type to filter by
            status: Optional status to filter by

        Returns:
            List of matching spec sheets
        """
        try:
            if self._storage_repository:
                # Use repository to get specs
                return await self._storage_repository.list_specs(project_id, spec_type, status)
            else:
                # Use in-memory specs
                specs = list(self._specs.values())

                if project_id:
                    specs = [s for s in specs if s.get("project_id") == project_id]

                if spec_type:
                    specs = [s for s in specs if s.get("type") == spec_type]

                if status:
                    specs = [s for s in specs if s.get("status") == status]

                return specs
        except Exception as e:
            self.logger.error(f"Error listing specs: {e}", exc_info=True)
            return []

    async def delete_spec(self, spec_id: str) -> bool:
        """
        Delete a spec sheet.

        Args:
            spec_id: ID of the spec to delete

        Returns:
            True if deleted successfully
        """
        try:
            async with self._lock:
                # Remove from memory cache
                if spec_id in self._specs:
                    del self._specs[spec_id]

                # Remove from relations
                if spec_id in self._spec_relations:
                    del self._spec_relations[spec_id]

                # Remove from storage
                if self._storage_repository:
                    return await self._storage_repository.delete_spec(spec_id)

                return True
        except Exception as e:
            self.logger.error(f"Error deleting spec {spec_id}: {e}", exc_info=True)
            return False

    async def add_relation(
        self, spec_id: str, related_spec_id: str, relation_type: str = "depends_on"
    ) -> bool:
        """
        Add a relation between two spec sheets.

        Args:
            spec_id: ID of the spec
            related_spec_id: ID of the related spec
            relation_type: Type of relation

        Returns:
            True if relation added successfully
        """
        try:
            async with self._lock:
                # Verify both specs exist
                spec = await self.get_spec(spec_id)
                related_spec = await self.get_spec(related_spec_id)

                if not spec or not related_spec:
                    self.logger.warning(
                        f"Cannot add relation: Spec {spec_id} or {related_spec_id} not found"
                    )
                    return False

                # Add relation to in-memory cache
                if spec_id not in self._spec_relations:
                    self._spec_relations[spec_id] = []

                # Add relation if it doesn't exist already
                relation = {"spec_id": related_spec_id, "type": relation_type}
                if relation not in self._spec_relations[spec_id]:
                    self._spec_relations[spec_id].append(relation)

                # Store in repository
                if self._storage_repository:
                    await self._storage_repository.store_spec_relation(
                        spec_id, related_spec_id, relation_type
                    )

                return True
        except Exception as e:
            self.logger.error(
                f"Error adding relation between {spec_id} and {related_spec_id}: {e}", exc_info=True
            )
            return False

    async def get_related_specs(self, spec_id: str) -> List[Dict[str, Any]]:
        """
        Get specs related to the given spec.

        Args:
            spec_id: ID of the spec to get related specs for

        Returns:
            List of related specs with relation types
        """
        try:
            if self._storage_repository:
                # Use repository to get related specs
                return await self._storage_repository.get_related_specs(spec_id)
            else:
                # Use in-memory relations
                if spec_id not in self._spec_relations:
                    return []

                related_specs = []
                for relation in self._spec_relations[spec_id]:
                    related_spec = await self.get_spec(relation["spec_id"])
                    if related_spec:
                        related_specs.append(
                            {"spec": related_spec, "relation_type": relation["type"]}
                        )

                return related_specs
        except Exception as e:
            self.logger.error(f"Error getting related specs for {spec_id}: {e}", exc_info=True)
            return []

    async def analyze_template_compatibility(
        self,
        spec_id: str,
        new_template_type: Optional[str] = None,
        new_template_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze compatibility between a spec and a template.

        This helps determine if a spec needs to be updated to match
        a newer version of its template or a different template type.

        Args:
            spec_id: ID of the spec to analyze
            new_template_type: Type of template to compare against (default: use spec's type)
            new_template_version: Version of template to compare against (default: latest)

        Returns:
            Analysis result with compatibility information
        """
        try:
            # Get the spec
            spec = await self.get_spec(spec_id)
            if not spec:
                raise ValueError(f"Spec with ID {spec_id} not found")

            # Determine template type to use
            template_type = new_template_type or spec.get("type")

            # Get current template fields
            current_fields = spec.get("fields", {})

            # Get new template
            new_template = None
            if self._storage_repository:
                new_template = await self._storage_repository.get_template(
                    template_type, new_template_version
                )

            # If no template found in storage, use hardcoded template
            if not new_template:
                new_template_fields = self.get_fields_for_spec_type(template_type)
                new_template = {
                    "type": template_type,
                    "version": "1.0",
                    "fields": new_template_fields,
                }

            new_template_fields = new_template.get("fields", {})

            # Analysis results
            missing_fields = []
            extra_fields = []
            type_mismatches = []
            constraint_changes = []
            compatible_fields = []

            # Check current fields against new template
            for field_name, field_data in current_fields.items():
                if field_name in new_template_fields:
                    # Field exists in new template, check compatibility
                    new_field = new_template_fields[field_name]

                    # Check type compatibility
                    if field_data.get("type") != new_field.get("type"):
                        type_mismatches.append(
                            {
                                "field_name": field_name,
                                "current_type": field_data.get("type"),
                                "new_type": new_field.get("type"),
                            }
                        )

                    # Check constraint changes
                    current_constraints = set(field_data.get("constraints", []))
                    new_constraints = set(new_field.get("constraints", []))

                    added_constraints = new_constraints - current_constraints
                    removed_constraints = current_constraints - new_constraints

                    if added_constraints or removed_constraints:
                        constraint_changes.append(
                            {
                                "field_name": field_name,
                                "added_constraints": list(added_constraints),
                                "removed_constraints": list(removed_constraints),
                            }
                        )

                    # If no issues, it's compatible
                    if (
                        field_data.get("type") == new_field.get("type")
                        and not added_constraints
                        and not removed_constraints
                    ):
                        compatible_fields.append(field_name)
                else:
                    # Field doesn't exist in new template
                    extra_fields.append(field_name)

            # Check for fields in new template that don't exist in current fields
            for field_name in new_template_fields:
                if field_name not in current_fields:
                    missing_fields.append(
                        {
                            "field_name": field_name,
                            "required": new_template_fields[field_name].get("required", False),
                        }
                    )

            # Determine compatibility level
            compatibility_level = "fully_compatible"
            required_missing_fields = [f for f in missing_fields if f["required"]]

            if required_missing_fields:
                compatibility_level = "incompatible"
            elif missing_fields or type_mismatches:
                compatibility_level = "needs_update"
            elif constraint_changes:
                compatibility_level = "needs_validation"

            # Create analysis result
            result = {
                "spec_id": spec_id,
                "template_type": template_type,
                "current_template_version": spec.get("template_version"),
                "new_template_version": new_template.get("version"),
                "compatibility_level": compatibility_level,
                "missing_fields": missing_fields,
                "extra_fields": extra_fields,
                "type_mismatches": type_mismatches,
                "constraint_changes": constraint_changes,
                "compatible_fields": compatible_fields,
            }

            return result
        except Exception as e:
            self.logger.error(
                f"Error analyzing template compatibility for {spec_id}: {e}", exc_info=True
            )
            raise

    async def evolve_spec(
        self,
        spec_id: str,
        new_template_type: Optional[str] = None,
        new_template_version: Optional[str] = None,
        keep_extra_fields: bool = True,
    ) -> Dict[str, Any]:
        """
        Evolve a spec to match a new template version.

        Args:
            spec_id: ID of the spec to evolve
            new_template_type: Type of the new template (default: use spec's type)
            new_template_version: Version of the new template (default: latest)
            keep_extra_fields: Whether to keep fields not in the new template

        Returns:
            Updated spec
        """
        try:
            # Get the spec
            spec = await self.get_spec(spec_id)
            if not spec:
                raise ValueError(f"Spec with ID {spec_id} not found")

            # Determine template type to use
            template_type = new_template_type or spec.get("type")

            # Get new template
            new_template = None
            if self._storage_repository:
                new_template = await self._storage_repository.get_template(
                    template_type, new_template_version
                )

            # If no template found in storage, use hardcoded template
            if not new_template:
                new_template_fields = self.get_fields_for_spec_type(template_type)
                new_template = {
                    "type": template_type,
                    "version": "1.0",
                    "fields": new_template_fields,
                }

            new_template_fields = new_template.get("fields", {})

            # Current fields
            current_fields = spec.get("fields", {})

            # Store original fields for change tracking
            spec["_original_fields"] = dict(current_fields)

            # Create updated fields
            updated_fields = {}

            # Add fields from new template
            for field_name, field_data in new_template_fields.items():
                if field_name in current_fields:
                    # Field exists in current spec
                    current_field = current_fields[field_name]

                    # Keep current value
                    updated_fields[field_name] = {**field_data, "value": current_field.get("value")}
                else:
                    # Field doesn't exist in current spec
                    updated_fields[field_name] = field_data

            # Add extra fields from current spec if keeping them
            if keep_extra_fields:
                for field_name, field_data in current_fields.items():
                    if field_name not in new_template_fields:
                        updated_fields[field_name] = field_data

            # Update spec
            spec["fields"] = updated_fields
            spec["type"] = template_type
            spec["template_version"] = new_template.get("version")
            spec["updated_at"] = datetime.now()
            spec["status"] = SpecStatus.EVOLVED

            # Validate the updated spec
            validation_errors = self._validate_spec(spec)
            spec["validation_errors"] = validation_errors

            if not validation_errors:
                spec["status"] = SpecStatus.VALIDATED

            # Update in storage
            async with self._lock:
                self._specs[spec_id] = spec

                if self._storage_repository:
                    await self._storage_repository.update_spec(spec)

            return spec
        except Exception as e:
            self.logger.error(f"Error evolving spec {spec_id}: {e}", exc_info=True)
            raise

    async def generate_template_from_spec(
        self, spec_id: str, template_name: Optional[str] = None, template_version: str = "1.0"
    ) -> Dict[str, Any]:
        """
        Generate a new template from an existing spec.

        This can be useful for creating specialized templates based on
        real-world usage patterns.

        Args:
            spec_id: ID of the spec to use as a basis
            template_name: Name for the new template (default: spec's type with "-custom" suffix)
            template_version: Version for the new template

        Returns:
            Created template
        """
        try:
            # Get the spec
            spec = await self.get_spec(spec_id)
            if not spec:
                raise ValueError(f"Spec with ID {spec_id} not found")

            # Generate template name if not provided
            if not template_name:
                template_name = f"{spec.get('type')}-custom"

            # Create template from spec's fields
            template_fields = {}
            for field_name, field_data in spec.get("fields", {}).items():
                # Copy field definition, but not the value
                template_fields[field_name] = {**field_data, "value": None}  # Clear value

            # Create template
            template = {
                "type": template_name,
                "version": template_version,
                "name": f"Custom template based on {spec.get('id')}",
                "description": f"Automatically generated from spec {spec.get('id')}",
                "fields": template_fields,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "is_active": True,
            }

            # Store the template
            if self._storage_repository:
                await self._storage_repository.store_template(
                    template_name,
                    template_fields,
                    template_version,
                    {
                        "source_spec_id": spec_id,
                        "name": template.get("name"),
                        "description": template.get("description"),
                    },
                )

            return template
        except Exception as e:
            self.logger.error(f"Error generating template from spec {spec_id}: {e}", exc_info=True)
            raise

    async def validate_spec_data(self, spec_id: str) -> ValidationResult:
        """
        Validate a spec and return structured validation result.

        Args:
            spec_id: ID of the spec to validate

        Returns:
            ValidationResult with validation status and errors
        """
        try:
            spec = await self.get_spec(spec_id)
            if not spec:
                raise ValueError(f"Spec with ID {spec_id} not found")

            validation_errors = self._validate_spec(spec)

            is_valid = len(validation_errors) == 0

            # Update spec with validation results
            spec["validation_errors"] = validation_errors
            spec["status"] = SpecStatus.VALIDATED if is_valid else SpecStatus.VALIDATION_FAILED

            # Update in memory and storage
            async with self._lock:
                self._specs[spec_id] = spec
                if self._storage_repository:
                    await self._storage_repository.update_spec(spec)

            return ValidationResult(
                spec_id=spec_id, is_valid=is_valid, validation_errors=validation_errors
            )
        except Exception as e:
            self.logger.error(f"Error validating spec {spec_id}: {e}", exc_info=True)
            raise
