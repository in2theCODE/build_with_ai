import asyncio
import json
import re
import time
import uuid
from datetime import datetime

from src.services.shared.logging.logger import get_logger

logger = get_logger(__name__)


class SpecRegistry:
    """
    Registry for managing specification sheets.

    Stores, retrieves, and manages spec sheets that define requirements
    for code generation. Implements constraint validation.
    """
Breakfast
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
            "string": self._validate_string,
            "text": self._validate_string,
            "int": self._validate_int,
            "float": self._validate_float,
            "boolean": self._validate_boolean,
            "list": self._validate_list,
            "json": self._validate_json,
            "code": self._validate_code,
        }

        # Constraint patterns with validators
        self._constraint_validators = [
            (re.compile(r'required'), self._validate_required),
            (re.compile(r'min_length\((\d+)\)'), self._validate_min_length),
            (re.compile(r'max_length\((\d+)\)'), self._validate_max_length),
            (re.compile(r'min_value\(([^)]+)\)'), self._validate_min_value),
            (re.compile(r'max_value\(([^)]+)\)'), self._validate_max_value),
            (re.compile(r'pattern\(([^)]+)\)'), self._validate_pattern),
        ]

    async def create_empty_spec(self, spec_type, project_id=None):
        """Create an empty spec sheet of the specified type."""
        spec_id = str(uuid.uuid4())
        spec = {
            "id": spec_id,
            "type": spec_type,
            "project_id": project_id,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "fields": self._get_fields_for_spec_type(spec_type),
            "status": "empty",
            "validation_errors": []
        }

        # Store spec
        async with self._lock:
            self._specs[spec_id] = spec
            if self._storage_repository:
                await self._storage_repository.store_spec(spec)

        return spec

    def _get_fields_for_spec_type(self, spec_type):
        """Get the field definitions for a spec type."""
        # Field definitions for different spec types with enhanced validation
        field_definitions = {
            "container": [
                {
                    "name": "container_name",
                    "type": "string",
                    "required": True,
                    "constraints": ["min_length(3)", "max_length(50)"]
                },
                {
                    "name": "description",
                    "type": "text",
                    "required": True,
                    "constraints": ["min_length(10)"]
                },
                {
                    "name": "dependencies",
                    "type": "list",
                    "required": False,
                    "constraints": []
                },
                {
                    "name": "event_handlers",
                    "type": "list",
                    "required": True,
                    "constraints": []
                },
                {
                    "name": "event_bus_config",
                    "type": "json",
                    "required": True,
                    "constraints": []
                },
                {
                    "name": "main_logic",
                    "type": "code",
                    "required": True,
                    "constraints": ["min_length(5)"]
                },
            ],
            "api": [
                {
                    "name": "api_name",
                    "type": "string",
                    "required": True,
                    "constraints": ["min_length(3)", "max_length(50)"]
                },
                {
                    "name": "endpoints",
                    "type": "list",
                    "required": True,
                    "constraints": []
                },
                {
                    "name": "auth_method",
                    "type": "string",
                    "required": False,
                    "constraints": []
                },
                {
                    "name": "documentation",
                    "type": "text",
                    "required": True,
                    "constraints": ["min_length(10)"]
                }
            ],
            "database": [
                {
                    "name": "db_name",
                    "type": "string",
                    "required": True,
                    "constraints": ["min_length(3)", "max_length(50)"]
                },
                {
                    "name": "db_type",
                    "type": "string",
                    "required": True,
                    "constraints": []
                },
                {
                    "name": "tables",
                    "type": "json",
                    "required": True,
                    "constraints": []
                },
                {
                    "name": "relationships",
                    "type": "json",
                    "required": False,
                    "constraints": []
                }
            ]
        }

        # If spec_type not found, return empty field definitions
        if spec_type not in field_definitions:
            self.logger.warning(f"No field definitions found for spec type: {spec_type}")
            return {}

        # Convert to the format expected by the spec system
        return {field["name"]: {
            "value": None,
            "type": field["type"],
            "required": field["required"],
            "constraints": field["constraints"]
        } for field in field_definitions[spec_type]}

    async def update_spec(self, spec_id, field_updates):
        """Update fields in a spec sheet."""
        async with self._lock:
            if spec_id not in self._specs:
                raise ValueError(f"Spec with ID {spec_id} not found")

            spec = self._specs[spec_id]

            for field_name, value in field_updates.items():
                if field_name in spec["fields"]:
                    spec["fields"][field_name]["value"] = value

            spec["updated_at"] = datetime.now().timestamp()
            spec["status"] = "updated"

            # Validate the spec after updates
            validation_errors = self._validate_spec(spec)
            spec["validation_errors"] = validation_errors

            if not validation_errors:
                spec["status"] = "validated"

            if self._storage_repository:
                await self._storage_repository.update_spec(spec)

            return spec

    def _validate_spec(self, spec):
        """Validate all fields in a spec sheet."""
        validation_errors = []

        for field_name, field_data in spec["fields"].items():
            # Skip validation for empty fields that aren't required
            if field_data["value"] is None or field_data["value"] == "":
                if field_data.get("required", False):
                    validation_errors.append(f"Field '{field_name}' is required but missing")
                continue

            # Validate type
            field_type = field_data.get("type", "string")
            type_validator = self._type_validators.get(field_type)
            if type_validator:
                error = type_validator(field_name, field_data["value"])
                if error:
                    validation_errors.append(error)

            # Validate constraints
            for constraint in field_data.get("constraints", []):
                for pattern, validator in self._constraint_validators:
                    match = pattern.match(constraint)
                    if match:
                        error = validator(field_name, field_data["value"], match)
                        if error:
                            validation_errors.append(error)
                        break

        return validation_errors

    # Validation functions
    def _validate_required(self, field_name, value, match):
        """Validate that a required field has a value."""
        if value is None or value == "":
            return f"Field '{field_name}' is required"
        return None

    def _validate_min_length(self, field_name, value, match):
        """Validate minimum length."""
        min_length = int(match.group(1))
        if isinstance(value, str) and len(value) < min_length:
            return f"Field '{field_name}' must be at least {min_length} characters long"
        if isinstance(value, list) and len(value) < min_length:
            return f"Field '{field_name}' must have at least {min_length} items"
        return None

    def _validate_max_length(self, field_name, value, match):
        """Validate maximum length."""
        max_length = int(match.group(1))
        if isinstance(value, str) and len(value) > max_length:
            return f"Field '{field_name}' must not exceed {max_length} characters"
        if isinstance(value, list) and len(value) > max_length:
            return f"Field '{field_name}' must not have more than {max_length} items"
        return None

    def _validate_min_value(self, field_name, value, match):
        """Validate minimum value."""
        min_value = float(match.group(1))
        if isinstance(value, (int, float)) and value < min_value:
            return f"Field '{field_name}' must be at least {min_value}"
        return None

    def _validate_max_value(self, field_name, value, match):
        """Validate maximum value."""
        max_value = float(match.group(1))
        if isinstance(value, (int, float)) and value > max_value:
            return f"Field '{field_name}' must not exceed {max_value}"
        return None

    def _validate_pattern(self, field_name, value, match):
        """Validate against a regex pattern."""
        pattern = match.group(1)
        if isinstance(value, str) and not re.match(pattern, value):
            return f"Field '{field_name}' does not match the required pattern"
        return None

    # Type validators
    def _validate_string(self, field_name, value):
        """Validate string type."""
        if not isinstance(value, str):
            return f"Field '{field_name}' must be a string"
        return None

    def _validate_int(self, field_name, value):
        """Validate integer type."""
        if not isinstance(value, int) or isinstance(value, bool):
            return f"Field '{field_name}' must be an integer"
        return None

    def _validate_float(self, field_name, value):
        """Validate float type."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"Field '{field_name}' must be a number"
        return None

    def _validate_boolean(self, field_name, value):
        """Validate boolean type."""
        if not isinstance(value, bool):
            return f"Field '{field_name}' must be a boolean"
        return None

    def _validate_list(self, field_name, value):
        """Validate list type."""
        if not isinstance(value, list):
            return f"Field '{field_name}' must be a list"
        return None

    def _validate_json(self, field_name, value):
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

    def _validate_code(self, field_name, value):
        """Validate code."""
        if not isinstance(value, str):
            return f"Field '{field_name}' must be a string containing code"
        return None

    async def get_spec(self, spec_id):
        """Get a spec by ID."""
        if spec_id in self._specs:
            return self._specs[spec_id]

        if self._storage_repository:
            spec = await self._storage_repository.get_spec(spec_id)
            if spec:
                self._specs[spec_id] = spec
                return spec

        return None

    async def list_specs(self, project_id=None, spec_type=None):
        """List specs, optionally filtered by project ID or type."""
        specs = list(self._specs.values())

        if project_id:
            specs = [s for s in specs if s["project_id"] == project_id]

        if spec_type:
            specs = [s for s in specs if s["type"] == spec_type]

        return specs

    async def delete_spec(self, spec_id):
        """Delete a spec sheet."""
        async with self._lock:
            if spec_id not in self._specs:
                return False

            del self._specs[spec_id]

            if self._storage_repository:
                await self._storage_repository.delete_spec(spec_id)

            return True

    async def add_relation(self, spec_id, related_spec_id, relation_type="depends_on"):
        """Add a relation between two spec sheets."""
        async with self._lock:
            if spec_id not in self._specs or related_spec_id not in self._specs:
                return False

            if spec_id not in self._spec_relations:
                self._spec_relations[spec_id] = []

            # Add relation if it doesn't exist already
            relation = {"spec_id": related_spec_id, "type": relation_type}
            if relation not in self._spec_relations[spec_id]:
                self._spec_relations[spec_id].append(relation)

            if self._storage_repository:
                await self._storage_repository.store_spec_relation(spec_id, related_spec_id, relation_type)

            return True

    async def get_related_specs(self, spec_id):
        """Get specs related to the given spec."""
        if spec_id not in self._spec_relations:
            return []

        related_specs = []
        for relation in self._spec_relations[spec_id]:
            related_spec = await self.get_spec(relation["spec_id"])
            if related_spec:
                related_specs.append({
                    "spec": related_spec,
                    "relation_type": relation["type"]
                })

        return related_specs