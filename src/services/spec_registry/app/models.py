# app/services/spec_registry/app/app.py
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
import uuid

from pydantic import BaseModel, Field, model_validator


# Field types for spec sheets
class FieldType(str, Enum):
    STRING = "string"
    TEXT = "text"
    INT = "int"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    JSON = "json"
    CODE = "code"
    DATETIME = "datetime"
    REFERENCE = "reference"


# Spec status
class SpecStatus(str, Enum):
    EMPTY = "empty"
    UPDATED = "updated"
    VALIDATED = "validated"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    GENERATING = "generating"
    GENERATED = "generated"
    EVOLVED = "evolved"
    VALIDATION_FAILED = "validation_failed"
    GENERATION_FAILED = "generation_failed"


# Base model for constraints
class FieldConstraint(BaseModel):
    type: str
    parameters: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


# Common constraints
class RequiredConstraint(FieldConstraint):
    type: Literal["required"] = "required"


class MinLengthConstraint(FieldConstraint):
    type: Literal["min_length"] = "min_length"
    parameters: Dict[str, int] = Field(..., example={"value": 3})


class MaxLengthConstraint(FieldConstraint):
    type: Literal["max_length"] = "max_length"
    parameters: Dict[str, int] = Field(..., example={"value": 50})


class PatternConstraint(FieldConstraint):
    type: Literal["pattern"] = "pattern"
    parameters: Dict[str, str] = Field(..., example={"pattern": "^[a-zA-Z0-9_-]+$"})


class MinValueConstraint(FieldConstraint):
    type: Literal["min_value"] = "min_value"
    parameters: Dict[str, Union[int, float]] = Field(..., example={"value": 0})


class MaxValueConstraint(FieldConstraint):
    type: Literal["max_value"] = "max_value"
    parameters: Dict[str, Union[int, float]] = Field(..., example={"value": 100})


# Field definition
class FieldDefinition(BaseModel):
    name: str
    type: FieldType
    label: Optional[str] = None
    description: Optional[str] = None
    required: bool = False
    default_value: Optional[Any] = None
    constraints: List[FieldConstraint] = Field(default_factory=list)
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    hidden: bool = False
    order: int = 0
    group: Optional[str] = None
    conditional: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Template model
class SpecTemplate(BaseModel):
    id: Optional[int] = None
    type: str
    version: str = "1.0"
    name: Optional[str] = None
    description: Optional[str] = None
    fields: List[FieldDefinition]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_field_names(self) -> "SpecTemplate":
        """Ensure that field names are unique."""
        field_names = set()
        for field in self.fields:
            if field.name in field_names:
                raise ValueError(f"Duplicate field name: {field.name}")
            field_names.add(field.name)
        return self

    def to_internal_format(self) -> Dict[str, Dict[str, Any]]:
        """Convert to the internal format used by the spec registry."""
        fields = {}
        for field in self.fields:
            fields[field.name] = {
                "value": field.default_value,
                "type": field.type.value if isinstance(field.type, FieldType) else field.type,
                "required": field.required,
                "constraints": [
                    # Convert constraints to strings
                    self._constraint_to_string(constraint)
                    for constraint in field.constraints
                ],
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
        return fields

    def _constraint_to_string(self, constraint: FieldConstraint) -> str:
        """Convert a constraint to its string representation."""
        if constraint.type == "required":
            return "required"
        elif constraint.type == "min_length":
            return f"min_length({constraint.parameters['value']})"
        elif constraint.type == "max_length":
            return f"max_length({constraint.parameters['value']})"
        elif constraint.type == "pattern":
            return f"pattern({constraint.parameters['pattern']})"
        elif constraint.type == "min_value":
            return f"min_value({constraint.parameters['value']})"
        elif constraint.type == "max_value":
            return f"max_value({constraint.parameters['value']})"
        else:
            return f"{constraint.type}({constraint.parameters})"

    @classmethod
    def from_internal_format(cls, data: Dict[str, Any]) -> "SpecTemplate":
        """Create a SpecTemplate from the internal format used by the spec registry."""
        fields = []
        for name, field_data in data.get("fields", {}).items():
            constraints = []
            for constraint_str in field_data.get("constraints", []):
                constraint = cls._string_to_constraint(constraint_str)
                if constraint:
                    constraints.append(constraint)

            fields.append(
                FieldDefinition(
                    name=name,
                    type=field_data.get("type", "string"),
                    label=field_data.get("label", name),
                    description=field_data.get("description", ""),
                    required=field_data.get("required", False),
                    default_value=field_data.get("value"),
                    constraints=constraints,
                    placeholder=field_data.get("placeholder"),
                    help_text=field_data.get("help_text"),
                    hidden=field_data.get("hidden", False),
                    order=field_data.get("order", 0),
                    group=field_data.get("group"),
                    conditional=field_data.get("conditional"),
                    metadata=field_data.get("metadata", {}),
                )
            )

        return cls(
            id=data.get("id"),
            type=data.get("type", ""),
            version=data.get("version", "1.0"),
            name=data.get("name"),
            description=data.get("description"),
            fields=fields,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            is_active=data.get("is_active", True),
            metadata={
                k: v
                for k, v in data.items()
                if k
                not in [
                    "id",
                    "type",
                    "version",
                    "name",
                    "description",
                    "fields",
                    "created_at",
                    "updated_at",
                    "is_active",
                ]
            },
        )

    @classmethod
    def _string_to_constraint(cls, constraint_str: str) -> Optional[FieldConstraint]:
        """Convert a constraint string to a FieldConstraint object."""
        if constraint_str == "required":
            return RequiredConstraint()
        elif constraint_str.startswith("min_length"):
            import re

            match = re.match(r"min_length\((\d+)\)", constraint_str)
            if match:
                return MinLengthConstraint(parameters={"value": int(match.group(1))})
        elif constraint_str.startswith("max_length"):
            import re

            match = re.match(r"max_length\((\d+)\)", constraint_str)
            if match:
                return MaxLengthConstraint(parameters={"value": int(match.group(1))})
        elif constraint_str.startswith("pattern"):
            import re

            match = re.match(r"pattern\(([^)]+)\)", constraint_str)
            if match:
                return PatternConstraint(parameters={"pattern": match.group(1)})
        elif constraint_str.startswith("min_value"):
            import re

            match = re.match(r"min_value\(([^)]+)\)", constraint_str)
            if match:
                try:
                    value = int(match.group(1))
                except ValueError:
                    value = float(match.group(1))
                return MinValueConstraint(parameters={"value": value})
        elif constraint_str.startswith("max_value"):
            import re

            match = re.match(r"max_value\(([^)]+)\)", constraint_str)
            if match:
                try:
                    value = int(match.group(1))
                except ValueError:
                    value = float(match.group(1))
                return MaxValueConstraint(parameters={"value": value})

        # If we can't parse it, return a generic constraint
        return FieldConstraint(type=constraint_str)


# Spec instance model
class Spec(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    project_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    fields: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    status: SpecStatus = SpecStatus.EMPTY
    validation_errors: List[str] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    template_version: Optional[str] = None
    generation_id: Optional[str] = None
    generation_error: Optional[str] = None
    generation_error_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Validation result model
class ValidationResult(BaseModel):
    spec_id: str
    is_valid: bool
    validation_errors: List[str] = Field(default_factory=list)
