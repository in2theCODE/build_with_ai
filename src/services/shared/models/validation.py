from enum import Enum
from typing import List
from pydantic import BaseModel, Field,  ConfigDict


class ValidationResult(BaseModel):
    """Result of a validation operation."""
    valid: bool = Field(True, description="Whether the validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation error messages")

    model_config = ConfigDict(
        frozen=True
    )

    @property
    def error_message(self) -> str:
        """Get a combined error message."""
        return "; ".join(self.errors)

    @classmethod
    def with_error(cls, error: str) -> 'ValidationResult':
        """Factory method to create a validation result with an error."""
        return cls(valid=False, errors=[error])

    @classmethod
    def merge(cls, results: List['ValidationResult']) -> 'ValidationResult':
        """Factory method to merge multiple validation results."""
        valid = all(result.valid for result in results)
        errors = []
        for result in results:
            errors.extend(result.errors)
        return cls(valid=valid, errors=errors)


class HealthStatus(str, Enum):
    """Health status constants."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"