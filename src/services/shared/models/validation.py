"""
Validation result models for data validation operations.

This module defines models related to validation operations in the system.
The central class is ValidationResult, which encapsulates the result of a
validation operation with success/failure status and error messages.

Classes:
    ValidationResult: Result of a validation operation
"""

from typing import List

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ValidationResult(BaseModel):
    """Result of a validation operation."""

    valid: bool = Field(True, description="Whether the validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation error messages")

    model_config = ConfigDict(frozen=True)

    @property
    def error_message(self) -> str:
        """Get a combined error message."""
        return "; ".join(self.errors)

    @classmethod
    def with_error(cls, error: str) -> "ValidationResult":
        """Factory method to create a validation result with an error."""
        return cls(valid=False, errors=[error])

    @classmethod
    def merge(cls, results: List["ValidationResult"]) -> "ValidationResult":
        """Factory method to merge multiple validation results."""
        valid = all(result.valid for result in results)
        errors = []
        for result in results:
            errors.extend(result.errors)
        return cls(valid=valid, errors=errors)
