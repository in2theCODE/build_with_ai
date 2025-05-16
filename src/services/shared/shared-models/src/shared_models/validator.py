#!/usr/bin/env python3
"""
Validation utility for the Program Synthesis System.

This module provides comprehensive validation tools for inputs, outputs,
data structures, and program behavior in the context of program synthesis.
"""

from functools import wraps
import inspect
from pathlib import Path
import re
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    Type,
    TypeVar,
    Union,
)


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import advanced logger
from src.services.shared.loggerService.loggingService import get_logger


# Setup logger
logger = get_logger(__name__)

# Type variables for generics
T = TypeVar("T")
R = TypeVar("R")


class ValidationResult:
    """Result of a validation operation."""

    def __init__(self, valid: bool = True, errors: Optional[List[str]] = None):
        """
        Initialize the validation result.

        Args:
            valid: Whether the validation passed.
            errors: List of validation error messages.
        """
        self.valid = valid
        self.errors = errors or []

    def __bool__(self) -> bool:
        """Convert to boolean (True if valid)."""
        return self.valid

    def add_error(self, error: str) -> None:
        """
        Add an error message.

        Args:
            error: Error message.
        """
        self.errors.append(error)
        self.valid = False

    def add_errors(self, errors: List[str]) -> None:
        """
        Add multiple error messages.

        Args:
            errors: List of error messages.
        """
        if errors:
            self.errors.extend(errors)
            self.valid = False

    def merge(self, other: "ValidationResult") -> None:
        """
        Merge with another validation result.

        Args:
            other: Other validation result.
        """
        if not other.valid:
            self.valid = False
            self.errors.extend(other.errors)

    @property
    def error_message(self) -> str:
        """Get a combined error message."""
        return "; ".join(self.errors)


class ValidationError(Exception):
    """Exception raised for validation errors."""

    def __init__(self, result: ValidationResult):
        """
        Initialize with a validation result.

        Args:
            result: Validation result.
        """
        self.result = result
        super().__init__(result.error_message)


class Validator:
    """Base class for validators."""

    def __init__(self, field_name: Optional[str] = None):
        """
        Initialize the validator.

        Args:
            field_name: Optional field name (for error messages).
        """
        self.field_name = field_name

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a value.

        Args:
            value: Value to validate.

        Returns:
            Validation result.
        """
        raise NotImplementedError("Subclasses must implement validate()")

    def __call__(self, value: Any) -> ValidationResult:
        """
        Call the validator.

        Args:
            value: Value to validate.

        Returns:
            Validation result.
        """
        return self.validate(value)


class TypeValidator(Validator):
    """Validates that a value is of the expected type."""

    def __init__(self, expected_type: Type, field_name: Optional[str] = None):
        """
        Initialize the validator.

        Args:
            expected_type: Expected type.
            field_name: Optional field name.
        """
        super().__init__(field_name)
        self.expected_type = expected_type

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate that a value is of the expected type.

        Args:
            value: Value to validate.

        Returns:
            Validation result.
        """
        result = ValidationResult()
        if not isinstance(value, self.expected_type):
            field_desc = f"{self.field_name} " if self.field_name else ""
            result.add_error(f"{field_desc}Expected type {self.expected_type.__name__}, got {type(value).__name__}")
        return result


class StringValidator(Validator):
    """Validates string values."""

    def __init__(
        self,
        min_length: int = 0,
        max_length: Optional[int] = None,
        pattern: Optional[Union[str, Pattern]] = None,
        field_name: Optional[str] = None,
    ):
        """
        Initialize the validator.

        Args:
            min_length: Minimum string length.
            max_length: Maximum string length (None for no limit).
            pattern: Regular expression pattern (string or compiled).
            field_name: Optional field name.
        """
        super().__init__(field_name)
        self.min_length = min_length
        self.max_length = max_length

        # Compile pattern if it's a string
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a string value.

        Args:
            value: Value to validate.

        Returns:
            Validation result.
        """
        result = ValidationResult()
        field_desc = f"{self.field_name} " if self.field_name else ""

        if not isinstance(value, str):
            result.add_error(f"{field_desc}Expected a string, got {type(value).__name__}")
            return result

        if len(value) < self.min_length:
            result.add_error(f"{field_desc}String is too short (minimum length: {self.min_length})")

        if self.max_length is not None and len(value) > self.max_length:
            result.add_error(f"{field_desc}String is too long (maximum length: {self.max_length})")

        if self.pattern and not self.pattern.match(value):
            result.add_error(f"{field_desc}String does not match the required pattern")

        return result


class NumberValidator(Validator):
    """Validates numeric values."""

    def __init__(
        self,
        min_: Union[int, float],
        max_: Optional[Union[int, float]] = None,
        field_name: Optional[str] = None,
    ):
        """
        Initialize the validator.

        Args:
            min_: Minimum value.
            max_: Maximum value (None for no limit).
            field_name: Optional field name.
        """
        super().__init__(field_name)
        self.min_ = min_
        self.max_ = max_

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a numeric value.

        Args:
            value: Value to validate.

        Returns:
            Validation result.
        """
        result = ValidationResult()
        field_desc = f"{self.field_name} " if self.field_name else ""

        if not isinstance(value, (int, float)):
            result.add_error(f"{field_desc}Expected a number, got {type(value).__name__}")
            return result

        if value < self.min_:
            result.add_error(f"{field_desc}Value {value} is less than minimum {self.min_}")

        if self.max_ is not None and value > self.max_:
            result.add_error(f"{field_desc}Value {value} is greater than maximum {self.max_}")

        return result


# Example usage of a decorator for function validation
def validate_input(validators: Dict[str, Validator]) -> Callable:
    """
    Decorator to validate function input arguments using specified validators.

    Args:
        validators: Dictionary mapping argument names to Validator instances.
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            errors = []
            for arg, validator in validators.items():
                value = bound_args.arguments.get(arg)
                result = validator.validate(value)
                if not result:
                    errors.append(f"Validation error in '{arg}': {result.error_message}")
            if errors:
                raise ValidationError(ValidationResult(valid=False, errors=errors))
            return func(*args, **kwargs)

        return wrapper

    return decorator


# If needed, you can add more specialized validators here

if __name__ == "__main__":
    # Example usage for testing the validators
    try:
        # Test TypeValidator
        type_validator = TypeValidator(int, "age")
        print("TypeValidator result for 25:", type_validator.validate(25).valid)
        print(
            "TypeValidator result for '25':",
            type_validator.validate("25").error_message,
        )

        # Test StringValidator
        string_validator = StringValidator(min_length=3, max_length=10, pattern=r"^[A-Za-z]+$", field_name="username")
        print(
            "StringValidator result for 'John':",
            string_validator.validate("John").valid,
        )
        print(
            "StringValidator result for 'Jo':",
            string_validator.validate("Jo").error_message,
        )
        print(
            "StringValidator result for 'John123':",
            string_validator.validate("John123").error_message,
        )

        # Test NumberValidator
        number_validator = NumberValidator(0, 100, field_name="score")
        print("NumberValidator result for 50:", number_validator.validate(50).valid)
        print(
            "NumberValidator result for -10:",
            number_validator.validate(-10).error_message,
        )
        print(
            "NumberValidator result for 150:",
            number_validator.validate(150).error_message,
        )
    except ValidationError as ve:
        logger.error("Validation failed: %s", ve)
