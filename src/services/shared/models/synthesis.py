"""
Code synthesis models and result representation.

This module defines models related to code synthesis, which is the process
of generating code from specifications. The central class is SynthesisResult,
which encapsulates the output of a code synthesis operation.

Classes:
    SynthesisResult: Result of a code synthesis operation
"""

from datetime import datetime
from datetime import timezone

# src/services/shared/models/synthesis.py
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


# Define strategy literals for better type safety
SynthesisStrategyType = Literal[
    "neural",
    "incremental",
    "sequential",
    "parallel",
    "conditional",
    "statistical",
    "hybrid",
    "combined",
    "incremental_sequential",
    "incremental_parallel",
    "incremental_conditional",
]


class SynthesisResult(BaseModel):
    """
    Represents the result of a code synthesis operation.

    This is the central class that encapsulates generated code, AST,
    confidence scores, performance metrics, and generation strategy.
    """

    # Core fields
    program_ast: Optional[Dict[str, Any]] = Field(
        default=None, description="Abstract Syntax Tree of the synthesized program"
    )
    code: Optional[str] = Field(
        default=None, description="String representation of the generated code"
    )
    # No alias here - we'll handle backwards compatibility differently
    confidence_score: float = Field(
        default=0.0, description="Confidence score of the synthesis result (0.0-1.0)"
    )
    time_taken: float = Field(default=0.0, description="Time taken for synthesis in seconds")

    # Strategy and metadata
    strategy: Optional[SynthesisStrategyType] = Field(
        default=None, description="Strategy used for synthesis"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the result was generated",
    )

    # Additional fields for enhanced functionality
    errors: List[str] = Field(
        default_factory=list, description="List of errors encountered during synthesis"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the synthesis process"
    )

    # Preserving fields that might be used by other components
    verification_result: Optional[Any] = Field(
        default=None, description="Verification result if available"
    )

    # Up-to-date model configuration
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        json_encoders={datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")},
        populate_by_name=True,  # For supporting various field names
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary with JSON-compatible types."""
        result = self.model_dump()

        # Handle datetime conversion
        if self.timestamp:
            result["timestamp"] = self.timestamp.isoformat().replace("+00:00", "Z")

        # Add confidence field for backward compatibility
        result["confidence"] = result["confidence_score"]

        return result

    # Handle backward compatibility for confidence at the model level
    @property
    def confidence(self) -> float:
        """Alias for confidence_score for backward compatibility."""
        return self.confidence_score

    # Handle backward compatibility for ast at the model level
    @property
    def ast(self) -> Optional[Dict[str, Any]]:
        """Alternative access for AST (aliases to program_ast)."""
        return self.program_ast

    # Method to create an instance with backward compatibility mappings
    @classmethod
    def create(cls, **data):
        """Create a SynthesisResult with backward compatibility for field names."""
        if "confidence" in data and "confidence_score" not in data:
            data["confidence_score"] = data.pop("confidence")
        if "ast" in data and "program_ast" not in data:
            data["program_ast"] = data.pop("ast")
        return cls(**data)

    def combine(self, component_results: List["SynthesisResult"]) -> "SynthesisResult":
        """
        Combine this result with other component results into a new combined result.

        Args:
            component_results: Results from other synthesis components

        Returns:
            A new SynthesisResult representing the combined solution
        """
        if not component_results:
            return self

        all_results = [self] + component_results
        valid_results = [r for r in all_results if r is not None]

        if not valid_results:
            # This would only happen if self is None AND all component_results are None
            return SynthesisResult()

        avg_confidence = sum(r.confidence_score for r in valid_results) / len(valid_results)
        total_time = sum(r.time_taken for r in valid_results)

        # Use proper dictionary construction
        combined_metadata = dict(self.metadata)
        for result in component_results:
            if result and result.metadata:
                combined_metadata.update(result.metadata)

        return SynthesisResult(
            program_ast=self.program_ast,  # Use primary result's AST
            code=self.code,
            confidence_score=avg_confidence,
            time_taken=total_time,
            strategy="combined",
            metadata=combined_metadata,
        )

    def with_code(self, code: str) -> "SynthesisResult":
        """
        Create a new SynthesisResult with the provided code.

        Args:
            code: The generated code as a string

        Returns:
            A new SynthesisResult with updated code
        """
        return SynthesisResult(
            program_ast=self.program_ast,
            code=code,
            confidence_score=self.confidence_score,
            time_taken=self.time_taken,
            strategy=self.strategy,
            metadata=dict(self.metadata),
            errors=list(self.errors),
        )
