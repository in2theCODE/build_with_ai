#!/usr/bin/env python3
"""
Type definitions for the constraint relaxation system.
"""

from datetime import timezone, datetime
import uuid
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from src.services.shared.models.base import BaseMessage


class VerificationResult(str, Enum):
    """Result of verification process."""
    COUNTEREXAMPLE_FOUND = "counterexample_found"
    VERIFIED = "verified"
    FALSIFIED = "falsified"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"


class FormalSpecification(BaseModel):
    """Represents a formal specification parsed from requirements."""
    ast: Any = Field(..., description="Abstract syntax tree or other internal representation")
    constraints: List[Any] = Field(..., description="List of constraints (e.g., Z3 expressions)")
    types: Dict[str, str] = Field(..., description="Type assignments for variables")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Input/output examples")

    model_config = ConfigDict(frozen=True)

    def is_decomposable(self) -> bool:
        """Check if this specification can be decomposed for incremental synthesis."""
        return len(self.constraints) > 3  # Simple heuristic

    def model_copy(self, **kwargs):
        """Create a deep copy of this specification with optional updates."""
        return self.model_copy(update=kwargs)


class VerificationReport(BaseModel):
    """Report from the verification process."""
    status: VerificationResult = Field(..., description="Result of verification")
    confidence: float = Field(..., description="Confidence level (0.0 to 1.0)")
    time_taken: float = Field(..., description="Time taken in seconds")
    counterexamples: List[Dict[str, Any]] = Field(default_factory=list, description="Found counterexamples")
    reason: Optional[str] = Field(None, description="Reason for failure if not verified")

    model_config = ConfigDict(frozen=True)



class ConstraintRelaxationRequest(BaseMessage):
    """Request for constraint relaxation."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request identifier")
    formal_spec: FormalSpecification = Field(..., description="Formal specification to relax")
    verification_result: Optional[VerificationReport] = Field(None, description="Verification result with counterexamples")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Request timestamp")
    max_constraints_to_remove: int = Field(default=None, description="Maximum number of constraints to remove")
    min_constraints_to_keep: int = Field(default=1, description="Minimum number of constraints to keep")
    strategy_preference: Optional[str] = Field(None, description="Preferred relaxation strategy")

    model_config = ConfigDict(frozen=True)


class ConstraintRelaxationResponse(BaseMessage):
    """Response with relaxed constraints."""
    request_id: str = Field(..., description="Original request identifier")
    success: bool = Field(..., description="Whether relaxation was successful")
    relaxed_spec: Optional[FormalSpecification] = Field(None, description="Relaxed specification")
    removed_constraints: List[Any] = Field(default_factory=list, description="Constraints that were removed")
    strategy_used: str = Field(default="unspecified", description="Relaxation strategy that was used")
    time_taken: float = Field(default=0.0, description="Time taken for relaxation")
    error: Optional[str] = Field(None, description="Error message if relaxation failed")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Response timestamp")

    model_config = ConfigDict(frozen=True)