from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
import time
import json


class EvolutionSuggestionType(str, Enum):
    """Types of spec sheet definition evolution suggestions"""
    ADD_FIELD = "add_field"
    REMOVE_FIELD = "remove_field"
    MODIFY_FIELD = "modify_field"
    REORDER_FIELDS = "reorder_fields"
    ADD_SECTION = "add_section"
    REMOVE_SECTION = "remove_section"
    MODIFY_SECTION = "modify_section"
    REORDER_SECTIONS = "reorder_sections"
    ADD_VALIDATION = "add_validation"
    REMOVE_VALIDATION = "remove_validation"
    MODIFY_VALIDATION = "modify_validation"
    SPLIT_SPEC_SHEET_DEFINITION = "split_spec_sheet_definition"
    MERGE_SPEC_SHEET_DEFINITIONS = "merge_spec_sheet_definitions"


class EvolutionSuggestion(BaseModel):
    """A suggestion for evolving a spec sheet definition"""
    suggestion_id: str
    suggestion_type: EvolutionSuggestionType
    spec_sheet_definition_id: str
    spec_sheet_definition_version: str
    description: str
    confidence: float
    impact_score: float
    created_at: int = Field(default_factory=lambda: int(time.time()))
    changes: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    applied: bool = False
    applied_at: Optional[int] = None
    applied_version: Optional[str] = None

    # Pydantic provides to_dict functionality through model_dump()
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()

    # JSON serialization is built-in with model_dump_json()
    def to_json(self) -> str:
        """Convert to JSON string"""
        return self.model_dump_json(indent=2)

    # You can still have custom factory methods
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionSuggestion':
        """Create from dictionary"""
        # Handle legacy field names
        if "template_id" in data and "spec_sheet_definition_id" not in data:
            data["spec_sheet_definition_id"] = data.pop("template_id")
        if "template_version" in data and "spec_sheet_definition_version" not in data:
            data["spec_sheet_definition_version"] = data.pop("template_version")

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'EvolutionSuggestion':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class FieldEvolutionData(BaseModel):
    """Evolution data for a field"""
    field_path: str
    field_name: str
    section_name: str
    usage_count: int = 0
    completion_rate: float = 0.0
    error_rate: float = 0.0
    avg_fill_time: float = 0.0
    common_values: List[Any] = Field(default_factory=list)
    value_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    correlations: Dict[str, float] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldEvolutionData':
        """Create from dictionary"""
        return cls(**data)


class SectionEvolutionData(BaseModel):
    """Evolution data for a section"""
    section_name: str
    usage_count: int = 0
    completion_rate: float = 0.0
    error_rate: float = 0.0
    avg_fill_time: float = 0.0
    field_evolution: Dict[str, FieldEvolutionData] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SectionEvolutionData':
        """Create from dictionary"""
        field_evolution = {}
        for k, v in data.get("field_evolution", {}).items():
            field_evolution[k] = FieldEvolutionData.from_dict(v)

        return cls(
            section_name=data["section_name"],
            usage_count=data.get("usage_count", 0),
            completion_rate=data.get("completion_rate", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_fill_time=data.get("avg_fill_time", 0.0),
            field_evolution=field_evolution
        )


class SpecSheetDefinitionEvolutionAnalysis(BaseModel):
    """Comprehensive evolution analysis for a spec sheet definition"""
    spec_sheet_definition_id: str
    spec_sheet_definition_version: str
    analysis_timestamp: int = Field(default_factory=lambda: int(time.time()))
    total_instances: int = 0
    evolution_score: float = 0.0
    section_evolution: Dict[str, SectionEvolutionData] = Field(default_factory=dict)
    field_correlations: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    suggestions: List[EvolutionSuggestion] = Field(default_factory=list)
    common_patterns: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to JSON string"""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpecSheetDefinitionEvolutionAnalysis':
        """Create from dictionary"""
        # Handle legacy field names
        if "template_id" in data and "spec_sheet_definition_id" not in data:
            data["spec_sheet_definition_id"] = data.pop("template_id")
        if "template_version" in data and "spec_sheet_definition_version" not in data:
            data["spec_sheet_definition_version"] = data.pop("template_version")

        # Process nested objects
        section_evolution = {}
        for k, v in data.get("section_evolution", {}).items():
            section_evolution[k] = SectionEvolutionData.from_dict(v)

        suggestions = []
        for s in data.get("suggestions", []):
            suggestions.append(EvolutionSuggestion.from_dict(s))

        # Create the instance with processed data
        return cls(
            spec_sheet_definition_id=data["spec_sheet_definition_id"],
            spec_sheet_definition_version=data["spec_sheet_definition_version"],
            analysis_timestamp=data.get("analysis_timestamp", int(time.time())),
            total_instances=data.get("total_instances", 0),
            evolution_score=data.get("evolution_score", 0.0),
            section_evolution=section_evolution,
            field_correlations=data.get("field_correlations", {}),
            suggestions=suggestions,
            common_patterns=data.get("common_patterns", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'SpecSheetDefinitionEvolutionAnalysis':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)