#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spec Sheet Definition Registry - Usage Analytics Models

This module defines models for spec sheet definition usage analytics.
"""

import json
import time
from typing import Dict, List, Any, Optional, Set, Union

from pydantic import BaseModel, Field, ConfigDict


class FieldUsageStats(BaseModel):
    """Usage statistics for a field"""
    field_path: str  # Format: "section_name.field_name"
    section_name: str
    field_name: str
    completion_rate: float = 0.0  # Percentage of instances where field is filled
    error_rate: float = 0.0  # Percentage of instances where field has validation errors
    avg_fill_time: float = 0.0  # Average time to fill the field in seconds
    common_values: List[Any] = Field(default_factory=list)  # Most common values
    common_errors: List[str] = Field(default_factory=list)  # Most common validation errors

    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldUsageStats':
        """Create from dictionary"""
        return cls(
            field_path=data["field_path"],
            section_name=data["section_name"],
            field_name=data["field_name"],
            completion_rate=data.get("completion_rate", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_fill_time=data.get("avg_fill_time", 0.0),
            common_values=data.get("common_values", []),
            common_errors=data.get("common_errors", [])
        )


class SectionUsageStats(BaseModel):
    """Usage statistics for a section"""
    section_name: str
    completion_rate: float = 0.0  # Percentage of instances where section is filled
    error_rate: float = 0.0  # Percentage of instances where section has validation errors
    avg_fill_time: float = 0.0  # Average time to fill the section in seconds
    field_stats: Dict[str, FieldUsageStats] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = self.model_dump()
        result["field_stats"] = {k: v.to_dict() for k, v in self.field_stats.items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SectionUsageStats':
        """Create from dictionary"""
        field_stats = {}
        for k, v in data.get("field_stats", {}).items():
            field_stats[k] = FieldUsageStats.from_dict(v)

        return cls(
            section_name=data["section_name"],
            completion_rate=data.get("completion_rate", 0.0),
            error_rate=data.get("error_rate", 0.0),
            avg_fill_time=data.get("avg_fill_time", 0.0),
            field_stats=field_stats
        )


class CompletionPathStats(BaseModel):
    """Statistics about the path users take to complete a spec sheet definition"""
    total_instances: int = 0
    avg_completion_time: float = 0.0
    section_order: List[str] = Field(default_factory=list)  # Most common section fill order
    field_order: List[str] = Field(default_factory=list)  # Most common field fill order
    common_start_sections: List[str] = Field(default_factory=list)
    common_end_sections: List[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompletionPathStats':
        """Create from dictionary"""
        return cls(
            total_instances=data.get("total_instances", 0),
            avg_completion_time=data.get("avg_completion_time", 0.0),
            section_order=data.get("section_order", []),
            field_order=data.get("field_order", []),
            common_start_sections=data.get("common_start_sections", []),
            common_end_sections=data.get("common_end_sections", [])
        )


class SpecSheetDefinitionUsageAnalytics(BaseModel):
    """Comprehensive usage analytics for a spec sheet definition"""
    spec_sheet_definition_id: str
    spec_sheet_definition_version: str
    analysis_timestamp: int = Field(default_factory=lambda: int(time.time()))
    total_instances: int = 0
    completed_instances: int = 0
    completion_rate: float = 0.0
    avg_completion_time: float = 0.0
    validation_success_rate: float = 0.0
    section_stats: Dict[str, SectionUsageStats] = Field(default_factory=dict)
    completion_path: CompletionPathStats = Field(default_factory=CompletionPathStats)
    user_segments: Dict[str, Any] = Field(default_factory=dict)
    generation_stats: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "spec_sheet_definition_id": self.spec_sheet_definition_id,
            "spec_sheet_definition_version": self.spec_sheet_definition_version,
            "analysis_timestamp": self.analysis_timestamp,
            "total_instances": self.total_instances,
            "completed_instances": self.completed_instances,
            "completion_rate": self.completion_rate,
            "avg_completion_time": self.avg_completion_time,
            "validation_success_rate": self.validation_success_rate,
            "section_stats": {k: v.to_dict() for k, v in self.section_stats.items()},
            "completion_path": self.completion_path.to_dict(),
            "user_segments": self.user_segments,
            "generation_stats": self.generation_stats
        }
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpecSheetDefinitionUsageAnalytics':
        """Create from dictionary"""
        # Handle legacy field names for backward compatibility
        if "template_id" in data and "spec_sheet_definition_id" not in data:
            data["spec_sheet_definition_id"] = data.pop("template_id")
        if "template_version" in data and "spec_sheet_definition_version" not in data:
            data["spec_sheet_definition_version"] = data.pop("template_version")

        section_stats = {}
        for k, v in data.get("section_stats", {}).items():
            section_stats[k] = SectionUsageStats.from_dict(v)

        completion_path = CompletionPathStats.from_dict(
            data.get("completion_path", {})
        )

        return cls(
            spec_sheet_definition_id=data["spec_sheet_definition_id"],
            spec_sheet_definition_version=data["spec_sheet_definition_version"],
            analysis_timestamp=data.get("analysis_timestamp", int(time.time())),
            total_instances=data.get("total_instances", 0),
            completed_instances=data.get("completed_instances", 0),
            completion_rate=data.get("completion_rate", 0.0),
            avg_completion_time=data.get("avg_completion_time", 0.0),
            validation_success_rate=data.get("validation_success_rate", 0.0),
            section_stats=section_stats,
            completion_path=completion_path,
            user_segments=data.get("user_segments", {}),
            generation_stats=data.get("generation_stats", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'SpecSheetDefinitionUsageAnalytics':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))