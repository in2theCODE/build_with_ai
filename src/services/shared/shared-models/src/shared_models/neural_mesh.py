"""
Context app for the Neural Context Mesh.

This module defines the app for context nodes, node states,
and context-related data structures used throughout the mesh.
"""

from .enums import ContextType, ActivationFunction, SynapseState
from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field

from .enums import EvolutionMechanism


class ContextNode(BaseModel):
    """Base model for all context nodes in the mesh."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique node identifier")
    context_type: ContextType = Field(..., description="Type of context node")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    activation_value: float = Field(default=0.0, description="Current activation value")
    activation_function: ActivationFunction = Field(
        default=ActivationFunction.SIGMOID, description="Activation function used by this node"
    )
    active: bool = Field(default=True, description="Whether this node is active")
    embedding_dimension: Optional[int] = Field(
        default=None, description="Dimension of the embedding vector if applicable"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        extra = "allow"


class CodePatternNode(ContextNode):
    """Context node specific to code patterns."""

    context_type: ContextType = ContextType.CODE_PATTERN
    pattern_code: str = Field(..., description="The code pattern")
    language: Optional[str] = Field(default=None, description="Programming language")
    embedding: Optional[List[float]] = Field(default=None, description="Pattern embedding vector")
    tree_structure: Optional[Dict[str, Any]] = Field(default=None, description="AST or tree structure of the pattern")
    usage_count: int = Field(default=0, description="Number of times this pattern was used")
    success_rate: float = Field(default=0.0, description="Success rate when used")


class KnowledgeNode(ContextNode):
    """Context node specific to knowledge items."""

    context_type: ContextType = ContextType.KNOWLEDGE
    content: str = Field(..., description="Knowledge content")
    content_type: str = Field(..., description="Type of content (code, documentation, etc.)")
    embedding: Optional[List[float]] = Field(default=None, description="Content embedding vector")
    source: Optional[str] = Field(default=None, description="Source of the knowledge")
    relevance_score: float = Field(default=1.0, description="Relevance score")
    access_count: int = Field(default=0, description="Number of times accessed")


class MetricNode(ContextNode):
    """Context node specific to metrics."""

    context_type: ContextType = ContextType.METRICS
    metric_name: str = Field(..., description="Name of the metric")
    metric_value: float = Field(..., description="Current value of the metric")
    metric_history: List[Dict[str, Any]] = Field(default_factory=list, description="Historical values")
    aggregation_interval: Optional[int] = Field(default=None, description="Aggregation interval in seconds")
    thresholds: Dict[str, float] = Field(default_factory=dict, description="Alert thresholds for this metric")


class EvolutionNode(ContextNode):
    """Context node specific to evolution tracking."""

    context_type: ContextType = ContextType.EVOLUTION
    template_id: str = Field(..., description="ID of the template being evolved")
    parent_ids: List[str] = Field(default_factory=list, description="IDs of parent templates")
    generation: int = Field(default=0, description="Generation number")
    fitness_score: float = Field(default=0.0, description="Current fitness score")
    mutation_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of mutations")
    embedding: Optional[List[float]] = Field(default=None, description="Embedding vector for similarity comparison")


class GlobalContextState(BaseModel):
    """Global state of the context mesh."""

    active_nodes: int = Field(default=0, description="Number of active nodes")
    active_synapses: int = Field(default=0, description="Number of active synapses")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    context_types_count: Dict[str, int] = Field(default_factory=dict, description="Count of active nodes by type")
    health_status: str = Field(default="healthy", description="Overall health status")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Key metrics")

    class Config:
        extra = "allow"


class ContextActivation(BaseModel):
    """Model for context activation events."""

    node_id: str = Field(..., description="ID of the activated node")
    activation_value: float = Field(..., description="Activation value")
    timestamp: datetime = Field(default_factory=datetime.now, description="Activation timestamp")
    trigger_type: str = Field(..., description="What triggered the activation")
    context_vector: Optional[List[float]] = Field(default=None, description="Context vector if applicable")
    propagation_path: List[str] = Field(default_factory=list, description="Path of propagation")

    # synapse model
    """
    Synapse app for the Neural Context Mesh.

    This module defines the app for synapses (connections between nodes),
    synapse states, and related data structures used throughout the mesh.
    """


class Synapse(BaseModel):
    """Model for synapses (connections between context nodes)."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique synapse identifier")
    from_node_id: str = Field(..., description="Source node ID")
    to_node_id: str = Field(..., description="Target node ID")
    weight: float = Field(default=0.5, description="Connection weight")
    state: SynapseState = Field(default=SynapseState.FORMING, description="Current state")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    last_activated: Optional[datetime] = Field(default=None, description="Last activation timestamp")
    activation_count: int = Field(default=0, description="Number of activations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        extra = "allow"


class PathwaySegment(BaseModel):
    """Model for a segment of a pathway through the mesh."""

    synapse_id: str = Field(..., description="Synapse ID")
    from_node_id: str = Field(..., description="Source node ID")
    to_node_id: str = Field(..., description="Target node ID")
    weight: float = Field(..., description="Connection weight")
    traversal_time: float = Field(default=0.0, description="Estimated traversal time")


class Pathway(BaseModel):
    """Model for a pathway through the mesh."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique pathway identifier")
    segments: List[PathwaySegment] = Field(..., description="Segments in the pathway")
    start_node_id: str = Field(..., description="Starting node ID")
    end_node_id: str = Field(..., description="Ending node ID")
    total_weight: float = Field(..., description="Total pathway weight")
    total_traversal_time: float = Field(..., description="Total estimated traversal time")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    usage_count: int = Field(default=0, description="Number of times this pathway was used")

    @property
    def node_sequence(self) -> List[str]:
        """Get the sequence of nodes in this pathway."""
        sequence = [self.start_node_id]
        for segment in self.segments:
            sequence.append(segment.to_node_id)
        return sequence


class SynapseActivity(BaseModel):
    """Model for tracking synapse activity."""

    synapse_id: str = Field(..., description="Synapse ID")
    activation_time: datetime = Field(default_factory=datetime.now, description="Activation timestamp")
    presynaptic_value: float = Field(..., description="Presynaptic activation value")
    postsynaptic_value: float = Field(..., description="Postsynaptic activation value")
    weight_before: float = Field(..., description="Weight before activation")
    weight_after: Optional[float] = Field(default=None, description="Weight after activation")
    context_id: Optional[str] = Field(default=None, description="Context ID if applicable")


# evolution model
"""
Evolution app for the Neural Context Mesh.

This module defines the app for template evolution, 
fitness evaluation, and emergence detection used in the mesh.
"""


class Template(BaseModel):
    """Model for an evolvable template."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique template identifier")
    name: str = Field(..., description="Template name")
    content: str = Field(..., description="Template content")
    language: Optional[str] = Field(default=None, description="Programming language")
    category: str = Field(..., description="Template category")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    version: int = Field(default=1, description="Template version")
    parent_ids: List[str] = Field(default_factory=list, description="Parent template IDs")
    generation: int = Field(default=0, description="Generation number")
    fitness_score: float = Field(default=0.0, description="Current fitness score")
    usage_count: int = Field(default=0, description="Number of times used")
    success_count: int = Field(default=0, description="Number of successful uses")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        extra = "allow"


class EvolutionEvent(BaseModel):
    """Model for a template evolution event."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique event identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    mechanism: EvolutionMechanism = Field(..., description="Evolution mechanism")
    parent_template_ids: List[str] = Field(..., description="Parent template IDs")
    child_template_id: str = Field(..., description="Child template ID")
    description: str = Field(..., description="Description of the evolution")
    fitness_before: float = Field(..., description="Fitness before evolution")
    fitness_after: Optional[float] = Field(default=None, description="Fitness after evolution")
    changes_summary: str = Field(..., description="Summary of changes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EmergentPattern(BaseModel):
    """Model for an emergent pattern."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique pattern identifier")
    pattern_type: str = Field(..., description="Type of emergent pattern")
    component_ids: List[str] = Field(..., description="Component IDs that form this pattern")
    detection_time: datetime = Field(default_factory=datetime.now, description="When pattern was detected")
    confidence: float = Field(..., description="Detection confidence")
    occurrence_count: int = Field(default=1, description="Number of occurrences observed")
    description: str = Field(..., description="Description of the pattern")
    embedding: Optional[List[float]] = Field(default=None, description="Pattern embedding if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FitnessEvaluation(BaseModel):
    """Model for fitness evaluation results."""

    template_id: str = Field(..., description="Template ID")
    evaluation_time: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    fitness_score: float = Field(..., description="Calculated fitness score")
    evaluation_context: Dict[str, Any] = Field(default_factory=dict, description="Context of evaluation")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Component metrics")
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Identified weaknesses")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
