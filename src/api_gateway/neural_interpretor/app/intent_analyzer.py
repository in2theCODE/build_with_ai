from datetime import datetime
from enum import Enum
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from src.services.shared.monitoring.embedding_client import EmbeddingClient
from src.services.shared.monitoring.metrics import track_inference_time
from uuid6 import uuid7

from .pattern_matcher import PatternMatcher


logger = logging.getLogger(__name__)


class ProcessingMode(str, Enum):
    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"


class QueryIntent(BaseModel):
    query_id: str
    embedding: List[float]
    processing_mode: ProcessingMode
    fast_path_match: Optional[Dict[str, Any]] = None
    complexity_score: float
    estimated_processing_time: float
    created_at: datetime = datetime.utcnow()


class IntentAnalyzer:
    def __init__(
        self,
        config: Dict[str, Any],
        embedding_client: EmbeddingClient,
        pattern_matcher: PatternMatcher,
    ):
        self.config = config
        self.embedding_client = embedding_client
        self.pattern_matcher = pattern_matcher
        self.complexity_threshold = config["intent_analysis"]["complexity_threshold"]

    @track_inference_time
    async def analyze_query(
        self,
        query: str,
        system_message: Optional[str] = None,
        require_delegation: bool = False,
    ) -> QueryIntent:
        """
        Analyze a query to determine its intent and complexity
        """
        # Extract embedding
        embedding = await self.embedding_client.embed_text(query)

        # Check for fast path patterns
        fast_path_match = await self.pattern_matcher.match(
            query_text=query,
            embedding=embedding,
            threshold=self.config["intent_analysis"].get("fast_path_confidence_threshold", 0.85),
        )

        # Detect complexity signals
        complexity_signals = {
            "token_count": len(query.split()),
            "question_marks": query.count("?"),
            "complexity_keywords": self._count_complexity_keywords(query),
            "conversational_markers": self._detect_conversational_markers(query),
            "explicit_delegation": require_delegation,
        }

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(complexity_signals)

        # Determine processing mode
        if require_delegation:
            processing_mode = ProcessingMode.DELIBERATIVE
            logger.info(f"Query {query[:50]}... delegated to deliberative mode by request")
        elif fast_path_match and fast_path_match["score"] > self.config["intent_analysis"].get(
            "fast_path_confidence_threshold", 0.85
        ):
            processing_mode = ProcessingMode.REACTIVE
            logger.info(
                f"Query {query[:50]}... matched pattern {fast_path_match['id']} with score {fast_path_match['score']}"
            )
        else:
            # Use complexity scoring
            processing_mode = (
                ProcessingMode.DELIBERATIVE if complexity_score > self.complexity_threshold else ProcessingMode.REACTIVE
            )
            logger.info(
                f"Query {query[:50]}... assigned to {processing_mode} mode with complexity {complexity_score:.2f}"
            )

        # Estimate processing time
        estimated_time = self._estimate_processing_time(processing_mode, complexity_signals)

        return QueryIntent(
            query_id=str(uuid7()),
            embedding=embedding,
            processing_mode=processing_mode,
            fast_path_match=fast_path_match,
            complexity_score=complexity_score,
            estimated_processing_time=estimated_time,
        )

    def _count_complexity_keywords(self, text: str) -> int:
        """Count words indicating complex reasoning"""
        keywords = [
            "explain",
            "analyze",
            "compare",
            "contrast",
            "evaluate",
            "synthesize",
            "interpret",
            "infer",
            "deduce",
            "hypothesize",
            "why",
            "how",
            "implications",
            "consequences",
            "trade-offs",
            "pros and cons",
            "advantages",
            "disadvantages",
            "alternatives",
        ]
        count = 0
        text_lower = text.lower()
        for word in keywords:
            if word in text_lower:
                count += 1
        return count

    def _detect_conversational_markers(self, text: str) -> int:
        """Detect markers of conversational queries"""
        markers = [
            "please",
            "could you",
            "would you",
            "I want",
            "I need",
            "help me",
            "can you",
            "tell me",
            "show me",
        ]
        count = 0
        text_lower = text.lower()
        for marker in markers:
            if marker in text_lower:
                count += 1
        return count

    def _calculate_complexity_score(self, signals: Dict[str, Any]) -> float:
        """Calculate query complexity score from signals"""
        weights = {
            "token_count": 0.2,
            "question_marks": 0.1,
            "complexity_keywords": 0.4,
            "conversational_markers": 0.1,
            "explicit_delegation": 0.2,
        }

        # Normalize token count (>200 tokens is complex)
        norm_token_count = min(
            signals["token_count"] / self.config["intent_analysis"].get("max_token_threshold", 200),
            1.0,
        )

        # Calculate weighted score
        score = (
            weights["token_count"] * norm_token_count
            + weights["question_marks"] * min(signals["question_marks"] / 3, 1.0)
            + weights["complexity_keywords"] * min(signals["complexity_keywords"] / 3, 1.0)
            + weights["conversational_markers"] * min(signals["conversational_markers"] / 3, 1.0)
            + weights["explicit_delegation"] * (1.0 if signals["explicit_delegation"] else 0.0)
        )

        return score

    def _estimate_processing_time(self, mode: ProcessingMode, signals: Dict[str, Any]) -> float:
        """Estimate processing time in seconds"""
        if mode == ProcessingMode.REACTIVE:
            # Fast path - base time + token factor
            return 1.0 + (signals["token_count"] * 0.01)
        else:
            # Complex path - much longer
            base_time = 5.0
            complexity_factor = 1.0 + (signals["complexity_keywords"] * 2.0)
            token_factor = signals["token_count"] * 0.02
            return base_time + (complexity_factor * token_factor)
