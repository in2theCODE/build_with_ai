#!/usr/bin/env python3
"""
Feedback collection component for the Program Synthesis System.

This component records and processes feedback from synthesis failures and successes,
helping to improve future synthesis performance through machine learning.
"""

import datetime
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.shared.constants.base_component import BaseComponent
from src.services.shared.constants.models import SynthesisResult, VerificationReport
from src.services.shared.constants.enums import VerificationResult

try:
    # Optional dependencies for advanced features
    import numpy as np
    from sklearn.cluster import DBSCAN
    HAVE_ML_DEPS = True
except ImportError:
    HAVE_ML_DEPS = False


def _sanitize_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize context data to remove sensitive information."""
    if not context:
        return {}

    # Create a copy to avoid modifying the original
    sanitized = context.copy()

    # Remove sensitive fields
    sensitive_fields = ["api_key", "password", "token", "secret", "credential"]
    for field in list(sanitized.keys()):
        for sensitive in sensitive_fields:
            if sensitive in field.lower():
                sanitized[field] = "***REDACTED***"

    return sanitized


def _categorize_error(verification_result: VerificationReport) -> str:
    """Categorize the type of verification error."""
    # If there's an explicit reason, use it for categorization
    if verification_result.reason:
        reason = verification_result.reason.lower()

        if "timeout" in reason:
            return "timeout"
        elif "memory" in reason:
            return "memory_limit"
        elif "syntax" in reason:
            return "syntax_error"
        elif "type" in reason:
            return "type_error"
        elif "boundary" in reason or "range" in reason:
            return "boundary_error"
        elif "assertion" in reason:
            return "assertion_failure"
        elif "infinite" in reason or "loop" in reason:
            return "infinite_loop"
        elif "constraint" in reason:
            return "constraint_violation"

    # Otherwise categorize based on status and counterexamples
    if verification_result.status == VerificationResult.TIMEOUT:
        return "timeout"
    elif verification_result.status == VerificationResult.ERROR:
        return "system_error"
    elif verification_result.status == VerificationResult.COUNTEREXAMPLE_FOUND:
        if verification_result.counterexamples:
            # Could do more detailed analysis of counterexamples here
            return "logical_error"
        else:
            return "unspecified_failure"

    return "unknown"


def _check_type_violations(counterexamples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check for type-related violations in counterexamples."""
    # In a real implementation, this would analyze counterexamples for type issues
    # For demonstration, we'll return empty results
    return {}


def _extract_common_values(counterexamples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract common values from counterexamples that might indicate patterns."""
    # In a real implementation, this would analyze counterexamples to find patterns
    # For demonstration, we'll return empty results
    return {}


def _is_negative_feedback(feedback_data: Dict[str, Any]) -> bool:
    """Determine if user feedback is negative."""
    # Look for indicators of negative feedback
    if "rating" in feedback_data:
        # Assuming rating is 1-5 with 5 being best
        return feedback_data["rating"] <= 2

    if "sentiment" in feedback_data:
        return feedback_data["sentiment"].lower() in ["negative", "very negative"]

    if "satisfied" in feedback_data:
        return not feedback_data["satisfied"]

    # Check for negative keywords in comments
    if "comments" in feedback_data and isinstance(feedback_data["comments"], str):
        negative_keywords = ["wrong", "incorrect", "bad", "error", "not working", "useless", "terrible"]
        comments = feedback_data["comments"].lower()
        for keyword in negative_keywords:
            if keyword in comments:
                return True

    return False


def _check_boundary_violations(counterexamples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check for boundary condition violations in counterexamples."""
    # In a real implementation, this would analyze counterexamples for boundary issues
    # For demonstration, we'll return empty results
    return {}


def _extract_error_patterns(specification: str,
                            verification_result: VerificationReport) -> Dict[str, Any]:
    """Extract patterns from verification failures."""
    error_patterns = {
        "verification_status": verification_result.status.value,
        "error_category": _categorize_error(verification_result),
        "counterexample_count": len(verification_result.counterexamples),
    }

    # Add specific patterns based on counterexamples
    if verification_result.counterexamples:
        error_patterns["boundary_violations"] = _check_boundary_violations(verification_result.counterexamples)
        error_patterns["type_violations"] = _check_type_violations(verification_result.counterexamples)
        error_patterns["common_values"] = _extract_common_values(verification_result.counterexamples)

    return error_patterns


def _get_memory_usage() -> float:
    """Get the current memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # In MB
    except ImportError:
        return 0.0


def _get_system_load() -> float:
    """Get the current system load average."""
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return 0.0


def get_feedback_statistics() -> Dict[str, Any]:
    """
    Get statistics on collected feedback data.

    Returns:
        Dictionary with feedback statistics
    """
    stats = {
        "success_count": 0,
        "failure_count": 0,
        "user_feedback_count": 0,
        "avg_success_confidence": 0.0,
        "avg_time_taken": 0.0,
        "top_failure_reasons": [],
        "top_synthesis_strategies": [],
    }

    # In a real implementation, this would compute statistics from the stored data
    # For demonstration, we'll return empty statistics

    return stats


class FeedbackCollector(BaseComponent):
    """Collects and analyzes feedback from synthesis processes."""

    def __init__(self, **params):
        """Initialize the feedback collector with storage and analysis options."""
        super().__init__(**params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Storage parameters
        self.storage_type = self.get_param("storage_type", "file")
        self.storage_path = self.get_param("storage_path", "feedback_data")
        self.max_feedback_size = self.get_param("max_feedback_size", 10000)
        self.compress_old_data = self.get_param("compress_old_data", True)

        # Analysis parameters
        self.perform_clustering = self.get_param("perform_clustering", HAVE_ML_DEPS)
        self.similarity_threshold = self.get_param("similarity_threshold", 0.7)
        self.min_cluster_size = self.get_param("min_cluster_size", 3)

        # Telemetry and feedback parameters
        self.collect_performance_metrics = self.get_param("collect_performance_metrics", True)
        self.collect_error_patterns = self.get_param("collect_error_patterns", True)
        self.collect_user_feedback = self.get_param("collect_user_feedback", False)

        # Multi-tenant support
        self.is_multi_tenant = self.get_param("is_multi_tenant", True)

        # Initialize storage
        self._initialize_storage()

        self.logger.info(f"Feedback collector initialized with {self.storage_type} storage")

    def _initialize_storage(self):
        """Initialize the feedback storage system."""
        if self.storage_type == "file":
            # Create the storage directory if it doesn't exist
            os.makedirs(self.storage_path, exist_ok=True)

            # Create subdirectories for different feedback types
            os.makedirs(os.path.join(self.storage_path, "successes"), exist_ok=True)
            os.makedirs(os.path.join(self.storage_path, "failures"), exist_ok=True)
            os.makedirs(os.path.join(self.storage_path, "user_feedback"), exist_ok=True)
            os.makedirs(os.path.join(self.storage_path, "analysis"), exist_ok=True)

            # Create multi-tenant directories if needed
            if self.is_multi_tenant:
                os.makedirs(os.path.join(self.storage_path, "tenants"), exist_ok=True)

            self.logger.info(f"Initialized file storage at {self.storage_path}")
        elif self.storage_type == "database":
            # In a real implementation, this would initialize database connections
            # For demonstration, we'll log a message
            self.logger.info(f"Database storage selected but not implemented")
            self.logger.info("Falling back to file storage")
            self.storage_type = "file"
            self._initialize_storage()
        else:
            self.logger.warning(f"Unknown storage type: {self.storage_type}")
            self.logger.info("Falling back to file storage")
            self.storage_type = "file"
            self._initialize_storage()

    def record_success(self, specification: str, context: Dict[str, Any], synthesis_result: SynthesisResult) -> None:
        """
        Record successful synthesis information for future learning.

        Args:
            specification: The specification text
            context: Additional context used during synthesis
            synthesis_result: The successful synthesis result
        """
        self.logger.info("Recording successful synthesis")

        # Create success record
        success_data = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "specification": specification,
            "context": _sanitize_context(context),
            "synthesis_strategy": synthesis_result.strategy,
            "confidence_score": synthesis_result.confidence_score,
            "time_taken": synthesis_result.time_taken,
            "used_relaxation": getattr(synthesis_result, 'used_relaxation', False),
        }

        # Add performance metrics if enabled
        if self.collect_performance_metrics:
            success_data["performance_metrics"] = {
                "memory_usage": _get_memory_usage(),
                "system_load": _get_system_load(),
            }

        # Store the record
        self._store_feedback("successes", success_data, context)

        # Trigger analysis if needed
        if self.perform_clustering and HAVE_ML_DEPS:
            self._schedule_pattern_analysis()

    def record_failure(self, specification: str, context: Dict[str, Any],
                       synthesis_result: SynthesisResult, verification_result: VerificationReport) -> None:
        """
        Record failure information for analysis and improvement.

        Args:
            specification: The specification text
            context: Additional context used during synthesis
            synthesis_result: The synthesis result (even though it failed)
            verification_result: The verification result showing why it failed
        """
        self.logger.info(f"Recording synthesis failure: {verification_result.status.value}")

        # Create failure record
        failure_data = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "specification": specification,
            "context": _sanitize_context(context),
            "synthesis_strategy": getattr(synthesis_result, 'strategy', 'unknown'),
            "verification_status": verification_result.status.value,
            "verification_reason": verification_result.reason,
            "time_taken": synthesis_result.time_taken + verification_result.time_taken,
        }

        # Add counterexamples if available
        if verification_result.counterexamples:
            failure_data["counterexamples"] = verification_result.counterexamples

        # Add error patterns if enabled
        if self.collect_error_patterns:
            failure_data["error_patterns"] = _extract_error_patterns(
                specification, verification_result
            )

        # Add performance metrics if enabled
        if self.collect_performance_metrics:
            failure_data["performance_metrics"] = {
                "memory_usage": _get_memory_usage(),
                "system_load": _get_system_load(),
            }

        # Store the record
        self._store_feedback("failures", failure_data, context)

        # Trigger analysis if needed
        if self.collect_error_patterns:
            self._analyze_failure_patterns()

    def record_user_feedback(self, synthesis_id: str, feedback_data: Dict[str, Any]) -> None:
        """
        Record feedback from end-users about synthesis results.

        Args:
            synthesis_id: ID of the synthesis result
            feedback_data: User feedback data
        """
        if not self.collect_user_feedback:
            self.logger.info("User feedback collection is disabled")
            return

        self.logger.info(f"Recording user feedback for synthesis {synthesis_id}")

        # Create user feedback record
        user_feedback = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "synthesis_id": synthesis_id,
            "feedback": feedback_data,
        }

        # Store the record
        self._store_feedback("user_feedback", user_feedback, {"synthesis_id": synthesis_id})

        # Check if feedback is negative, potentially triggering analysis
        is_negative = _is_negative_feedback(feedback_data)
        if is_negative:
            self.logger.info("Detected negative user feedback, triggering analysis")
            self._analyze_negative_feedback(synthesis_id, feedback_data)

    def export_feedback_for_training(self, output_path: str) -> bool:
        """
        Export feedback data in a format suitable for model training.

        Args:
            output_path: Path to write the training data

        Returns:
            True if export was successful
        """
        self.logger.info(f"Exporting feedback data for training to {output_path}")

        try:
            # Collect successful and failed synthesis examples
            success_examples = self._load_feedback_data("successes")
            failure_examples = self._load_feedback_data("failures")

            # Prepare training data format
            training_data = {
                "positive_examples": success_examples,
                "negative_examples": failure_examples,
                "export_date": datetime.datetime.now().isoformat(),
                "total_examples": len(success_examples) + len(failure_examples),
            }

            # Write training data to output file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(training_data, f, indent=2)

            self.logger.info(f"Exported {training_data['total_examples']} examples for training")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export feedback for training: {e}")
            return False

    def _store_feedback(self, feedback_type: str, data: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Store feedback data in the appropriate storage.
        :type feedback_type: object
        """
        if self.storage_type == "file":
            # Determine the storage path based on tenant if multi-tenant
            storage_path = self.storage_path
            if self.is_multi_tenant and "tenant_id" in context:
                tenant_id = context["tenant_id"]
                storage_path = os.path.join(self.storage_path, "tenants", tenant_id)
                os.makedirs(os.path.join(storage_path, feedback_type), exist_ok=True)

            # Create a filename based on ID and timestamp
            filename = f"{data['id']}.json"
            file_path = os.path.join(storage_path, feedback_type, filename)

            # Write the data
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.debug(f"Stored {feedback_type} feedback to {file_path}")

            # Check if we need to compress old data
            if self.compress_old_data:
                self._maybe_compress_old_data(os.path.join(storage_path, feedback_type))

        elif self.storage_type == "database":
            # In a real implementation, this would store data in a database
            self.logger.debug(f"Would store {feedback_type} feedback in database")
            pass

    def _maybe_compress_old_data(self, directory: str) -> None:
        """Check if we need to compress old feedback data and do so if needed."""
        import glob
        import zipfile
        from datetime import datetime

        # Get all JSON files in the directory
        json_files = glob.glob(os.path.join(directory, "*.json"))

        # If we have more than the max number of files, compress older ones
        if len(json_files) > self.max_feedback_size:
            self.logger.info(f"Found {len(json_files)} feedback files, compressing older ones")

            # Sort files by modification time (oldest first)
            json_files.sort(key=os.path.getmtime)

            # Determine how many files to compress
            num_to_compress = len(json_files) - self.max_feedback_size
            files_to_compress = json_files[:num_to_compress]

            if files_to_compress:
                # Create a zip file with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = os.path.join(directory, f"archived_{timestamp}.zip")

                with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in files_to_compress:
                        zipf.write(file, os.path.basename(file))
                        # Remove the original file
                        os.remove(file)

                self.logger.info(f"Compressed {len(files_to_compress)} old feedback files to {zip_filename}")

    def _load_feedback_data(self, feedback_type: str) -> List[Dict[str, Any]]:
        """Load feedback data of the specified type."""
        data = []

        if self.storage_type == "file":
            import glob

            # Get all JSON files in the directory
            json_files = glob.glob(os.path.join(self.storage_path, feedback_type, "*.json"))

            # Load each file
            for file_path in json_files:
                try:
                    with open(file_path, 'r') as f:
                        data.append(json.load(f))
                except Exception as e:
                    self.logger.warning(f"Failed to load feedback file {file_path}: {e}")

            self.logger.debug(f"Loaded {len(data)} {feedback_type} feedback records")

        elif self.storage_type == "database":
            # In a real implementation, this would query the database
            self.logger.debug(f"Would load {feedback_type} feedback from database")
            pass

        return data

    def _analyze_negative_feedback(self, synthesis_id: str, feedback_data: Dict[str, Any]) -> None:
        """Analyze negative user feedback for insights."""
        # In a real implementation, this would analyze negative feedback for patterns
        # and potentially trigger alerts or add to the knowledge base

        self.logger.info(f"Analyzing negative feedback for synthesis {synthesis_id}")

        # Log analysis results
        analysis_data = {
            "synthesis_id": synthesis_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "feedback_data": feedback_data,
            "analysis_type": "negative_feedback",
            "findings": {}  # In a real implementation, this would contain analysis results
        }

        # Store the analysis
        analysis_path = os.path.join(self.storage_path, "analysis")
        analysis_file = os.path.join(analysis_path, f"analysis_{uuid.uuid4()}.json")

        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)

    def _analyze_failure_patterns(self) -> None:
        """Analyze failure patterns for insights."""
        # In a real implementation, this would analyze failure records for patterns
        # and potentially update the knowledge base or synthesis strategies

        self.logger.info("Analyzing failure patterns")

        # This would be implemented in a real system to find common patterns
        pass

    def _schedule_pattern_analysis(self) -> None:
        """Schedule an asynchronous pattern analysis."""
        # In a real implementation, this would queue an analysis job
        # for execution in the background

        self.logger.info("Scheduled pattern analysis")

        # This would be implemented in a real system to trigger async analysis
        pass

