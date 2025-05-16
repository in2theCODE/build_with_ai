#!/usr/bin/env python3
"""
Version manager component for the Program Synthesis System.

This component manages versioning of specifications, generated code, and synthesis
artifacts, enabling history tracking, comparison, and intelligent selection of
related previous results.
"""

import datetime
from difflib import SequenceMatcher
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from program_synthesis_system.src.shared import BaseComponent


class VersionManager(BaseComponent):
    """Manages versions of specifications and synthesis results."""

    def __init__(self, **params):
        """Initialize the version manager with storage parameters."""
        super().__init__(**params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Storage parameters
        self.storage_path = self.get_param("storage_path", "versions")
        self.max_versions = self.get_param("max_versions", 100)
        self.similarity_threshold = self.get_param("similarity_threshold", 0.8)

        # Advanced parameters
        self.use_semantic_similarity = self.get_param("use_semantic_similarity", False)
        self.track_parameter_changes = self.get_param("track_parameter_changes", True)
        self.fingerprint_algorithm = self.get_param("fingerprint_algorithm", "sha256")
        self.branch_naming_convention = self.get_param("branch_naming_convention", "auto")

        # Initialize storage
        self._initialize_storage()

        # Version index
        self.version_index = {}
        self._load_version_index()

        self.logger.info(f"Version manager initialized with storage path {self.storage_path}")

    def _initialize_storage(self) -> None:
        """Initialize storage directories."""
        # Create main storage directory
        os.makedirs(self.storage_path, exist_ok=True)

        # Create subdirectories
        os.makedirs(os.path.join(self.storage_path, "specifications"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "code"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "relations"), exist_ok=True)

        # Create index file if it doesn't exist
        index_path = os.path.join(self.storage_path, "index.json")
        if not os.path.exists(index_path):
            with open(index_path, "w") as f:
                json.dump({"versions": {}}, f)

    def _load_version_index(self) -> None:
        """Load the version index from storage."""
        try:
            index_path = os.path.join(self.storage_path, "index.json")
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    index_data = json.load(f)
                    self.version_index = index_data.get("versions", {})

                self.logger.info(f"Loaded {len(self.version_index)} versions from index")
            else:
                self.version_index = {}
        except Exception as e:
            self.logger.error(f"Failed to load version index: {e}")
            self.version_index = {}

    def _save_version_index(self) -> None:
        """Save the version index to storage."""
        try:
            index_path = os.path.join(self.storage_path, "index.json")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(index_path), exist_ok=True)

            with open(index_path, "w") as f:
                json.dump({"versions": self.version_index}, f, indent=2)

            self.logger.debug("Saved version index")
        except Exception as e:
            self.logger.error(f"Failed to save version index: {e}")

    def record_new_version(
        self,
        version_id: str,
        specification: str,
        context: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Record a new version of a specification and its synthesis result.

        Args:
            version_id: Unique identifier for this version
            specification: The specification text
            context: Additional context information
            metadata: Metadata about the synthesis process
        """
        self.logger.info(f"Recording new version: {version_id}")

        try:
            # Compute specification fingerprint
            fingerprint = self._compute_fingerprint(specification)

            # Create version record
            timestamp = datetime.datetime.now().isoformat()

            # Extract key information
            function_name = context.get("function_name", "unknown_function")
            domain = context.get("domain", "general")

            # Create version record
            version_record = {
                "id": version_id,
                "fingerprint": fingerprint,
                "timestamp": timestamp,
                "function_name": function_name,
                "domain": domain,
                "metadata": metadata,
            }

            # Store specification
            spec_path = os.path.join(self.storage_path, "specifications", f"{version_id}.txt")
            with open(spec_path, "w") as f:
                f.write(specification)

            # Store code if available
            if "code" in metadata:
                code_path = os.path.join(self.storage_path, "code", f"{version_id}.py")
                with open(code_path, "w") as f:
                    f.write(metadata["code"])

            # Store full metadata
            combined_metadata = {
                "version": version_record,
                "context": context,
                "synthesis_metadata": metadata,
            }

            metadata_path = os.path.join(self.storage_path, "metadata", f"{version_id}.json")
            with open(metadata_path, "w") as f:
                json.dump(combined_metadata, f, indent=2)

            # Find related versions
            related_versions = self._find_related_versions(specification, fingerprint, function_name, domain)

            # Store relations
            if related_versions:
                relations_path = os.path.join(self.storage_path, "relations", f"{version_id}.json")
                with open(relations_path, "w") as f:
                    json.dump({"related_versions": related_versions}, f, indent=2)

            # Update version index
            self.version_index[version_id] = {
                "fingerprint": fingerprint,
                "timestamp": timestamp,
                "function_name": function_name,
                "domain": domain,
                "related_versions": [rv["id"] for rv in related_versions],
            }

            # Save updated index
            self._save_version_index()

            # Prune old versions if needed
            if len(self.version_index) > self.max_versions:
                self._prune_old_versions()

            self.logger.info(f"Version {version_id} recorded successfully")

        except Exception as e:
            self.logger.error(f"Failed to record version {version_id}: {e}")

    def record_usage(self, version_id: str, usage_data: Dict[str, Any]) -> None:
        """
        Record usage of an existing version.

        Args:
            version_id: Identifier of the version
            usage_data: Data about the usage
        """
        self.logger.info(f"Recording usage of version: {version_id}")

        try:
            # Check if version exists
            if version_id not in self.version_index:
                self.logger.warning(f"Version {version_id} not found in index")
                return

            # Load existing metadata
            metadata_path = os.path.join(self.storage_path, "metadata", f"{version_id}.json")
            if not os.path.exists(metadata_path):
                self.logger.warning(f"Metadata file not found for version {version_id}")
                return

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Update usage information
            if "usage" not in metadata:
                metadata["usage"] = []

            usage_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "data": usage_data,
            }

            metadata["usage"].append(usage_entry)

            # Save updated metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Recorded usage of version {version_id}")

        except Exception as e:
            self.logger.error(f"Failed to record usage of version {version_id}: {e}")

    def find_prior_versions(self, specification: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Find prior versions related to a specification.

        Args:
            specification: The specification text
            context: Additional context information

        Returns:
            List of related prior versions
        """
        self.logger.info("Finding prior versions for specification")

        # Extract context information
        function_name = "unknown_function"
        domain = "general"

        if context:
            function_name = context.get("function_name", function_name)
            domain = context.get("domain", domain)

        # Compute fingerprint for quick comparison
        fingerprint = self._compute_fingerprint(specification)

        # Find related versions
        related_versions = self._find_related_versions(specification, fingerprint, function_name, domain)

        return related_versions

    def get_version_history(self, version_id: str) -> List[Dict[str, Any]]:
        """
        Get the history of related versions for a specific version.

        Args:
            version_id: Identifier of the version

        Returns:
            List of related versions in chronological order
        """
        self.logger.info(f"Getting version history for {version_id}")

        if version_id not in self.version_index:
            self.logger.warning(f"Version {version_id} not found in index")
            return []

        # Get related versions from index
        related_ids = self.version_index[version_id].get("related_versions", [])

        # Load complete records for related versions
        history = []
        for related_id in related_ids:
            try:
                metadata_path = os.path.join(self.storage_path, "metadata", f"{related_id}.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    history.append(
                        {
                            "id": related_id,
                            "timestamp": metadata["version"]["timestamp"],
                            "metadata": metadata["version"]["metadata"],
                        }
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load metadata for version {related_id}: {e}")

        # Sort by timestamp
        history.sort(key=lambda v: v.get("timestamp", ""))

        return history

    def get_version_details(self, version_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific version.

        Args:
            version_id: Identifier of the version

        Returns:
            Dictionary with version details
        """
        self.logger.info(f"Getting details for version {version_id}")

        result = {"id": version_id, "exists": False}

        try:
            # Check if version exists in index
            if version_id not in self.version_index:
                self.logger.warning(f"Version {version_id} not found in index")
                return result

            # Get basic info from index
            result.update(self.version_index[version_id])
            result["exists"] = True

            # Load specification
            spec_path = os.path.join(self.storage_path, "specifications", f"{version_id}.txt")
            if os.path.exists(spec_path):
                with open(spec_path, "r") as f:
                    result["specification"] = f.read()

            # Load code
            code_path = os.path.join(self.storage_path, "code", f"{version_id}.py")
            if os.path.exists(code_path):
                with open(code_path, "r") as f:
                    result["code"] = f.read()

            # Load full metadata
            metadata_path = os.path.join(self.storage_path, "metadata", f"{version_id}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    result["full_metadata"] = json.load(f)

            # Load relations
            relations_path = os.path.join(self.storage_path, "relations", f"{version_id}.json")
            if os.path.exists(relations_path):
                with open(relations_path, "r") as f:
                    result["relations"] = json.load(f)

            return result

        except Exception as e:
            self.logger.error(f"Failed to get details for version {version_id}: {e}")
            return result

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        Compare two versions and identify differences.

        Args:
            version_id1: First version identifier
            version_id2: Second version identifier

        Returns:
            Dictionary with comparison results
        """
        self.logger.info(f"Comparing versions {version_id1} and {version_id2}")

        comparison = {
            "version1": version_id1,
            "version2": version_id2,
            "differences": {},
        }

        try:
            # Load details for both versions
            details1 = self.get_version_details(version_id1)
            details2 = self.get_version_details(version_id2)

            # Check if both versions exist
            if not details1.get("exists", False) or not details2.get("exists", False):
                self.logger.warning(f"One or both versions not found: {version_id1}, {version_id2}")
                comparison["error"] = "One or both versions not found"
                return comparison

            # Compare specifications
            if "specification" in details1 and "specification" in details2:
                spec_diff = self._compute_text_diff(details1["specification"], details2["specification"])
                comparison["differences"]["specification"] = spec_diff

            # Compare code
            if "code" in details1 and "code" in details2:
                code_diff = self._compute_text_diff(details1["code"], details2["code"])
                comparison["differences"]["code"] = code_diff

            # Compare metadata
            if "full_metadata" in details1 and "full_metadata" in details2:
                metadata_diff = self._compare_metadata(
                    details1["full_metadata"]["synthesis_metadata"],
                    details2["full_metadata"]["synthesis_metadata"],
                )
                comparison["differences"]["metadata"] = metadata_diff

            # Calculate overall similarity
            similarity = self._calculate_version_similarity(details1, details2)
            comparison["similarity"] = similarity

            return comparison

        except Exception as e:
            self.logger.error(f"Failed to compare versions {version_id1} and {version_id2}: {e}")
            comparison["error"] = str(e)
            return comparison

    def _compute_fingerprint(self, text: str) -> str:
        """Compute a fingerprint for a text to enable quick similarity checks."""
        # Normalize text: lowercase, remove whitespace and punctuation
        normalized = re.sub(r"[\s\n\r\t.,;:!?()]+", "", text.lower())

        # Use selected algorithm to create fingerprint
        if self.fingerprint_algorithm == "sha256":
            return hashlib.sha256(normalized.encode()).hexdigest()
        elif self.fingerprint_algorithm == "md5":
            return hashlib.md5(normalized.encode()).hexdigest()
        else:
            # Default to SHA-256
            return hashlib.sha256(normalized.encode()).hexdigest()

    def _find_related_versions(
        self, specification: str, fingerprint: str, function_name: str, domain: str
    ) -> List[Dict[str, Any]]:
        """Find versions related to a specification based on similarity."""
        related_versions = []

        # First, check for exact fingerprint matches
        exact_matches = [
            version_id
            for version_id, version_data in self.version_index.items()
            if version_data.get("fingerprint") == fingerprint
        ]

        if exact_matches:
            for version_id in exact_matches:
                # Load full metadata for exact matches
                try:
                    metadata_path = os.path.join(self.storage_path, "metadata", f"{version_id}.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                        related_versions.append(
                            {
                                "id": version_id,
                                "similarity": 1.0,
                                "relationship": "exact_match",
                                "timestamp": metadata["version"]["timestamp"],
                                "metadata": metadata["version"]["metadata"],
                            }
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to load metadata for version {version_id}: {e}")

        # Next, look for function name matches
        function_matches = [
            version_id
            for version_id, version_data in self.version_index.items()
            if version_data.get("function_name") == function_name and version_id not in exact_matches
        ]

        # For function matches, compute text similarity
        for version_id in function_matches:
            try:
                # Load specification for comparison
                spec_path = os.path.join(self.storage_path, "specifications", f"{version_id}.txt")
                if not os.path.exists(spec_path):
                    continue

                with open(spec_path, "r") as f:
                    other_spec = f.read()

                # Compute similarity
                similarity = self._compute_text_similarity(specification, other_spec)

                # Include if similarity is above threshold
                if similarity >= self.similarity_threshold:
                    # Load metadata
                    metadata_path = os.path.join(self.storage_path, "metadata", f"{version_id}.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                        related_versions.append(
                            {
                                "id": version_id,
                                "similarity": similarity,
                                "relationship": "similar_function",
                                "timestamp": metadata["version"]["timestamp"],
                                "metadata": metadata["version"]["metadata"],
                            }
                        )
            except Exception as e:
                self.logger.warning(f"Failed to process version {version_id}: {e}")

        # Finally, check domain matches (if not enough results yet)
        if len(related_versions) < 3:
            domain_matches = [
                version_id
                for version_id, version_data in self.version_index.items()
                if version_data.get("domain") == domain
                and version_id not in exact_matches
                and version_id not in function_matches
            ]

            # Add up to 3 domain matches without detailed similarity check
            for version_id in domain_matches[:3]:
                try:
                    metadata_path = os.path.join(self.storage_path, "metadata", f"{version_id}.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                        related_versions.append(
                            {
                                "id": version_id,
                                "similarity": 0.5,  # Arbitrary similarity for domain matches
                                "relationship": "same_domain",
                                "timestamp": metadata["version"]["timestamp"],
                                "metadata": metadata["version"]["metadata"],
                            }
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to load metadata for version {version_id}: {e}")

        # Sort by similarity (descending)
        related_versions.sort(key=lambda v: v.get("similarity", 0), reverse=True)

        return related_versions

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two text strings."""
        # Use SequenceMatcher for similarity computation
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    def _compute_text_diff(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compute differences between two text strings."""
        # Use SequenceMatcher to compute differences
        matcher = SequenceMatcher(None, text1, text2)

        diff = {"similarity": matcher.ratio(), "changes": []}

        # Extract opcodes (operations needed to transform text1 into text2)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                change = {
                    "operation": tag,
                    "text1_start": i1,
                    "text1_end": i2,
                    "text2_start": j1,
                    "text2_end": j2,
                }

                if tag == "replace":
                    change["text1"] = text1[i1:i2]
                    change["text2"] = text2[j1:j2]
                elif tag == "delete":
                    change["text1"] = text1[i1:i2]
                elif tag == "insert":
                    change["text2"] = text2[j1:j2]

                diff["changes"].append(change)

        return diff

    def _compare_metadata(self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two metadata dictionaries and identify differences."""
        diff = {}

        # Find all keys in either dictionary
        all_keys = set(metadata1.keys()) | set(metadata2.keys())

        for key in all_keys:
            # Check for keys unique to one dictionary
            if key not in metadata1:
                diff[key] = {"status": "only_in_second", "value": metadata2[key]}
            elif key not in metadata2:
                diff[key] = {"status": "only_in_first", "value": metadata1[key]}
            # Compare values for common keys
            elif metadata1[key] != metadata2[key]:
                diff[key] = {
                    "status": "different",
                    "value1": metadata1[key],
                    "value2": metadata2[key],
                }

        return diff

    def _calculate_version_similarity(self, details1: Dict[str, Any], details2: Dict[str, Any]) -> float:
        """Calculate overall similarity between two versions."""
        similarity_components = []

        # Include specification similarity if available
        if "specification" in details1 and "specification" in details2:
            spec_similarity = self._compute_text_similarity(details1["specification"], details2["specification"])
            similarity_components.append(spec_similarity * 0.5)  # Weight: 50%

        # Include code similarity if available
        if "code" in details1 and "code" in details2:
            code_similarity = self._compute_text_similarity(details1["code"], details2["code"])
            similarity_components.append(code_similarity * 0.3)  # Weight: 30%

        # Include metadata similarity if available
        if "full_metadata" in details1 and "full_metadata" in details2:
            # Simple metadata similarity based on synthesis strategy
            strategy1 = details1["full_metadata"]["synthesis_metadata"].get("strategy", "")
            strategy2 = details2["full_metadata"]["synthesis_metadata"].get("strategy", "")

            metadata_similarity = 1.0 if strategy1 == strategy2 else 0.5
            similarity_components.append(metadata_similarity * 0.2)  # Weight: 20%

        # Calculate overall similarity (or default to 0)
        if similarity_components:
            return sum(similarity_components) / sum(
                0.5 if i == 0 else 0.3 if i == 1 else 0.2 for i in range(len(similarity_components))
            )
        else:
            return 0.0

    def _prune_old_versions(self) -> None:
        """Prune old versions if the index exceeds the maximum size."""
        self.logger.info(f"Pruning old versions (max: {self.max_versions})")

        # Get all versions sorted by timestamp
        versions = [(v_id, data.get("timestamp", "")) for v_id, data in self.version_index.items()]
        versions.sort(key=lambda v: v[1])  # Sort by timestamp (oldest first)

        # Determine how many to remove
        to_remove = len(versions) - self.max_versions

        if to_remove <= 0:
            return

        self.logger.info(f"Removing {to_remove} old versions")

        # Remove oldest versions first
        for i in range(to_remove):
            version_id, _ = versions[i]

            try:
                # Remove from storage
                self._remove_version(version_id)

                # Remove from index
                if version_id in self.version_index:
                    del self.version_index[version_id]
            except Exception as e:
                self.logger.error(f"Failed to remove version {version_id}: {e}")

        # Save updated index
        self._save_version_index()

    def _remove_version(self, version_id: str) -> None:
        """Remove a version from storage."""
        # Remove specification file
        spec_path = os.path.join(self.storage_path, "specifications", f"{version_id}.txt")
        if os.path.exists(spec_path):
            os.remove(spec_path)

        # Remove code file
        code_path = os.path.join(self.storage_path, "code", f"{version_id}.py")
        if os.path.exists(code_path):
            os.remove(code_path)

        # Remove metadata file
        metadata_path = os.path.join(self.storage_path, "metadata", f"{version_id}.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        # Remove relations file
        relations_path = os.path.join(self.storage_path, "relations", f"{version_id}.json")
        if os.path.exists(relations_path):
            os.remove(relations_path)

        self.logger.debug(f"Removed version {version_id} from storage")

    def get_version_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about version history.

        Returns:
            Dictionary with version statistics
        """
        self.logger.info("Generating version statistics")

        stats = {
            "total_versions": len(self.version_index),
            "versions_by_domain": {},
            "versions_by_function": {},
            "versions_by_strategy": {},
            "versions_by_month": {},
            "average_similarity": 0.0,
            "most_revised_functions": [],
        }

        try:
            # Count by domain and function
            domain_counts = {}
            function_counts = {}
            strategy_counts = {}

            # Track timestamps for monthly breakdown
            timestamps = []

            # Track all similarity scores for average calculation
            similarities = []

            for version_id, version_data in self.version_index.items():
                # Count by domain
                domain = version_data.get("domain", "unknown")
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

                # Count by function
                function_name = version_data.get("function_name", "unknown")
                function_counts[function_name] = function_counts.get(function_name, 0) + 1

                # Add timestamp
                timestamp = version_data.get("timestamp", "")
                if timestamp:
                    timestamps.append(timestamp)

                # Get synthesis strategy
                try:
                    metadata_path = os.path.join(self.storage_path, "metadata", f"{version_id}.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                        strategy = metadata["version"]["metadata"].get("strategy", "unknown")
                        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                        # Get similarity scores from relations
                        if "related_versions" in version_data and version_data["related_versions"]:
                            relations_path = os.path.join(self.storage_path, "relations", f"{version_id}.json")
                            if os.path.exists(relations_path):
                                with open(relations_path, "r") as f:
                                    relations = json.load(f)

                                for related in relations.get("related_versions", []):
                                    if "similarity" in related and related["similarity"] < 1.0:  # Exclude exact matches
                                        similarities.append(related["similarity"])
                except Exception as e:
                    self.logger.warning(f"Failed to load metadata for version {version_id}: {e}")

            # Store counts in statistics
            stats["versions_by_domain"] = domain_counts
            stats["versions_by_function"] = function_counts
            stats["versions_by_strategy"] = strategy_counts

            # Calculate versions by month
            if timestamps:
                for timestamp in timestamps:
                    try:
                        # Extract year and month from timestamp
                        date = datetime.datetime.fromisoformat(timestamp)
                        month_key = f"{date.year}-{date.month:02d}"
                        stats["versions_by_month"][month_key] = stats["versions_by_month"].get(month_key, 0) + 1
                    except Exception:
                        pass

            # Calculate average similarity
            if similarities:
                stats["average_similarity"] = sum(similarities) / len(similarities)

            # Find most revised functions (functions with most versions)
            most_revised = sorted(function_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            stats["most_revised_functions"] = [{"name": name, "count": count} for name, count in most_revised]

            return stats

        except Exception as e:
            self.logger.error(f"Failed to generate version statistics: {e}")
            return stats
