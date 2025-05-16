#!/usr/bin/env python3
"""
Meta learning component for the Program Synthesis System.

This component analyzes synthesis results to learn more effective synthesis
strategies over time, enabling the system to adapt to different problem domains.
"""

import hashlib
from collections import defaultdict
import datetime
import json
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, Optional


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.shared.models.base import BaseComponent
from src.services.shared.models.enums import SynthesisStrategy


try:
    # Optional dependencies for advanced features
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    HAVE_ML_DEPS = True
except ImportError:
    HAVE_ML_DEPS = False


def _extract_domain(specification: str, context: Dict[str, Any]) -> str:
    """Extract domain from specification and context."""
    # Try to get domain from context
    if context and "domain" in context:
        return context["domain"]

    # Try to infer domain from specification content
    # This could be enhanced with more sophisticated NLP techniques
    domain_keywords = {
        "math": ["math", "arithmetic", "calculation", "formula", "number", "algebra"],
        "string": ["string", "text", "character", "substring", "concat", "replace"],
        "array": ["array", "list", "sequence", "sort", "filter", "map", "reduce"],
        "io": ["file", "input", "output", "read", "write", "stream", "buffer"],
        "graphics": ["image", "graphic", "draw", "color", "pixel", "render"],
        "web": ["http", "html", "url", "api", "request", "response", "endpoint"],
        "database": ["database", "query", "sql", "table", "join", "select", "insert"],
    }

    domain_scores = defaultdict(int)
    spec_lower = specification.lower()

    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            if keyword in spec_lower:
                domain_scores[domain] += 1

    if domain_scores:
        best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        if domain_scores[best_domain] >= 2:  # Threshold for confidence
            return best_domain

    # Default to "general" if no specific domain is identified
    return "general"


class MetaLearner(BaseComponent):
    """Learns synthesis strategies from past experiences."""

    def __init__(self, strategy_pool, **params):
        """Initialize the meta learner with learning parameters."""
        self.strategy_pool = strategy_pool or {}
        self.success_counts = {}
        self.failure_counts = {}
        super().__init__(**params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Learning parameters
        self.learning_rate = self.get_param("learning_rate", 0.1)
        self.exploration_rate = self.get_param("exploration_rate", 0.2)
        self.discount_factor = self.get_param("discount_factor", 0.9)
        self.min_examples = self.get_param("min_examples", 5)
        self.max_history_size = self.get_param("max_history_size", 1000)
        self.enable_clustering = self.get_param("enable_clustering", HAVE_ML_DEPS)
        self.feature_extraction_method = self.get_param("feature_extraction_method", "tfidf")
        self.num_clusters = self.get_param("num_clusters", 5)

        # Storage parameters
        self.storage_path = self.get_param("storage_path", "meta_learning_data")
        self.persistence_enabled = self.get_param("persistence_enabled", True)

        # Initialize strategy performance tracking
        self.strategy_stats = defaultdict(
            lambda: {
                "success_count": 0,
                "failure_count": 0,
                "total_time": 0.0,
                "avg_confidence": 0.0,
                "examples": [],
            }
        )

        # Initialize domain-specific strategy mapping
        self.domain_strategies = defaultdict(dict)

        # Initialize problem clusters if using clustering
        self.problem_clusters = {}
        self.vectorizer = None

        # Load previous learning data if available
        if self.persistence_enabled:
            self._load_learning_data()

        self.logger.info(f"Meta learner initialized with learning rate {self.learning_rate}")

    def _load_learning_data(self):
        """Load previous learning data from persistent storage."""
        try:
            # Create storage directory if it doesn't exist
            os.makedirs(self.storage_path, exist_ok=True)

            # Load strategy stats if available
            stats_path = os.path.join(self.storage_path, "strategy_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, "r") as f:
                    loaded_stats = json.load(f)

                # Convert loaded data to defaultdict
                for strategy, stats in loaded_stats.items():
                    self.strategy_stats[strategy] = stats

                self.logger.info(f"Loaded strategy stats for {len(loaded_stats)} strategies")

            # Load domain strategies if available
            domains_path = os.path.join(self.storage_path, "domain_strategies.json")
            if os.path.exists(domains_path):
                with open(domains_path, "r") as f:
                    loaded_domains = json.load(f)

                # Convert loaded data to defaultdict
                for domain, strategies in loaded_domains.items():
                    self.domain_strategies[domain] = strategies

                self.logger.info(f"Loaded domain strategies for {len(loaded_domains)} domains")

            # Load problem clusters if available and clustering is enabled
            if self.enable_clustering and HAVE_ML_DEPS:
                clusters_path = os.path.join(self.storage_path, "problem_clusters.json")
                if os.path.exists(clusters_path):
                    with open(clusters_path, "r") as f:
                        self.problem_clusters = json.load(f)

                    # Create a new vectorizer
                    if "vectors" in self.problem_clusters:
                        examples = [ex["specification"] for ex in self.problem_clusters["vectors"]]
                        self.vectorizer = TfidfVectorizer()
                        self.vectorizer.fit(examples)

                    self.logger.info(
                        f"Loaded problem clusters with {len(self.problem_clusters.get('clusters', {}))} clusters"
                    )

        except Exception as e:
            self.logger.warning(f"Failed to load learning data: {e}")
            # Continue with default initialization

    def _save_learning_data(self):
        """Save learning data to persistent storage."""
        if not self.persistence_enabled:
            return

        try:
            # Create storage directory if it doesn't exist
            os.makedirs(self.storage_path, exist_ok=True)

            # Save strategy stats
            stats_path = os.path.join(self.storage_path, "strategy_stats.json")
            with open(stats_path, "w") as f:
                json.dump(dict(self.strategy_stats), f, indent=2)

            # Save domain strategies
            domains_path = os.path.join(self.storage_path, "domain_strategies.json")
            with open(domains_path, "w") as f:
                json.dump(dict(self.domain_strategies), f, indent=2)

            # Save problem clusters if clustering is enabled
            if self.enable_clustering and self.problem_clusters:
                clusters_path = os.path.join(self.storage_path, "problem_clusters.json")
                with open(clusters_path, "w") as f:
                    json.dump(self.problem_clusters, f, indent=2)

            self.logger.info("Saved learning data to persistent storage")

        except Exception as e:
            self.logger.warning(f"Failed to save learning data: {e}")

    def suggest_strategy(self, specification: str, context: Dict[str, Any]) -> str:
        """
        Suggest a synthesis strategy based on learning.

        Args:
            specification: The specification text
            context: Additional context for synthesis

        Returns:
            Name of the suggested synthesis strategy
        """
        self.logger.info("Suggesting synthesis strategy")

        # Extract domain from context if available
        domain = _extract_domain(specification, context)

        # Check if we should explore randomly
        if random.random() < self.exploration_rate:
            self.logger.info("Using exploration for strategy selection")
            return self._explore_strategies()

        # If clustering is enabled, use problem clustering
        if self.enable_clustering and HAVE_ML_DEPS and self.vectorizer is not None:
            cluster = self._get_problem_cluster(specification)

            if cluster is not None and cluster in self.problem_clusters.get("clusters", {}):
                # Use cluster-specific strategies
                cluster_strategies = self.problem_clusters["clusters"][cluster]

                if cluster_strategies:
                    best_strategy = max(cluster_strategies.items(), key=lambda x: x[1])[0]
                    self.logger.info(f"Selected strategy {best_strategy} from cluster {cluster}")
                    return best_strategy

        # Check if domain-specific strategies are available
        if domain in self.domain_strategies and self.domain_strategies[domain]:
            # Sort strategies by success rate
            domain_stats = {
                strategy: self._calculate_strategy_success_rate(strategy) for strategy in self.domain_strategies[domain]
            }

            best_strategy = max(domain_stats.items(), key=lambda x: x[1])[0]
            self.logger.info(f"Selected domain-specific strategy {best_strategy}")
            return best_strategy

        # Use overall best strategy
        best_strategy = self._get_best_overall_strategy()
        self.logger.info(f"Selected overall best strategy {best_strategy}")

        return best_strategy

    def record_success(self, specification: str, context: Dict[str, Any], strategy: str) -> None:
        """
        Record a successful synthesis result.

        Args:
            specification: The specification text
            context: Additional context for synthesis
            strategy: The synthesis strategy used
        """
        self.logger.info(f"Recording successful synthesis with strategy {strategy}")

        # Update strategy stats
        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = {
                "success_count": 0,
                "failure_count": 0,
                "total_time": 0.0,
                "avg_confidence": 0.0,
                "examples": [],
            }

        self.strategy_stats[strategy]["success_count"] += 1

        # Add example if needed
        if len(self.strategy_stats[strategy]["examples"]) < self.max_history_size:
            example = {
                "specification": specification,
                "context": self._sanitize_context(context),
                "outcome": "success",
                "timestamp": datetime.datetime.now().isoformat(),
            }
            self.strategy_stats[strategy]["examples"].append(example)

        # Extract domain and update domain-specific strategies
        domain = _extract_domain(specification, context)
        if domain:
            if strategy not in self.domain_strategies[domain]:
                self.domain_strategies[domain][strategy] = 0

            # Apply reinforcement learning update
            current_value = self.domain_strategies[domain][strategy]
            # Reward is 1 for success
            reward = 1

            # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ·maxQ(s',a') - Q(s,a)]
            # Simplified here since we're only tracking immediate rewards
            self.domain_strategies[domain][strategy] = current_value + self.learning_rate * (reward - current_value)

        # Update clusters if using clustering
        if self.enable_clustering and HAVE_ML_DEPS:
            self._update_problem_clusters(specification, strategy, True)

        # Save updated learning data
        if self.persistence_enabled:
            self._save_learning_data()

    def record_failure(self, specification: str, context: Dict[str, Any], strategy: str) -> None:
        """
        Record a failed synthesis result.

        Args:
            specification: The specification text
            context: Additional context for synthesis
            strategy: The synthesis strategy used
        """
        self.logger.info(f"Recording failed synthesis with strategy {strategy}")

        # Update strategy stats
        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = {
                "success_count": 0,
                "failure_count": 0,
                "total_time": 0.0,
                "avg_confidence": 0.0,
                "examples": [],
            }

        def recommend_strategy(self, specification, context):
            """Recommend the best synthesis strategy for a given problem."""
            problem_type = self._determine_problem_type(specification, context)

            if problem_type not in self.success_counts:
                return "default"  # No data for this problem type

            # Calculate success rates for each strategy
            success_rates = {}
            for strategy in self.success_counts[problem_type]:
                successes = self.success_counts[problem_type].get(strategy, 0)
                failures = self.failure_counts[problem_type].get(strategy, 0)

                if successes + failures == 0:
                    success_rates[strategy] = 0
                else:
                    success_rates[strategy] = successes / (successes + failures)

            # Return the strategy with the highest success rate
            if not success_rates:
                return "default"

            return max(success_rates.items(), key=lambda x: x[1])[0]

        def _determine_problem_type(spec):
            """Determine the type of problem based on the specification and context."""
            # For simplicity, we'll just return a hash of the specification
            return hashlib.md5(spec.encode()).hexdigest()[:8]

        self.strategy_stats[strategy]["failure_count"] += 1

        # Add example if needed
        if len(self.strategy_stats[strategy]["examples"]) < self.max_history_size:
            example = {
                "specification": specification,
                "context": self._sanitize_context(context),
                "outcome": "failure",
                "timestamp": datetime.datetime.now().isoformat(),
            }
            self.strategy_stats[strategy]["examples"].append(example)

        # Extract domain and update domain-specific strategies
        domain = _extract_domain(specification, context)
        if domain:
            if strategy not in self.domain_strategies[domain]:
                self.domain_strategies[domain][strategy] = 0

            # Apply reinforcement learning update
            current_value = self.domain_strategies[domain][strategy]
            # Penalty is -1 for failure
            reward = -1

            # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ·maxQ(s',a') - Q(s,a)]
            # Simplified here since we're only tracking immediate rewards
            self.domain_strategies[domain][strategy] = current_value + self.learning_rate * (reward - current_value)

        # Update clusters if using clustering
        if self.enable_clustering and HAVE_ML_DEPS:
            self._update_problem_clusters(specification, strategy, False)

        # Save updated learning data
        if self.persistence_enabled:
            self._save_learning_data()

    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all strategies.

        Returns:
            Dictionary mapping strategy names to performance metrics
        """
        performance = {}

        for strategy, stats in self.strategy_stats.items():
            success_count = stats["success_count"]
            failure_count = stats["failure_count"]
            total_count = success_count + failure_count

            if total_count > 0:
                success_rate = success_count / total_count
            else:
                success_rate = 0.0

            performance[strategy] = {
                "success_count": success_count,
                "failure_count": failure_count,
                "total_count": total_count,
                "success_rate": success_rate,
                "avg_confidence": stats["avg_confidence"],
            }

        return performance

    def get_domain_strategies(self, domain: str) -> Dict[str, float]:
        """
        Get strategy rankings for a specific domain.

        Args:
            domain: The domain name

        Returns:
            Dictionary mapping strategy names to scores
        """
        if domain in self.domain_strategies:
            return self.domain_strategies[domain]

        return {}

    def analyze_strategy_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in strategy performance across domains.

        Returns:
            Dictionary with analysis results
        """
        # Skip if not enough data
        if not self.strategy_stats or not self.domain_strategies:
            return {
                "has_data": False,
                "message": "Not enough data for pattern analysis",
            }

        # Calculate strategy versatility
        strategy_versatility = {}
        for strategy in self.strategy_stats:
            # Count domains where this strategy is used
            domain_count = 0
            for domain, strategies in self.domain_strategies.items():
                if strategy in strategies:
                    domain_count += 1

            strategy_versatility[strategy] = {
                "domain_count": domain_count,
                "versatility_score": domain_count / max(1, len(self.domain_strategies)),
            }

        # Find domain similarities based on strategy preferences
        domain_similarities = {}
        domains = list(self.domain_strategies.keys())

        for i, domain1 in enumerate(domains):
            for domain2 in domains[i + 1 :]:
                similarity = self._calculate_domain_similarity(domain1, domain2)
                domain_similarities[f"{domain1}-{domain2}"] = similarity

        # Calculate strategy complementarity
        strategy_complementarity = {}
        strategies = list(self.strategy_stats.keys())

        for i, strategy1 in enumerate(strategies):
            for strategy2 in strategies[i + 1 :]:
                complementarity = self._calculate_strategy_complementarity(strategy1, strategy2)
                strategy_complementarity[f"{strategy1}-{strategy2}"] = complementarity

        return {
            "has_data": True,
            "strategy_versatility": strategy_versatility,
            "domain_similarities": domain_similarities,
            "strategy_complementarity": strategy_complementarity,
        }

    def _explore_strategies(self) -> str:
        """Randomly explore a strategy for learning purposes."""
        # List of all available strategies
        all_strategies = [strategy.value for strategy in SynthesisStrategy]

        # If we have performance data, bias exploration toward better strategies
        if self.strategy_stats:
            strategy_scores = {}
            for strategy in all_strategies:
                if strategy in self.strategy_stats:
                    stats = self.strategy_stats[strategy]
                    total = stats["success_count"] + stats["failure_count"]

                    if total > 0:
                        score = stats["success_count"] / total
                    else:
                        score = 0.5  # Default score for untried strategies
                else:
                    score = 0.5  # Default score for untried strategies

                strategy_scores[strategy] = score

            # Use weighted random selection
            total_score = sum(strategy_scores.values())
            if total_score > 0:
                rand_val = random.random() * total_score
                cumulative = 0

                for strategy, score in strategy_scores.items():
                    cumulative += score
                    if cumulative >= rand_val:
                        return strategy

            # Fallback to uniform random if scores don't add up properly
            return random.choice(all_strategies)
        else:
            # Simple random selection if no performance data
            return random.choice(all_strategies)

    def _get_best_overall_strategy(self) -> str:
        """Get the best overall strategy based on success rate."""
        best_strategy = None
        best_score = -1.0

        for strategy in self.strategy_stats:
            success_count = self.strategy_stats[strategy]["success_count"]
            failure_count = self.strategy_stats[strategy]["failure_count"]
            total = success_count + failure_count

            if total >= self.min_examples:
                score = success_count / total if total > 0 else 0

                if score > best_score:
                    best_score = score
                    best_strategy = strategy

        # If no strategy has enough examples, use a default
        if best_strategy is None:
            best_strategy = SynthesisStrategy.NEURAL_GUIDED.value

        return best_strategy

    def _calculate_strategy_success_rate(self, strategy: str) -> float:
        """Calculate success rate for a given strategy."""
        if strategy not in self.strategy_stats:
            return 0.0

        stats = self.strategy_stats[strategy]
        success_count = stats["success_count"]
        failure_count = stats["failure_count"]
        total = success_count + failure_count

        if total == 0:
            return 0.0

        return success_count / total

    def _update_problem_clusters(self, specification: str, strategy: str, success: bool) -> None:
        """Update problem clusters with new data."""
        if not self.enable_clustering or not HAVE_ML_DEPS:
            return

        # Initialize clusters if needed
        if not self.problem_clusters:
            self.problem_clusters = {"clusters": {}, "vectors": []}

        # Add this example to vectors
        example = {
            "specification": specification,
            "strategy": strategy,
            "success": success,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        self.problem_clusters["vectors"].append(example)

        # Re-cluster if we have enough examples
        if len(self.problem_clusters["vectors"]) >= self.min_examples:
            self._perform_clustering()

    def _perform_clustering(self) -> None:
        """Perform clustering on problem specifications."""
        if not HAVE_ML_DEPS:
            return

        self.logger.info("Performing problem clustering")

        # Extract specifications
        examples = [ex["specification"] for ex in self.problem_clusters["vectors"]]

        # Create or update the vectorizer
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=100)
            X = self.vectorizer.fit_transform(examples)
        else:
            X = self.vectorizer.transform(examples)

        # Cluster the vectors
        kmeans = KMeans(n_clusters=min(self.num_clusters, len(examples)), random_state=42)
        clusters = kmeans.fit_predict(X)

        # Update cluster information
        self.problem_clusters["clusters"] = {}

        for i, (example, cluster) in enumerate(zip(self.problem_clusters["vectors"], clusters)):
            cluster_id = str(cluster)
            strategy = example["strategy"]
            success = example["success"]

            # Initialize cluster if needed
            if cluster_id not in self.problem_clusters["clusters"]:
                self.problem_clusters["clusters"][cluster_id] = {}

            # Initialize strategy in cluster if needed
            if strategy not in self.problem_clusters["clusters"][cluster_id]:
                self.problem_clusters["clusters"][cluster_id][strategy] = 0

            # Update strategy score in cluster
            current_score = self.problem_clusters["clusters"][cluster_id][strategy]

            # Increase score for success, decrease for failure
            reward = 1 if success else -1

            # Apply Q-learning update
            self.problem_clusters["clusters"][cluster_id][strategy] = current_score + self.learning_rate * (
                reward - current_score
            )

        self.logger.info(f"Clustered {len(examples)} examples into {len(self.problem_clusters['clusters'])} clusters")

    def _get_problem_cluster(self, specification: str) -> Optional[str]:
        """Get the cluster for a given problem specification."""
        if not HAVE_ML_DEPS or self.vectorizer is None:
            return None

        try:
            # Transform the specification
            X = self.vectorizer.transform([specification])

            # Find the nearest cluster
            kmeans = KMeans(n_clusters=len(self.problem_clusters["clusters"]), random_state=42)

            # Fit on existing cluster centers
            # This is a simplification; in practice, you'd want to use a pre-trained model
            examples = [ex["specification"] for ex in self.problem_clusters["vectors"]]
            vectors = self.vectorizer.transform(examples)
            kmeans.fit(vectors)

            # Predict cluster
            cluster = kmeans.predict(X)[0]

            return str(cluster)
        except Exception as e:
            self.logger.warning(f"Failed to predict problem cluster: {e}")
            return None

    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate similarity between two domains based on strategy preferences."""
        if domain1 not in self.domain_strategies or domain2 not in self.domain_strategies:
            return 0.0

        strategies1 = set(self.domain_strategies[domain1].keys())
        strategies2 = set(self.domain_strategies[domain2].keys())

        # Calculate Jaccard similarity (intersection over union)
        intersection = len(strategies1.intersection(strategies2))
        union = len(strategies1.union(strategies2))

        if union == 0:
            return 0.0

        return intersection / union

    def _calculate_strategy_complementarity(self, strategy1: str, strategy2: str) -> float:
        """Calculate how complementary two strategies are across domains."""
        if strategy1 not in self.strategy_stats or strategy2 not in self.strategy_stats:
            return 0.0

        # Find domains where both strategies have been tried
        common_domains = []

        for domain in self.domain_strategies:
            if strategy1 in self.domain_strategies[domain] and strategy2 in self.domain_strategies[domain]:
                common_domains.append(domain)

        if not common_domains:
            return 0.0

        # Calculate correlation between strategy preferences
        preferences1 = [self.domain_strategies[domain][strategy1] for domain in common_domains]
        preferences2 = [self.domain_strategies[domain][strategy2] for domain in common_domains]

        # Use numpy's corrcoef if available
        if HAVE_ML_DEPS:
            correlation = np.corrcoef(preferences1, preferences2)[0, 1]
            # Invert correlation to get complementarity (negative correlation = high complementarity)
            return 1.0 - abs(correlation)
        else:
            # Simple computation if numpy not available
            n = len(common_domains)
            if n <= 1:
                return 0.5  # Neutral if not enough data

            mean1 = sum(preferences1) / n
            mean2 = sum(preferences2) / n

            cov = sum((preferences1[i] - mean1) * (preferences2[i] - mean2) for i in range(n)) / n
            var1 = sum((x - mean1) ** 2 for x in preferences1) / n
            var2 = sum((x - mean2) ** 2 for x in preferences2) / n

            if var1 == 0 or var2 == 0:
                return 0.5  # Neutral if no variance

            correlation = cov / ((var1 * var2) ** 0.5)

            # Invert correlation to get complementarity
            return 1.0 - abs(correlation)

    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize context for storage (remove sensitive info)."""
        if not context:
            return {}

        # Create a copy to avoid modifying the original
        sanitized = {}

        # Sensitive fields to filter out
        sensitive_fields = ["api_key", "token", "secret", "password", "credential"]

        for key, value in context.items():
            # Skip sensitive fields
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                sanitized[key] = "*** REDACTED ***"
            else:
                sanitized[key] = value

        return sanitized
