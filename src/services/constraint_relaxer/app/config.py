#!/usr/bin/env python3
"""
Configuration for the Constraint Relaxer Service.
"""

import os


class AppConfig:
    """Configuration class for the Constraint Relaxer Service."""

    def __init__(self):
        """Initialize configuration from environment variables with defaults."""
        # Event bus configuration
        self.EVENT_BUS_HOST = os.getenv("EVENT_BUS_HOST", "pulsar")
        self.EVENT_BUS_PORT = int(os.getenv("EVENT_BUS_PORT", "6650"))
        self.EVENT_BUS_TENANT = os.getenv("EVENT_BUS_TENANT", "public")
        self.EVENT_BUS_NAMESPACE = os.getenv("EVENT_BUS_NAMESPACE", "default")

        # Topics
        self.RELAXATION_REQUEST_TOPIC = os.getenv(
            "RELAXATION_REQUEST_TOPIC", "constraint.relaxation.requests"
        )
        self.RELAXATION_RESPONSE_TOPIC = os.getenv(
            "RELAXATION_RESPONSE_TOPIC", "constraint.relaxation.responses"
        )

        # Constraint relaxer configuration
        self.MAX_RELAXATION_ITERATIONS = int(os.getenv("MAX_RELAXATION_ITERATIONS", "5"))
        self.TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))
        self.USE_OPTIMIZATION = os.getenv("USE_OPTIMIZATION", "true").lower() == "true"
        self.USE_UNSAT_CORE = os.getenv("USE_UNSAT_CORE", "true").lower() == "true"
        self.USE_MAXSAT = os.getenv("USE_MAXSAT", "true").lower() == "true"
        self.MIN_CONSTRAINTS_TO_KEEP = int(os.getenv("MIN_CONSTRAINTS_TO_KEEP", "1"))
