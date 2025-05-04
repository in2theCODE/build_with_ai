"""
Pulsar Configuration Module

This module manages the Pulsar client configuration and topic settings for the generator
microservice. It handles the initialization of connection parameters, authentication,
and management of topic schemas.

The configuration is designed to be environment-aware, loading different settings based
on the deployment environment (development, staging, production).
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class PulsarAuthConfig(BaseModel):
    """Pulsar authentication configuration."""
    auth_type: str = Field(default="none", description="Authentication type (none, token, oauth2)")
    token: Optional[str] = Field(default=None, description="Authentication token if using token auth")
    oauth_params: Optional[Dict[str, str]] = Field(default=None, description="OAuth2 parameters if using oauth2")


class PulsarTopicConfig(BaseModel):
    """Configuration for a single Pulsar topic."""
    name: str = Field(..., description="Topic name")
    subscription_name: str = Field(..., description="Subscription name for this consumer")
    subscription_type: str = Field(default="Exclusive", description="Subscription type (Exclusive, Shared, Failover)")
    schema_type: str = Field(default="JSON", description="Message schema type")
    batch_size: int = Field(default=100, description="Consumer batch size")
    consumer_type: str = Field(default="Listener", description="Consumer implementation type")
    retry_topic: Optional[str] = Field(default=None, description="Retry topic name if any")
    dlq_topic: Optional[str] = Field(default=None, description="Dead letter queue topic if any")
    max_retries: int = Field(default=3, description="Maximum retry attempts before moving to DLQ")


class PulsarConfig(BaseModel):
    """Pulsar client and topic configuration."""
    service_url: str = Field(..., description="Pulsar service URL")
    connection_timeout_ms: int = Field(default=30000, description="Connection timeout in milliseconds")
    operation_timeout_ms: int = Field(default=30000, description="Operation timeout in milliseconds")
    io_threads: int = Field(default=1, description="Number of IO threads")
    message_listener_threads: int = Field(default=1, description="Number of message listener threads")
    concurrent_lookup_requests: int = Field(default=50000, description="Max concurrent lookup requests")
    auth: PulsarAuthConfig = Field(default_factory=PulsarAuthConfig, description="Authentication configuration")
    use_tls: bool = Field(default=False, description="Whether to use TLS for connection")
    tls_trust_certs_file_path: Optional[str] = Field(default=None, description="TLS trust certs file path if using TLS")
    tls_allow_insecure_connection: bool = Field(default=False, description="Allow insecure TLS connection")
    consumer_topics: List[PulsarTopicConfig] = Field(default_factory=list, description="List of topics to consume from")
    producer_topics: List[PulsarTopicConfig] = Field(default_factory=list, description="List of topics to produce to")


class PulsarConfigManager:
    """
    Manages Pulsar configuration across the application.

    This class loads Pulsar configuration from the specified YAML file and provides
    access to the configuration for both consumers and producers.

    EXTENSION POINT: Add custom authentication methods and schema handling by extending
    the get_auth_params and get_schema_params methods.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Pulsar configuration manager.

        Args:
            config_path: Path to the Pulsar configuration YAML file.
                         If None, will try to load from environment variable or default location.
        """
        self.config_path = config_path or os.environ.get("PULSAR_CONFIG_PATH") or "config/pulsar_topics.yaml"
        self.config: Optional[PulsarConfig] = None

    def load(self) -> PulsarConfig:
        """
        Load the Pulsar configuration from the YAML file.

        Returns:
            The loaded PulsarConfig object.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValidationError: If the configuration file contains invalid values.
        """
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            self.config = PulsarConfig(**config_data)
            logger.info(f"Loaded Pulsar configuration from {self.config_path}")
            return self.config
        except Exception as e:
            logger.error(f"Failed to load Pulsar configuration: {str(e)}")
            raise

    def get_client_config(self) -> Dict[str, Any]:
        """
        Get the Pulsar client configuration parameters.

        Returns:
            Dictionary of client configuration parameters.
        """
        if not self.config:
            self.load()

        client_config = {
            "service_url": self.config.service_url,
            "connection_timeout_ms": self.config.connection_timeout_ms,
            "operation_timeout_ms": self.config.operation_timeout_ms,
            "io_threads": self.config.io_threads,
            "message_listener_threads": self.config.message_listener_threads,
            "concurrent_lookup_requests": self.config.concurrent_lookup_requests,
        }

        # Add TLS configuration if enabled
        if self.config.use_tls:
            client_config["use_tls"] = True
            if self.config.tls_trust_certs_file_path:
                client_config["tls_trust_certs_file_path"] = self.config.tls_trust_certs_file_path
            client_config["tls_allow_insecure_connection"] = self.config.tls_allow_insecure_connection

        # Add authentication if configured
        auth_params = self.get_auth_params()
        if auth_params:
            client_config.update(auth_params)

        return client_config

    def get_auth_params(self) -> Dict[str, Any]:
        """
        Get authentication parameters based on the configured authentication type.

        Returns:
            Dictionary of authentication parameters.

        EXTENSION POINT: Add custom authentication methods by extending this method.
        """
        if not self.config:
            self.load()

        auth_params = {}

        if self.config.auth.auth_type == "token":
            if not self.config.auth.token:
                raise ValueError("Token authentication specified but no token provided")
            auth_params["authentication"] = {
                "type": "token",
                "token": self.config.auth.token
            }
        elif self.config.auth.auth_type == "oauth2":
            if not self.config.auth.oauth_params:
                raise ValueError("OAuth2 authentication specified but no parameters provided")
            auth_params["authentication"] = {
                "type": "oauth2",
                "params": self.config.auth.oauth_params
            }

        return auth_params

    def get_consumer_topics(self) -> List[PulsarTopicConfig]:
        """
        Get the list of consumer topics.

        Returns:
            List of consumer topic configurations.
        """
        if not self.config:
            self.load()
        return self.config.consumer_topics

    def get_producer_topics(self) -> List[PulsarTopicConfig]:
        """
        Get the list of producer topics.

        Returns:
            List of producer topic configurations.
        """
        if not self.config:
            self.load()
        return self.config.producer_topics

