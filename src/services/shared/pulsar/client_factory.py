from typing import Optional

from pulsar import AuthenticationTLS
from pulsar import Client


def create_pulsar_client(
    service_url: str,
    ca_cert_path: Optional[str] = None,
    client_cert_path: Optional[str] = None,
    client_key_path: Optional[str] = None,
    allow_insecure: bool = False,
) -> Client:
    """
    Create a Pulsar client according to official Pulsar documentation.

    Args:
        service_url: Pulsar service URL (pulsar:// or pulsar+ssl://)
        ca_cert_path: Path to CA certificate for server verification
        client_cert_path: Path to client certificate for mTLS authentication
        client_key_path: Path to client private key for mTLS authentication
        allow_insecure: Whether to allow insecure connections (not recommended)

    Returns:
        Configured Pulsar client
    """
    # Base configuration
    config = {}

    # Configure TLS if CA certificate provided
    if ca_cert_path:
        config["tls_trust_certs_file_path"] = ca_cert_path
        config["tls_allow_insecure_connection"] = allow_insecure

    # Configure mTLS authentication if both cert and key provided
    if client_cert_path and client_key_path:
        config["authentication"] = AuthenticationTLS(client_cert_path, client_key_path)

    # Create and return the client
    return Client(service_url, **config)
