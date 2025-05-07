#!/usr/bin/env python3
import argparse
import base64
import datetime
import hashlib
import hmac
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import urllib.parse

import requests


class HmacSigner:
    """
    HMAC signature generator for API requests to securely authenticate with the API gateway.
    Works with Traefik HMAC authentication middleware.
    """

    def __init__(self, key_id: str, secret_key: str):
        """
        Initialize the HMAC signer with credentials.

        Args:
            key_id: The identifier for the key used in the HMAC middleware
            secret_key: The shared secret key for signing requests
        """
        self.key_id = key_id
        self.secret_key = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key

    def sign_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[str, bytes]] = None,
        expiry_seconds: int = 300,
    ) -> Dict[str, str]:
        """
        Sign an HTTP request with HMAC for Traefik authentication.

        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            url: Full URL for the request
            headers: Optional dictionary of headers to include in signature
            body: Optional request body for digest calculation
            expiry_seconds: Seconds until the signature expires

        Returns:
            Dictionary of headers with HMAC signature added
        """
        if headers is None:
            headers = {}

        # Ensure headers keys are lowercase for consistency
        headers = {k.lower(): v for k, v in headers.items()}

        # Parse URL to get path and query parameters
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path or "/"
        if parsed_url.query:
            path = f"{path}?{parsed_url.query}"

        # Create timestamps (seconds since epoch)
        created = int(time.time())
        expires = created + expiry_seconds

        # Construct string to sign
        string_parts = [
            f"(request-target): {method.lower()} {path}",
            f"(created): {created}",
            f"(expires): {expires}",
        ]

        # Add host if present
        if "host" in headers:
            string_parts.append(f"host: {headers['host']}")
        elif parsed_url.netloc:
            host = parsed_url.netloc
            # Remove port if present
            if ":" in host:
                host = host.split(":")[0]
            string_parts.append(f"host: {host}")
            headers["host"] = host

        # Add digest if body is present
        if body:
            if isinstance(body, str):
                body_bytes = body.encode("utf-8")
            else:
                body_bytes = body

            digest = base64.b64encode(hashlib.sha256(body_bytes).digest()).decode("utf-8")

            headers["digest"] = f"SHA-256={digest}"
            string_parts.append(f"digest: SHA-256={digest}")

        # Create final string to sign
        string_to_sign = "\n".join(string_parts)

        # Generate signature
        signature = base64.b64encode(
            hmac.new(self.secret_key, string_to_sign.encode("utf-8"), hashlib.sha256).digest()
        ).decode("utf-8")

        # Build headers list for the Authorization header
        headers_list = "(request-target) (created) (expires)"
        if "host" in headers:
            headers_list += " host"
        if "digest" in headers:
            headers_list += " digest"

        # Add Authorization header
        headers["authorization"] = (
            f'Hmac keyId="{self.key_id}",'
            f'algorithm="hmac-sha256",'
            f'headers="{headers_list}",'
            f'signature="{signature}",'
            f'created="{created}",'
            f'expires="{expires}"'
        )

        return headers

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict] = None,
        data: Optional[Union[str, bytes]] = None,
        **kwargs,
    ) -> requests.Response:
        """
        Make a signed HTTP request using the requests library.

        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            url: Full URL for the request
            headers: Optional dictionary of headers
            json: Optional JSON body (will be serialized)
            data: Optional request body
            **kwargs: Additional arguments to pass to requests

        Returns:
            requests.Response object
        """
        if headers is None:
            headers = {}

        # Prepare body for signing
        body = None
        if json is not None:
            import json as json_lib

            body = json_lib.dumps(json).encode("utf-8")
        elif data is not None:
            body = data

        # Sign the request
        signed_headers = self.sign_request(method, url, headers, body)

        # Make the request
        return requests.request(method, url, headers=signed_headers, json=json, data=data, **kwargs)


# Example usage when script is run directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign API requests with HMAC")
    parser.add_argument("--key-id", required=True, help="HMAC key ID")
    parser.add_argument("--secret-key", required=True, help="HMAC secret key")
    parser.add_argument("--method", default="GET", help="HTTP method")
    parser.add_argument("--url", required=True, help="Request URL")
    parser.add_argument("--header", action="append", help="Headers in format key:value")
    parser.add_argument("--data", help="Request body")

    args = parser.parse_args()

    # Parse headers
    headers = {}
    if args.header:
        for header in args.header:
            key, value = header.split(":", 1)
            headers[key.strip()] = value.strip()

    # Create signer and sign request
    signer = HmacSigner(args.key_id, args.secret_key)
    signed_headers = signer.sign_request(args.method, args.url, headers, args.data)

    # Output headers for use in curl or other tools
    for key, value in signed_headers.items():
        print(f"{key}: {value}")
