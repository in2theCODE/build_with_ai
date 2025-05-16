#!/usr/bin/env python3
"""
Template Request Client
Send template generation requests to Pulsar
"""

import argparse
from datetime import datetime
import json
import sys
import time
import uuid


# Import Pulsar client
try:
    import pulsar
except ImportError:
    print("Pulsar client not installed. Run: pip install pulsar-client")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Send template generation request")
    parser.add_argument(
        "--broker-url", default="pulsar://localhost:6650", help="Pulsar broker URL"
    )
    parser.add_argument("--category", required=True, help="Template category")
    parser.add_argument("--name", required=True, help="Template name")
    parser.add_argument("--description", required=True, help="Template description")
    parser.add_argument(
        "--services-file", required=True, help="JSON file with component definitions"
    )
    parser.add_argument("--audience", default="system", help="Target audience")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    parser.add_argument(
        "--wait-response", action="store_true", help="Wait for response"
    )

    args = parser.parse_args()

    # Load services
    try:
        with open(args.components_file, "r") as f:
            components = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading services: {e}")
        return 1

    # Create request
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "category": args.category,
        "name": args.name,
        "description": args.description,
        "services": components,
        "audience": args.audience,
    }

    # Create Pulsar client
    client = pulsar.Client(args.broker_url)

    try:
        # Create producer
        producer = client.create_producer("template.generation.request")

        # Send request
        producer.send(json.dumps(request).encode("utf-8"))
        print(f"Sent template generation request: {request_id}")

        # Wait for response if requested
        if args.wait_response:
            # Create consumer for responses
            consumer = client.subscribe(
                topic="template.generation.response",
                subscription_name=f"temp-client-{request_id}",
                consumer_type=pulsar.ConsumerType.Exclusive,
            )

            print(f"Waiting for response (timeout: {args.timeout}s)...")
            start_time = time.time()

            while time.time() - start_time < args.timeout:
                try:
                    # Try to receive message with timeout
                    msg = consumer.receive(timeout_millis=1000)

                    # Parse response
                    response = json.loads(msg.data().decode("utf-8"))

                    # Check if this is our response
                    if response.get("request_id") == request_id:
                        print("Received response:")
                        print(json.dumps(response, indent=2))

                        # Acknowledge and break
                        consumer.acknowledge(msg)
                        break
                    else:
                        # Not our response, negative acknowledge
                        consumer.negative_acknowledge(msg)

                except Exception as e:
                    if "timeout" not in str(e).lower():
                        print(f"Error receiving response: {e}")
                    # Continue waiting

            else:
                print(f"Timeout waiting for response after {args.timeout}s")

            # Clean up consumer
            consumer.close()

    finally:
        # Clean up resources
        if "producer" in locals():
            producer.close()
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
