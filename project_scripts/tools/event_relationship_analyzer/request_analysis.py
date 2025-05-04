#!/usr/bin/env python3
"""
Relationship Analysis Request Client
Send relationship analysis requests to Pulsar
"""
import json
import uuid
import argparse
from datetime import datetime
import time
import sys

# Import Pulsar client
try:
    import pulsar
except ImportError:
    print("Pulsar client not installed. Run: pip install pulsar-client")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Send relationship analysis request")
    parser.add_argument("--broker-url", default="pulsar://localhost:6650", help="Pulsar broker URL")
    parser.add_argument("--action", choices=["analyze", "apply"], default="analyze", help="Action to perform")
    parser.add_argument("--apply", action="store_true", help="Apply discovered relationships")
    parser.add_argument("--relationships-file", help="JSON file with relationships (for apply action)")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")

    args = parser.parse_args()

    # Create request
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "action": args.action
    }

    # Add action-specific data
    if args.action == "analyze":
        request["apply"] = args.apply
    elif args.action == "apply":
        # Load relationships from file
        if not args.relationships_file:
            print("Error: --relationships-file is required for 'apply' action")
            return 1

        try:
            with open(args.relationships_file, 'r') as f:
                relationships = json.load(f)
                request["relationships"] = relationships
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading relationships: {e}")
            return 1

    # Create Pulsar client
    client = pulsar.Client(args.broker_url)

    try:
        # Create producer
        producer = client.create_producer("template.relationship.request")

        # Send request
        producer.send(json.dumps(request).encode('utf-8'))
        print(f"Sent relationship analysis request: {request_id}")

        # Create consumer for responses
        consumer = client.subscribe(
            topic="template.relationship.response",
            subscription_name=f"temp-client-{request_id}",
            consumer_type=pulsar.ConsumerType.Exclusive
        )

        print(f"Waiting for response (timeout: {args.timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < args.timeout:
            try:
                # Try to receive message with timeout
                msg = consumer.receive(timeout_millis=1000)

                # Parse response
                response = json.loads(msg.data().decode('utf-8'))

                # Check if this is our response
                if response.get("request_id") == request_id:
                    print("Received response:")

                    result = response.get("result", {})
                    print(f"Status: {result.get('status')}")

                    if result.get("status") == "success":
                        if args.action == "analyze":
                            print(f"Templates analyzed: {result.get('template_count', 0)}")
                            print(f"Relationships found: {result.get('relationship_count', 0)}")

                            if args.apply:
                                print(f"Templates updated: {result.get('updated_templates', 0)}")

                            # Save relationships to file if successful
                            relationships = result.get("relationships")
                            if relationships:
                                output_file = f"relationships_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
                                with open(output_file, 'w') as f:
                                    json.dump(relationships, f, indent=2)
                                print(f"Relationships saved to: {output_file}")
                        elif args.action == "apply":
                            print(f"Templates updated: {result.get('updated_templates', 0)}")
                    else:
                        print(f"Error:
                        else:
                        print(f"Error: {result.get('message', 'Unknown error')}")

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
        if 'producer' in locals():
            producer.close()
        client.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
