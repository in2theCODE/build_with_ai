#!/bin/bash
# event_template_generator.sh - Create an event-driven template generator

# Set up environment
TEMPLATE_DIR="./templates"
EVENT_GENERATOR_DIR="./event_templates"

# Create the necessary directories
mkdir -p $EVENT_GENERATOR_DIR

# Create the event-driven template generator
cat > $EVENT_GENERATOR_DIR/event_template_generator.py << 'EOF'
#!/usr/bin/env python3
"""
Event-Driven Template Generator
Creates templates based on events received from Pulsar
"""
import os
import json
import uuid
import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("event_template_generator")

# Import Pulsar client
try:
    import pulsar
except ImportError:
    logger.error("Pulsar client not installed. Run: pip install pulsar-client")
    sys.exit(1)

class EventTemplateGenerator:
    """Generate templates based on events from Pulsar"""

    def __init__(self, template_dir: str, broker_url: str):
        self.template_dir = template_dir
        self.broker_url = broker_url
        self.client = None
        self.consumer = None
        self.producer = None
        self.running = False

    async def initialize(self):
        """Initialize Pulsar client and consumer"""
        try:
            # Create Pulsar client
            self.client = pulsar.Client(self.broker_url)

            # Create consumer for template generation requests
            self.consumer = self.client.subscribe(
                topic='template.generation.request',
                subscription_name='template-generator',
                consumer_type=pulsar.ConsumerType.Shared
            )

            # Create producer for template generation responses
            self.producer = self.client.create_producer(
                topic='template.generation.response'
            )

            logger.info(f"Connected to Pulsar broker at {self.broker_url}")
            logger.info("Listening for template generation requests...")

            self.running = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Pulsar client: {str(e)}")
            return False

    def shutdown(self):
        """Shutdown Pulsar client and connections"""
        self.running = False

        if self.consumer:
            self.consumer.close()

        if self.producer:
            self.producer.close()

        if self.client:
            self.client.close()

        logger.info("Shut down Pulsar connections")

    async def process_events(self):
        """Process events from Pulsar"""
        while self.running:
            try:
                # Receive message with timeout
                msg = self.consumer.receive(timeout_millis=1000)

                # Process message
                try:
                    # Parse message data
                    data = json.loads(msg.data().decode('utf-8'))
                    logger.info(f"Received template generation request: {data.get('request_id', 'unknown')}")

                    # Generate template
                    result = await self.generate_template(data)

                    # Send response
                    self._send_response(data.get('request_id'), result)

                    # Acknowledge message
                    self.consumer.acknowledge(msg)

                except Exception as e:
                    logger.error(f"Failed to process message: {str(e)}")
                    # Negative acknowledge so message can be redelivered
                    self.consumer.negative_acknowledge(msg)

            except Exception as e:
                if "timeout" not in str(e).lower():
                    logger.error(f"Error receiving message: {str(e)}")
                # Small delay before trying again
                await asyncio.sleep(0.1)

    def _send_response(self, request_id: str, result: Dict[str, Any]):
        """Send response message back to Pulsar"""
        response = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result
        }

        self.producer.send(json.dumps(response).encode('utf-8'))
        logger.info(f"Sent response for request {request_id}")

    async def generate_template(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate template based on request data"""
        try:
            # Extract template details from request
            category = request_data.get('category', '')
            name = request_data.get('name', '')
            description = request_data.get('description', '')
            components = request_data.get('components', [])
            audience = request_data.get('audience', 'system')

            # Validate required fields
            if not category or not name or not description or not components:
                return {
                    "status": "error",
                    "message": "Missing required fields: category, name, description, components"
                }

            # Create template ID
            template_id = f"{category}-{name.replace(' ', '-').lower()}-{uuid.uuid4().hex[:8]}"

            # Create metadata
            metadata = {
                "name": name,
                "description": description,
                "version": "1.0.0",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "category": category,
                "subcategory": request_data.get('subcategory', ''),
                "audience": audience,
                "tags": request_data.get('tags', [category, "event-generated"]),
                "complexity": self._calculate_complexity(components)
            }

            # Create template
            template = {
                "id": template_id,
                "metadata": metadata,
                "source_type": "event-generated",
                "source_location": "",
                "relationships": [],
                "components": components,
                "variables": request_data.get('variables', []),
                "is_cached": False,
                "cache_path": ""
            }

            # Save template to appropriate directory
            template_path = self._save_template(template, category)

            return {
                "status": "success",
                "template_id": template_id,
                "template": template,
                "template_path": template_path
            }

        except Exception as e:
            logger.error(f"Template generation error: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate template: {str(e)}"
            }

    def _calculate_complexity(self, components: List[Dict[str, Any]]) -> int:
        """Calculate template complexity based on components"""
        # Simple complexity calculation based on component count and required status
        base_complexity = min(len(components), 10)
        required_components = sum(1 for comp in components if comp.get("required", False))

        # Scale complexity between 1-10
        return max(1, min(10, base_complexity + (required_components // 2)))

    def _save_template(self, template: Dict[str, Any], category: str) -> str:
        """Save template to appropriate directory"""
        category_dir = os.path.join(self.template_dir, "categories", category)
        os.makedirs(category_dir, exist_ok=True)

        template_path = os.path.join(category_dir, f"{template['id']}.json")
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)

        logger.info(f"Template saved to: {template_path}")
        return template_path

async def run_service(generator):
    """Run the template generator service"""
    # Initialize the generator
    if not await generator.initialize():
        return

    # Process events until shutdown
    await generator.process_events()

def main():
    parser = argparse.ArgumentParser(description="Event-driven template generator service")
    parser.add_argument("--template-dir", default="./templates", help="Template directory")
    parser.add_argument("--broker-url", default="pulsar://localhost:6650", help="Pulsar broker URL")

    args = parser.parse_args()

    # Create generator
    generator = EventTemplateGenerator(args.template_dir, args.broker_url)

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Shutting down...")
        generator.shutdown()
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Run service
        loop.run_until_complete(run_service(generator))
    finally:
        loop.close()

if __name__ == "__main__":
    main()
EOF

chmod +x $EVENT_GENERATOR_DIR/event_template_generator.py

# Create a client script to send requests
cat > $EVENT_GENERATOR_DIR/request_template.py << 'EOF'
#!/usr/bin/env python3
"""
Template Request Client
Send template generation requests to Pulsar
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
    parser = argparse.ArgumentParser(description="Send template generation request")
    parser.add_argument("--broker-url", default="pulsar://localhost:6650", help="Pulsar broker URL")
    parser.add_argument("--category", required=True, help="Template category")
    parser.add_argument("--name", required=True, help="Template name")
    parser.add_argument("--description", required=True, help="Template description")
    parser.add_argument("--components-file", required=True, help="JSON file with component definitions")
    parser.add_argument("--audience", default="system", help="Target audience")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    parser.add_argument("--wait-response", action="store_true", help="Wait for response")

    args = parser.parse_args()

    # Load components
    try:
        with open(args.components_file, 'r') as f:
            components = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading components: {e}")
        return 1

    # Create request
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "category": args.category,
        "name": args.name,
        "description": args.description,
        "components": components,
        "audience": args.audience
    }

    # Create Pulsar client
    client = pulsar.Client(args.broker_url)

    try:
        # Create producer
        producer = client.create_producer("template.generation.request")

        # Send request
        producer.send(json.dumps(request).encode('utf-8'))
        print(f"Sent template generation request: {request_id}")

        # Wait for response if requested
        if args.wait_response:
            # Create consumer for responses
            consumer = client.subscribe(
                topic="template.generation.response",
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
        if 'producer' in locals():
            producer.close()
        client.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x $EVENT_GENERATOR_DIR/request_template.py

echo "Event-driven Template Generator created successfully"