#!/bin/bash
# event_relationship_analyzer.sh - Create an event-driven relationship analyzer

# Set up directories
TOOLS_DIR="./tools"
EVENT_REL_DIR="$TOOLS_DIR/event_relationship_analyzer"

mkdir -p $EVENT_REL_DIR

# Create the event-driven relationship analyzer
cat > $EVENT_REL_DIR/event_relationship_analyzer.py << 'EOF'
#!/usr/bin/env python3
"""
Event-Driven Template Relationship Analyzer
Discovers and manages relationships between templates via Pulsar events
"""
import os
import json
import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Any, Set
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("event_relationship_analyzer")

# Import Pulsar client
try:
    import pulsar
except ImportError:
    logger.error("Pulsar client not installed. Run: pip install pulsar-client")
    sys.exit(1)

class EventRelationshipAnalyzer:
    """Analyze and manage relationships between templates via events"""

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

            # Create consumer for relationship analysis requests
            self.consumer = self.client.subscribe(
                topic='template.relationship.request',
                subscription_name='template-relationship-analyzer',
                consumer_type=pulsar.ConsumerType.Shared
            )

            # Create producer for relationship analysis responses
            self.producer = self.client.create_producer(
                topic='template.relationship.response'
            )

            logger.info(f"Connected to Pulsar broker at {self.broker_url}")
            logger.info("Listening for relationship analysis requests...")

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
                    logger.info(f"Received relationship analysis request: {data.get('request_id', 'unknown')}")

                    # Process the analysis request
                    if data.get('action') == 'analyze':
                        result = await self.analyze_relationships(data)
                    elif data.get('action') == 'apply':
                        result = await self.apply_relationships(data)
                    else:
                        result = {
                            "status": "error",
                            "message": f"Unknown action: {data.get('action')}"
                        }

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

    async def analyze_relationships(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between templates"""
        try:
            # Load all templates
            templates = self._load_templates()

            # Find relationships
            relationships = self._discover_relationships(templates)

            # Apply relationships if requested
            if request_data.get('apply', False):
                updated_count = self._apply_discovered_relationships(templates, relationships)

                return {
                    "status": "success",
                    "relationships": relationships,
                    "template_count": len(templates),
                    "relationship_count": sum(len(rels) for rels in relationships.values()),
                    "updated_templates": updated_count
                }
            else:
                return {
                    "status": "success",
                    "relationships": relationships,
                    "template_count": len(templates),
                    "relationship_count": sum(len(rels) for rels in relationships.values())
                }

        except Exception as e:
            logger.error(f"Relationship analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze relationships: {str(e)}"
            }

    async def apply_relationships(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specified relationships to templates"""
        try:
            relationships = request_data.get('relationships', {})

            if not relationships:
                return {
                    "status": "error",
                    "message": "No relationships specified"
                }

            # Load all templates
            templates = self._load_templates()

            # Apply the relationships
            updated_count = self._apply_discovered_relationships(templates, relationships)

            return {
                "status": "success",
                "updated_templates": updated_count
            }

        except Exception as e:
            logger.error(f"Apply relationships error: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to apply relationships: {str(e)}"
            }

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all templates from the directory"""
        templates = {}

        for root, _, files in os.walk(self.template_dir):
            for file in files:
                if file.endswith('.json'):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            template = json.load(f)
                            template_id = template.get('id', os.path.splitext(file)[0])
                            templates[template_id] = {
                                'path': file_path,
                                'data': template
                            }
                    except Exception as e:
                        logger.error(f"Error loading template {file}: {e}")

        logger.info(f"Loaded {len(templates)} templates")
        return templates

    def _discover_relationships(self, templates: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Discover relationships between templates"""
        relationships = defaultdict(list)

        # Group templates by category
        categories = defaultdict(list)
        for template_id, template_info in templates.items():
            template = template_info['data']
            category = template.get('metadata', {}).get('category', '')
            if category:
                categories[category].append(template_id)

        # Find complementary templates in same category
        for category, template_ids in categories.items():
            for i, template_id1 in enumerate(template_ids):
                for template_id2 in template_ids[i+1:]:
                    # Check if templates have complementary components
                    if self._are_complementary(
                        templates[template_id1]['data'],
                        templates[template_id2]['data']
                    ):
                        # Add relationship both ways
                        relationships[template_id1].append({
                            'related_id': template_id2,
                            'relationship_type': 'complements',
                            'description': f"Complements {template_id2}"
                        })

                        relationships[template_id2].append({
                            'related_id': template_id1,
                            'relationship_type': 'complements',
                            'description': f"Complements {template_id1}"
                        })

        # Find dependency relationships based on categories
        category_dependencies = {
            'adaptive-workflows': {'depends-on': ['decision-making']},
            'task-automation': {'depends-on': ['decision-making']},
            'context-evolution': {'depends-on': ['self-learning']},
            'research-modes': {'depends-on': ['knowledge-acquisition']}
        }

        for template_id, template_info in templates.items():
            template = template_info['data']
            category = template.get('metadata', {}).get('category', '')

            # Skip if no category dependencies defined
            if category not in category_dependencies:
                continue

            # Check dependencies for this category
            for rel_type, dep_categories in category_dependencies[category].items():
                for dep_category in dep_categories:
                    # Find templates in dependency category
                    for other_id in categories.get(dep_category, []):
                        if template_id != other_id:
                            relationships[template_id].append({
                                'related_id': other_id,
                                'relationship_type': rel_type,
                                'description': f"{rel_type.replace('-', ' ')} {dep_category} template {other_id}"
                            })

        return dict(relationships)

    def _are_complementary(self, template1: Dict[str, Any], template2: Dict[str, Any]) -> bool:
        """Check if two templates are complementary"""
        # Templates are complementary if they have different components
        # that could work together

        # Get component names
        components1 = {comp.get('name') for comp in template1.get('components', [])}
        components2 = {comp.get('name') for comp in template2.get('components', [])}

        # If no overlap but in same category, likely complementary
        return len(components1.intersection(components2)) < 2 and len(components1) > 0 and len(components2) > 0

    def _apply_discovered_relationships(self, templates: Dict[str, Dict[str, Any]],
                                       relationships: Dict[str, List[Dict[str, Any]]]) -> int:
        """Apply discovered relationships to templates"""
        updated_count = 0

        for template_id, related_templates in relationships.items():
            if template_id not in templates:
                continue

            template_info = templates[template_id]
            template = template_info['data']
            path = template_info['path']

            # Get existing relationships
            existing_relationships = template.get('relationships', [])
            existing_related_ids = {rel.get('related_id') for rel in existing_relationships}

            # Add new relationships
            added = False
            for rel in related_templates:
                related_id = rel.get('related_id')
                if related_id and related_id not in existing_related_ids:
                    existing_relationships.append(rel)
                    existing_related_ids.add(related_id)
                    added = True

            if added:
                # Update template
                template['relationships'] = existing_relationships

                # Save updated template
                with open(path, 'w') as f:
                    json.dump(template, f, indent=2)

                updated_count += 1

        return updated_count

async def run_service(analyzer):
    """Run the relationship analyzer service"""
    # Initialize the analyzer
    if not await analyzer.initialize():
        return

    # Process events until shutdown
    await analyzer.process_events()

def main():
    parser = argparse.ArgumentParser(description="Event-driven template relationship analyzer service")
    parser.add_argument("--template-dir", default="./templates", help="Template directory")
    parser.add_argument("--broker-url", default="pulsar://localhost:6650", help="Pulsar broker URL")

    args = parser.parse_args()

    # Create analyzer
    analyzer = EventRelationshipAnalyzer(args.template_dir, args.broker_url)

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Shutting down...")
        analyzer.shutdown()
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Run service
        loop.run_until_complete(run_service(analyzer))
    finally:
        loop.close()

if __name__ == "__main__":
    main()
EOF

chmod +x $EVENT_REL_DIR/event_relationship_analyzer.py

# Create a client script to send relationship analysis requests
cat > $EVENT_REL_DIR/request_analysis.py << 'EOF'
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
EOF

chmod +x $EVENT_REL_DIR/request_analysis.py

echo "Event-driven Relationship Analyzer created successfully"