#!/usr/bin/env python3
"""
Simplified script to send a code generation request to your neural code generator.
This uses direct Pulsar communication without the security layers.
"""

import json
import uuid
import pulsar
from datetime import datetime

# Connect to Pulsar
client = pulsar.Client("pulsar://localhost:6655")

# Create a producer for the code generation requests topic
# This should match the input_topic configured in your EnhancedNeuralCodeGenerator
producer = client.create_producer("code-generation-requests")

# Create a sample code generation request
message = {
    "event_id": str(uuid.uuid4()),
    "event_type": "code_generation_requested",
    "source_container": "test-script",
    "timestamp": datetime.now().isoformat(),
    "payload": {
        "spec_sheet": {
            "function_name": "calculate_fibonacci",
            "description": "Calculate the nth Fibonacci number efficiently",
            "parameters": [
                {
                    "name": "n",
                    "type": "int",
                    "description": "Position in Fibonacci sequence",
                }
            ],
            "return_type": "int",
            "constraints": [
                "n must be a non-negative integer",
                "Return the nth Fibonacci number where F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2)",
            ],
            "examples": [
                {"input": {"n": 0}, "output": 0},
                {"input": {"n": 1}, "output": 1},
                {"input": {"n": 5}, "output": 5},
                {"input": {"n": 10}, "output": 55},
            ],
        },
        "target_language": "python",
    },
    "correlation_id": f"test-{datetime.now().timestamp()}",
    "metadata": {"test": True},
}

# Send the message
producer.send(json.dumps(message).encode("utf-8"))
print(f"Sent message with ID: {message['event_id']}")
print("Message sent to topic: code-generation-requests")

# Clean up
producer.close()
client.close()
