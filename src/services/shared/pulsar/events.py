import json
import logging
import pulsar
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class EventPublisher:
    """
    Publishes events to Pulsar topics for neural interpreter functions.
    Handles event serialization, batching, and error handling.
    """
    
    def __init__(
        self,
        pulsar_service_url: str,
        inference_topic: str,
        workflow_topic: str
    ):
        """
        Initialize the event publisher
        
        Args:
            pulsar_service_url: Pulsar service URL
            inference_topic: Topic for inference requests
            workflow_topic: Topic for workflow execution requests
        """
        self.pulsar_service_url = pulsar_service_url
        self.inference_topic = inference_topic
        self.workflow_topic = workflow_topic
        self.client = None
        self.producers = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the Pulsar client and producers"""
        try:
            # Create Pulsar client
            self.client = pulsar.Client(self.pulsar_service_url)
            
            # Create producers
            self.producers["inference"] = self.client.create_producer(
                topic=self.inference_topic,
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=10
            )
            
            self.producers["workflow"] = self.client.create_producer(
                topic=self.workflow_topic,
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=10
            )
            
            logger.info(f"Initialized Pulsar producers for topics: {self.inference_topic}, {self.workflow_topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Pulsar client: {e}")
            # Clean up if initialization fails
            await self.close()
            raise
    
    async def close(self):
        """Close Pulsar client and producers"""
        try:
            # Close producers
            for producer_name, producer in self.producers.items():
                try:
                    producer.close()
                except Exception as e:
                    logger.error(f"Error closing producer {producer_name}: {e}")
            
            # Close client
            if self.client:
                self.client.close()
            
            # Clear references
            self.producers = {}
            self.client = None
        except Exception as e:
            logger.error(f"Error closing Pulsar client: {e}")
    
    async def publish_inference_request(
        self,
        query: str,
        system_message: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        pattern_match: Optional[Dict[str, Any]] = None
    ):
        """
        Publish an inference request event to Pulsar
        
        Args:
            query: User query text
            system_message: Optional system context message
            session_id: Optional session identifier
            request_id: Optional request identifier
            pattern_match: Optional pattern match result
        """
        try:
            event = self._create_base_event("INFERENCE_REQUEST")
            event["payload"] = {
                "query": query,
                "system_message": system_message,
                "session_id": session_id,
                "request_id": request_id or event["event_id"],
                "pattern_match": pattern_match
            }
            
            event["metadata"]["priority"] = "high" if pattern_match else "medium"
            event["metadata"]["path"] = "reactive"
            
            # Publish event
            await self._publish_event("inference", event)
            logger.info(f"Published inference request {event['event_id']} to {self.inference_topic}")
        except Exception as e:
            logger.error(f"Failed to publish inference request: {e}", exc_info=True)
            # Re-raise to allow for retry logic
            raise
    
    async def publish_workflow_request(
        self,
        query: str,
        system_message: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        complexity_score: float = 0.0
    ):
        """
        Publish a workflow execution request event to Pulsar
        
        Args:
            query: User query text
            system_message: Optional system context message
            session_id: Optional session identifier
            request_id: Optional request identifier
            complexity_score: Query complexity score
        """
        try:
            event = self._create_base_event("WORKFLOW_EXECUTION_REQUEST")
            event["payload"] = {
                "query": query,
                "system_message": system_message,
                "session_id": session_id,
                "request_id": request_id or event["event_id"],
                "complexity_score": complexity_score
            }
            
            event["metadata"]["priority"] = "high" if complexity_score > 0.8 else "medium"
            event["metadata"]["path"] = "deliberative"
            
            # Publish event
            await self._publish_event("workflow", event)
            logger.info(f"Published workflow request {event['event_id']} to {self.workflow_topic}")
        except Exception as e:
            logger.error(f"Failed to publish workflow request: {e}", exc_info=True)
            # Re-raise to allow for retry logic
            raise
    
    async def publish_error_event(
        self,
        error_message: str,
        error_code: str,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Publish an error event to Pulsar
        
        Args:
            error_message: Error message
            error_code: Error code
            request_id: Optional request identifier
            context: Optional error context
        """
        try:
            event = self._create_base_event("ERROR")
            event["payload"] = {
                "error_message": error_message,
                "error_code": error_code,
                "request_id": request_id,
                "context": context or {}
            }
            
            event["metadata"]["priority"] = "high"
            event["metadata"]["error"] = True
            
            # Publish event - use workflow topic for errors
            await self._publish_event("workflow", event)
            logger.info(f"Published error event {event['event_id']} to {self.workflow_topic}")
        except Exception as e:
            logger.error(f"Failed to publish error event: {e}", exc_info=True)
    
    def _create_base_event(self, event_type: str) -> Dict[str, Any]:
        """
        Create a base event structure
        
        Args:
            event_type: Type of event
            
        Returns:
            Base event dictionary
        """
        return {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "source_container": "neural-interpreter",
            "event_type": event_type,
            "priority": 1,  # Higher number = higher priority
            "payload": {},
            "metadata": {},
            "version": "1.0"
        }
    
    async def _publish_event(self, producer_key: str, event: Dict[str, Any]):
        """
        Publish an event to the appropriate topic
        
        Args:
            producer_key: Producer key ("inference" or "workflow")
            event: Event to publish
        """
        if producer_key not in self.producers:
            raise ValueError(f"Unknown producer key: {producer_key}")
        
        # Serialize event
        event_json = json.dumps(event)
        
        # Get producer
        producer = self.producers[producer_key]
        
        # Publish
        producer.send(event_json.encode("utf-8"))

