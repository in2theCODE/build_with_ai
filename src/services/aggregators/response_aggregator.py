import asyncio
import time
from typing import Dict, Any, List, Union, Callable

from src.services.shared.logging import logger
from src.services.shared.events.base_event import BaseEvent, EventType
from src.services.shared.constants.base_component import BaseComponent
from src.services.shared.logging.logger import get_logger

class ResponseAggregator(BaseComponent):

  logger = get_logger("ResponseAggregator")

  def __init__(self, **params):
      super().__init__(params)
      self.response_processors = {}

  def register_response_processor(self, event_type: Union[str, EventType], processor: Callable):
    """
    Register a custom processor for specific event types.

    Args:
        event_type: The event type to process
        processor: Function to process the event
    """
    if isinstance(event_type, EventType):
        event_type_str = event_type.value
    else:
        event_type_str = event_type

    self.response_processors[event_type_str] = processor
    self.logger.info(f"Registered custom processor for event type: {event_type_str}")


async def start(self):
    """Start the response aggregator."""
    if self.running:
        self.logger.info("ResponseAggregator is already running")
        return

    self.running = True
    self.logger.info("Starting ResponseAggregator")

    # Start the event listener
    await self.event_listener.start()

    # Start the main processing loop
    asyncio.create_task(self._process_events())
    asyncio.create_task(self._cleanup_stale_requests())


async def stop(self):
    """Stop the response aggregator."""
    self.logger.info("Stopping ResponseAggregator")
    self.running = False
    await self.event_listener.stop()


async def _process_events(self):
    """Process incoming events from the event listener."""
    while self.running:
        try:
            event = await self.event_listener.receive_event(timeout=1.0)
            if event:
                await self._handle_event(event)
        except Exception as e:
            self.logger.error(f"Error processing event: {str(e)}", exc_info=True)
            await asyncio.sleep(0.1)


async def _handle_event(self, event: BaseEvent):
    """
    Handle an incoming event.

    Args:
        event: The event to handle
    """
    self.logger.debug(f"Handling event: {event.event_type} with correlation_id: {event.correlation_id}")

    # Get the correlation ID from the event
    correlation_id = event.correlation_id

    # Check if this is part of an ongoing request
    if correlation_id not in self.pending_requests:
        # Initialize a new request entry
        self.pending_requests[correlation_id] = {
            'request_data': event.metadata.get('request_data', {}),
            'responses': [],
            'timestamp': time.time(),
            'timeout': event.metadata.get('timeout', 300),  # Default 5 minutes
            'expected_responses': event.metadata.get('expected_responses', 1)
        }

    # Add this response to the pending request
    self.pending_requests[correlation_id]['responses'].append({
        'event_type': event.event_type,
        'data': event.data,
        'timestamp': time.time()
    })

    # Process with custom handler if registered
    if event.event_type in self.response_processors:
        try:
            await self.response_processors[event.event_type](event, self.pending_requests[correlation_id])
        except Exception as e:
            self.logger.error(f"Error in custom processor for {event.event_type}: {str(e)}", exc_info=True)

    # Check if we have all expected responses
    await self._check_request_completion(correlation_id)


async def _check_request_completion(self, correlation_id: str):
    """
    Check if a request is complete and process it if so.

    Args:
        correlation_id: The correlation ID of the request
    """
    request_data = self.pending_requests.get(correlation_id)
    if not request_data:
        return

    responses = request_data['responses']
    expected = request_data.get('expected_responses', 1)

    if len(responses) >= expected:
        # Request is complete, process the aggregated response
        await self._process_completed_request(correlation_id)


async def _process_completed_request(self, correlation_id: str):
    """
    Process a completed request.

    Args:
        correlation_id: The correlation ID of the completed request
    """
    request_data = self.pending_requests.pop(correlation_id, None)
    if not request_data:
        return

    # Aggregate the responses
    aggregated_result = self._aggregate_responses(request_data['responses'])

    # Store the result in the database
    await self._store_result(correlation_id, aggregated_result)

    # Emit a completion event
    await self.event_emitter.emit_event(
        event_type=EventType.AGGREGATION_COMPLETED,
        correlation_id=correlation_id,
        data=aggregated_result,
        metadata={'original_request': request_data['request_data']}
    )

    self.logger.info(f"Completed request processing for correlation_id: {correlation_id}")


def _aggregate_responses(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple responses into a single result.

    Args:
        responses: List of response data

    Returns:
        Aggregated result dictionary
    """
    # Sort responses by timestamp
    sorted_responses = sorted(responses, key=lambda x: x['timestamp'])

    # Basic aggregation - can be extended for more complex logic
    result = {
        'status': 'success',
        'timestamp': time.time(),
        'response_count': len(responses),
        'responses': sorted_responses
    }

    # Check for failures
    failures = [r for r in responses if r['event_type'] == EventType.CODE_GENERATION_FAILED.value]
    if failures:
        result['status'] = 'partial_failure' if len(failures) < len(responses) else 'failure'
        result['failures'] = failures

    return result


async def _store_result(self, correlation_id: str, result: Dict[str, Any]):
    """
    Store the aggregated result in the database.

    Args:
        correlation_id: The correlation ID of the request
        result: The aggregated result
    """
    # Store in primary database if available
    if 'primary' in self.db_connections:
        try:
            db = self.db_connections['primary']
            # Implementation depends on database type
            # Example for a document database:
            await db.results.update_one(
                {'correlation_id': correlation_id},
                {'$set': result},
                upsert=True
            )
            logger.info(f"Stored result for correlation_id: {correlation_id}")
        except Exception as e:
            self.logger.error(f"Error storing result: {str(e)}", exc_info=True)


async def _cleanup_stale_requests(self):
    """Periodically clean up stale requests that have timed out."""
    while self.running:
        try:
            current_time = time.time()
            stale_ids = []

            for correlation_id, request_data in self.pending_requests.items():
                timeout = request_data.get('timeout', 300)  # Default 5 minutes
                if current_time - request_data['timestamp'] > timeout:
                    stale_ids.append(correlation_id)

            for correlation_id in stale_ids:
                self.logger.warning(f"Request timed out: {correlation_id}")
                request_data = self.pending_requests.pop(correlation_id)

                # Emit timeout event
                await self.event_emitter.emit_event(
                    event_type=EventType.AGGREGATION_TIMEOUT,
                    correlation_id=correlation_id,
                    data={'status': 'timeout', 'partial_responses': request_data['responses']},
                    metadata={'original_request': request_data['request_data']}
                )

        except Exception as e:
            self.logger.error(f"Error in cleanup task: {str(e)}", exc_info=True)

        await asyncio.sleep(10)  # Check every 10 seconds