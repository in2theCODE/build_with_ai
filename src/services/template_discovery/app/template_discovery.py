import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import pulsar
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TemplateRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    domain: Optional[str] = None
    language: Optional[str] = None
    framework: Optional[str] = None
    component: Optional[str] = None
    pattern: Optional[str] = None
    generator_type: str
    formal_spec: Dict[str, Any]
    additional_criteria: Dict[str, Any] = Field(default_factory=dict)


class TemplateResponse(BaseModel):
    request_id: str
    template_id: Optional[str]
    template_path: Optional[str]
    match_confidence: float = 0.0
    template_metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


def plate_metadata(file_path: str, relative_path: str) -> Dict[str, Any]:
    path_parts = relative_path.split(os.sep)
    metadata = {
        "path": file_path,
        "relative_path": relative_path,
        "domain": None,
        "language": None,
        "framework": None,
        "component": None,
        "pattern": None,
        "tags": []
    }
    try:
        for i, part in enumerate(path_parts):
            if i > 0 and path_parts[i - 1] == "domains":
                metadata["domain"] = part
            elif i > 0 and path_parts[i - 1] == "languages":
                metadata["language"] = part
            elif i > 0 and path_parts[i - 1] == "frameworks":
                metadata["framework"] = part
            elif i > 0 and path_parts[i - 1] == "services":
                metadata["component"] = part
            elif i > 0 and path_parts[i - 1] == "patterns":
                metadata["pattern"] = part

        with open(file_path, 'r') as f:
            header = ''.join([f.readline() for _ in range(10)])

            if "@template-info" in header:
                info_match = re.search(r'@template-info\s*({[^}]+})', header)
                if info_match:
                    try:
                        template_info = json.loads(info_match.group(1))
                        metadata.update(template_info)
                    except json.JSONDecodeError:
                        pass

            if "@tags" in header:
                tags_match = re.search(r'@tags\s*:?\s*([\w\s,]+)', header)
                if tags_match:
                    tags = [tag.strip() for tag in tags_match.group(1).split(',')]
                    metadata["tags"] = tags
    except:
        pass

    return metadata


def _score_template_match(request: TemplateRequest, metadata: Dict[str, Any]) -> float:
    score = 0.0

    if request.language:
        if metadata["language"] == request.language:
            score += 30.0
        elif metadata["language"] is None:
            score += 10.0
        else:
            return 0.0

    if request.domain and metadata["domain"] == request.domain:
        score += 20.0

    if request.framework and metadata["framework"] == request.framework:
        score += 15.0

    if request.component and metadata["component"] == request.component:
        score += 15.0

    if request.pattern and metadata["pattern"] == request.pattern:
        score += 10.0

    score += min(10.0, metadata["relative_path"].count(os.sep) * 2.0)

    if "function_name" in request.formal_spec and "function" in metadata["relative_path"].lower():
        score += 5.0

    if "class_name" in request.formal_spec and "class" in metadata["relative_path"].lower():
        score += 5.0

    for tag in metadata.get("tags", []):
        if tag in request.additional_criteria.get("tags", []):
            score += 3.0

    return score


class TemplateDiscoveryService:
    def __init__(self, template_root="src/templates", pulsar_url="pulsar://localhost:6650",
                 request_topic="persistent://public/default/template_requests",
                 response_topic="persistent://public/default/template_responses",
                 metrics_topic="persistent://public/default/metrics"):
        self.template_root = template_root
        self.pulsar_url = pulsar_url
        self.request_topic = request_topic
        self.response_topic = response_topic
        self.metrics_topic = metrics_topic
        self.templates = {}
        self.template_metadata = {}
        self.client = None
        self.producers = {}
        self.consumers = {}
        self.scan_templates()

    def scan_templates(self):
        logger.info(f"Scanning templates in {self.template_root}")
        for root, _, files in os.walk(self.template_root):
            for file in files:
                if file == ".gitkeep":
                    continue
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.template_root)
                template_id = relative_path.replace(os.sep, '.')
                metadata = plate_metadata(file_path, relative_path)
                self.templates[template_id] = file_path
                self.template_metadata[template_id] = metadata
        logger.info(f"Found {len(self.templates)} templates")

    async def initialize(self):
        try:
            self.client = pulsar.Client(self.pulsar_url)
            self.producers["responses"] = self.client.create_producer(
                topic=self.response_topic,
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=10
            )
            self.producers["metrics"] = self.client.create_producer(
                topic=self.metrics_topic,
                block_if_queue_full=True,
                batching_enabled=True,
                batching_max_publish_delay_ms=100
            )
            self.consumers["requests"] = self.client.subscribe(
                topic=self.request_topic,
                subscription_name="template-discovery-service",
                consumer_type=pulsar.ConsumerType.Shared
            )
            logger.info(f"Initialized Pulsar connections to {self.pulsar_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Pulsar client: {e}")
            await self.close()
            raise

    async def close(self):
        try:
            for producer_name, producer in self.producers.items():
                try:
                    producer.close()
                except Exception as e:
                    logger.error(f"Error closing producer {producer_name}: {e}")
            for consumer_name, consumer in self.consumers.items():
                try:
                    consumer.close()
                except Exception as e:
                    logger.error(f"Error closing consumer {consumer_name}: {e}")
            if self.client:
                self.client.close()
            self.producers = {}
            self.consumers = {}
            self.client = None
            logger.info("Closed all Pulsar connections")
        except Exception as e:
            logger.error(f"Error closing Pulsar client: {e}")

    def find_template(self, request: TemplateRequest) -> TemplateResponse:
        start_time = datetime.now(timezone.utc)
        try:
            scored_templates = []
            for template_id, template_path in self.templates.items():
                metadata = self.template_metadata[template_id]
                score = _score_template_match(request, metadata)
                if score > 0:
                    scored_templates.append((template_id, template_path, metadata, score))
            scored_templates.sort(key=lambda x: x[3], reverse=True)
            if scored_templates:
                best_match = scored_templates[0]
                response = TemplateResponse(
                    request_id=request.request_id,
                    template_id=best_match[0],
                    template_path=best_match[1],
                    match_confidence=best_match[3] / 100.0,
                    template_metadata=best_match[2]
                )
            else:
                response = TemplateResponse(
                    request_id=request.request_id,
                    template_id=None,
                    template_path=None,
                    match_confidence=0.0,
                    error="No matching template found"
                )
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"Found template for request {request.request_id} in {processing_time:.3f}s "
                        f"(confidence: {response.match_confidence:.2f})")
            return response
        except Exception as e:
            logger.error(f"Error finding template: {e}")
            return TemplateResponse(
                request_id=request.request_id,
                template_id=None,
                template_path=None,
                match_confidence=0.0,
                error=str(e)
            )

    async def publish_response(self, response: TemplateResponse):
        if "responses" not in self.producers:
            raise RuntimeError("Response producer not initialized")
        response_json = json.dumps(response.model_dump())
        self.producers["responses"].send(
            response_json.encode("utf-8"),
            properties={"request_id": response.request_id}
        )
        logger.info(f"Published response for request {response.request_id}")

    async def publish_metrics(self, metrics: Dict[str, Any]):
        if "metrics" not in self.producers:
            logger.warning("Metrics producer not initialized")
            return
        metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
        metrics["component"] = "template-discovery-service"
        metrics_json = json.dumps(metrics)
        self.producers["metrics"].send(metrics_json.encode("utf-8"))

    async def start_request_consumer(self):
        if "requests" not in self.consumers:
            raise RuntimeError("Requests consumer not initialized")
        logger.info(f"Starting to consume requests from {self.request_topic}")
        while True:
            try:
                msg = self.consumers["requests"].receive(timeout_millis=10000)
                try:
                    payload = json.loads(msg.data().decode("utf-8"))
                    try:
                        request = TemplateRequest(**payload)
                    except Exception as e:
                        logger.error(f"Invalid request format: {e}")
                        self.consumers["requests"].negative_acknowledge(msg)
                        continue
                    logger.info(f"Received request {request.request_id}")
                    start_time = datetime.now(timezone.utc)
                    response = self.find_template(request)
                    processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                    await self.publish_response(response)
                    await self.publish_metrics({
                        "event": "template_request_processed",
                        "request_id": request.request_id,
                        "processing_time_ms": int(processing_time * 1000),
                        "match_confidence": response.match_confidence,
                        "template_found": response.template_id is not None
                    })
                    self.consumers["requests"].acknowledge(msg)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON in message")
                    self.consumers["requests"].negative_acknowledge(msg)
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    self.consumers["requests"].negative_acknowledge(msg)
            except Exception as e:
                if "timeout" not in str(e).lower():
                    logger.error(f"Error receiving requests: {e}")
                    await asyncio.sleep(1)


async def run_service():
    service = TemplateDiscoveryService()
    try:
        service.scan_templates()
        await service.initialize()
        await service.start_request_consumer()
    except KeyboardInterrupt:
        logger.info("Shutting down Template Discovery Service...")
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(run_service())
