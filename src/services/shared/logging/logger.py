#!/usr/bin/env python3
"""
Dedicated logging service that subscribes to log events and stores them in MongoDB.
"""
import asyncio
import os
import json
import signal
import sys
from datetime import datetime, timedelta
import pulsar
from motor.motor_asyncio import AsyncIOMotorClient
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("logger_service")


class LoggerService:
    """Service that subscribes to log events and stores them in MongoDB."""

    def __init__(self):
        """Initialize the logger service."""
        # Configuration from environment
        self.mongodb_uri = os.environ.get(
            "MONGODB_URI",
            "mongodb://admin:mongodb@event-aggregator-mongodb:27017/event_system?authSource=admin",
        )
        self.mongodb_database = os.environ.get("MONGODB_DATABASE", "event_system")
        self.mongodb_collection = os.environ.get("MONGODB_COLLECTION", "application_logs")
        self.pulsar_service_url = os.environ.get("PULSAR_SERVICE_URL", "pulsar://pulsar:6650")
        self.pulsar_topic = os.environ.get("PULSAR_TOPIC", "log.entry")
        self.pulsar_subscription = os.environ.get("PULSAR_SUBSCRIPTION", "logger-service")
        self.log_retention_days = int(os.environ.get("LOG_RETENTION_DAYS", "30"))

        # State
        self.running = False
        self.mongodb_client = None
        self.db = None
        self.collection = None
        self.pulsar_client = None
        self.consumer = None
        self.stats = {
            "received_logs": 0,
            "stored_logs": 0,
            "errors": 0,
            "start_time": datetime.utcnow(),
        }

    async def start(self):
        """Start the logger service."""
        if self.running:
            logger.info("Logger service is already running")
            return

        logger.info("Starting logger service")
        self.running = True

        # Connect to MongoDB
        try:
            self.mongodb_client = AsyncIOMotorClient(self.mongodb_uri)
            await self.mongodb_client.admin.command("ping")
            self.db = self.mongodb_client[self.mongodb_database]
            self.collection = self.db[self.mongodb_collection]
            logger.info(f"Connected to MongoDB: {self.mongodb_database}.{self.mongodb_collection}")

            # Create indexes
            await self._create_indexes()
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}")
            self.running = False
            raise

        # Connect to Pulsar
        try:
            self.pulsar_client = pulsar.Client(self.pulsar_service_url)
            self.consumer = self.pulsar_client.subscribe(
                topic=self.pulsar_topic,
                subscription_name=self.pulsar_subscription,
                consumer_type=pulsar.ConsumerType.Shared,
            )
            logger.info(f"Connected to Pulsar: {self.pulsar_topic}")
        except Exception as e:
            logger.error(f"Pulsar connection error: {e}")
            self.running = False
            if self.mongodb_client:
                self.mongodb_client.close()
            raise

        # Start background tasks
        asyncio.create_task(self._process_logs())
        asyncio.create_task(self._cleanup_old_logs())
        asyncio.create_task(self._report_stats())

        logger.info("Logger service started successfully")

    async def _create_indexes(self):
        """Create MongoDB indexes for efficient querying."""
        logger.info("Creating MongoDB indexes")

        # Basic indexes
        await self.collection.create_index("timestamp")
        await self.collection.create_index("level")
        await self.collection.create_index("logger")

        # Context-based indexes
        await self.collection.create_index("correlation_id")
        await self.collection.create_index("request_id")
        await self.collection.create_index("container")

        # Compound indexes for common queries
        await self.collection.create_index([("timestamp", 1), ("level", 1)])

        # TTL index for automatic cleanup
        # This will delete logs older than the specified retention period
        await self.collection.create_index(
            "timestamp", expireAfterSeconds=self.log_retention_days * 24 * 60 * 60
        )

        logger.info("MongoDB indexes created successfully")

    async def _process_logs(self):
        """Process log messages from Pulsar."""
        logger.info("Starting log processing loop")

        while self.running:
            try:
                # Receive message with timeout
                message = self.consumer.receive(timeout_millis=1000)

                if message:
                    # Parse log data
                    log_data = json.loads(message.data())

                    # Store in MongoDB
                    await self._store_log(log_data)

                    # Acknowledge the message
                    self.consumer.acknowledge(message)

                    # Update stats
                    self.stats["received_logs"] += 1
                    self.stats["stored_logs"] += 1

            except pulsar.Timeout:
                # No message received, just continue
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error processing log message: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(1)  # Back off on error

    async def _store_log(self, log_data):
        """Store a log entry in MongoDB."""
        try:
            # Add received timestamp
            log_data["received_at"] = datetime.utcnow().isoformat()

            # Ensure timestamp is in the right format for TTL index
            if "timestamp" in log_data and isinstance(log_data["timestamp"], str):
                try:
                    # Parse the timestamp string to a datetime
                    dt = datetime.fromisoformat(log_data["timestamp"].replace("Z", "+00:00"))
                    # Store as a datetime object for MongoDB TTL index
                    log_data["timestamp"] = dt
                except ValueError:
                    # If parsing fails, use current time
                    log_data["timestamp"] = datetime.utcnow()
            else:
                # Default to current time if no timestamp
                log_data["timestamp"] = datetime.utcnow()

            # Insert into MongoDB
            await self.collection.insert_one(log_data)
        except Exception as e:
            logger.error(f"Error storing log: {e}")
            self.stats["errors"] += 1
            raise

    async def _cleanup_old_logs(self):
        """Clean up old logs (backup for TTL index)."""
        while self.running:
            try:
                # Calculate cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=self.log_retention_days)

                # Delete old logs
                result = await self.collection.delete_many({"timestamp": {"$lt": cutoff_date}})

                if result.deleted_count > 0:
                    logger.info(f"Deleted {result.deleted_count} old log entries")

                # Run once per day
                await asyncio.sleep(24 * 60 * 60)

            except Exception as e:
                logger.error(f"Error cleaning up old logs: {e}")
                await asyncio.sleep(60 * 60)  # Retry after an hour

    async def _report_stats(self):
        """Report statistics periodically."""
        while self.running:
            try:
                uptime = datetime.utcnow() - self.stats["start_time"]
                uptime_str = str(uptime).split(".")[0]  # Remove microseconds

                logger.info(
                    f"Stats - Uptime: {uptime_str}, "
                    f"Received: {self.stats['received_logs']}, "
                    f"Stored: {self.stats['stored_logs']}, "
                    f"Errors: {self.stats['errors']}"
                )

                # Report every 5 minutes
                await asyncio.sleep(5 * 60)

            except Exception as e:
                logger.error(f"Error reporting stats: {e}")
                await asyncio.sleep(60)  # Retry after a minute

    async def stop(self):
        """Stop the logger service."""
        if not self.running:
            return

        logger.info("Stopping logger service")
        self.running = False

        # Close connections
        if self.consumer:
            self.consumer.close()

        if self.pulsar_client:
            self.pulsar_client.close()

        if self.mongodb_client:
            self.mongodb_client.close()

        logger.info("Logger service stopped")


# Run the service
async def main():
    """Run the logger service."""
    service = LoggerService()

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(service)))

    try:
        await service.start()
        # Keep running until stopped
        while service.running:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        await service.stop()


async def shutdown(service):
    """Shut down the service gracefully."""
    logger.info("Shutdown signal received")
    await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
