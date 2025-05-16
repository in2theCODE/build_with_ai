"""
Dedicated logging service that subscribes to log events from Pulsar
and stores them in a database (e.g., MongoDB).
"""

import asyncio
import os
import json
import signal
import sys
from datetime import (
    datetime,
    timedelta,
    timezone,
)  # Ensure timezone aware for MongoDB TTL
import logging

try:
    import pulsar
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError:
    print(
        "Required libraries (pulsar-client, motor) not found. Please install them: pip install pulsar-client motor",
        file=sys.stderr,
    )
    sys.exit(1)

# --- LoggerService's Own Logging Configuration ---
# This service logs its own operational messages to stdout.
# It does NOT use PulsarLogHandler to send its own logs back to Pulsar.
LOG_LEVEL_SERVICE = os.environ.get("LOGGER_SERVICE_LOG_LEVEL", "INFO").upper()
numeric_log_level_service = getattr(logging, LOG_LEVEL_SERVICE, logging.INFO)

logging.basicConfig(
    level=numeric_log_level_service,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
# Dedicated logger for this service's internal messages
service_logger = logging.getLogger("LoggerService")


# --- End of LoggerService's Own Logging Configuration ---


class LoggerService:
    """
    Service that subscribes to log events from a Pulsar topic
    and stores them in MongoDB.
    """

    def __init__(self):
        """Initialize the logger service."""
        self.mongodb_uri = os.environ.get(
            "MONGODB_URI",
            "mongodb://admin:mongodb@event-aggregator-mongodb:27017/event_system?authSource=admin",
        )
        self.mongodb_database_name = os.environ.get("MONGODB_DATABASE", "event_system_logs")
        self.mongodb_collection_name = os.environ.get("MONGODB_LOG_COLLECTION", "application_logs")

        self.pulsar_service_url = os.environ.get("PULSAR_SERVICE_URL", "pulsar://pulsar:6650")
        self.pulsar_topic = os.environ.get("PULSAR_LOG_TOPIC", "log.entry")  # Topic to consume from
        self.pulsar_subscription_name = os.environ.get("PULSAR_LOG_SUBSCRIPTION", "logger-service-subscription")

        self.log_retention_days = int(os.environ.get("LOG_RETENTION_DAYS", 30))

        self.running = False
        self.mongodb_client: AsyncIOMotorClient = None
        self.db = None
        self.log_collection = None
        self.pulsar_client: pulsar.Client = None
        self.consumer: pulsar.Consumer = None

        self.stats = {
            "received_count": 0,
            "stored_count": 0,
            "error_count": 0,
            "start_time": datetime.now(timezone.utc),
        }

    async def start(self):
        """Start the logger service."""
        if self.running:
            service_logger.info("Logger service is already running.")
            return

        service_logger.info("Starting LoggerService...")
        self.running = True

        # Connect to MongoDB
        try:
            self.mongodb_client = AsyncIOMotorClient(self.mongodb_uri)
            # Ping to confirm connection
            await self.mongodb_client.admin.command("ping")
            self.db = self.mongodb_client[self.mongodb_database_name]
            self.log_collection = self.db[self.mongodb_collection_name]
            service_logger.info(f"Connected to MongoDB: {self.mongodb_database_name}.{self.mongodb_collection_name}")
            await self._create_mongodb_indexes()
        except Exception as e:
            service_logger.error(f"MongoDB connection error: {e}", exc_info=True)
            self.running = False
            return  # Critical failure

        # Connect to Pulsar
        try:
            self.pulsar_client = pulsar.Client(self.pulsar_service_url)
            self.consumer = self.pulsar_client.subscribe(
                topic=self.pulsar_topic,
                subscription_name=self.pulsar_subscription_name,
                consumer_type=pulsar.ConsumerType.Shared,  # Allow multiple instances
            )
            service_logger.info(
                f"Subscribed to Pulsar topic '{self.pulsar_topic}' with subscription '{self.pulsar_subscription_name}'"
            )
        except Exception as e:
            service_logger.error(f"Pulsar connection error: {e}", exc_info=True)
            if self.mongodb_client:
                self.mongodb_client.close()
            self.running = False
            return  # Critical failure

        # Start background tasks
        asyncio.create_task(self._process_log_messages())
        if self.log_retention_days > 0:  # Only run cleanup if retention is enabled
            # Manual cleanup is a backup; TTL index is primary
            asyncio.create_task(self._cleanup_old_logs_periodically())
        asyncio.create_task(self._report_stats_periodically())

        service_logger.info("LoggerService started successfully.")

    async def _create_mongodb_indexes(self):
        """Create MongoDB indexes for efficient querying and TTL."""
        service_logger.info("Ensuring MongoDB indexes exist...")
        try:
            await self.log_collection.create_index("timestamp")
            await self.log_collection.create_index("level")
            await self.log_collection.create_index("service_name")
            await self.log_collection.create_index("logger_name")
            # For searching by correlation IDs or other custom fields if added
            await self.log_collection.create_index("correlation_id", sparse=True)
            await self.log_collection.create_index("request_id", sparse=True)

            # TTL index for automatic log deletion
            if self.log_retention_days > 0:
                ttl_seconds = self.log_retention_days * 24 * 60 * 60
                # Check if TTL index exists or needs update
                existing_indexes = await self.log_collection.index_information()
                ttl_index_name = "timestamp_ttl"

                create_new_ttl = True
                if ttl_index_name in existing_indexes:
                    if existing_indexes[ttl_index_name].get("expireAfterSeconds") == ttl_seconds:
                        create_new_ttl = False
                    else:
                        # TTL value changed, drop old index
                        await self.log_collection.drop_index(ttl_index_name)
                        service_logger.info(
                            f"Dropped existing TTL index '{ttl_index_name}' to update retention period."
                        )

                if create_new_ttl:
                    await self.log_collection.create_index(
                        "timestamp",
                        name=ttl_index_name,
                        expireAfterSeconds=ttl_seconds,
                    )
                    service_logger.info(f"TTL index on 'timestamp' created/updated for {self.log_retention_days} days.")
            else:
                service_logger.info("Log retention (TTL index) is disabled as LOG_RETENTION_DAYS is 0 or less.")

            service_logger.info("MongoDB indexes ensured.")
        except Exception as e:
            service_logger.error(f"Error creating MongoDB indexes: {e}", exc_info=True)

    async def _process_log_messages(self):
        """Continuously process log messages from Pulsar."""
        service_logger.info("Starting log message processing loop...")
        while self.running:
            try:
                msg = self.consumer.receive(timeout_millis=1000)  # Timeout to allow checking self.running
                if msg:
                    try:
                        log_data_str = msg.data().decode("utf-8")
                        log_entry = json.loads(log_data_str)

                        # Add a received_at timestamp by the logger service
                        log_entry["_logger_service_received_at"] = datetime.now(timezone.utc).isoformat()

                        # Ensure timestamp is a datetime object for MongoDB TTL and queries
                        if "timestamp" in log_entry and isinstance(log_entry["timestamp"], str):
                            try:
                                # Attempt to parse ISO format, make it timezone-aware if not
                                parsed_ts = datetime.fromisoformat(log_entry["timestamp"].replace("Z", "+00:00"))
                                if parsed_ts.tzinfo is None:
                                    parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)  # Assume UTC if no tz
                                log_entry["timestamp"] = parsed_ts
                            except ValueError:
                                service_logger.warning(
                                    f"Could not parse timestamp: {log_entry['timestamp']}. Using current time."
                                )
                                log_entry["timestamp"] = datetime.now(timezone.utc)
                        elif "timestamp" not in log_entry:
                            log_entry["timestamp"] = datetime.now(timezone.utc)

                        await self.log_collection.insert_one(log_entry)
                        self.consumer.acknowledge(msg)
                        self.stats["received_count"] += 1
                        self.stats["stored_count"] += 1
                    except json.JSONDecodeError as je:
                        service_logger.error(f"Failed to decode JSON log message: {log_data_str[:200]}... Error: {je}")
                        self.consumer.acknowledge(msg)  # Acknowledge unparseable message to avoid reprocessing
                        self.stats["error_count"] += 1
                    except Exception as e:
                        service_logger.error(
                            f"Error processing or storing log message: {e}",
                            exc_info=True,
                        )
                        # Depending on error, might not acknowledge or might implement dead-letter queue
                        self.stats["error_count"] += 1
                        await asyncio.sleep(0.1)  # Brief pause on error
            except pulsar.Timeout:
                continue  # Normal timeout, check self.running and loop
            except Exception as e:
                service_logger.error(f"Pulsar receive error: {e}", exc_info=True)
                self.stats["error_count"] += 1
                await asyncio.sleep(1)  # Wait a bit longer if Pulsar client has issues

    async def _cleanup_old_logs_periodically(self):
        """Periodically clean up old logs (backup for TTL, or if TTL is not used)."""
        service_logger.info("Starting periodic old log cleanup task.")
        while self.running:
            await asyncio.sleep(24 * 60 * 60)  # Run once a day
            if not self.running:
                break

            try:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.log_retention_days)
                service_logger.info(f"Running manual cleanup for logs older than {cutoff_date.isoformat()}...")
                result = await self.log_collection.delete_many({"timestamp": {"$lt": cutoff_date}})
                if result.deleted_count > 0:
                    service_logger.info(f"Manually deleted {result.deleted_count} old log entries.")
            except Exception as e:
                service_logger.error(f"Error during periodic log cleanup: {e}", exc_info=True)

    async def _report_stats_periodically(self):
        """Report operational statistics periodically."""
        service_logger.info("Starting periodic stats reporting task.")
        while self.running:
            await asyncio.sleep(5 * 60)  # Report every 5 minutes
            if not self.running:
                break

            uptime = datetime.now(timezone.utc) - self.stats["start_time"]
            service_logger.info(
                f"Stats | Uptime: {str(uptime).split('.')[0]} | "
                f"Received: {self.stats['received_count']} | "
                f"Stored: {self.stats['stored_count']} | "
                f"Errors: {self.stats['error_count']}"
            )

    async def stop(self):
        """Stop the logger service gracefully."""
        if not self.running:
            return
        service_logger.info("Stopping LoggerService...")
        self.running = False  # Signal background tasks to stop

        # Give tasks a moment to finish
        await asyncio.sleep(1.5)

        if self.consumer:
            try:
                self.consumer.close()
                service_logger.info("Pulsar consumer closed.")
            except Exception as e:
                service_logger.error(f"Error closing Pulsar consumer: {e}", exc_info=True)
        if self.pulsar_client:
            try:
                self.pulsar_client.close()
                service_logger.info("Pulsar client closed.")
            except Exception as e:
                service_logger.error(f"Error closing Pulsar client: {e}", exc_info=True)
        if self.mongodb_client:
            try:
                self.mongodb_client.close()
                service_logger.info("MongoDB client closed.")
            except Exception as e:
                service_logger.error(f"Error closing MongoDB client: {e}", exc_info=True)

        service_logger.info("LoggerService stopped.")


async def main():
    """Main entry point for the LoggerService."""
    service = LoggerService()

    loop = asyncio.get_event_loop()

    def signal_handler():
        service_logger.info("Shutdown signal received. Initiating graceful shutdown...")
        asyncio.create_task(service.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await service.start()
        while service.running:  # Keep main alive while service is running
            await asyncio.sleep(1)
    except Exception as e:
        service_logger.critical(f"LoggerService encountered a critical error in main: {e}", exc_info=True)
    finally:
        if service.running:  # If stop wasn't called by signal
            service_logger.info("Main loop exited, ensuring service stop.")
            await service.stop()
        service_logger.info("LoggerService has shut down.")


if __name__ == "__main__":
    asyncio.run(main())
