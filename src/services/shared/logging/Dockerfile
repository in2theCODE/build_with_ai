FROM python:3.13-alpine

WORKDIR /

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the logger code
COPY logger.py .
COPY db_handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MONGODB_URI=mongodb://admin:mongodb@event-aggregator-mongodb:27017/event_system?authSource=admin \
    MONGODB_DATABASE=event_system \
    MONGODB_COLLECTION=application_logs \
    PULSAR_SERVICE_URL=pulsar://pulsar:6650 \
    PULSAR_SUBSCRIPTION=logger-service \
    LOG_RETENTION_DAYS=30

# Run the logger service
CMD ["python", "logger.py"]