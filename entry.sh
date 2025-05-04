#!/bin/bash
set -e

# Configure environment based on container environment variables
export PYTHONPATH=/app:$PYTHONPATH

# Set up logging
LOG_LEVEL=${LOG_LEVEL:-INFO}
echo "Setting log level to $LOG_LEVEL"

# Wait for Pulsar to be ready
if [ -n "$PULSAR_SERVICE_URL" ]; then
    echo "Waiting for Pulsar at $PULSAR_SERVICE_URL to be ready..."

    # Extract host and port from Pulsar URL
    if [[ "$PULSAR_SERVICE_URL" =~ pulsar://([^:]+):([0-9]+) ]]; then
        PULSAR_HOST=${BASH_REMATCH[1]}
        PULSAR_PORT=${BASH_REMATCH[2]}

        # Wait for Pulsar port to be available
        until nc -z $PULSAR_HOST $PULSAR_PORT; do
            echo "Pulsar is unavailable - sleeping"
            sleep 2
        done

        echo "Pulsar is up - continuing"
    else
        echo "Could not parse Pulsar URL, skipping connection check"
    fi
fi

# Download models if they don't exist and MODEL_DOWNLOAD_URL is set
if [ -n "$MODEL_DOWNLOAD_URL" ] && [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Downloading model from $MODEL_DOWNLOAD_URL to $MODEL_PATH"
    mkdir -p "$MODEL_PATH"
    curl -L "$MODEL_DOWNLOAD_URL" | tar -xz -C "$MODEL_PATH"
fi

# Start the health check API in the background
if [ "${ENABLE_HEALTH_CHECK:-true}" = "true" ]; then
    echo "Starting health check API on port 8000"
    python3 -m program_synthesis_system.shared.health_check &
fi

# Run the actual service based on execution mode
EXEC_MODE=${EXEC_MODE:-service}

case "$EXEC_MODE" in
    service)
        echo "Starting Neural Code Generator as a service"
        python3 -m program_synthesis_system.components.neural_code_generator.service
        ;;
    standalone)
        echo "Starting Neural Code Generator in standalone mode"
        python3 -m program_synthesis_system.components.neural_code_generator.standalone "$@"
        ;;
    *)
        echo "Unknown execution mode: $EXEC_MODE"
        exit 1
        ;;
esac