# deploy_pulsar.sh - Complete Pulsar deployment script

#!/bin/bash
set -e

# Create network
docker network create workflow-net 2>/dev/null || true

# Deploy ZooKeeper
docker run -d \
  --name workflow-zookeeper \
  --network workflow-net \
  -e ZOOKEEPER_CLIENT_PORT=2181 \
  -e ZOOKEEPER_TICK_TIME=2000 \
  -p 2181:2181 \
  wurstmeister/zookeeper:latest

# Deploy BookKeeper
docker run -d \
  --name workflow-bookkeeper \
  --network workflow-net \
  -e ZK_SERVERS=workflow-zookeeper:2181 \
  -e BOOKKEEPER_START_SERVICE=true \
  -e BOOKKEEPER_HTTP_SERVER_PORT=8080 \
  -p 8080:8080 \
  -p 4181:4181 \
  apachepulsar/pulsar:latest \
  bash -c "bin/apply-config-from-env.py conf/bookkeeper.conf && exec bin/bookkeeper bookie"

# Deploy Pulsar broker
docker run -d \
  --name workflow-pulsar \
  --network workflow-net \
  -e PULSAR_ZK_SERVERS=workflow-zookeeper:2181 \
  -e PULSAR_MEM="-Xms512m -Xmx512m" \
  -e PULSAR_METADATA_STORE_URL=zk:workflow-zookeeper:2181 \
  -e PULSAR_CONFIGURATION_METADATA_STORE=zk:workflow-zookeeper:2181 \
  -p 6650:6650 \
  -p 8081:8080 \
  apachepulsar/pulsar:latest \
  bin/pulsar broker

# Configure tenants and namespaces
sleep 10 # Wait for Pulsar to start
docker exec -it workflow-pulsar bin/pulsar-admin tenants create workflow
docker exec -it workflow-pulsar bin/pulsar-admin namespaces create workflow/codegen

# Create topics for the workflow stages
# Stage 1: Spec Generation
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/spec-generation-requests
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/spec-templates-generated

# Stage 2: Spec Completion
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/spec-completion-requests
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/specs-completed

# Stage 3: Code Generation
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/code-generation-requests
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/code-generated

# Stage 4: Integration
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/integration-requests
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/integration-completed

# Ad-hoc assistance
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/assistance-requests
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/assistance-responses

# System topics
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/system-errors
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/system-health

# Create a topic for state events (for event sourcing)
docker exec -it workflow-pulsar bin/pulsar-admin topics create persistent://workflow/codegen/state-events
docker exec -it workflow-pulsar bin/pulsar-admin topics create-partitioned-topic workflow/codegen/state-events -p 16

echo "Pulsar deployment complete"