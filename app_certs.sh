#!/bin/bash
# Generate TLS certificates for all services in the code generation system

# Exit on error
set -e

# Define base directory for certificates
BASE_DIR="/opt/certs"

# Define service list from the project structure
SERVICES=(
  "api_gateway"
  "neural_interpretor"
  "ast_code_generator"
  "constraint_relaxer"
  "feedback_collector"
  "grafana"
  "incremental_synthesis"
  "knowledge_base"
  "language_interop"
  "llm_volume"
  "meta_learner"
  "neural_code_generator"
  "project_manager"
  "prometheus"
  "spec_inference"
  "spec_registry"
  "specification_parser"
  "synthesis_engine"
  "template_discovery"
  "template_lib_volume"
  "verifier"
  "version_manager"
  "workflow_orchestrator"
  "workflow_registry"
)

# Infrastructure components
INFRA_COMPONENTS=(
  "traefik"
  "pulsar"
)

# Create main certificates directory
echo "Creating certificates directory structure..."
sudo mkdir -p $BASE_DIR
cd $BASE_DIR

# Generate root CA
echo "Generating Root CA..."
sudo openssl genrsa -out ca.key 4096
sudo openssl req -new -x509 -days 365 -key ca.key -out ca.crt -subj "/CN=Code Gen CA"

# Generate certificates for infrastructure components
for COMPONENT in "${INFRA_COMPONENTS[@]}"; do
  echo "Generating certificates for ${COMPONENT}..."
  sudo openssl genrsa -out ${COMPONENT}.key 2048
  sudo openssl req -new -key ${COMPONENT}.key -out ${COMPONENT}.csr -subj "/CN=${COMPONENT}.codegen.internal"
  sudo openssl x509 -req -days 365 -in ${COMPONENT}.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out ${COMPONENT}.crt

  # Convert key to PKCS8 format for Pulsar compatibility
  sudo openssl pkcs8 -topk8 -inform PEM -outform PEM -in ${COMPONENT}.key -out ${COMPONENT}.key-pk8.pem -nocrypt
done

# Generate certificates for each service
for SERVICE in "${SERVICES[@]}"; do
  echo "Generating certificates for ${SERVICE}..."

  # Create service directory
  sudo mkdir -p ${SERVICE}

  # Generate key and certificate
  sudo openssl genrsa -out ${SERVICE}/client.key 2048
  sudo openssl req -new -key ${SERVICE}/client.key -out ${SERVICE}/client.csr -subj "/CN=${SERVICE}"
  sudo openssl x509 -req -days 365 -in ${SERVICE}/client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out ${SERVICE}/client.crt

  # Convert key to PKCS8 format for Pulsar compatibility
  sudo openssl pkcs8 -topk8 -inform PEM -outform PEM -in ${SERVICE}/client.key -out ${SERVICE}/client.key-pk8.pem -nocrypt

  # Copy CA certificate to the service directory for convenience
  sudo cp ca.crt ${SERVICE}/ca.crt
done

# Create additional client certificate (for API clients)
echo "Generating general client certificates..."
sudo mkdir -p client
sudo openssl genrsa -out client/client.key 2048
sudo openssl req -new -key client/client.key -out client/client.csr -subj "/CN=api-client"
sudo openssl x509 -req -days 365 -in client/client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client/client.crt
sudo openssl pkcs8 -topk8 -inform PEM -outform PEM -in client/client.key -out client/client.key-pk8.pem -nocrypt
sudo cp ca.crt client/ca.crt

# Set appropriate permissions
echo "Setting file permissions..."
sudo chmod 644 ca.crt
sudo chmod 600 ca.key
sudo find $BASE_DIR -name "*.crt" -exec sudo chmod 644 {} \;
sudo find $BASE_DIR -name "*.key" -exec sudo chmod 600 {} \;
sudo find $BASE_DIR -name "*.key-pk8.pem" -exec sudo chmod 600 {} \;

echo "Certificate generation complete!"
echo "Generated certificates for ${#SERVICES[@]} services and ${#INFRA_COMPONENTS[@]} infrastructure components."
echo "All certificates are valid for 365 days."
echo "Certificates are stored in $BASE_DIR"