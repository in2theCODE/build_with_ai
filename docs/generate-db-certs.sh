#!/bin/bash
# Database Certificate Generator
# This script generates self-signed certificates for database services
# to be used in development until Let's Encrypt is implemented

# Create base directories
mkdir -p ./docker/certs/{spec-registry-db,project-manager-db,workflow-engine-db,meta-learner-db,neo4j-vector,evolution-analyzer-vector-db,event-aggregator-mongodb}

# Common information for certificates
COUNTRY="US"
STATE="YourState"
LOCALITY="YourCity"
ORGANIZATION="YourOrganization"
ORGANIZATIONAL_UNIT="Development"
EMAIL="admin@example.com"
VALIDITY_DAYS=365

# Function to generate certificates for a service
generate_certs() {
    local service=$1
    local dir="./docker/certs/${service}"

    echo "Generating certificates for ${service}..."

    # Generate private key
    openssl genrsa -out "${dir}/private.key" 2048

    # Generate CSR (Certificate Signing Request)
    openssl req -new -key "${dir}/private.key" -out "${dir}/certificate.csr" -subj "/C=${COUNTRY}/ST=${STATE}/L=${LOCALITY}/O=${ORGANIZATION}/OU=${ORGANIZATIONAL_UNIT}/CN=${service}/emailAddress=${EMAIL}"

    # Generate self-signed certificate
    openssl x509 -req -days ${VALIDITY_DAYS} -in "${dir}/certificate.csr" -signkey "${dir}/private.key" -out "${dir}/certificate.crt"

    # Create combined file for services that need it
    cat "${dir}/certificate.crt" "${dir}/private.key" > "${dir}/fullchain.pem"

    # Clean up CSR
    rm "${dir}/certificate.csr"

    # Set appropriate permissions
    chmod 600 "${dir}/private.key"
    chmod 644 "${dir}/certificate.crt"
    chmod 600 "${dir}/fullchain.pem"

    echo "✅ Certificates for ${service} created successfully"
}

# Generate certificates for all database services
generate_certs "spec-registry-db"
generate_certs "project-manager-db"
generate_certs "workflow-engine-db"
generate_certs "meta-learner-db"
generate_certs "neo4j-vector"
generate_certs "evolution-analyzer-vector-db"
generate_certs "event-aggregator-mongodb"

echo "All database certificates generated successfully!"
echo "These are self-signed certificates for development use only."
echo "For production, Let's Encrypt certificates should be used."