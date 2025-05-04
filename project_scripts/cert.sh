#!/bin/bash
# Generate TLS certificates for the code generation API gateway

# Exit on error
set -e

# Create required directories
mkdir -p ./certs
cd ./certs

# Generate root CA
echo "Generating Root CA..."
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 365 -key ca.key -out ca.crt -subj "/CN=Code Gen CA"

# Generate server certificates for Traefik
echo "Generating certificates for Traefik..."
openssl genrsa -out traefik.key 2048
openssl req -new -key traefik.key -out traefik.csr -subj "/CN=api.yourdomain.com"
openssl x509 -req -days 365 -in traefik.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out traefik.crt

# Generate certificates for Pulsar
echo "Generating certificates for Pulsar..."
openssl genrsa -out pulsar.key 2048
openssl req -new -key pulsar.key -out pulsar.csr -subj "/CN=pulsar.yourdomain.com"
openssl x509 -req -days 365 -in pulsar.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out pulsar.crt

# Generate client certificates
echo "Generating client certificates..."
openssl genrsa -out client.key 2048
openssl req -new -key client.key -out client.csr -subj "/CN=client"
openssl x509 -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt

# Convert client key to PKCS8 format for Pulsar
openssl pkcs8 -topk8 -inform PEM -outform PEM -in client.key -out client.key-pk8.pem -nocrypt

echo "Certificate generation complete!"