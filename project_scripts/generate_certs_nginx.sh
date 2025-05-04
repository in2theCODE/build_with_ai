#!/bin/bash

# Generate CA private key and certificate
openssl genrsa -out ca.key 4096
openssl req -new -x509 -key ca.key -sha256 -subj "/C=US/ST=CA/O=Local Dev/CN=Local Root CA" -days 365 -out ca.crt

# Generate client certs
openssl genrsa -out client.key 4096
openssl req -new -key client.key -out client.csr -subj "/CN=internal-client"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -days 365

# Generate server private key and CSR
openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr -config <(
cat <<-EOF
[req]
default_bits = 4096
prompt = no
default_md = sha256
req_extensions = req_ext
distinguished_name = dn

[ dn ]
C=US
ST=California
L=San Francisco
O=Local Development
OU=Development
CN=localhost

[ req_ext ]
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = localhost
DNS.2 = *.localhost
IP.1 = 127.0.0.1
EOF
)

# Generate server certificate
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out server.crt -days 365 -sha256 \
    -extfile <(printf "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1")
