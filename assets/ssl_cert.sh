#!/bin/bash

# Create the ssl directory if it doesn't exist
mkdir -p ssl

# Generate CA private key
openssl genpkey -algorithm RSA -out ssl/ca.key

# Generate CA certificate
openssl req -x509 -new -nodes -key ssl/ca.key -sha256 -days 1024 -out ssl/ca.crt -subj "/C=US/ST=State/L=City/O=MyOrg/OU=MyUnit/CN=MyCA"

# Generate server private key
openssl genpkey -algorithm RSA -out ssl/server.key

# Create a configuration file for the server certificate in the ssl directory
cat > ssl/server.cnf << EOF
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = MyOrg
OU = MyUnit
CN = 192.168.0.126

[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = IP:192.168.0.126
EOF

# Generate a certificate signing request (CSR) for the server
openssl req -new -key ssl/server.key -out ssl/server.csr -config ssl/server.cnf

# Use the CA to sign the server's CSR and generate the server certificate
openssl x509 -req -in ssl/server.csr -CA ssl/ca.crt -CAkey ssl/ca.key -CAcreateserial -out ssl/server.crt -days 365 -sha256 -extfile ssl/server.cnf -extensions v3_req

# Clean up the CSR file
rm ssl/server.csr

echo "CA and server certificates generated successfully in the ssl/ directory!"
