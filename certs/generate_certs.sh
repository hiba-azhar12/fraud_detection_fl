#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  Génération des certificats mTLS X.509 — FL Finance Platform
#  Zero Trust : chaque nœud a son propre certificat signé par la CA
#  Usage : cd certs && bash generate_certs.sh
# ══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

echo "================================================="
echo "  Génération des certificats mTLS X.509"
echo "  FL Finance Platform — 4 banques + 1 serveur"
echo "================================================="

# ── Certificate Authority (CA) ──────────────────────────────
echo ""
echo "1. Génération de la CA (Certificate Authority)..."
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 365 \
    -out ca.crt -subj '/CN=FL-Finance-CA/O=FL-Finance/C=MA'
echo "   ca.crt ✓"

# ── Serveur d'agrégation ────────────────────────────────────
echo ""
echo "2. Certificat du serveur FL (aggregator)..."
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr \
    -subj '/CN=fl-server/O=FL-Finance/C=MA'
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
    -CAcreateserial -out server.crt -days 365 -sha256
rm server.csr
echo "   server.crt ✓"

# ── Certificats des 4 banques ──────────────────────────────
echo ""
echo "3. Certificats des 4 banques..."
for bank in bank_a bank_b bank_c bank_d; do
    openssl genrsa -out ${bank}.key 2048
    openssl req -new -key ${bank}.key -out ${bank}.csr \
        -subj "/CN=${bank}/O=FL-Finance/C=MA"
    openssl x509 -req -in ${bank}.csr -CA ca.crt -CAkey ca.key \
        -CAcreateserial -out ${bank}.crt -days 365 -sha256
    rm ${bank}.csr
    echo "   ${bank}.crt ✓"
done

echo ""
echo "================================================="
echo "  Certificats générés avec succès !"
echo "================================================="
echo ""
ls -la *.crt *.key
echo ""
echo "Test de sécurité : un 5ème container sans certificat"
echo "sera automatiquement rejeté par le serveur FL."
