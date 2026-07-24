#!/bin/bash
# Mihomo VPN Setup Script for C500 Server
set -e

MIHOMO_VERSION="v1.19.4"
MIHOMO_URL="https://github.com/MetaCubeX/mihomo/releases/download/${MIHOMO_VERSION}/mihomo-linux-amd64-${MIHOMO_VERSION}.gz"

echo "=== Installing Mihomo VPN ==="

# 1. Download and install mihomo binary
if ! command -v mihomo &> /dev/null; then
    echo "[1/4] Downloading mihomo ${MIHOMO_VERSION}..."
    wget -O /tmp/mihomo.gz "${MIHOMO_URL}"
    gunzip /tmp/mihomo.gz
    mv /tmp/mihomo /usr/local/bin/mihomo
    chmod +x /usr/local/bin/mihomo
    echo "  mihomo installed to /usr/local/bin/mihomo"
else
    echo "[1/4] mihomo already installed: $(which mihomo)"
fi

# 2. Create config directory
echo "[2/4] Creating config directory..."
mkdir -p /etc/mihomo

# 3. Copy config (EDIT THIS FILE FIRST!)
echo "[3/4] Config file location: /etc/mihomo/config.yaml"
if [ ! -f /etc/mihomo/config.yaml ]; then
    if [ -f setup/vpn/config.yaml.example ]; then
        cp setup/vpn/config.yaml.example /etc/mihomo/config.yaml
        echo "  Template copied. EDIT /etc/mihomo/config.yaml with your proxy credentials!"
    else
        echo "  WARNING: config.yaml.example not found. Create /etc/mihomo/config.yaml manually."
    fi
else
    echo "  /etc/mihomo/config.yaml already exists, skipping."
fi

# 4. Download GeoIP data (optional, for GEOIP rules)
echo "[4/4] Downloading GeoIP data..."
if [ ! -f /etc/mihomo/geoip.dat ]; then
    wget -O /etc/mihomo/geoip.dat https://github.com/Loyalsoldier/v2ray-rules-dat/releases/latest/download/geoip.dat
fi
if [ ! -f /etc/mihomo/geosite.dat ]; then
    wget -O /etc/mihomo/geosite.dat https://github.com/Loyalsoldier/v2ray-rules-dat/releases/latest/download/geosite.dat
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start VPN:"
echo "  mihomo -d /etc/mihomo"
echo ""
echo "To test:"
echo "  export http_proxy=http://127.0.0.1:7897"
echo "  export https_proxy=http://127.0.0.1:7897"
echo "  curl -I https://www.google.com"
echo ""
echo "To run in background:"
echo "  nohup mihomo -d /etc/mihomo > /var/log/mihomo.log 2>&1 &"
