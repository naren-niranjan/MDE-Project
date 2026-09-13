#!/usr/bin/env bash
# Jetson network tuning for Lucid GigE cameras.
# Usage:  sudo ./setup_network.sh <iface>     (default iface: eth0)
#
# This script:
#   1. Sets MTU 9000 on the chosen NIC (jumbo frames).
#   2. Bumps kernel socket buffer sizes so GigE Vision streams don't drop packets.
#   3. Persists the sysctl settings.
# It does NOT install the Arena SDK — see the README for that.

set -euo pipefail

IFACE="${1:-eth0}"

if [[ $EUID -ne 0 ]]; then
    echo "Please run as root (sudo)." >&2
    exit 1
fi

echo "Setting MTU 9000 on $IFACE ..."
ip link set dev "$IFACE" mtu 9000 || {
    echo "Could not set MTU on $IFACE. Check interface name with: ip a"
    exit 1
}

echo "Applying socket buffer tuning ..."
sysctl -w net.core.rmem_default=33554432
sysctl -w net.core.rmem_max=536870912
sysctl -w net.core.wmem_default=33554432
sysctl -w net.core.wmem_max=536870912
sysctl -w net.core.netdev_max_backlog=2000

# Persist (idempotent)
SYSCTL_BLOCK="
# --- Lucid GigE Vision tuning ---
net.core.rmem_default=33554432
net.core.rmem_max=536870912
net.core.wmem_default=33554432
net.core.wmem_max=536870912
net.core.netdev_max_backlog=2000
"
if ! grep -q "Lucid GigE Vision tuning" /etc/sysctl.conf; then
    echo "$SYSCTL_BLOCK" >> /etc/sysctl.conf
    echo "Appended tuning to /etc/sysctl.conf"
fi

echo
echo "Network tuning done."
echo "To make MTU 9000 persistent, edit your netplan / NetworkManager config so that"
echo "$IFACE always comes up with MTU 9000 (this varies by Jetson L4T release)."
