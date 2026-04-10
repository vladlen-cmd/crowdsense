#!/bin/bash
# CrowdSense — Raspberry Pi 5 Setup Script
# Run once after cloning: bash setup.sh

set -e

echo "=========================================="
echo "  CrowdSense Setup — Raspberry Pi 5"
echo "=========================================="

# ---- System packages ----
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-dev python3-venv \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev libopenjp2-7 \
    mosquitto mosquitto-clients \
    git wget curl

# ---- Python environment ----
echo "[2/5] Setting up Python environment..."
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt --break-system-packages 2>/dev/null || \
pip install -r requirements.txt

# ---- TFLite runtime ----
echo "[3/5] Installing TFLite runtime..."
ARCH=$(uname -m)

if [ "$ARCH" = "aarch64" ]; then
    # Pi 5 (Bookworm / Python 3.11+): use ai-edge-litert (Google's replacement for tflite-runtime)
    pip install ai-edge-litert 2>/dev/null || \
    pip install tflite-runtime 2>/dev/null || \
    echo "  → TFLite install failed. Try: pip install ai-edge-litert"
else
    pip install tflite-runtime
fi

# ---- YOLOv5n model ----
echo "[4/5] Downloading YOLOv5n TFLite model..."
mkdir -p models
if [ ! -f models/yolov5n-int8.tflite ]; then
    wget -q --show-progress \
        https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-int8.tflite \
        -O models/yolov5n-int8.tflite
    echo "  → YOLOv5n INT8 model downloaded."
else
    echo "  → Model already exists, skipping."
fi

# ---- Enable Pi Camera ----
echo "[5/5] Configuring Pi Camera..."
# Pi 5 (Bookworm) uses /boot/firmware/config.txt; older Pi OS uses /boot/config.txt
if [ -f /boot/firmware/config.txt ]; then
    BOOT_CFG="/boot/firmware/config.txt"
else
    BOOT_CFG="/boot/config.txt"
fi
if ! grep -q "camera_auto_detect=1" "$BOOT_CFG" 2>/dev/null; then
    echo "camera_auto_detect=1" | sudo tee -a "$BOOT_CFG" > /dev/null
    echo "  → Pi Camera enabled in $BOOT_CFG. Reboot required."
else
    echo "  → Pi Camera already enabled."
fi

# ---- MQTT broker (for digital signage) ----
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
echo "  → MQTT broker started on port 1883."

# ---- Create log directory ----
mkdir -p logs data

# ---- Systemd service files ----
echo "[6/6] Installing systemd services..."

INSTALL_DIR="$(pwd)"
VENV_PYTHON="$INSTALL_DIR/.venv/bin/python"
SERVICE_USER="$(whoami)"

# crowdsense-monitor.service — main detection loop
sudo tee /etc/systemd/system/crowdsense-monitor.service > /dev/null <<SERVICE
[Unit]
Description=CrowdSense Crowd Monitor
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV_PYTHON main.py --zones cafeteria library auditorium
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE

# crowdsense-api.service — Flask REST API / dashboard backend
sudo tee /etc/systemd/system/crowdsense-api.service > /dev/null <<SERVICE
[Unit]
Description=CrowdSense API Server
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV_PYTHON api_server.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable crowdsense-monitor crowdsense-api
echo "  → Services installed. Start with:"
echo "    sudo systemctl start crowdsense-monitor crowdsense-api"
echo "  → View logs with:"
echo "    journalctl -u crowdsense-monitor -f"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  Manual usage:"
echo "    source .venv/bin/activate"
echo "    python main.py --zones cafeteria --display"
echo "    python api_server.py   (in a separate terminal)"
echo ""
echo "  Or start as system services (auto-run on boot):"
echo "    sudo systemctl start crowdsense-monitor crowdsense-api"
echo ""
echo "  Alert delivery is configured in config.yaml."
echo ""
echo "  If camera was just enabled, reboot first:"
echo "    sudo reboot"
echo "=========================================="
