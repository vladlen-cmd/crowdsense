#!/bin/bash
# CrowdSense — Raspberry Pi 5 Setup Script
# Run once after cloning: bash setup.sh

set -e   # abort on unhandled errors in critical sections (camera installs use || to handle failures)

echo "=========================================="
echo "  CrowdSense Setup — Raspberry Pi 4 / 5"
echo "=========================================="

# ---- System packages ----
echo "[1/5] Installing system packages..."
sudo apt-get update -qq

# Required — script will abort if these fail
sudo apt-get install -y \
    python3-pip python3-dev python3-venv \
    libatlas-base-dev \
    libjpeg-dev libopenjp2-7 \
    ffmpeg \
    v4l-utils \
    mosquitto mosquitto-clients \
    git wget curl

# Optional OpenCV video backend libraries — install best-effort, don't abort on failure
sudo apt-get install -y libavdevice-dev libavcodec-dev libavformat-dev libswscale-dev 2>/dev/null \
    || echo "  → libav packages unavailable — OpenCV will use a basic video backend"

# Camera support — package names differ between Pi 4 and Pi 5 / Bookworm versions
echo "  → Installing camera stack..."
# Pi 5 / newer Bookworm uses rpicam-apps; Pi 4 / older uses libcamera-apps
sudo apt-get install -y rpicam-apps       2>/dev/null \
    || sudo apt-get install -y libcamera-apps 2>/dev/null \
    || echo "  → libcamera-apps not available (may already be bundled in the OS image)"

# libcamera dev headers — optional, only needed if building OpenCV from source
sudo apt-get install -y libcamera-dev 2>/dev/null \
    || echo "  → libcamera-dev not installed (not required at runtime)"

# picamera2 — Python API for Pi Camera modules via libcamera
sudo apt-get install -y python3-picamera2 2>/dev/null \
    || echo "  → python3-picamera2 not available via apt (install manually if using Pi Camera)"

# Enable the V4L2 compat kernel module so libcamera cameras appear as /dev/videoN
# This is what lets OpenCV VideoCapture(0) see the Pi Camera Module
if ! grep -q "v4l2-compat" /etc/modules 2>/dev/null && \
   ! [ -f /etc/modules-load.d/crowdsense-camera.conf ]; then
    echo "v4l2-compat" | sudo tee /etc/modules-load.d/crowdsense-camera.conf > /dev/null
    echo "  → V4L2 compat layer configured (will activate after reboot)"
fi
sudo modprobe v4l2-compat 2>/dev/null \
    && echo "  → V4L2 compat module loaded — Pi Camera should now appear in /dev/video*" \
    || echo "  → Could not load v4l2-compat now (will load on next boot)"

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

_download_model() {
    local url="$1"
    echo "  → Trying: $url"
    wget -q --show-progress "$url" -O models/yolov5n-int8.tflite
    # Validate: a real TFLite file must be > 1 MB
    local size
    size=$(stat -c%s models/yolov5n-int8.tflite 2>/dev/null || echo 0)
    if [ "$size" -gt 1000000 ]; then
        echo "  → Model downloaded OK (${size} bytes)."
        return 0
    else
        echo "  → Download invalid (${size} bytes) — trying next source."
        rm -f models/yolov5n-int8.tflite
        return 1
    fi
}

if [ ! -f models/yolov5n-int8.tflite ]; then
    # Try Ultralytics assets CDN first, then the v7.0 release tag
    _download_model "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n-int8.tflite" || \
    _download_model "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-int8.tflite" || \
    echo "  ⚠ Model download failed. Export it manually and place at models/yolov5n-int8.tflite"
else
    size=$(stat -c%s models/yolov5n-int8.tflite 2>/dev/null || echo 0)
    if [ "$size" -gt 1000000 ]; then
        echo "  → Model already exists and looks valid (${size} bytes), skipping."
    else
        echo "  ⚠ Existing model file is too small (${size} bytes) — likely corrupt. Re-downloading..."
        rm -f models/yolov5n-int8.tflite
        _download_model "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n-int8.tflite" || \
        _download_model "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-int8.tflite" || \
        echo "  ⚠ Model download failed. Export it manually and place at models/yolov5n-int8.tflite"
    fi
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

# ---- Detect available cameras ----
echo ""
echo "  Detected video devices:"
if ls /dev/video* 2>/dev/null | head -10; then
    echo ""
    echo "  V4L2 camera info (run 'v4l2-ctl --list-devices' for full detail):"
    v4l2-ctl --list-devices 2>/dev/null | head -20 || true
else
    echo "  No /dev/video* devices found yet. Plug in cameras and reboot if needed."
fi
echo ""
echo "  → Update camera_id values in config.yaml to match the numbers above."
echo "    Example: /dev/video0 → camera_id: 0"

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