# CrowdSense
**Crowd Density & Social Distancing Monitor**  
Raspberry Pi 4 · Pi Camera · YOLOv5n · OpenCV · TFLite · Python 3

---

## Hardware

| Component | Spec |
|-----------|------|
| Raspberry Pi 4 Model B | 4GB or 8GB RAM |
| Pi Camera Module | V2 or HQ Camera |
| MicroSD | 32GB+ Class 10 |
| Power Supply | 5V 3A USB-C |

---

## Project Structure

```
crowdsense/
├── main.py          # Entry point — camera loop, detection, signage
├── detector.py      # YOLOv5n TFLite inference + social distance check
├── heatmap.py       # Density heatmap accumulation + rendering
├── capacity.py      # Zone capacity management + digital signage
├── predictor.py     # LSTM crowd forecasting model
├── api_server.py    # Flask REST API for web dashboard
├── models/
│   └── yolov5n-int8.tflite   # Download via setup.sh
├── logs/
│   └── metrics.json
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone and run setup (once)
git clone <your-repo> crowdsense
cd crowdsense
bash setup.sh

# 2. Activate environment
source .venv/bin/activate

# 3. Run detection (single zone, with display)
python main.py --zones cafeteria --display

# 4. Run all zones
python main.py --zones cafeteria library auditorium

# 5. Start the API server (in a separate terminal)
python api_server.py
```

---

## Key Modules

### `detector.py` — YOLOv5n Inference
- Loads `yolov5n-int8.tflite` via TFLite runtime (4-thread inference)
- Input: 320×320 BGR frame from Pi Camera
- Output: list of `Detection` objects (x1,y1,x2,y2, confidence)
- Social distance: Euclidean distance between bounding box centres vs. `min_distance_px`
- INT8 quantization: ~24 FPS at 320×320 on Pi 4 (no overclock needed)

### `heatmap.py` — Density Heatmap
- Maintains a decaying float32 accumulation buffer
- Stamps a 2D Gaussian kernel at each person's **foot position** (bottom-centre of box)
- `decay_factor=0.95` per frame — old data fades over ~60 frames
- `get_high_traffic_zones()` divides frame into grid, returns hottest cells
- Colormap: `cv2.COLORMAP_JET` (green→yellow→red)

### `capacity.py` — Zone Management
- Smooths raw counts over a sliding window (default: 10 frames) to reduce jitter
- Thresholds: OPEN (<50%), MODERATE (50–74%), NEAR_FULL (75–89%), FULL (≥90%)
- `SignageController` fires callbacks on status change
- Output modes: `opencv` overlay | `mqtt` broker | `http` REST POST

### `predictor.py` — Crowd Forecasting
- Lightweight NumPy LSTM (32 hidden units) — no framework overhead on Pi
- Input features: normalized count, hour_sin/cos, day_of_week_sin/cos
- Predicts 12 × 5-minute steps ahead (1 hour forecast)
- Train offline on a laptop with `train_model()`, copy weights JSON to Pi

### `api_server.py` — REST API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/zones` | GET | All zone snapshots |
| `/api/zones/<n>` | GET | Single zone |
| `/api/zones/<n>` | POST | Push metrics from main.py |
| `/api/zones/<n>/forecast` | GET | LSTM forecast |
| `/api/zones/<n>/heatmap` | GET | Heatmap JPEG |
| `/api/metrics/history` | GET | Rolling history |
| `/api/alerts` | GET | Active alerts |

---

## Performance on Pi 4 (8GB)

| Model | Input Size | FPS | Latency |
|-------|-----------|-----|---------|
| YOLOv5n INT8 | 320×320 | ~24 | ~42ms |
| YOLOv5n FP16 | 320×320 | ~16 | ~62ms |
| YOLOv5n INT8 | 640×640 | ~8  | ~125ms |

Recommended: **320×320 INT8** for real-time monitoring.

---

## MQTT Digital Signage

```bash
# Subscribe to all zone updates on any device in the same network
mosquitto_sub -h <pi-ip> -t "crowdsense/+/status"

# Example payload:
{
  "zone": "cafeteria",
  "count": 72,
  "capacity": 92,
  "occupancy": 78.3,
  "status": "NEAR_FULL",
  "message": "Nearing capacity — consider another area."
}
```

---

## Training the Forecast Model (on laptop)

```python
from predictor import train_model

# Your CSV: columns = timestamp, count
train_model(
    training_csv="data/cafeteria_history.csv",
    zone_name="cafeteria",
    max_capacity=92,
    output_weights_path="models/cafeteria_weights.json",
    epochs=150,
)
# Then copy models/cafeteria_weights.json to the Pi
```

---

## Adding a New Zone

1. Add entry to `ZONE_CONFIGS` in `main.py`:
   ```python
   "lobby": ZoneConfig(name="lobby", max_capacity=30, camera_id=3),
   ```
2. Connect the camera to USB port 3 (or configure additional Pi Camera via CSI)
3. Run: `python main.py --zones cafeteria library auditorium lobby`

---

## License
MIT — free to use and modify for educational and institutional purposes.
