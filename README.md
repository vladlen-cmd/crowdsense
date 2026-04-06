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
