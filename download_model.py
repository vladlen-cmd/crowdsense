#!/usr/bin/env python3
"""
Download the YOLOv5n INT8 TFLite model required by CrowdSense.
Run:  python download_model.py
"""
import sys
import struct
import urllib.request
import urllib.error
from pathlib import Path

MODEL_PATH = Path("models/yolov5n-int8.tflite")
MIN_SIZE   = 1_500_000   # valid model is ~3.8 MB; reject anything smaller

# Sources tried in order
URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n-int8.tflite",
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-int8.tflite",
]


def is_valid_tflite(path: Path) -> bool:
    """Check file size and FlatBuffer magic bytes (TFL3 at offset 4)."""
    if not path.exists() or path.stat().st_size < MIN_SIZE:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        # FlatBuffer: bytes [4:8] == b'TFL3'
        return header[4:8] == b"TFL3"
    except Exception:
        return False


def download(url: str, dest: Path) -> bool:
    print(f"  Trying {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CrowdSense/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := resp.read(65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  {pct:3d}%  {downloaded//1024} KB / {total//1024} KB", end="", flush=True)
        print()
        if is_valid_tflite(dest):
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  Model OK — {size_mb:.1f} MB")
            return True
        print(f"  Downloaded file is not a valid TFLite model (may be a 404 page). Skipping.")
        dest.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"  Failed: {e}")
        dest.unlink(missing_ok=True)
        return False


def try_ultralytics_export():
    """Last resort: use the ultralytics package to export the model."""
    print("  Trying ultralytics export (requires PyTorch — may be slow)...")
    try:
        from ultralytics import YOLO
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO("yolov5n.pt")
        model.export(format="tflite", int8=True, imgsz=320)
        # ultralytics exports to yolov5n_saved_model/yolov5n_full_integer_quant.tflite
        for candidate in Path(".").rglob("*int8*.tflite"):
            candidate.rename(MODEL_PATH)
            print(f"  Exported model moved to {MODEL_PATH}")
            return True
        print("  Export succeeded but could not locate output file.")
        return False
    except ImportError:
        print("  ultralytics not installed — skipping export.")
        return False
    except Exception as e:
        print(f"  Export failed: {e}")
        return False


def main():
    print("\n=== CrowdSense Model Download ===\n")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Already valid?
    if is_valid_tflite(MODEL_PATH):
        size_mb = MODEL_PATH.stat().st_size / 1_048_576
        print(f"Model already present and valid ({size_mb:.1f} MB): {MODEL_PATH}")
        return 0

    if MODEL_PATH.exists():
        print(f"Existing file is corrupt or too small — removing and re-downloading.")
        MODEL_PATH.unlink()

    # Try each URL
    for url in URLS:
        if download(url, MODEL_PATH):
            print(f"\nModel saved to {MODEL_PATH}")
            return 0

    # Last resort
    if try_ultralytics_export():
        return 0

    print("\nAll download attempts failed.")
    print("Manual fix — run this on the Pi:")
    print()
    print("  pip install ultralytics")
    print("  python -c \"")
    print("  from ultralytics import YOLO")
    print("  YOLO('yolov5n.pt').export(format='tflite', int8=True, imgsz=320)")
    print("  \"")
    print()
    print("Or copy yolov5n-int8.tflite manually to:  models/yolov5n-int8.tflite")
    return 1


if __name__ == "__main__":
    sys.exit(main())
