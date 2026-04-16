#!/usr/bin/env python3
"""
Download the YOLOv5n INT8 TFLite model required by CrowdSense.
Run:  python download_model.py
"""
import sys
import urllib.request
import urllib.error
from pathlib import Path

MODEL_PATH = Path("models/yolov5n-int8.tflite")
MIN_SIZE   = 1_500_000   # valid model is ~3.8 MB; reject anything smaller

# Sources tried in order — ordered by reliability
URLS = [
    # Ultralytics GitHub releases (multiple tags in case one is missing)
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-int8.tflite",
    "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n-int8.tflite",
    "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n-int8.tflite",
    # Ultralytics CDN / assets repo
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n-int8.tflite",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5n-int8.tflite",
]


def is_valid_tflite(path: Path) -> bool:
    """Check file size and FlatBuffer magic bytes (TFL3 at offset 4)."""
    if not path.exists() or path.stat().st_size < MIN_SIZE:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(8)
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


def _pip_install(package: str) -> bool:
    import subprocess
    print(f"  Installing {package} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package, "-q"],
        timeout=300,
    )
    return result.returncode == 0


def try_ultralytics_export() -> bool:
    """Use the ultralytics package to export the model (auto-installs if missing)."""
    print("  Trying ultralytics export ...")
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        print("  ultralytics not installed — attempting auto-install (this may take a minute)...")
        if not _pip_install("ultralytics"):
            print("  pip install ultralytics failed.")
            return False
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError:
            print("  Still cannot import ultralytics after install.")
            return False

    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO("yolov5n.pt")
        model.export(format="tflite", int8=True, imgsz=320)
        # Locate the output file — ultralytics saves it as *int8*.tflite
        for candidate in sorted(Path(".").rglob("*int8*.tflite"), key=lambda p: p.stat().st_mtime, reverse=True):
            if candidate.resolve() != MODEL_PATH.resolve():
                candidate.rename(MODEL_PATH)
            print(f"  Exported model moved to {MODEL_PATH}")
            return True
        # Maybe it was already written to MODEL_PATH
        if is_valid_tflite(MODEL_PATH):
            print(f"  Model already at {MODEL_PATH}")
            return True
        print("  Export succeeded but could not locate output file.")
        return False
    except Exception as e:
        print(f"  Export failed: {e}")
        return False


def main():
    print("\n=== CrowdSense Model Download ===\n")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if is_valid_tflite(MODEL_PATH):
        size_mb = MODEL_PATH.stat().st_size / 1_048_576
        print(f"Model already present and valid ({size_mb:.1f} MB): {MODEL_PATH}")
        return 0

    if MODEL_PATH.exists():
        print(f"Existing file is corrupt or too small — removing and re-downloading.")
        MODEL_PATH.unlink()

    for url in URLS:
        if download(url, MODEL_PATH):
            print(f"\nModel saved to {MODEL_PATH}")
            return 0

    print("\nAll direct downloads failed — trying ultralytics export fallback...")
    if try_ultralytics_export():
        return 0

    print("\n" + "="*60)
    print("All automatic methods failed.")
    print("="*60)
    print("\nOption 1 — export on the Pi (installs PyTorch, ~1 GB):")
    print("  pip install ultralytics")
    print("  python -c \"from ultralytics import YOLO; YOLO('yolov5n.pt').export(format='tflite', int8=True, imgsz=320)\"")
    print("  # Then find the output file and copy it:")
    print("  find . -name '*int8*.tflite' -exec cp {} models/yolov5n-int8.tflite \\;")
    print()
    print("Option 2 — export on a PC/Mac, then copy to the Pi:")
    print("  # On your Mac/PC:")
    print("  pip install ultralytics")
    print("  python -c \"from ultralytics import YOLO; YOLO('yolov5n.pt').export(format='tflite', int8=True, imgsz=320)\"")
    print("  find . -name '*int8*.tflite'   # note the path")
    print("  scp <path>/yolov5n-int8.tflite pi@<PI_IP>:~/crowdsense/models/")
    print()
    print("Option 3 — copy models/yolov5n-int8.tflite directly to the Pi via USB/SD card")
    return 1


if __name__ == "__main__":
    sys.exit(main())
