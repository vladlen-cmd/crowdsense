import argparse
import time
import logging
import threading
import signal
import sys
import json
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from detector  import YOLOv5TFLiteDetector, FrameResult
from heatmap   import HeatmapAccumulator, ZoneHeatmapManager
from capacity  import CapacityManager, SignageController, ZoneConfig, CapacityStatus
from predictor import CrowdPredictor
from alerts    import AlertManager
import api_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('crowdsense.main')

def _cv2_has_gui() -> bool:
    """Return False if OpenCV was built without GUI support (headless install)."""
    try:
        # A harmless probe — will raise if highgui is not compiled in
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False

CV2_GUI = _cv2_has_gui()

def load_config(path: str = "config.yaml") -> dict:
    defaults = {
        "zones": {
            "cafeteria":  {"max_capacity": 92,  "camera_id": 0},
            "library":    {"max_capacity": 60,  "camera_id": 1},
            "auditorium": {"max_capacity": 100, "camera_id": 2},
        },
        "camera":    {"resolution": [640, 480]},
        "detection": {
            "model_path": "models/yolov5n-int8.tflite",
            "input_size": 320,
            "conf_threshold": 0.45,
            "min_distance_px": 50,
        },
        "signage": {
            "mode": "opencv",
            "mqtt_host": "localhost",
            "mqtt_port": 1883,
            "http_url": "http://localhost:5000",
        },
        "alerts": {
            "cooldown_minutes": 5,
            "email": {"enabled": False},
            "ntfy":  {"enabled": False},
        },
    }

    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.warning(f"config.yaml not found — using built-in defaults.")
        return defaults

    try:
        import yaml
        with open(cfg_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        for key, val in user_cfg.items():
            if isinstance(val, dict) and key in defaults and isinstance(defaults[key], dict):
                defaults[key].update(val)
            else:
                defaults[key] = val
        return defaults
    except Exception as e:
        logger.error(f"Failed to parse config.yaml: {e}. Using defaults.")
        return defaults
    
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except Exception:
    HAS_PICAMERA2 = False


class CameraThread(threading.Thread):
    """
    Camera capture thread with three-tier fallback for Raspberry Pi:
      1. picamera2  — best for Pi Camera Module (CSI) on Bookworm/libcamera stack
      2. GStreamer libcamerasrc pipeline  — alternative libcamera path via OpenCV
      3. OpenCV VideoCapture (V4L2)  — USB webcams and legacy Pi camera with v4l2-compat
    """

    def __init__(self, zone_name: str, camera_id: int, resolution: tuple[int, int]):
        super().__init__(daemon=True, name=f'cam-{zone_name}')
        self.zone_name  = zone_name
        self.camera_id  = camera_id
        self.resolution = resolution
        self.latest_frame: Optional[np.ndarray] = None
        self._lock    = threading.Lock()
        self._running = False

    def run(self):
        self._running = True
        w, h = self.resolution

        # ── Tier 1: picamera2 (Pi Camera Module on Bookworm) ──────────────────
        if HAS_PICAMERA2:
            try:
                self._run_picamera2(w, h)
                return
            except Exception as e:
                logger.warning(f'[{self.zone_name}] picamera2 failed ({e}), trying GStreamer...')

        # ── Tier 2: GStreamer libcamerasrc pipeline ────────────────────────────
        gst = (
            f'libcamerasrc ! '
            f'video/x-raw,width={w},height={h},framerate=30/1 ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                logger.info(f'[{self.zone_name}] Camera {self.camera_id} opened via GStreamer/libcamera.')
                self._run_cap(cap)
                return
            cap.release()
        logger.warning(f'[{self.zone_name}] GStreamer pipeline failed, falling back to V4L2...')

        # ── Tier 3: standard OpenCV V4L2 (USB webcams / v4l2-compat) ─────────
        self._run_v4l2(w, h)

    def _run_picamera2(self, w: int, h: int):
        cam = Picamera2(self.camera_id)
        try:
            # Use RGB888 — picamera2's most reliable format — then convert to BGR for OpenCV
            cfg = cam.create_video_configuration(
                main={'size': (w, h), 'format': 'RGB888'},
                controls={'FrameRate': 30},
            )
            cam.configure(cfg)
            cam.start()
        except Exception:
            # Make sure we release the camera handle before re-raising so
            # it doesn't stay locked for the GStreamer / V4L2 fallback.
            try:
                cam.close()
            except Exception:
                pass
            raise

        logger.info(f'[{self.zone_name}] Camera {self.camera_id} opened via picamera2.')
        first_frame = True
        try:
            while self._running:
                frame = cam.capture_array("main")
                # Log shape/dtype once so format issues are immediately visible in logs
                if first_frame:
                    logger.info(
                        f'[{self.zone_name}] picamera2 frame shape={frame.shape} dtype={frame.dtype}'
                    )
                    first_frame = False
                # Handle both 3-channel RGB and 4-channel XRGB output
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]   # drop alpha / X channel → still RGB
                if frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                with self._lock:
                    self.latest_frame = frame
        finally:
            cam.stop()
            cam.close()

    def _run_v4l2(self, w: int, h: int):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.error(
                f'[{self.zone_name}] Cannot open camera {self.camera_id}. '
                f'Run  python main.py --scan-cameras  to see available IDs.'
            )
            return
        logger.info(f'[{self.zone_name}] Camera {self.camera_id} opened via V4L2/OpenCV.')
        self._run_cap(cap)

    def _run_cap(self, cap: cv2.VideoCapture):
        consecutive_failures = 0
        while self._running:
            ret, frame = cap.read()
            if ret:
                consecutive_failures = 0
                with self._lock:
                    self.latest_frame = frame
            else:
                consecutive_failures += 1
                if consecutive_failures == 30:
                    logger.error(
                        f'[{self.zone_name}] 30 consecutive read failures — camera is locked or '
                        f'disconnected. Kill all python/libcamera processes and restart:\n'
                        f'  pkill -f "python main.py" && pkill -f libcamera\n'
                        f'  sudo systemctl restart crowdsense-monitor\n'
                        f'Camera thread exiting.'
                    )
                    break
                time.sleep(0.01)
        cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self._running = False

class ZoneProcessor:
    def __init__(
        self,
        config: ZoneConfig,
        detector: YOLOv5TFLiteDetector,
        capacity_manager: CapacityManager,
        signage: SignageController,
        alert_manager: AlertManager,
        show_display: bool = False,
    ):
        self.config         = config
        self.detector       = detector
        self.capacity       = capacity_manager
        self.signage        = signage
        self.alert_manager  = alert_manager
        self.show_display   = show_display and CV2_GUI

        self.heatmap = HeatmapAccumulator(
            frame_size=tuple(reversed(config._resolution)),
            decay_factor=0.95,
            gaussian_radius=28,
            colormap='jet',
        )

        self.predictor = CrowdPredictor(
            zone_name=config.name,
            max_capacity=config.max_capacity,
            model_path=f'models/{config.name}_weights.json',
        )

        self._frame_count = 0
        self._last_prediction_push = 0

    def process(self, frame: np.ndarray) -> dict:
        result: FrameResult = self.detector.process_frame(frame, self.heatmap)

        snapshot = self.capacity.update(
            zone_name=self.config.name,
            count=result.count,
            violations=len(result.violations),
        )
        self.alert_manager.on_violations(self.config.name, len(result.violations))

        now = time.time()
        if now - self._last_prediction_push >= 60:
            self.predictor.push_observation(result.count)
            self._last_prediction_push = now

        annotated = self._annotate_frame(frame, result, snapshot)

        if self.show_display:
            cv2.imshow(f'CrowdSense — {self.config.name}', annotated)

        api_server.update_frame(self.config.name, annotated)
        api_server.update_heatmap(self.config.name, self.heatmap.render())

        self._frame_count += 1

        metrics = {
            'zone':       self.config.name,
            'count':      result.count,
            'capacity':   self.config.max_capacity,
            'occupancy':  snapshot.occupancy_pct,
            'status':     snapshot.status.value,
            'violations': len(result.violations),
            'fps':        round(result.fps, 1),
            'latency_ms': round(result.latency_ms, 1),
        }
        api_server.update_zone_data(self.config.name, metrics)
        return metrics

    def _annotate_frame(self, frame, result: FrameResult, snapshot) -> np.ndarray:
        annotated = self.heatmap.overlay_on_frame(frame)

        for i, det in enumerate(result.detections):
            color = (0, 255, 0)
            for a, b in result.violations:
                if i == a or i == b:
                    color = (0, 0, 220)
                    break
            cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            cv2.putText(
                annotated, f'{det.confidence:.2f}',
                (det.x1, det.y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

        for a_idx, b_idx in result.violations:
            a = result.detections[a_idx]
            b = result.detections[b_idx]
            cv2.line(
                annotated,
                (int(a.center_x), int(a.center_y)),
                (int(b.center_x), int(b.center_y)),
                (0, 0, 220), 2,
            )

        annotated = self.signage.render_overlay(annotated, self.config.name)

        cv2.putText(
            annotated,
            f'FPS: {result.fps:.0f}  Latency: {result.latency_ms:.0f}ms',
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
        )

        return annotated

    def get_forecast(self) -> list[dict]:
        return self.predictor.predict()

    def get_peak_warning(self) -> Optional[dict]:
        return self.predictor.peak_warning(warn_threshold=0.80)

    def get_heatmap_frame(self):
        return self.heatmap.render()

class CrowdSenseApp:
    def __init__(self, zone_names: list[str], cfg: dict, show_display: bool = False):
        self.zone_names   = zone_names
        self.cfg          = cfg
        self.show_display = show_display
        self._running     = False
        all_zone_cfgs = {}
        for name, z in cfg["zones"].items():
            zc = ZoneConfig(
                name=name,
                max_capacity=z["max_capacity"],
                camera_id=z["camera_id"],
                thresholds=z.get("thresholds", {}),
            )
            zc._resolution = cfg["camera"]["resolution"]  # stash for ZoneProcessor
            all_zone_cfgs[name] = zc

        self.zone_configs = [all_zone_cfgs[n] for n in zone_names if n in all_zone_cfgs]
        if not self.zone_configs:
            raise ValueError(f'No valid zones in {zone_names}. Available: {list(all_zone_cfgs)}')

        det = cfg["detection"]
        self.detector = YOLOv5TFLiteDetector(
            model_path=det["model_path"],
            input_size=det["input_size"],
            conf_threshold=det["conf_threshold"],
        )

        self.capacity_manager = CapacityManager(zones=self.zone_configs)
        sig_cfg = cfg["signage"]
        self.signage = SignageController(self.capacity_manager, output_mode=sig_cfg["mode"])
        alert_cfg = cfg["alerts"]
        self.alert_manager = AlertManager(
            cooldown_minutes=alert_cfg.get("cooldown_minutes", 5),
            email_cfg=alert_cfg.get("email"),
            ntfy_cfg=alert_cfg.get("ntfy"),
        )
        self.capacity_manager.register_status_callback(self.alert_manager.on_status_change)

        self.processors: dict[str, ZoneProcessor] = {
            zc.name: ZoneProcessor(
                zc, self.detector, self.capacity_manager,
                self.signage, self.alert_manager, show_display,
            )
            for zc in self.zone_configs
        }

        self.cameras: dict[str, CameraThread] = {
            zc.name: CameraThread(zc.name, zc.camera_id, tuple(cfg["camera"]["resolution"]))
            for zc in self.zone_configs
        }

        self._metrics_log = []

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def run(self):
        logger.info(f'Starting CrowdSense for zones: {self.zone_names}')
        if self.show_display and not CV2_GUI:
            logger.warning(
                '--display requested but OpenCV has no GUI support (headless build). '
                'Display disabled. Use the web dashboard at http://0.0.0.0:5000/ instead.'
            )
            self.show_display = False

        api_server.run_in_thread()

        for cam in self.cameras.values():
            cam.start()

        time.sleep(2.0)
        self._running = True
        frame_count   = 0

        try:
            while self._running:
                for zone_name, cam in self.cameras.items():
                    frame = cam.get_frame()
                    if frame is None:
                        continue

                    metrics = self.processors[zone_name].process(frame)
                    self._metrics_log.append(metrics)

                    frame_count += 1

                    if frame_count % 30 == 0:
                        logger.info(json.dumps(metrics))

                        warning = self.processors[zone_name].get_peak_warning()
                        if warning:
                            occ = warning['occupancy_pct']
                            mins = warning['minutes_ahead']
                            logger.warning(
                                f'[PEAK WARNING] {zone_name}: {occ}% predicted in {mins} min'
                            )
                            self.alert_manager.on_peak_warning(zone_name, occ, mins)

                if self.show_display and cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.04)

        finally:
            self._cleanup()

    def _shutdown(self, *args):
        logger.info('Shutting down CrowdSense...')
        self._running = False

    def _cleanup(self):
        for cam in self.cameras.values():
            cam.stop()
        if self.show_display and CV2_GUI:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

        log_path = Path('logs/metrics.json')
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(self._metrics_log[-1000:], f, indent=2)
        logger.info(f'Metrics saved → {log_path}')

def main():
    parser = argparse.ArgumentParser(description='CrowdSense — Crowd Density Monitor')
    parser.add_argument(
        '--zones', nargs='*', default=None,
        help='Zones to monitor (space-separated). Defaults to all zones in config.yaml.',
    )
    parser.add_argument(
        '--display', action='store_true',
        help='Show live OpenCV window (requires a monitor connected to the Pi).',
    )
    parser.add_argument(
        '--config', default='config.yaml',
        help='Path to config file (default: config.yaml).',
    )
    parser.add_argument(
        '--scan-cameras', action='store_true',
        help='Probe all /dev/video* devices and print which camera_ids are usable, then exit.',
    )
    args = parser.parse_args()

    if args.scan_cameras:
        _scan_cameras()
        return

    cfg = load_config(args.config)
    zone_names = args.zones if args.zones else list(cfg["zones"].keys())
    app = CrowdSenseApp(zone_names=zone_names, cfg=cfg, show_display=args.display)
    app.run()


def _scan_cameras():
    """Probe every /dev/video* device and report which camera_ids work with OpenCV."""
    import glob
    print("\nScanning for available cameras...\n")
    found = []
    devices = sorted(glob.glob('/dev/video*'))
    if not devices:
        # Fallback: probe indices 0-9 directly
        devices = [str(i) for i in range(10)]

    for dev in devices:
        try:
            idx = int(dev.replace('/dev/video', '')) if '/dev/video' in str(dev) else int(dev)
        except ValueError:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, _ = cap.read()
            status = "readable" if ret else "opened but no frame"
            print(f"  [OK]  camera_id: {idx}  ({w}x{h})  — {status}")
            found.append(idx)
        else:
            print(f"  [--]  camera_id: {idx}  — could not open")
        cap.release()

    print()
    if found:
        print(f"  Working camera IDs: {found}")
        print("  Update camera_id values in config.yaml to match.")
    else:
        print("  No working cameras found. Check connections and try rebooting.")
    print()

if __name__ == '__main__':
    main()
