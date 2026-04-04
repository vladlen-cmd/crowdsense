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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('crowdsense.main')

ZONE_CONFIGS = {
    'cafeteria': ZoneConfig(name='cafeteria',   max_capacity=92,  camera_id=0),
    'library':   ZoneConfig(name='library',     max_capacity=60,  camera_id=1),
    'auditorium':ZoneConfig(name='auditorium',  max_capacity=100, camera_id=2),
}

CAMERA_RESOLUTION = (640, 480)
DETECTION_MODEL   = 'models/yolov5n-int8.tflite'
SIGNAGE_MODE      = 'opencv'

class CameraThread(threading.Thread):
    def __init__(self, zone_name: str, camera_id: int, resolution: tuple[int, int]):
        super().__init__(daemon=True, name=f'cam-{zone_name}')
        self.zone_name = zone_name
        self.camera_id = camera_id
        self.resolution = resolution
        self.latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

    def run(self):
        self._running = True
        self._cap = cv2.VideoCapture(self.camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            logger.error(f'[{self.zone_name}] Cannot open camera {self.camera_id}')
            return

        logger.info(f'[{self.zone_name}] Camera {self.camera_id} opened.')

        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

        self._cap.release()

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
        show_display: bool = False,
    ):
        self.config   = config
        self.detector = detector
        self.capacity = capacity_manager
        self.signage  = signage
        self.show_display = show_display

        self.heatmap = HeatmapAccumulator(
            frame_size=CAMERA_RESOLUTION[::-1],
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

        now = time.time()
        if now - self._last_prediction_push >= 60:
            self.predictor.push_observation(result.count)
            self._last_prediction_push = now

        if self.show_display:
            annotated = self._annotate_frame(frame, result, snapshot)
            cv2.imshow(f'CrowdSense — {self.config.name}', annotated)

        self._frame_count += 1

        return {
            'zone':       self.config.name,
            'count':      result.count,
            'capacity':   self.config.max_capacity,
            'occupancy':  snapshot.occupancy_pct,
            'status':     snapshot.status.value,
            'violations': len(result.violations),
            'fps':        round(result.fps, 1),
            'latency_ms': round(result.latency_ms, 1),
        }

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


class CrowdSenseApp:
    def __init__(self, zone_names: list[str], show_display: bool = False):
        self.zone_names  = zone_names
        self.show_display = show_display
        self._running = False

        self.zone_configs = [ZONE_CONFIGS[n] for n in zone_names if n in ZONE_CONFIGS]
        if not self.zone_configs:
            raise ValueError(f'No valid zones in {zone_names}. Available: {list(ZONE_CONFIGS)}')

        self.detector = YOLOv5TFLiteDetector(
            model_path=DETECTION_MODEL,
            input_size=320,
            conf_threshold=0.45,
        )

        self.capacity_manager = CapacityManager(zones=self.zone_configs)
        self.signage = SignageController(self.capacity_manager, output_mode=SIGNAGE_MODE)

        self.processors: dict[str, ZoneProcessor] = {
            cfg.name: ZoneProcessor(cfg, self.detector, self.capacity_manager, self.signage, show_display)
            for cfg in self.zone_configs
        }

        self.cameras: dict[str, CameraThread] = {
            cfg.name: CameraThread(cfg.name, cfg.camera_id, CAMERA_RESOLUTION)
            for cfg in self.zone_configs
        }

        self._metrics_log = []

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def run(self):
        logger.info(f'Starting CrowdSense for zones: {self.zone_names}')
        for cam in self.cameras.values():
            cam.start()

        time.sleep(2.0)
        self._running = True

        fps_clock = time.perf_counter()
        frame_count = 0

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
                            occupancy_pct = warning['occupancy_pct']
                            minutes_ahead = warning['minutes_ahead']
                            logger.warning(
                                f'[PEAK WARNING] {zone_name}: {occupancy_pct}% '
                                f'predicted in {minutes_ahead} min'
                            )

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
        if self.show_display:
            cv2.destroyAllWindows()

        log_path = Path('logs/metrics.json')
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(self._metrics_log[-1000:], f, indent=2)
        logger.info(f'Metrics saved → {log_path}')

def main():
    parser = argparse.ArgumentParser(description='CrowdSense — Crowd Density Monitor')
    parser.add_argument(
        '--zones', nargs='+', default=['cafeteria'],
        choices=list(ZONE_CONFIGS.keys()),
        help='Zones to monitor (space-separated)',
    )
    parser.add_argument(
        '--display', action='store_true',
        help='Show live OpenCV window (requires monitor connected to Pi)',
    )
    args = parser.parse_args()

    app = CrowdSenseApp(zone_names=args.zones, show_display=args.display)
    app.run()

if __name__ == '__main__':
    main()
