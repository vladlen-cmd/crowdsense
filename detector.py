import cv2
import numpy as np
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    center_x: float = field(init=False)
    center_y: float = field(init=False)

    def __post_init__(self):
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height


@dataclass
class FrameResult:
    count: int
    detections: list[Detection]
    violations: list[tuple[int, int]]
    fps: float
    latency_ms: float
    heatmap: Optional[np.ndarray] = None


class YOLOv5TFLiteDetector:
    PERSON_CLASS_ID = 0
    def __init__(
        self,
        model_path: str = "models/yolov5n-int8.tflite",
        input_size: int = 320,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        min_distance_meters: float = 1.0,
        pixels_per_meter: float = 50.0,
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_distance_px = min_distance_meters * pixels_per_meter
        self.pixels_per_meter = pixels_per_meter

        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._load_model()

    def _load_model(self):
        try:
            import tflite_runtime.interpreter as tflite
            self._interpreter = tflite.Interpreter(
                model_path=str(self.model_path),
                num_threads=4,
            )
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            logger.info(f"TFLite model loaded: {self.model_path}")
            logger.info(f"Input shape: {self._input_details[0]['shape']}")
        except ImportError:
            logger.warning("tflite_runtime not found. Install: pip install tflite-runtime")
            self._interpreter = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def _postprocess(
        self, output: np.ndarray, orig_w: int, orig_h: int
    ) -> list[Detection]:
        predictions = output[0]
        detections = []

        for pred in predictions:
            obj_conf = float(pred[4])
            if obj_conf < self.conf_threshold:
                continue

            class_probs = pred[5:]
            class_id = int(np.argmax(class_probs))
            class_conf = float(class_probs[class_id])
            score = obj_conf * class_conf

            if class_id != self.PERSON_CLASS_ID or score < self.conf_threshold:
                continue

            cx, cy, bw, bh = pred[:4]
            scale_x = orig_w / self.input_size
            scale_y = orig_h / self.input_size

            x1 = int((cx - bw / 2) * self.input_size * scale_x)
            y1 = int((cy - bh / 2) * self.input_size * scale_y)
            x2 = int((cx + bw / 2) * self.input_size * scale_x)
            y2 = int((cy + bh / 2) * self.input_size * scale_y)

            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            detections.append(Detection(x1, y1, x2, y2, score))

        return self._non_max_suppression(detections)

    def _non_max_suppression(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        detections.sort(key=lambda d: d.confidence, reverse=True)
        kept = []

        for det in detections:
            suppress = False
            for kept_det in kept:
                iou = self._compute_iou(det, kept_det)
                if iou > self.iou_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(det)

        return kept

    def _compute_iou(self, a: Detection, b: Detection) -> float:
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = a.area + b.area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def detect_social_distance_violations(
        self, detections: list[Detection]
    ) -> list[tuple[int, int]]:
        violations = []
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                a, b = detections[i], detections[j]
                dist = math.sqrt(
                    (a.center_x - b.center_x) ** 2 + (a.center_y - b.center_y) ** 2
                )
                if dist < self.min_distance_px:
                    violations.append((i, j))
        return violations

    def run_inference(self, frame: np.ndarray) -> list[Detection]:
        if self._interpreter is None:
            logger.error("Model not loaded.")
            return []

        orig_h, orig_w = frame.shape[:2]
        input_tensor = self._preprocess(frame)

        if self._input_details[0]["dtype"] == np.uint8:
            scale, zero_point = self._input_details[0]["quantization"]
            input_tensor = (input_tensor / scale + zero_point).astype(np.uint8)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])

        if self._output_details[0]["dtype"] == np.uint8:
            scale, zero_point = self._output_details[0]["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale

        return self._postprocess(output, orig_w, orig_h)

    def process_frame(self, frame: np.ndarray, heatmap_accumulator=None) -> FrameResult:
        t_start = time.perf_counter()

        detections = self.run_inference(frame)
        violations = self.detect_social_distance_violations(detections)

        latency_ms = (time.perf_counter() - t_start) * 1000
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0

        heatmap = None
        if heatmap_accumulator is not None:
            heatmap_accumulator.update(detections, frame.shape[:2])
            heatmap = heatmap_accumulator.render()

        return FrameResult(
            count=len(detections),
            detections=detections,
            violations=violations,
            fps=fps,
            latency_ms=latency_ms,
            heatmap=heatmap,
        )
