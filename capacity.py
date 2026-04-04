import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
from collections import deque

logger = logging.getLogger(__name__)

class CapacityStatus(Enum):
    OPEN       = "OPEN"
    MODERATE   = "MODERATE"
    NEAR_FULL  = "NEAR_FULL"
    FULL       = "FULL"

STATUS_COLORS = {
    CapacityStatus.OPEN:      (0, 200, 100),
    CapacityStatus.MODERATE:  (0, 165, 255),
    CapacityStatus.NEAR_FULL: (0, 100, 255),
    CapacityStatus.FULL:      (0, 0, 220),
}

STATUS_MESSAGES = {
    CapacityStatus.OPEN:      "Welcome — plenty of space available.",
    CapacityStatus.MODERATE:  "Moderate occupancy — please maintain distance.",
    CapacityStatus.NEAR_FULL: "Nearing capacity — consider another area.",
    CapacityStatus.FULL:      "At capacity — please use an alternative space.",
}

@dataclass
class ZoneConfig:
    name: str
    max_capacity: int
    camera_id: int
    thresholds: dict = field(default_factory=lambda: {
        "open":      0.50,
        "moderate":  0.75,
        "near_full": 0.90,
    })

@dataclass
class ZoneSnapshot:
    zone_name: str
    count: int
    max_capacity: int
    violations: int
    status: CapacityStatus
    occupancy_pct: float
    timestamp: float = field(default_factory=time.time)

    @property
    def message(self) -> str:
        return STATUS_MESSAGES[self.status]

    @property
    def color_bgr(self) -> tuple[int, int, int]:
        return STATUS_COLORS[self.status]

class CapacityManager:
    def __init__(
        self,
        zones: list[ZoneConfig],
        history_window: int = 300,
        smoothing_frames: int = 10,
    ):
        self.zones = {z.name: z for z in zones}
        self.smoothing_frames = smoothing_frames

        self._count_buffers: dict[str, deque] = {
            z.name: deque(maxlen=smoothing_frames) for z in zones
        }

        self._snapshots: dict[str, ZoneSnapshot] = {}

        self._history: dict[str, deque] = {
            z.name: deque(maxlen=history_window * 2) for z in zones
        }

        self._on_status_change: list[Callable] = []

        self._prev_status: dict[str, CapacityStatus] = {}

    def register_status_callback(self, fn: Callable):
        self._on_status_change.append(fn)

    def update(self, zone_name: str, count: int, violations: int = 0) -> ZoneSnapshot:
        if zone_name not in self.zones:
            raise ValueError(f"Unknown zone: {zone_name}")

        config = self.zones[zone_name]

        self._count_buffers[zone_name].append(count)
        smoothed = round(sum(self._count_buffers[zone_name]) / len(self._count_buffers[zone_name]))

        occupancy_pct = smoothed / config.max_capacity if config.max_capacity > 0 else 0.0
        status = self._compute_status(occupancy_pct, config.thresholds)

        snapshot = ZoneSnapshot(
            zone_name=zone_name,
            count=smoothed,
            max_capacity=config.max_capacity,
            violations=violations,
            status=status,
            occupancy_pct=round(occupancy_pct * 100, 1),
        )

        self._snapshots[zone_name] = snapshot
        self._history[zone_name].append((snapshot.timestamp, smoothed))

        prev = self._prev_status.get(zone_name)
        if prev != status:
            for fn in self._on_status_change:
                try:
                    fn(zone_name, prev, status, snapshot)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")
            self._prev_status[zone_name] = status

        return snapshot

    @staticmethod
    def _compute_status(pct: float, thresholds: dict) -> CapacityStatus:
        if pct >= thresholds["near_full"]:
            return CapacityStatus.FULL
        elif pct >= thresholds["moderate"]:
            return CapacityStatus.NEAR_FULL
        elif pct >= thresholds["open"]:
            return CapacityStatus.MODERATE
        else:
            return CapacityStatus.OPEN

    def get_snapshot(self, zone_name: str) -> Optional[ZoneSnapshot]:
        return self._snapshots.get(zone_name)

    def get_all_snapshots(self) -> dict[str, ZoneSnapshot]:
        return dict(self._snapshots)

    def get_history(self, zone_name: str, last_n_seconds: int = 3600) -> list[tuple]:
        if zone_name not in self._history:
            return []
        cutoff = time.time() - last_n_seconds
        return [(t, c) for t, c in self._history[zone_name] if t >= cutoff]

    def total_occupancy(self) -> int:
        return sum(s.count for s in self._snapshots.values())

class SignageController:
    def __init__(self, capacity_manager: CapacityManager, output_mode: str = "opencv"):
        self.manager = capacity_manager
        self.output_mode = output_mode
        capacity_manager.register_status_callback(self._on_status_change)

    def _on_status_change(self, zone_name, old_status, new_status, snapshot):
        logger.info(
            f"[SIGNAGE] {zone_name}: {old_status} → {new_status} "
            f"({snapshot.count}/{snapshot.max_capacity} = {snapshot.occupancy_pct}%)"
        )
        if self.output_mode == "mqtt":
            self._publish_mqtt(zone_name, snapshot)
        elif self.output_mode == "http":
            self._post_http(zone_name, snapshot)

    def render_overlay(self, frame, zone_name: str):
        import cv2
        snapshot = self.manager.get_snapshot(zone_name)
        if snapshot is None:
            return frame

        h, w = frame.shape[:2]
        banner_h = 80
        banner = frame.copy()

        overlay = banner.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), snapshot.color_bgr, -1)
        cv2.addWeighted(overlay, 0.7, banner, 0.3, 0, banner)

        cv2.putText(
            banner,
            f"{zone_name.upper()}  {snapshot.count}/{snapshot.max_capacity}",
            (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        cv2.putText(
            banner,
            snapshot.status.value,
            (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        )

        bar_x = w - 220
        bar_y = 30
        bar_w = 180
        bar_h_px = 16
        filled_w = int(bar_w * snapshot.occupancy_pct / 100)
        cv2.rectangle(banner, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (80, 80, 80), -1)
        cv2.rectangle(banner, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h_px), snapshot.color_bgr, -1)
        cv2.putText(
            banner,
            f"{snapshot.occupancy_pct}%",
            (bar_x, bar_y + bar_h_px + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
        )

        return banner

    def _publish_mqtt(self, zone_name: str, snapshot: ZoneSnapshot):
        try:
            import paho.mqtt.publish as publish
            import json
            payload = {
                "zone":       zone_name,
                "count":      snapshot.count,
                "capacity":   snapshot.max_capacity,
                "occupancy":  snapshot.occupancy_pct,
                "status":     snapshot.status.value,
                "message":    snapshot.message,
                "timestamp":  snapshot.timestamp,
            }
            publish.single(
                topic=f"crowdsense/{zone_name}/status",
                payload=json.dumps(payload),
                hostname="localhost",
                port=1883,
            )
        except Exception as e:
            logger.error(f"MQTT publish failed: {e}")

    def _post_http(self, zone_name: str, snapshot: ZoneSnapshot):
        """POST zone status to a REST API endpoint."""
        try:
            import requests, json
            requests.post(
                url=f"http://localhost:5000/api/zone/{zone_name}",
                json={
                    "count":     snapshot.count,
                    "status":    snapshot.status.value,
                    "occupancy": snapshot.occupancy_pct,
                    "message":   snapshot.message,
                },
                timeout=1.0,
            )
        except Exception as e:
            logger.error(f"HTTP signage update failed: {e}")
