import cv2
import numpy as np
from collections import deque

class HeatmapAccumulator:
    COLORMAP_OPTIONS = {
                'jet':      cv2.COLORMAP_JET,
                'hot':      cv2.COLORMAP_HOT,
                'inferno':  cv2.COLORMAP_INFERNO,
                'plasma':   cv2.COLORMAP_PLASMA,
    }

    def __init__(
        self,
        frame_size: tuple[int, int],
        decay_factor: float = 0.95,
        gaussian_radius: int = 30,
        colormap: str = 'jet',
        history_frames: int = 60,
        alpha: float = 0.6,
    ):
        self.h, self.w = frame_size
        self.decay_factor = decay_factor
        self.gaussian_radius = gaussian_radius
        self.colormap_id = self.COLORMAP_OPTIONS.get(colormap, cv2.COLORMAP_JET)
        self.alpha = alpha

        self._heat = np.zeros((self.h, self.w), dtype=np.float32)
        self._heat_normalized = self._heat.copy()   # safe initial state for render()

        self._history: deque = deque(maxlen=history_frames * 20)

        k_size = self.gaussian_radius * 2 + 1
        self._kernel = self._make_gaussian_kernel(k_size, sigma=self.gaussian_radius / 3)

    @staticmethod
    def _make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        ax = np.arange(-(size // 2), size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return (kernel / kernel.max()).astype(np.float32)

    def update(self, detections, frame_shape: tuple[int, int]):
        self._heat *= self.decay_factor

        for det in detections:
            cx = int(det.center_x)
            cy = int(det.y2)
            self._stamp_heat(cx, cy)
            self._history.append((cx, cy))

        max_val = self._heat.max()
        if max_val > 0:
            self._heat_normalized = self._heat / max_val
        else:
            self._heat_normalized = self._heat.copy()

    def _stamp_heat(self, cx: int, cy: int):
        r = self.gaussian_radius
        k = self._kernel
        k_h, k_w = k.shape

        x1 = cx - r;  y1 = cy - r
        x2 = cx + r + 1;  y2 = cy + r + 1

        hm_x1 = max(x1, 0);  hm_y1 = max(y1, 0)
        hm_x2 = min(x2, self.w);  hm_y2 = min(y2, self.h)

        k_x1 = hm_x1 - x1;  k_y1 = hm_y1 - y1
        k_x2 = k_w - (x2 - hm_x2);  k_y2 = k_h - (y2 - hm_y2)

        if hm_x2 > hm_x1 and hm_y2 > hm_y1:
            self._heat[hm_y1:hm_y2, hm_x1:hm_x2] += k[k_y1:k_y2, k_x1:k_x2]

    def render(self) -> np.ndarray:
        heat_u8 = (self._heat_normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat_u8, self.colormap_id)
        return colored

    def overlay_on_frame(self, frame: np.ndarray) -> np.ndarray:
        heatmap = self.render()

        gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        mask = (gray > 15).astype(np.float32)

        result = frame.copy()
        for c in range(3):
            result[:, :, c] = (
                frame[:, :, c] * (1 - mask * self.alpha)
                + heatmap[:, :, c] * mask * self.alpha
            ).astype(np.uint8)

        return result

    def get_high_traffic_zones(
        self, grid_rows: int = 4, grid_cols: int = 4, threshold: float = 0.5
    ) -> list[dict]:
        zones = []
        cell_h = self.h // grid_rows
        cell_w = self.w // grid_cols

        for r in range(grid_rows):
            for c in range(grid_cols):
                cell = self._heat_normalized[
                    r * cell_h:(r + 1) * cell_h,
                    c * cell_w:(c + 1) * cell_w,
                ]
                avg_heat = float(cell.mean())
                if avg_heat > threshold:
                    zones.append({
                        'row': r, 'col': c,
                        'heat': round(avg_heat, 3),
                        'bbox': (c * cell_w, r * cell_h, (c+1) * cell_w, (r+1) * cell_h),
                    })

        return sorted(zones, key=lambda z: z['heat'], reverse=True)

    def reset(self):
        self._heat.fill(0)
        self._history.clear()

class ZoneHeatmapManager:
    def __init__(self, zones: dict[str, tuple[int, int]], **heatmap_kwargs):
        self.accumulators: dict[str, HeatmapAccumulator] = {
            name: HeatmapAccumulator(size, **heatmap_kwargs)
            for name, size in zones.items()
        }

    def update(self, zone_name: str, detections, frame_shape: tuple[int, int]):
        if zone_name in self.accumulators:
            self.accumulators[zone_name].update(detections, frame_shape)

    def get_heatmap(self, zone_name: str) -> np.ndarray | None:
        acc = self.accumulators.get(zone_name)
        return acc.render() if acc else None

    def get_high_traffic(self, zone_name: str) -> list[dict]:
        acc = self.accumulators.get(zone_name)
        return acc.get_high_traffic_zones() if acc else []
