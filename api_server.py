import json
import time
import threading
import logging
import base64
from pathlib import Path
try:
    from flask import Flask, jsonify, request, Response
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("Flask not installed. Run: pip install flask flask-cors")

logger = logging.getLogger(__name__)

_zone_store: dict = {}
_history_store: dict = {}
_heatmap_store: dict = {}

HISTORY_MAX = 720

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/api/zones", methods=["GET"])
    def get_all_zones():
        return jsonify({
            "zones": list(_zone_store.values()),
            "total_people": sum(z.get("count", 0) for z in _zone_store.values()),
            "timestamp": time.time(),
        })

    @app.route("/api/zones/<zone_name>", methods=["GET"])
    def get_zone(zone_name):
        zone = _zone_store.get(zone_name)
        if not zone:
            return jsonify({"error": f"Zone '{zone_name}' not found"}), 404
        return jsonify(zone)

    @app.route("/api/zones/<zone_name>", methods=["POST"])
    def update_zone(zone_name):
        data = request.get_json()
        _zone_store[zone_name] = {**data, "timestamp": time.time()}

        if zone_name not in _history_store:
            _history_store[zone_name] = []
        _history_store[zone_name].append({
            "t": time.time(),
            "count": data.get("count", 0),
            "occupancy": data.get("occupancy", 0),
            "violations": data.get("violations", 0),
        })
        if len(_history_store[zone_name]) > HISTORY_MAX:
            _history_store[zone_name] = _history_store[zone_name][-HISTORY_MAX:]

        return jsonify({"ok": True})

    @app.route("/api/zones/<zone_name>/forecast", methods=["GET"])
    def get_forecast(zone_name):
        mock_forecast = [
            {"minutes_ahead": i * 5, "occupancy_pct": min(95, 40 + i * 3)}
            for i in range(1, 13)
        ]
        return jsonify({"zone": zone_name, "forecast": mock_forecast})

    @app.route("/api/zones/<zone_name>/heatmap", methods=["GET"])
    def get_heatmap(zone_name):
        heatmap_bytes = _heatmap_store.get(zone_name)
        if not heatmap_bytes:
            return jsonify({"error": "No heatmap available yet"}), 404
        return Response(heatmap_bytes, mimetype="image/jpeg")

    @app.route("/api/zones/<zone_name>/heatmap.b64", methods=["GET"])
    def get_heatmap_b64(zone_name):
        heatmap_bytes = _heatmap_store.get(zone_name)
        if not heatmap_bytes:
            return jsonify({"error": "No heatmap available yet"}), 404
        encoded = base64.b64encode(heatmap_bytes).decode()
        return jsonify({"image": f"data:image/jpeg;base64,{encoded}"})

    @app.route("/api/metrics/history", methods=["GET"])
    def get_history():
        zone = request.args.get("zone")
        minutes = int(request.args.get("minutes", 60))
        cutoff = time.time() - minutes * 60

        if zone:
            data = [e for e in _history_store.get(zone, []) if e["t"] >= cutoff]
            return jsonify({"zone": zone, "history": data})
        else:
            result = {}
            for zname, entries in _history_store.items():
                result[zname] = [e for e in entries if e["t"] >= cutoff]
            return jsonify(result)

    @app.route("/api/alerts", methods=["GET"])
    def get_alerts():
        alerts = []
        for zone_name, zone in _zone_store.items():
            status = zone.get("status", "OPEN")
            if status in ("NEAR_FULL", "FULL"):
                alerts.append({
                    "level": "danger" if status == "FULL" else "warn",
                    "zone": zone_name,
                    "message": f"{zone_name.title()} at {zone.get('occupancy', 0)}% capacity.",
                    "ts": zone.get("timestamp", time.time()),
                })
            if zone.get("violations", 0) > 0:
                alerts.append({
                    "level": "danger",
                    "zone": zone_name,
                    "message": f"{zone.get('violations')} social distancing violation(s) detected.",
                    "ts": zone.get("timestamp", time.time()),
                })
        return jsonify({"alerts": alerts})

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "uptime": time.time(), "zones": list(_zone_store.keys())})
    return app

def update_heatmap(zone_name: str, heatmap_bgr):
    import cv2
    _, buf = cv2.imencode(".jpg", heatmap_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    _heatmap_store[zone_name] = bytes(buf)

def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    if not HAS_FLASK:
        logger.error("Flask not installed.")
        return
    app = create_app()
    logger.info(f"API server starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == "__main__":
    _zone_store = {
        "cafeteria": {"zone": "cafeteria", "count": 72, "capacity": 92, "occupancy": 78.3, "status": "NEAR_FULL", "violations": 2, "fps": 24.1},
        "library": {"zone": "library", "count": 38, "capacity": 60, "occupancy": 63.3, "status": "MODERATE", "violations": 0, "fps": 23.8},
        "auditorium": {"zone": "auditorium", "count": 37, "capacity": 100, "occupancy": 37.0, "status": "OPEN", "violations": 0, "fps": 24.2},
    }
    run_server(debug=True)