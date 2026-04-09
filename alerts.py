import json
import logging
import smtplib
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ALERT_HISTORY_FILE = Path("logs/alerts.json")
ALERT_HISTORY_MAX = 500

@dataclass
class Alert:
    zone: str
    level: str        # "info" | "warn" | "danger"
    alert_type: str   # "capacity" | "violation" | "peak_warning"
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

class AlertManager:
    def __init__(
        self,
        cooldown_minutes: float = 5.0,
        email_cfg: Optional[dict] = None,
        ntfy_cfg: Optional[dict] = None,
    ):
        self.cooldown_seconds = cooldown_minutes * 60
        self.email_cfg = email_cfg or {}
        self.ntfy_cfg = ntfy_cfg or {}
        self._last_triggered: dict[tuple, float] = defaultdict(float)
        self._history: list[Alert] = []
        self._load_history()

    def trigger(self, zone: str, level: str, message: str, alert_type: str) -> bool:
        key = (zone, alert_type)
        now = time.time()

        if now - self._last_triggered[key] < self.cooldown_seconds:
            return False

        self._last_triggered[key] = now
        alert = Alert(zone=zone, level=level, alert_type=alert_type, message=message)

        self._history.append(alert)
        if len(self._history) > ALERT_HISTORY_MAX:
            self._history = self._history[-ALERT_HISTORY_MAX:]

        logger.warning(f"[ALERT] [{level.upper()}] {zone}: {message}")

        if self.email_cfg.get("enabled"):
            self._send_email(alert)
        if self.ntfy_cfg.get("enabled"):
            self._send_ntfy(alert)

        self._save_history()
        return True

    def get_history(self, limit: int = 50) -> list[dict]:
        return [a.to_dict() for a in self._history[-limit:]]

    def on_status_change(self, zone_name: str, old_status, new_status, snapshot):
        from capacity import CapacityStatus

        if new_status == CapacityStatus.FULL:
            self.trigger(
                zone=zone_name,
                level="danger",
                alert_type="capacity",
                message=(
                    f"{zone_name.title()} is at full capacity "
                    f"({snapshot.count}/{snapshot.max_capacity} people)."
                ),
            )
        elif new_status == CapacityStatus.NEAR_FULL:
            self.trigger(
                zone=zone_name,
                level="warn",
                alert_type="capacity",
                message=(
                    f"{zone_name.title()} is nearing capacity "
                    f"({snapshot.occupancy_pct}% — {snapshot.count}/{snapshot.max_capacity})."
                ),
            )

    def on_violations(self, zone_name: str, violation_count: int):
        if violation_count > 0:
            self.trigger(
                zone=zone_name,
                level="warn",
                alert_type="violation",
                message=(
                    f"{violation_count} social distancing violation(s) detected in {zone_name}."
                ),
            )

    def on_peak_warning(self, zone_name: str, occupancy_pct: float, minutes_ahead: int):
        self.trigger(
            zone=zone_name,
            level="warn",
            alert_type="peak_warning",
            message=(
                f"{zone_name.title()} forecast: {occupancy_pct}% occupancy "
                f"expected in {minutes_ahead} minute(s)."
            ),
        )

    def _send_email(self, alert: Alert):
        cfg = self.email_cfg
        to_addrs = cfg.get("to_addrs", [])
        if not to_addrs:
            logger.warning("[ALERT] Email enabled but no to_addrs configured.")
            return
        try:
            body = (
                f"CrowdSense Alert\n"
                f"Zone:  {alert.zone}\n"
                f"Level: {alert.level.upper()}\n"
                f"Type:  {alert.alert_type}\n"
                f"Time:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}\n\n"
                f"{alert.message}"
            )
            msg = MIMEText(body)
            msg["Subject"] = f"[CrowdSense] {alert.level.upper()} — {alert.zone}"
            msg["From"] = cfg.get("from_addr", "crowdsense@localhost")
            msg["To"] = ", ".join(to_addrs)

            with smtplib.SMTP(cfg.get("smtp_host", "localhost"), cfg.get("smtp_port", 25)) as smtp:
                if cfg.get("smtp_port") == 587:
                    smtp.starttls()
                if cfg.get("username"):
                    smtp.login(cfg["username"], cfg["password"])
                smtp.send_message(msg)

            logger.info(f"[ALERT] Email sent for {alert.zone}/{alert.alert_type}")
        except Exception as e:
            logger.error(f"[ALERT] Email delivery failed: {e}")

    def _send_ntfy(self, alert: Alert):
        cfg = self.ntfy_cfg
        priority_map = {"info": "default", "warn": "high", "danger": "urgent"}
        try:
            import requests

            requests.post(
                f"{cfg.get('server', 'https://ntfy.sh')}/{cfg.get('topic', 'crowdsense')}",
                data=alert.message.encode(),
                headers={
                    "Title": f"CrowdSense \u2014 {alert.zone}",
                    "Priority": priority_map.get(alert.level, "default"),
                    "Tags": alert.alert_type,
                },
                timeout=5,
            )
            logger.info(f"[ALERT] ntfy notification sent for {alert.zone}/{alert.alert_type}")
        except Exception as e:
            logger.error(f"[ALERT] ntfy delivery failed: {e}")

    def _load_history(self):
        if ALERT_HISTORY_FILE.exists():
            try:
                raw = json.loads(ALERT_HISTORY_FILE.read_text())
                self._history = [Alert(**d) for d in raw]
                logger.debug(f"[ALERT] Loaded {len(self._history)} past alerts from disk.")
            except Exception as e:
                logger.warning(f"[ALERT] Could not load alert history: {e}")

    def _save_history(self):
        try:
            ALERT_HISTORY_FILE.parent.mkdir(exist_ok=True)
            ALERT_HISTORY_FILE.write_text(
                json.dumps([a.to_dict() for a in self._history], indent=2)
            )
        except Exception as e:
            logger.error(f"[ALERT] Could not save alert history: {e}")
