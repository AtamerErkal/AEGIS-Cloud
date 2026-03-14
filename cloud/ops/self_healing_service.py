"""
AEGIS-Cloud — Cloud / Ops / Self-Healing Service
===================================================
Module:   cloud.ops.self_healing_service
Platform: Azure (serverless / container)

Analyses Edge telemetry (IMU vibration, inference latency, sensor health)
and issues Cloud-to-Device (C2D) motor-speed adjustment commands back to
the Edge layer via Azure IoT Hub — closing the AIOps self-healing loop.

    Edge sensors → D2C telemetry → SelfHealingService → C2D command → Edge motors

EU AI Act Article 14 (Human Oversight):
  All motor-speed commands require human approval before dispatch.
  Pending commands are logged to mlops/compliance/audit_trail.log.

Simulation mode activates when AEGIS_IOT_HUB_SERVICE_CONN_STR is absent.
C2D commands are written to data/sim_samples/c2d_commands.json instead.
"""

import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from azure.iot.hub import IoTHubRegistryManager
    _IOT_HUB_SDK_AVAILABLE = True
except ImportError:
    _IOT_HUB_SDK_AVAILABLE = False

_AUDIT_TRAIL_PATH = Path("mlops/compliance/audit_trail.log")
_SIM_COMMANDS_PATH = Path("data/sim_samples/c2d_commands.json")


class SelfHealingService:
    """
    AIOps self-healing service for closed-loop motor control.

    Attributes
    ----------
    vibration_threshold : float
        RMS vibration level (g) above which corrective action triggers.
    rpm_limits : tuple
        (min_rpm, max_rpm) safe operating range.
    human_approval_required : bool
        If True, commands are held for operator approval (EU AI Act Article 14).
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        motor_cfg = cfg.get("motor", {})
        aiops_cfg = cfg.get("aiops", {})

        self.vibration_threshold: float = float(
            motor_cfg.get("vibration_safe_threshold_g", 2.5)
        )
        self.rpm_limits: tuple = (
            int(motor_cfg.get("speed_min_rpm", 200)),
            int(motor_cfg.get("speed_max_rpm", 8000)),
        )
        self.human_approval_required: bool = True
        self.anomaly_alert_threshold: int = int(
            aiops_cfg.get("anomaly_alert_threshold", 3)
        )

        self._consecutive_anomalies: int = 0
        self._sim_mode: bool = not os.getenv("AEGIS_IOT_HUB_SERVICE_CONN_STR")
        self._device_id: str = (
            cfg.get("nato_metadata", {}).get("station_id", "aegis-jetson-nano")
        )

        _AUDIT_TRAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SIM_COMMANDS_PATH.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("AEGIS.SelfHealingService")
        self.logger.info(
            f"[AEGIS][SelfHeal] Initialized. sim={self._sim_mode} "
            f"vib_thresh={self.vibration_threshold}g rpm={self.rpm_limits}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_telemetry(self, telemetry_batch: list[dict]) -> dict | None:
        """
        Analyse a batch of PlatformState telemetry records.

        Returns a corrective action dict if an anomaly is detected,
        or None if the platform is operating within normal parameters.
        """
        if not telemetry_batch:
            return None

        # --- Vibration check ---
        vibration_readings = [
            t.get("vibration_rms_g", 0.0)
            for t in telemetry_batch
            if t.get("vibration_rms_g") is not None
        ]
        if vibration_readings and self._detect_vibration_anomaly(vibration_readings):
            self._consecutive_anomalies += 1
            avg_vib = sum(vibration_readings) / len(vibration_readings)
            current_rpm = int(telemetry_batch[-1].get("motor_rpm", self.rpm_limits[1]))
            corrective_rpm = self._calculate_corrective_rpm(current_rpm, avg_vib)
            self.logger.warning(
                f"[AEGIS][SelfHeal] Vibration anomaly: {avg_vib:.2f}g "
                f"(threshold={self.vibration_threshold}g). "
                f"Corrective RPM: {current_rpm} → {corrective_rpm}"
            )
            return {
                "trigger": "vibration_anomaly",
                "measured_value_g": round(avg_vib, 3),
                "threshold_g": self.vibration_threshold,
                "current_rpm": current_rpm,
                "corrective_rpm": corrective_rpm,
                "consecutive_anomalies": self._consecutive_anomalies,
            }

        # --- Latency spike check ---
        latencies = [t.get("latency_ms", 0.0) for t in telemetry_batch]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        if avg_latency > 500:
            self._consecutive_anomalies += 1
            self.logger.warning(
                f"[AEGIS][SelfHeal] Latency spike: {avg_latency:.0f}ms. "
                "Recommending safe-mode RPM."
            )
            return {
                "trigger": "latency_spike",
                "measured_value_ms": round(avg_latency, 1),
                "threshold_ms": 500,
                "current_rpm": int(telemetry_batch[-1].get("motor_rpm", self.rpm_limits[1])),
                "corrective_rpm": self.rpm_limits[0],
                "consecutive_anomalies": self._consecutive_anomalies,
            }

        # All clear
        self._consecutive_anomalies = 0
        return None

    def issue_command(self, action: dict) -> dict:
        """
        Format and dispatch a C2D motor-speed adjustment command.

        All commands are held for human approval by default (EU AI Act Article 14).

        Returns
        -------
        dict — {command_id, status: "dispatched" | "pending_approval", action}
        """
        command_id = f"cmd-{uuid.uuid4().hex[:8]}"

        if self.human_approval_required:
            self._request_human_approval({"command_id": command_id, "action": action})
            self.logger.info(
                f"[AEGIS][SelfHeal] Command {command_id} held for HITL approval."
            )
            return {"command_id": command_id, "status": "pending_approval", "action": action}

        # Auto-dispatch path (human_approval_required=False in test/sim environments)
        c2d_payload = {
            "command_id": command_id,
            "command_type": "motor_speed_adjustment",
            "target_rpm": action.get("corrective_rpm"),
            "trigger": action.get("trigger"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        if self._sim_mode:
            self._sim_dispatch(c2d_payload)
        else:
            self._iot_dispatch(c2d_payload)

        self.logger.info(
            f"[AEGIS][SelfHeal] C2D dispatched: {command_id} "
            f"→ RPM={action.get('corrective_rpm')}"
        )
        return {"command_id": command_id, "status": "dispatched", "action": action}

    # ------------------------------------------------------------------
    # Anomaly Detection & RPM Calculation
    # ------------------------------------------------------------------

    def _detect_vibration_anomaly(self, readings: list[float]) -> bool:
        """Check if vibration RMS exceeds the safe threshold."""
        rms = math.sqrt(sum(v ** 2 for v in readings) / len(readings))
        return rms > self.vibration_threshold

    def _calculate_corrective_rpm(self, current_rpm: int, vibration_rms: float) -> int:
        """
        Compute corrective RPM proportional to vibration excess.
        Reduction is capped at 60% to avoid stall conditions.
        """
        excess_ratio = vibration_rms / self.vibration_threshold
        reduction = min(0.6, (excess_ratio - 1.0) * 0.3)
        corrective = int(current_rpm * (1.0 - reduction))
        return max(self.rpm_limits[0], min(corrective, self.rpm_limits[1]))

    # ------------------------------------------------------------------
    # Human-in-the-Loop & Dispatch
    # ------------------------------------------------------------------

    def _request_human_approval(self, command: dict):
        """
        Log a pending-approval record to the audit trail.
        Operator dashboard reads this file for HITL gating.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        action = command["action"]
        audit_entry = (
            f"{timestamp} | PENDING_APPROVAL | {action.get('trigger', 'unknown')} | "
            f"cmd_id={command['command_id']} | "
            f"value={action.get('measured_value_g', action.get('measured_value_ms', 'N/A'))} | "
            f"corrective_rpm={action.get('corrective_rpm', 'N/A')} | "
            f"human_decision=AWAITING\n"
        )
        with _AUDIT_TRAIL_PATH.open("a") as f:
            f.write(audit_entry)

    def _sim_dispatch(self, payload: dict):
        """Write C2D command to local JSON buffer (simulation)."""
        commands = []
        if _SIM_COMMANDS_PATH.exists():
            try:
                with _SIM_COMMANDS_PATH.open("r") as f:
                    commands = json.load(f)
            except (json.JSONDecodeError, IOError):
                commands = []
        commands.append(payload)
        with _SIM_COMMANDS_PATH.open("w") as f:
            json.dump(commands, f, indent=2)

    def _iot_dispatch(self, payload: dict):
        """Dispatch C2D message via Azure IoT Hub Registry Manager."""
        conn_str = os.getenv("AEGIS_IOT_HUB_SERVICE_CONN_STR", "")
        if not conn_str or not _IOT_HUB_SDK_AVAILABLE:
            self.logger.error(
                "[AEGIS][SelfHeal] IoT Hub service SDK or connection string missing. "
                "Falling back to simulation dispatch."
            )
            self._sim_dispatch(payload)
            return
        try:
            manager = IoTHubRegistryManager.from_connection_string(conn_str)
            manager.send_c2d_message(self._device_id, json.dumps(payload))
        except Exception as e:
            self.logger.error(f"[AEGIS][SelfHeal] C2D dispatch failed: {e}")
