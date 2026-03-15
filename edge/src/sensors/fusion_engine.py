"""
AEGIS-Cloud — Edge / Sensors / Fusion Engine
==============================================
Module:   edge.src.sensors.fusion_engine
Platform: NVIDIA Jetson Nano (JetPack 5.x)

PURPOSE
-------
Synchronises and fuses data from the ICM-20948 9-DoF IMU (orientation,
acceleration, gyroscope) and the VL6180X Time-of-Flight range sensor to
produce a unified ``PlatformState`` object consumed by the Vision Node
(image stabilisation hints) and the Cloud Sync gateway (telemetry).

The Fusion Engine is the single source of truth for the physical state
of the AEGIS guardian platform (turret orientation, range-to-target,
vibration level).

DESIGN PRINCIPLES
-----------------
1. **Simulation-First**
   When ``SIMULATION_MODE=True``, the engine reads pre-recorded sensor
   traces from ``data/sim_samples/`` via the ``DriverInterface``
   simulation stubs, enabling full pipeline testing without hardware.

2. **AIOps Integration — Self-Healing Loop**
   - Publishes fused telemetry (orientation, range, vibration RMS) to
     ``data/logs/`` for Cloud consumption.
   - The Cloud ``self_healing_service.py`` analyses these readings and
     may issue motor-speed adjustment commands back to the Edge via
     ``cloud_sync.py``.  The Fusion Engine applies these commands to
     re-calibrate sensor sampling rates or trigger safe-mode.

3. **NATO-Standard Logging**
   Sensor anomalies (jitter spikes, range dropouts) are logged as
   Incident Reports with the format:
       Timestamp | Lat/Long | Sensor_ID | Anomaly_Type | Severity

4. **EU AI Act — Data Provenance**
   Every fused ``PlatformState`` record carries a provenance hash
   linking back to the raw sensor readings, ensuring reproducibility
   for audit purposes.

INTERFACES
----------
- Input:  Raw IMU + ToF readings via ``DriverInterface``.
- Output: ``PlatformState`` dataclass → consumed by Vision Node and
          Cloud Sync gateway.

SPRINT ASSIGNMENT
-----------------
Day 1:   Define ``PlatformState`` schema and sensor sampling loop.
Day 2:   Implement Kalman-filter fusion for IMU + ToF alignment.
Day 3:   Wire simulation-mode playback from ``data/sim_samples/``.
Day 4:   Integrate AIOps telemetry publishing and self-healing hooks.
"""

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from edge.src.sensors.driver_interface import ICM20948Driver, VL6180XDriver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data schema
# ---------------------------------------------------------------------------

@dataclass
class PlatformState:
    """
    Fused sensor output representing the physical state of the AEGIS
    guardian platform at a single point in time.

    Fields
    ------
    timestamp_utc : str       ISO-8601 timestamp.
    orientation   : tuple     (roll, pitch, yaw) in degrees.
    acceleration  : tuple     (x, y, z) in m/s².
    range_mm      : int       ToF distance reading in millimetres.
    vibration_rms : float     RMS vibration magnitude (g).
    provenance    : str       SHA-256 hash of the raw sensor readings.
    """
    timestamp_utc: str = ""
    orientation: tuple = (0.0, 0.0, 0.0)
    acceleration: tuple = (0.0, 0.0, 0.0)
    range_mm: int = 0
    vibration_rms: float = 0.0
    provenance: str = ""


# ---------------------------------------------------------------------------
# Internal: 1-D Kalman filter for range smoothing
# ---------------------------------------------------------------------------

class _KalmanFilter1D:
    """Lightweight scalar Kalman filter (constant-velocity model)."""

    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 5.0):
        self._q = process_noise       # process noise covariance
        self._r = measurement_noise   # measurement noise covariance
        self._x = 0.0                 # state estimate
        self._p = 1.0                 # estimate error covariance
        self._initialized = False

    def update(self, measurement: float) -> float:
        """Feed one measurement; return the filtered estimate."""
        if not self._initialized:
            self._x = measurement
            self._initialized = True
            return self._x
        # Predict
        self._p += self._q
        # Kalman gain
        k = self._p / (self._p + self._r)
        # Update
        self._x += k * (measurement - self._x)
        self._p *= (1.0 - k)
        return self._x


# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """
    IMU + ToF sensor fusion engine.

    Attributes
    ----------
    driver : DriverInterface
        Hardware abstraction layer for sensor access.
    simulation_mode : bool
        If True, replays recorded sensor traces.
    sample_rate_hz : int
        Target fusion loop frequency (default: 100 Hz).
    """

    # Complementary filter coefficient: how much we trust gyro vs accel
    # Higher α → smoother but slower response; lower α → faster but noisier
    _CF_ALPHA = 0.98

    def __init__(self, config_path: str = "edge/config/edge_settings.yaml"):
        """Initialise drivers and configure the fusion loop."""
        self._config = self._load_config(config_path)
        sensor_cfg = self._config.get("sensors", {})
        self._sim = bool(self._config.get("simulation_mode", True))
        self._sample_rate_hz: int = int(sensor_cfg.get("sample_rate_hz", 100))
        self._safe_mode: bool = False

        # Build a flat config dict for the driver constructors
        driver_cfg = {
            "simulation_mode": self._sim,
            "imu": sensor_cfg.get("imu", {}),
            "tof": sensor_cfg.get("tof", {}),
        }
        self._imu = ICM20948Driver(driver_cfg)
        self._tof = VL6180XDriver(driver_cfg)

        # Range smoother
        self._kf_range = _KalmanFilter1D(process_noise=0.5, measurement_noise=4.0)

        # Orientation state for complementary filter (degrees)
        self._roll: float = 0.0
        self._pitch: float = 0.0
        self._yaw: float = 0.0
        self._last_ts: float = time.time()

        # Vibration RMS sliding window (last N accel magnitudes)
        self._vib_window: list = []
        self._vib_window_size: int = max(1, self._sample_rate_hz // 10)  # 0.1 s

        # Telemetry output path
        aiops_cfg = self._config.get("aiops", {})
        self._telemetry_dir = Path(aiops_cfg.get("telemetry_buffer_path", "data/logs/"))
        self._telemetry_dir.mkdir(parents=True, exist_ok=True)
        self._telemetry_path = self._telemetry_dir / "fusion_telemetry.jsonl"

        # NATO metadata
        nato = self._config.get("nato_metadata", {})
        self._station_id: str = nato.get("station_id", "AEGIS-EDGE-001")

        logger.info(
            "FusionEngine ready | sim=%s | rate=%d Hz | safe_mode=%s",
            self._sim, self._sample_rate_hz, self._safe_mode,
        )

    # ------------------------------------------------------------------
    # Config loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> dict:
        """Load YAML configuration; return empty dict on any error."""
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("FusionEngine: could not load config %s — %s", path, exc)
            return {}

    # ------------------------------------------------------------------
    # Orientation helpers (complementary filter)
    # ------------------------------------------------------------------

    def _update_orientation(self, accel: tuple, gyro: tuple, dt: float) -> tuple:
        """
        Fuse accelerometer tilt angles and gyroscope integration via a
        complementary filter.

        Returns (roll, pitch, yaw) in degrees.
        Yaw is gyro-only (no magnetometer fusion in this revision).
        """
        ax, ay, az = accel
        gx, gy, gz = gyro

        # Accelerometer-derived roll / pitch (degrees)
        accel_roll  = math.degrees(math.atan2(ay, az))
        accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))

        # Complementary filter
        alpha = self._CF_ALPHA
        self._roll  = alpha * (self._roll  + gx * dt) + (1.0 - alpha) * accel_roll
        self._pitch = alpha * (self._pitch + gy * dt) + (1.0 - alpha) * accel_pitch
        self._yaw  += gz * dt   # pure gyro integration; drifts over time

        return (round(self._roll, 3), round(self._pitch, 3), round(self._yaw, 3))

    # ------------------------------------------------------------------
    # Vibration RMS helper
    # ------------------------------------------------------------------

    def _update_vibration(self, accel: tuple) -> float:
        """
        Maintain a sliding window of accel magnitudes and return the RMS
        vibration level in g (gravity units).
        """
        ax, ay, az = accel
        # Remove gravity component from Z
        az_body = az - 9.81
        mag = math.sqrt(ax**2 + ay**2 + az_body**2) / 9.81   # in g

        self._vib_window.append(mag)
        if len(self._vib_window) > self._vib_window_size:
            self._vib_window.pop(0)

        rms = math.sqrt(sum(v**2 for v in self._vib_window) / len(self._vib_window))
        return round(rms, 4)

    # ------------------------------------------------------------------
    # Provenance hash
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_provenance(imu_data: dict, tof_data: dict) -> str:
        """SHA-256 hash of the raw sensor readings for EU AI Act audit trail."""
        raw = json.dumps(
            {"imu": imu_data, "tof": tof_data},
            sort_keys=True, default=str
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(self) -> PlatformState:
        """
        Read latest sensor data, apply Kalman filter and complementary
        filter, and return a fused ``PlatformState`` snapshot.
        """
        now = time.time()
        dt = max(now - self._last_ts, 1e-6)
        self._last_ts = now

        imu_data = self._imu.read()
        tof_data = self._tof.read()

        accel = imu_data.get("accel", (0.0, 0.0, 9.81))
        gyro  = imu_data.get("gyro",  (0.0, 0.0, 0.0))

        orientation = self._update_orientation(accel, gyro, dt)
        vibration_rms = self._update_vibration(accel)

        raw_range = float(tof_data.get("range_mm", 0))
        filtered_range = int(round(self._kf_range.update(raw_range)))

        # NATO anomaly check: vibration above safe threshold
        vib_limit = self._config.get("motor", {}).get("vibration_safe_threshold_g", 2.5)
        if vibration_rms > vib_limit:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            logger.warning(
                "%s | %s | VIBRATION_SPIKE | rms=%.3f g > limit=%.1f g | severity=HIGH",
                ts, self._station_id, vibration_rms, vib_limit,
            )

        state = PlatformState(
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            orientation=orientation,
            acceleration=tuple(round(v, 4) for v in accel),
            range_mm=filtered_range,
            vibration_rms=vibration_rms,
            provenance=self._compute_provenance(imu_data, tof_data),
        )

        self._publish_telemetry(state)
        return state

    def apply_cloud_command(self, command: dict):
        """
        Apply a self-healing command received from the Cloud layer.

        Supported command types
        -----------------------
        ``recalibrate`` : Reset orientation and vibration window.
        ``safe_mode``   : Enter safe mode — reduce sample rate.
        ``set_rate``    : Adjust sample_rate_hz.
        ``resume``      : Exit safe mode and restore normal rate.
        """
        cmd_type = command.get("type", "")
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if cmd_type == "recalibrate":
            self._roll = self._pitch = self._yaw = 0.0
            self._vib_window.clear()
            logger.info("%s | FusionEngine | RECALIBRATE | orientation and vib window reset", ts)

        elif cmd_type == "safe_mode":
            self._safe_mode = True
            self._sample_rate_hz = max(10, self._sample_rate_hz // 5)
            logger.warning(
                "%s | FusionEngine | SAFE_MODE_ACTIVATED | new_rate=%d Hz",
                ts, self._sample_rate_hz,
            )

        elif cmd_type == "set_rate":
            new_rate = int(command.get("sample_rate_hz", self._sample_rate_hz))
            new_rate = max(1, min(new_rate, 1000))
            self._sample_rate_hz = new_rate
            self._vib_window_size = max(1, new_rate // 10)
            logger.info("%s | FusionEngine | SET_RATE | rate=%d Hz", ts, new_rate)

        elif cmd_type == "resume":
            self._safe_mode = False
            default_rate = self._config.get("sensors", {}).get("sample_rate_hz", 100)
            self._sample_rate_hz = default_rate
            self._vib_window_size = max(1, default_rate // 10)
            logger.info(
                "%s | FusionEngine | RESUME | restored_rate=%d Hz", ts, default_rate
            )

        else:
            logger.warning(
                "%s | FusionEngine | UNKNOWN_COMMAND | type=%s", ts, cmd_type
            )

    def _publish_telemetry(self, state: PlatformState):
        """Write the fused state to the AIOps telemetry buffer (JSONL)."""
        record = asdict(state)
        # Convert tuples to lists for JSON serialisation
        record["orientation"] = list(record["orientation"])
        record["acceleration"] = list(record["acceleration"])
        record["sample_rate_hz"] = self._sample_rate_hz
        record["safe_mode"] = self._safe_mode
        record["station_id"] = self._station_id

        try:
            with self._telemetry_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("FusionEngine: telemetry write failed: %s", exc)
