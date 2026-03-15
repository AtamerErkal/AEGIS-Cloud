"""
AEGIS-Cloud — Edge / Sensors / Driver Interface
=================================================
Module:   edge.src.sensors.driver_interface
Platform: NVIDIA Jetson Nano (JetPack 5.x)

PURPOSE
-------
Hardware Abstraction Layer (HAL) providing a unified API for all
physical sensors on the AEGIS guardian platform.  Currently supports:

  • **ICM-20948** — 9-DoF IMU (accelerometer, gyroscope, magnetometer)
  • **VL6180X**  — Time-of-Flight proximity / ambient light sensor

The HAL isolates physical I²C/SPI communication from the business logic
in ``FusionEngine``, enabling transparent swap-out of sensor hardware
and, critically, transparent simulation-mode substitution.

DESIGN PRINCIPLES
-----------------
1. **Simulation-First**
   Each driver method checks ``SIMULATION_MODE`` from configuration.
   When True, the driver returns synthetic sensor data read from
   CSV/JSON trace files in ``data/sim_samples/``, enabling full
   end-to-end pipeline testing without any connected hardware.

   This is critical for CI/CD: the GitHub Actions pipeline
   (``mlops/pipelines/github_actions_ci.yml``) runs the entire Edge
   stack in simulation mode inside a container.

2. **AIOps Integration**
   - Each driver tracks consecutive read failures and publishes a
     ``sensor_health`` metric to the telemetry buffer.
   - Failure counts are consumed by the Cloud self-healing loop  which
     may issue recalibration or failover commands.

3. **NATO-Standard Logging**
   Hardware faults (I²C NACK, range-sensor timeout) are logged as
   Incident Reports:
       Timestamp | Sensor_ID | Fault_Code | Recovery_Action

INTERFACES
----------
- Consumed by: ``FusionEngine``
- Configuration: ``edge/config/edge_settings.yaml``

SPRINT ASSIGNMENT
-----------------
Day 1:   Define abstract ``SensorDriver`` protocol and I²C helpers.
Day 2:   Implement ICM-20948 read loop (real + simulation stubs).
Day 3:   Implement VL6180X read loop (real + simulation stubs).
Day 4:   Integration test with ``FusionEngine`` in simulation mode.
"""

import csv
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional I²C library — not available in CI/CD containers
# ---------------------------------------------------------------------------
try:
    import smbus2
    _SMBUS_AVAILABLE = True
except ImportError:
    _SMBUS_AVAILABLE = False


def _nato_log(sensor_id: str, fault_code: str, recovery: str) -> None:
    """Emit a NATO Incident Report log line."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logger.warning("%s | %s | %s | %s", ts, sensor_id, fault_code, recovery)


class SensorDriver(ABC):
    """
    Abstract base class for all AEGIS sensor drivers.

    Subclasses must implement ``read()`` and ``self_test()``.
    """

    @abstractmethod
    def read(self) -> dict:
        """Return the latest sensor reading as a dict."""
        ...

    @abstractmethod
    def self_test(self) -> bool:
        """Run a hardware self-test.  Return True if healthy."""
        ...


# ---------------------------------------------------------------------------
# ICM-20948 9-DoF IMU Driver
# ---------------------------------------------------------------------------

class ICM20948Driver(SensorDriver):
    """
    ICM-20948 9-DoF IMU driver.

    Returns
    -------
    dict
        {
            "accel": (ax, ay, az),  # m/s²
            "gyro":  (gx, gy, gz),  # deg/s
            "mag":   (mx, my, mz),  # µT
        }
    """

    # Register map (bank 0)
    _REG_WHO_AM_I     = 0x00   # Expected: 0xEA
    _REG_PWR_MGMT_1   = 0x06
    _REG_ACCEL_XOUT_H = 0x2D
    _REG_GYRO_XOUT_H  = 0x33

    # Scale factors
    _ACCEL_SCALE = 9.81 / 2048.0   # ±16 g → m/s²
    _GYRO_SCALE  = 1.0 / 131.0     # ±250 °/s → deg/s
    _MAG_SCALE   = 0.15            # µT per LSB

    def __init__(self, config: dict):
        """Initialise I²C bus or load simulation trace."""
        self._config = config
        self._sim = bool(config.get("simulation_mode", True))
        self._bus_num = config.get("imu", {}).get("i2c_bus", 1)
        self._addr = config.get("imu", {}).get("i2c_address", 0x69)
        self._trace_path = config.get("imu", {}).get(
            "sim_trace_path", "data/sim_samples/imu_trace.csv"
        )
        self._fail_count = 0
        self._bus = None
        self._trace_rows: list = []
        self._trace_idx: int = 0
        self._t0 = time.time()

        if self._sim:
            self._load_trace()
            logger.info("ICM20948: simulation mode — trace: %s", self._trace_path)
        else:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Hardware initialisation
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        """Open I²C bus and wake the ICM-20948."""
        if not _SMBUS_AVAILABLE:
            logger.warning("smbus2 not available — ICM20948 falling back to simulation.")
            self._sim = True
            self._load_trace()
            return
        try:
            self._bus = smbus2.SMBus(self._bus_num)
            # Wake device: clear SLEEP bit in PWR_MGMT_1, use auto clock
            self._bus.write_byte_data(self._addr, self._REG_PWR_MGMT_1, 0x01)
            time.sleep(0.05)
            who = self._bus.read_byte_data(self._addr, self._REG_WHO_AM_I)
            if who != 0xEA:
                raise IOError(f"ICM20948 WHO_AM_I={who:#x} — expected 0xEA")
            logger.info("ICM20948 detected on bus %d addr %#x", self._bus_num, self._addr)
        except Exception as exc:
            _nato_log("ICM20948", "I2C_INIT_FAIL", f"fallback_sim: {exc}")
            self._sim = True
            self._load_trace()

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _load_trace(self) -> None:
        """Load a CSV trace file; fall back to synthetic generator if absent."""
        path = Path(self._trace_path)
        if path.exists():
            with path.open() as f:
                reader = csv.DictReader(f)
                self._trace_rows = [
                    {k: float(v) for k, v in row.items()} for row in reader
                ]
            logger.info("ICM20948 trace loaded: %d rows", len(self._trace_rows))
        else:
            logger.info("ICM20948 trace not found — using synthetic generator")
            self._trace_rows = []

    def _synthetic_read(self) -> dict:
        """Return realistic sinusoidal IMU data for bench testing."""
        t = time.time() - self._t0
        ax = 0.05 * math.sin(2 * math.pi * 0.5 * t)
        ay = 0.03 * math.cos(2 * math.pi * 0.3 * t)
        az = 9.81 + 0.02 * math.sin(2 * math.pi * 1.0 * t)
        gx = 0.10 * math.sin(2 * math.pi * 0.20 * t) + random.gauss(0, 0.01)
        gy = 0.10 * math.cos(2 * math.pi * 0.15 * t) + random.gauss(0, 0.01)
        gz = 0.05 * math.sin(2 * math.pi * 0.10 * t) + random.gauss(0, 0.005)
        mx = 22.0 + random.gauss(0, 0.3)
        my = -5.0 + random.gauss(0, 0.3)
        mz = 41.0 + random.gauss(0, 0.3)
        return {
            "accel": (ax, ay, az),
            "gyro":  (gx, gy, gz),
            "mag":   (mx, my, mz),
        }

    def _sim_read(self) -> dict:
        """Return the next row from the trace (wraps around) or synthetic data."""
        if not self._trace_rows:
            return self._synthetic_read()
        row = self._trace_rows[self._trace_idx % len(self._trace_rows)]
        self._trace_idx += 1
        return {
            "accel": (row.get("ax", 0.0), row.get("ay", 0.0), row.get("az", 9.81)),
            "gyro":  (row.get("gx", 0.0), row.get("gy", 0.0), row.get("gz", 0.0)),
            "mag":   (row.get("mx", 22.0), row.get("my", -5.0), row.get("mz", 41.0)),
        }

    # ------------------------------------------------------------------
    # Hardware read helpers
    # ------------------------------------------------------------------

    def _to_signed(self, hi: int, lo: int) -> int:
        val = (hi << 8) | lo
        return val - 65536 if val > 32767 else val

    def _read_raw_i2c(self) -> dict:
        """Read 6-axis IMU data (accel + gyro) from ICM-20948 registers."""
        raw = self._bus.read_i2c_block_data(self._addr, self._REG_ACCEL_XOUT_H, 12)
        ax = self._to_signed(raw[0],  raw[1])  * self._ACCEL_SCALE
        ay = self._to_signed(raw[2],  raw[3])  * self._ACCEL_SCALE
        az = self._to_signed(raw[4],  raw[5])  * self._ACCEL_SCALE
        gx = self._to_signed(raw[6],  raw[7])  * self._GYRO_SCALE
        gy = self._to_signed(raw[8],  raw[9])  * self._GYRO_SCALE
        gz = self._to_signed(raw[10], raw[11]) * self._GYRO_SCALE
        # Magnetometer via I²C master passthrough (best-effort)
        mx = my = mz = 0.0
        try:
            m_raw = self._bus.read_i2c_block_data(self._addr, 0x3B, 6)
            mx = self._to_signed(m_raw[1], m_raw[0]) * self._MAG_SCALE
            my = self._to_signed(m_raw[3], m_raw[2]) * self._MAG_SCALE
            mz = self._to_signed(m_raw[5], m_raw[4]) * self._MAG_SCALE
        except Exception:
            pass  # Magnetometer is optional
        return {
            "accel": (ax, ay, az),
            "gyro":  (gx, gy, gz),
            "mag":   (mx, my, mz),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> dict:
        """Return the latest IMU reading. Falls back to simulation on I²C error."""
        if self._sim:
            return self._sim_read()
        try:
            data = self._read_raw_i2c()
            self._fail_count = 0
            data["sensor_health"] = {"imu_fail_count": self._fail_count}
            return data
        except Exception as exc:
            self._fail_count += 1
            _nato_log("ICM20948", f"I2C_READ_FAIL_{self._fail_count}", "fallback_synthetic")
            logger.error("ICM20948 read error (fail #%d): %s", self._fail_count, exc)
            data = self._synthetic_read()
            data["sensor_health"] = {"imu_fail_count": self._fail_count}
            return data

    def self_test(self) -> bool:
        """
        Run a hardware self-test.

        Simulation: always True.
        Hardware: verifies WHO_AM_I register and checks accel Z is near ±g.
        """
        if self._sim:
            return True
        try:
            who = self._bus.read_byte_data(self._addr, self._REG_WHO_AM_I)
            if who != 0xEA:
                _nato_log("ICM20948", "SELF_TEST_FAIL", f"WHO_AM_I={who:#x}")
                return False
            # Sanity: az should be near ±9.81 m/s² at rest
            az = self.read()["accel"][2]
            if not (5.0 < abs(az) < 15.0):
                _nato_log("ICM20948", "SELF_TEST_FAIL", f"az={az:.2f} out of range")
                return False
            logger.info("ICM20948 self-test PASSED")
            return True
        except Exception as exc:
            _nato_log("ICM20948", "SELF_TEST_EXCEPTION", str(exc))
            return False


# ---------------------------------------------------------------------------
# VL6180X Time-of-Flight Driver
# ---------------------------------------------------------------------------

class VL6180XDriver(SensorDriver):
    """
    VL6180X Time-of-Flight range / ambient light sensor driver.

    Returns
    -------
    dict
        {
            "range_mm": int,      # Measured distance in mm
            "ambient_lux": float  # Ambient light level in lux
        }
    """

    # 16-bit register map
    _REG_MODEL_ID              = 0x000   # Expected: 0xB4
    _REG_FRESH_OUT_OF_RESET    = 0x016
    _REG_SYSRANGE_START        = 0x018
    _REG_RESULT_RANGE_STATUS   = 0x04D
    _REG_RESULT_RANGE_VAL      = 0x062
    _REG_SYSALS_START          = 0x038
    _REG_RESULT_ALS_STATUS     = 0x04E
    _REG_RESULT_ALS_VAL        = 0x050

    def __init__(self, config: dict):
        """Initialise I²C bus or load simulation trace."""
        self._config = config
        self._sim = bool(config.get("simulation_mode", True))
        self._bus_num = config.get("tof", {}).get("i2c_bus", 1)
        self._addr = config.get("tof", {}).get("i2c_address", 0x29)
        self._trace_path = config.get("tof", {}).get(
            "sim_trace_path", "data/sim_samples/tof_trace.csv"
        )
        self._fail_count = 0
        self._bus = None
        self._trace_rows: list = []
        self._trace_idx: int = 0
        self._t0 = time.time()
        self._last_range_mm: int = 100

        if self._sim:
            self._load_trace()
            logger.info("VL6180X: simulation mode — trace: %s", self._trace_path)
        else:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Hardware initialisation
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        """Open I²C bus and initialise the VL6180X."""
        if not _SMBUS_AVAILABLE:
            logger.warning("smbus2 not available — VL6180X falling back to simulation.")
            self._sim = True
            self._load_trace()
            return
        try:
            self._bus = smbus2.SMBus(self._bus_num)
            model_id = self._read_reg(self._REG_MODEL_ID)
            if model_id != 0xB4:
                raise IOError(f"VL6180X model_id={model_id:#x} — expected 0xB4")
            if self._read_reg(self._REG_FRESH_OUT_OF_RESET) == 0x01:
                self._apply_mandatory_settings()
                self._write_reg(self._REG_FRESH_OUT_OF_RESET, 0x00)
            logger.info("VL6180X detected on bus %d addr %#x", self._bus_num, self._addr)
        except Exception as exc:
            _nato_log("VL6180X", "I2C_INIT_FAIL", f"fallback_sim: {exc}")
            self._sim = True
            self._load_trace()

    def _read_reg(self, reg: int) -> int:
        """Read a single byte from a 16-bit register address."""
        self._bus.write_byte_data(self._addr, reg >> 8, reg & 0xFF)
        return self._bus.read_byte(self._addr)

    def _write_reg(self, reg: int, val: int) -> None:
        """Write a single byte to a 16-bit register address."""
        self._bus.write_i2c_block_data(self._addr, reg >> 8, [reg & 0xFF, val])

    def _apply_mandatory_settings(self) -> None:
        """Apply STMicro mandatory initialisation settings (AN4545 §3.1)."""
        mandatory = [
            (0x0207, 0x01), (0x0208, 0x01), (0x0096, 0x00), (0x0097, 0xFD),
            (0x00E3, 0x01), (0x00E4, 0x03), (0x00E5, 0x02), (0x00E6, 0x01),
            (0x00E7, 0x03), (0x00F5, 0x02), (0x00D9, 0x05), (0x00DB, 0xCE),
            (0x00DC, 0x03), (0x00DD, 0xF8), (0x009F, 0x00), (0x00A3, 0x3C),
            (0x00B7, 0x00), (0x00BB, 0x3C), (0x00B2, 0x09), (0x00CA, 0x09),
            (0x0198, 0x01), (0x01B0, 0x17), (0x01AD, 0x00), (0x00FF, 0x05),
            (0x0100, 0x05), (0x0199, 0x05), (0x01A6, 0x1B), (0x01AC, 0x3E),
            (0x01A7, 0x1F), (0x0030, 0x00),
        ]
        for reg, val in mandatory:
            self._write_reg(reg, val)

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _load_trace(self) -> None:
        """Load a CSV trace or fall back to the synthetic generator."""
        path = Path(self._trace_path)
        if path.exists():
            with path.open() as f:
                reader = csv.DictReader(f)
                self._trace_rows = [
                    {k: float(v) for k, v in row.items()} for row in reader
                ]
            logger.info("VL6180X trace loaded: %d rows", len(self._trace_rows))
        else:
            logger.info("VL6180X trace not found — using synthetic generator")
            self._trace_rows = []

    def _synthetic_read(self) -> dict:
        """Generate a realistic ToF measurement (sinusoidal range + noise)."""
        t = time.time() - self._t0
        base_range = 120 + 40 * math.sin(2 * math.pi * 0.1 * t)
        range_mm = max(10, int(base_range + random.gauss(0, 2.0)))
        ambient = 500.0 + 200.0 * math.sin(2 * math.pi * 0.01 * t) + random.gauss(0, 5)
        return {"range_mm": range_mm, "ambient_lux": round(ambient, 1)}

    def _sim_read(self) -> dict:
        if not self._trace_rows:
            return self._synthetic_read()
        row = self._trace_rows[self._trace_idx % len(self._trace_rows)]
        self._trace_idx += 1
        return {
            "range_mm": int(row.get("range_mm", 100)),
            "ambient_lux": float(row.get("ambient_lux", 500.0)),
        }

    # ------------------------------------------------------------------
    # Hardware read helpers
    # ------------------------------------------------------------------

    def _read_hardware(self) -> dict:
        """Trigger a single-shot range + ALS measurement and return results."""
        # Range measurement
        self._write_reg(self._REG_SYSRANGE_START, 0x01)
        for _ in range(50):
            time.sleep(0.002)
            if self._read_reg(self._REG_RESULT_RANGE_STATUS) & 0x01:
                break
        range_mm = self._read_reg(self._REG_RESULT_RANGE_VAL)

        # Ambient light measurement
        self._write_reg(self._REG_SYSALS_START, 0x01)
        for _ in range(50):
            time.sleep(0.002)
            if self._read_reg(self._REG_RESULT_ALS_STATUS) & 0x01:
                break
        als_hi = self._read_reg(self._REG_RESULT_ALS_VAL)
        als_lo = self._read_reg(self._REG_RESULT_ALS_VAL + 1)
        als_raw = (als_hi << 8) | als_lo
        # Convert raw → lux (100 ms integration, gain 1.0, per datasheet §2.10)
        als_lux = 0.32 * (als_raw / 0.0135) if als_raw > 0 else 0.0
        return {"range_mm": range_mm, "ambient_lux": round(als_lux, 1)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> dict:
        """Return the latest ToF reading. Falls back to simulation on I²C error."""
        if self._sim:
            return self._sim_read()
        try:
            data = self._read_hardware()
            self._fail_count = 0
            self._last_range_mm = data["range_mm"]
            data["sensor_health"] = {"tof_fail_count": self._fail_count}
            return data
        except Exception as exc:
            self._fail_count += 1
            _nato_log(
                "VL6180X",
                f"I2C_READ_FAIL_{self._fail_count}",
                f"returning_last_valid={self._last_range_mm}mm",
            )
            logger.error("VL6180X read error (fail #%d): %s", self._fail_count, exc)
            return {
                "range_mm": self._last_range_mm,
                "ambient_lux": 0.0,
                "sensor_health": {"tof_fail_count": self._fail_count},
            }

    def self_test(self) -> bool:
        """
        Verify the sensor model ID and a single live range reading.

        Simulation: always True.
        """
        if self._sim:
            return True
        try:
            model_id = self._read_reg(self._REG_MODEL_ID)
            if model_id != 0xB4:
                _nato_log("VL6180X", "SELF_TEST_FAIL", f"model_id={model_id:#x}")
                return False
            reading = self.read()
            if not (5 <= reading["range_mm"] <= 200):
                _nato_log(
                    "VL6180X", "SELF_TEST_FAIL",
                    f"range_mm={reading['range_mm']} outside 5–200 mm window"
                )
                return False
            logger.info("VL6180X self-test PASSED")
            return True
        except Exception as exc:
            _nato_log("VL6180X", "SELF_TEST_EXCEPTION", str(exc))
            return False
