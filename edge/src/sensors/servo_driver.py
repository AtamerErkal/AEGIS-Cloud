"""
AEGIS-Cloud — Edge / Sensors / Servo Driver (PCA9685 Pan-Tilt)
===============================================================
Module:   edge.src.sensors.servo_driver
Platform: NVIDIA Jetson Nano (JetPack 5.x)

PURPOSE
-------
Hardware Abstraction Layer for the PCA9685 16-channel PWM servo controller.
Drives a 2-axis pan-tilt gimbal used for target search and tracking.

When a Detection arrives from VisionNode, the ``track()`` method computes
the angular error between the target bbox centre and the frame centre,
then applies a PD controller to smoothly steer the gimbal toward the target.

DESIGN PRINCIPLES
-----------------
1. **Simulation-First**
   When ``simulation_mode=true`` in config, no I²C hardware is accessed.
   All angle commands are logged as NATO-format lines — identical to
   the ICM-20948 / VL6180X drivers.

2. **PD Controller**
   Proportional + Derivative control is sufficient for a pan-tilt gimbal.
   Integral term is intentionally omitted to prevent wind-up on a
   resource-constrained Jetson Nano.

3. **Deadzone**
   A configurable deadzone (normalised frame fraction) suppresses micro-
   jitter when the target is already near the frame centre.

4. **AIOps**
   ``read()`` returns current pan/tilt angles for telemetry publishing.
   Consecutive I²C failures trigger NATO fault logs.

INTERFACES
----------
- Consumed by: ``VisionNode``
- Configuration: ``edge/config/edge_settings.yaml``  (servo section)

PCA9685 WIRING (Jetson Nano)
-----------------------------
  PCA9685 VCC  → 3.3 V (pin 1)
  PCA9685 GND  → GND   (pin 6)
  PCA9685 SDA  → SDA1  (pin 3 / I2C bus 1)
  PCA9685 SCL  → SCL1  (pin 5 / I2C bus 1)
  PCA9685 V+   → 5–6 V external servo supply
  Servo Pan    → channel 0
  Servo Tilt   → channel 1
"""

import logging
import time
from typing import Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional hardware library — not available in CI/CD containers
# ---------------------------------------------------------------------------
try:
    from adafruit_servokit import ServoKit as _ServoKit
    _SERVOKIT_AVAILABLE = True
except ImportError:
    _ServoKit = None  # type: ignore[assignment,misc]
    _SERVOKIT_AVAILABLE = False


def _nato_log(sensor_id: str, fault_code: str, recovery: str) -> None:
    """Emit a NATO Incident Report log line."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logger.warning("%s | %s | %s | %s", ts, sensor_id, fault_code, recovery)


# ---------------------------------------------------------------------------
# PD Controller (stateful, per axis)
# ---------------------------------------------------------------------------

class _PDController:
    """
    Minimal Proportional-Derivative controller for a single servo axis.

    The integral term is excluded to prevent wind-up on a slow VLM-
    interrupted Jetson loop where dt can be highly variable.
    """

    def __init__(self, kp: float, kd: float, deadzone: float) -> None:
        self._kp = kp
        self._kd = kd
        self._deadzone = deadzone
        self._prev_error: float = 0.0

    def reset(self) -> None:
        self._prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        """
        Compute the control output (delta angle in degrees).

        Parameters
        ----------
        error : float
            Normalised error in [−1, +1].  Positive = target right/below.
        dt : float
            Elapsed time since last call (seconds).  Used for derivative term.

        Returns
        -------
        float
            Delta angle to apply to the servo (degrees).
            Returns 0.0 when |error| < deadzone.
        """
        if abs(error) < self._deadzone:
            self._prev_error = 0.0
            return 0.0

        derivative = (error - self._prev_error) / max(dt, 1e-3)
        output = self._kp * error + self._kd * derivative
        self._prev_error = error
        return output


# ---------------------------------------------------------------------------
# Servo Driver
# ---------------------------------------------------------------------------

class PanTiltServoDriver:
    """
    PCA9685 pan-tilt servo driver with PD-controlled target tracking.

    Parameters
    ----------
    config : dict
        The full edge config dict (reads the ``servo`` and
        ``simulation_mode`` keys).

    Usage
    -----
    ::

        driver = PanTiltServoDriver(config)
        driver.center()                  # go to home position
        driver.track([x1, y1, x2, y2])  # bbox-driven tracking step
        state = driver.read()            # {"pan_deg": …, "tilt_deg": …}
        driver.self_test()               # sweep test
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._sim = bool(config.get("simulation_mode", True))
        srv = config.get("servo", {})

        # Hardware config
        self._i2c_addr: int = int(srv.get("i2c_address", 0x40))
        self._freq_hz: int  = int(srv.get("frequency_hz", 50))
        self._pan_ch: int   = int(srv.get("pan_channel", 0))
        self._tilt_ch: int  = int(srv.get("tilt_channel", 1))

        # Angle limits (degrees, absolute 0–180 range of standard servo)
        self._pan_center: float  = float(srv.get("pan_center_deg", 90.0))
        self._tilt_center: float = float(srv.get("tilt_center_deg", 90.0))
        pan_range  = srv.get("pan_range_deg",  [-90, 90])
        tilt_range = srv.get("tilt_range_deg", [-45, 45])
        self._pan_min: float  = self._pan_center  + float(pan_range[0])
        self._pan_max: float  = self._pan_center  + float(pan_range[1])
        self._tilt_min: float = self._tilt_center + float(tilt_range[0])
        self._tilt_max: float = self._tilt_center + float(tilt_range[1])

        # Current angles (initialised at centre)
        self._pan_deg: float  = self._pan_center
        self._tilt_deg: float = self._tilt_center

        # PD controllers
        pid = srv.get("pid", {})
        kp       = float(pid.get("kp", 0.4))
        kd       = float(pid.get("kd", 0.05))
        deadzone = float(pid.get("deadzone_norm", 0.03))
        self._pan_pd  = _PDController(kp, kd, deadzone)
        self._tilt_pd = _PDController(kp, kd, deadzone)

        # AIOps
        self._fail_count: int = 0
        self._last_track_ts: float = time.perf_counter()

        # Hardware init
        self._kit = None
        if self._sim:
            logger.info(
                "PanTiltServo: simulation mode — all commands logged, no I²C access."
            )
        else:
            self._init_hardware()

        # Move to home
        self.center()

    # ------------------------------------------------------------------
    # Hardware initialisation
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        """Open I²C bus and initialise PCA9685 via adafruit-servokit."""
        if not _SERVOKIT_AVAILABLE:
            _nato_log(
                "PCA9685", "LIBRARY_MISSING",
                "adafruit-servokit not installed — falling back to simulation. "
                "Run: pip install adafruit-circuitpython-servokit"
            )
            self._sim = True
            return
        try:
            self._kit = _ServoKit(channels=16, address=self._i2c_addr)
            # Configure PWM frequency (50 Hz is standard for RC servos)
            self._kit.frequency = self._freq_hz
            logger.info(
                "PCA9685 initialised — addr=0x%02X  freq=%dHz  "
                "pan_ch=%d  tilt_ch=%d",
                self._i2c_addr, self._freq_hz, self._pan_ch, self._tilt_ch,
            )
        except Exception as exc:
            _nato_log("PCA9685", "I2C_INIT_FAIL", f"fallback_sim: {exc}")
            self._sim = True

    # ------------------------------------------------------------------
    # Angle application
    # ------------------------------------------------------------------

    def _set_angle(self, channel: int, angle_deg: float) -> None:
        """Write a servo angle to the PCA9685 channel (or log in sim mode)."""
        angle_deg = round(float(angle_deg), 1)
        if self._sim:
            ch_name = "PAN" if channel == self._pan_ch else "TILT"
            logger.debug(
                "[SIM] PCA9685 ch%d (%s) → %.1f°", channel, ch_name, angle_deg
            )
            return
        try:
            self._kit.servo[channel].angle = angle_deg
            self._fail_count = 0
        except Exception as exc:
            self._fail_count += 1
            _nato_log(
                "PCA9685",
                f"WRITE_FAIL_ch{channel}_{self._fail_count}",
                f"angle={angle_deg}  err={exc}",
            )
            logger.error(
                "PCA9685 write error ch%d (fail #%d): %s",
                channel, self._fail_count, exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def center(self) -> None:
        """Move both axes to the configured centre position (home / search start)."""
        self._pan_deg  = self._pan_center
        self._tilt_deg = self._tilt_center
        self._pan_pd.reset()
        self._tilt_pd.reset()
        self._set_angle(self._pan_ch,  self._pan_deg)
        self._set_angle(self._tilt_ch, self._tilt_deg)
        logger.info(
            "PanTiltServo: centred → pan=%.1f°  tilt=%.1f°",
            self._pan_deg, self._tilt_deg,
        )

    def track(self, bbox: list) -> Tuple[float, float]:
        """
        Drive the gimbal toward the target defined by a normalised bbox.

        Parameters
        ----------
        bbox : list[float]
            Normalised bounding box ``[x1, y1, x2, y2]`` in [0, 1].

        Returns
        -------
        tuple[float, float]
            ``(pan_deg, tilt_deg)`` — the new absolute angles commanded.
        """
        now = time.perf_counter()
        dt  = now - self._last_track_ts
        self._last_track_ts = now

        # Compute normalised error: target centre − frame centre
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        pan_err  = cx - 0.5   # positive → target is right of centre
        tilt_err = cy - 0.5   # positive → target is below centre

        # PD output in degrees
        d_pan  = self._pan_pd.compute(pan_err,  dt)
        d_tilt = self._tilt_pd.compute(tilt_err, dt)

        # Apply and clamp
        self._pan_deg  = max(self._pan_min,  min(self._pan_max,  self._pan_deg  + d_pan))
        self._tilt_deg = max(self._tilt_min, min(self._tilt_max, self._tilt_deg + d_tilt))

        self._set_angle(self._pan_ch,  self._pan_deg)
        self._set_angle(self._tilt_ch, self._tilt_deg)

        logger.debug(
            "PanTiltServo: track  bbox_cx=%.3f cy=%.3f  "
            "err=(%.3f, %.3f)  → pan=%.1f°  tilt=%.1f°",
            cx, cy, pan_err, tilt_err, self._pan_deg, self._tilt_deg,
        )
        return self._pan_deg, self._tilt_deg

    def read(self) -> dict:
        """
        Return current gimbal state (satisfies SensorDriver-like interface).

        Returns
        -------
        dict
            ``{"pan_deg": float, "tilt_deg": float,
               "servo_fail_count": int}``
        """
        return {
            "pan_deg":          round(self._pan_deg,  1),
            "tilt_deg":         round(self._tilt_deg, 1),
            "servo_fail_count": self._fail_count,
        }

    def self_test(self) -> bool:
        """
        Sweep pan and tilt axes to ±15° from centre and return to home.

        Simulation: always returns True without hardware access.
        Hardware: verifies both channels respond without I²C errors.
        """
        logger.info("PanTiltServo: self-test starting…")
        if self._sim:
            logger.info("PanTiltServo: self-test PASSED (simulation)")
            return True
        try:
            for pan_off, tilt_off in [(15, 0), (-15, 0), (0, 15), (0, -15), (0, 0)]:
                self._set_angle(self._pan_ch,  self._pan_center  + pan_off)
                self._set_angle(self._tilt_ch, self._tilt_center + tilt_off)
                time.sleep(0.3)
            self.center()
            if self._fail_count > 0:
                _nato_log("PCA9685", "SELF_TEST_FAIL",
                          f"fail_count={self._fail_count}")
                return False
            logger.info("PanTiltServo: self-test PASSED")
            return True
        except Exception as exc:
            _nato_log("PCA9685", "SELF_TEST_EXCEPTION", str(exc))
            return False
