"""
AEGIS-Cloud — Servo Driver Unit Tests
======================================
Tests PanTiltServoDriver in simulation mode.
No hardware required — runs without PCA9685.

Run:
    python tests/test_servo_driver.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge.src.sensors.servo_driver import PanTiltServoDriver

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

# Simulation config — no hardware
CFG = {
    "simulation_mode": True,
    "servo": {
        "i2c_address": 0x40,
        "frequency_hz": 50,
        "pan_channel": 0,
        "tilt_channel": 1,
        "pan_center_deg": 90.0,
        "tilt_center_deg": 90.0,
        "pan_range_deg":  [-90, 90],
        "tilt_range_deg": [-45, 45],
        "pid": {"kp": 0.4, "kd": 0.05, "deadzone_norm": 0.03},
    },
}


def _ok(label, condition):
    status = PASS if condition else FAIL
    print(f"  {status} {label}")
    return condition


def test_center():
    """After center(), angles must equal home values."""
    print("\n[TEST 1] center() — home position")
    d = PanTiltServoDriver(CFG)
    d.center()
    s = d.read()
    ok1 = _ok("pan_deg == 90.0",  s["pan_deg"]  == 90.0)
    ok2 = _ok("tilt_deg == 90.0", s["tilt_deg"] == 90.0)
    return ok1 and ok2


def test_track_right():
    """Target right of centre (cx > 0.5) -> pan must increase."""
    print("\n[TEST 2] track() — target on the right")
    d = PanTiltServoDriver(CFG)
    d.center()
    pan_before = d.read()["pan_deg"]
    # bbox centre x = 0.7 -> right of frame
    d.track([0.6, 0.4, 0.8, 0.6])
    pan_after = d.read()["pan_deg"]
    return _ok(f"pan increased ({pan_before:.1f} -> {pan_after:.1f})", pan_after > pan_before)


def test_track_left():
    """Target left of centre (cx < 0.5) -> pan must decrease."""
    print("\n[TEST 3] track() — target on the left")
    d = PanTiltServoDriver(CFG)
    d.center()
    pan_before = d.read()["pan_deg"]
    # bbox centre x = 0.2 -> left of frame
    d.track([0.1, 0.4, 0.3, 0.6])
    pan_after = d.read()["pan_deg"]
    return _ok(f"pan decreased ({pan_before:.1f} -> {pan_after:.1f})", pan_after < pan_before)


def test_track_deadzone():
    """Target at frame centre -> inside deadzone, angles must not change."""
    print("\n[TEST 4] track() — deadzone (target near centre)")
    d = PanTiltServoDriver(CFG)
    d.center()
    pan_before  = d.read()["pan_deg"]
    tilt_before = d.read()["tilt_deg"]
    # bbox centre = (0.50, 0.50) -> exactly on centre
    d.track([0.45, 0.45, 0.55, 0.55])
    s = d.read()
    ok1 = _ok(f"pan unchanged ({s['pan_deg']:.1f})",  s["pan_deg"]  == pan_before)
    ok2 = _ok(f"tilt unchanged ({s['tilt_deg']:.1f})", s["tilt_deg"] == tilt_before)
    return ok1 and ok2


def test_angle_clamp():
    """Extreme error -> angle must not exceed configured limit."""
    print("\n[TEST 5] angle clamp — limits must not be exceeded")
    d = PanTiltServoDriver(CFG)
    # Push hard right 50 times
    for _ in range(50):
        d.track([0.9, 0.5, 1.0, 0.6])
    s = d.read()
    pan_max = CFG["servo"]["pan_center_deg"] + CFG["servo"]["pan_range_deg"][1]  # 180
    return _ok(f"pan <= {pan_max} (actual: {s['pan_deg']:.1f})", s["pan_deg"] <= pan_max)


def test_self_test():
    """
    Hardware self_test() — performs a physical sweep if PCA9685 is connected.

    Case 1 — adafruit-servokit not installed or PCA9685 not connected:
        Driver falls back to simulation automatically -> self_test() returns True.
        Passes in CI/CD without any hardware.

    Case 2 — PCA9685 present on I2C (Nano + connected hardware):
        Real +/-15 degree sweep is performed -> self_test() returns True.
        Check servo wiring: VCC, GND, SDA, SCL.
    """
    print("\n[TEST 6] self_test() — hardware (simulation_mode=False)")

    hw_cfg = {**CFG, "simulation_mode": False}
    d = PanTiltServoDriver(hw_cfg)

    if d._sim:
        print("    [INFO] adafruit-servokit missing or PCA9685 not found -> sim fallback")
        return _ok("self_test() == True (sim fallback)", d.self_test() is True)
    else:
        print("    [INFO] PCA9685 connected -> physical sweep test")
        return _ok("self_test() == True (hardware)", d.self_test() is True)


def test_read_keys():
    """read() must return a dict with the expected keys."""
    print("\n[TEST 7] read() — return dict structure")
    d = PanTiltServoDriver(CFG)
    s = d.read()
    ok1 = _ok("'pan_deg' key present",          "pan_deg"          in s)
    ok2 = _ok("'tilt_deg' key present",         "tilt_deg"         in s)
    ok3 = _ok("'servo_fail_count' key present", "servo_fail_count" in s)
    return ok1 and ok2 and ok3


if __name__ == "__main__":
    tests = [
        test_center,
        test_track_right,
        test_track_left,
        test_track_deadzone,
        test_angle_clamp,
        test_self_test,
        test_read_keys,
    ]

    results = [t() for t in tests]
    passed  = sum(results)
    total   = len(results)

    print(f"\n{'='*50}")
    print(f"  Result: {passed}/{total} tests passed")
    print(f"{'='*50}")
    sys.exit(0 if passed == total else 1)
