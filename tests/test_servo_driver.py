"""
AEGIS-Cloud — Servo Driver Unit Test
=====================================
Simülasyon modunda PanTiltServoDriver'ı test eder.
Donanım gerekmez — PCA9685 olmadan çalışır.

Run:
    python tests/test_servo_driver.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge.src.sensors.servo_driver import PanTiltServoDriver

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

# Simülasyon config — donanım yok
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


def _ok(name, condition):
    status = PASS if condition else FAIL
    print(f"  {status} {name}")
    return condition


def test_center():
    """center() sonrası açılar home değerinde olmalı."""
    print("\n[TEST 1] center() — home pozisyonu")
    d = PanTiltServoDriver(CFG)
    d.center()
    s = d.read()
    ok1 = _ok("pan_deg == 90.0",  s["pan_deg"]  == 90.0)
    ok2 = _ok("tilt_deg == 90.0", s["tilt_deg"] == 90.0)
    return ok1 and ok2


def test_track_right():
    """Hedef sağda (cx > 0.5) → pan artmalı (sağa dönmeli)."""
    print("\n[TEST 2] track() — hedef sağ tarafta")
    d = PanTiltServoDriver(CFG)
    d.center()
    pan_before = d.read()["pan_deg"]
    # bbox: [x1, y1, x2, y2] — merkez x = 0.7 → sağda
    d.track([0.6, 0.4, 0.8, 0.6])
    pan_after = d.read()["pan_deg"]
    return _ok(f"pan arttı ({pan_before:.1f}° → {pan_after:.1f}°)", pan_after > pan_before)


def test_track_left():
    """Hedef solda (cx < 0.5) → pan azalmalı (sola dönmeli)."""
    print("\n[TEST 3] track() — hedef sol tarafta")
    d = PanTiltServoDriver(CFG)
    d.center()
    pan_before = d.read()["pan_deg"]
    # bbox merkez x = 0.2 → solda
    d.track([0.1, 0.4, 0.3, 0.6])
    pan_after = d.read()["pan_deg"]
    return _ok(f"pan azaldı ({pan_before:.1f}° → {pan_after:.1f}°)", pan_after < pan_before)


def test_track_deadzone():
    """Hedef tam merkezde → deadzone içinde, açı değişmemeli."""
    print("\n[TEST 4] track() — deadzone (merkeze yakın hedef)")
    d = PanTiltServoDriver(CFG)
    d.center()
    pan_before = d.read()["pan_deg"]
    tilt_before = d.read()["tilt_deg"]
    # bbox merkez = (0.50, 0.50) → tam centre
    d.track([0.45, 0.45, 0.55, 0.55])
    s = d.read()
    ok1 = _ok(f"pan değişmedi ({s['pan_deg']:.1f}°)",  s["pan_deg"]  == pan_before)
    ok2 = _ok(f"tilt değişmedi ({s['tilt_deg']:.1f}°)", s["tilt_deg"] == tilt_before)
    return ok1 and ok2


def test_angle_clamp():
    """Aşırı büyük hata → açı sınır değerini geçmemeli."""
    print("\n[TEST 5] angle clamp — sınırlar aşılmamalı")
    d = PanTiltServoDriver(CFG)
    # 50 adım en sağa doğru it
    for _ in range(50):
        d.track([0.9, 0.5, 1.0, 0.6])
    s = d.read()
    pan_max = CFG["servo"]["pan_center_deg"] + CFG["servo"]["pan_range_deg"][1]  # 180
    return _ok(f"pan <= {pan_max}° (gerçek: {s['pan_deg']:.1f}°)", s["pan_deg"] <= pan_max)


def test_self_test():
    """Simülasyon modunda self_test() True dönmeli."""
    print("\n[TEST 6] self_test() — simülasyon")
    d = PanTiltServoDriver(CFG)
    return _ok("self_test() == True", d.self_test() is True)


def test_read_keys():
    """read() doğru anahtarları içermeli."""
    print("\n[TEST 7] read() — dönen dict yapısı")
    d = PanTiltServoDriver(CFG)
    s = d.read()
    ok1 = _ok("'pan_deg' anahtarı var",          "pan_deg"          in s)
    ok2 = _ok("'tilt_deg' anahtarı var",         "tilt_deg"         in s)
    ok3 = _ok("'servo_fail_count' anahtarı var", "servo_fail_count" in s)
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
    passed = sum(results)
    total  = len(results)

    print(f"\n{'='*50}")
    print(f"  Sonuç: {passed}/{total} test geçti")
    print(f"{'='*50}")
    sys.exit(0 if passed == total else 1)
