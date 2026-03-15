"""
AEGIS-Cloud — PCA9685 Physical Servo Test
==========================================
Run on Nano. Servos must move physically.

Wiring:
  PCA9685 SDA -> Nano pin 3
  PCA9685 SCL -> Nano pin 5
  PCA9685 VCC -> Nano pin 1 (3.3V)
  PCA9685 GND -> Nano pin 6
  PCA9685 V+  -> External 5V power supply (for servos)
  Pan  servo  -> channel 0
  Tilt servo  -> channel 1

Run:
  pip install adafruit-circuitpython-servokit
  python tests/test_servo_hardware.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Check PCA9685 connection ──────────────────────────────────────────────────
try:
    from adafruit_servokit import ServoKit
except ImportError:
    print("\n[ERROR] adafruit-servokit is not installed.")
    print("  Fix: pip install adafruit-circuitpython-servokit\n")
    sys.exit(1)

try:
    kit = ServoKit(channels=16, address=0x40)
    kit.frequency = 50
    print("\n[OK] PCA9685 connected — I2C 0x40\n")
except Exception as e:
    print(f"\n[ERROR] PCA9685 not found: {e}")
    print("  Check: SDA/SCL cables, 3.3V VCC, GND\n")
    sys.exit(1)

PAN  = 0    # channel 0
TILT = 1    # channel 1
WAIT = 0.6  # seconds between each move


def move(pan_deg, tilt_deg, label):
    pan_deg  = max(0, min(180, pan_deg))
    tilt_deg = max(0, min(180, tilt_deg))
    kit.servo[PAN].angle  = pan_deg
    kit.servo[TILT].angle = tilt_deg
    print(f"  -> {label:20s}  pan={pan_deg:5.1f}  tilt={tilt_deg:5.1f}")
    time.sleep(WAIT)


# ── Test sequence ─────────────────────────────────────────────────────────────
print("=" * 50)
print("  PCA9685 Physical Servo Test")
print("=" * 50)

print("\n[1] Centre (home)")
move(90, 90, "centre")

print("\n[2] Pan test")
move(135, 90, "right  +45")
move( 45, 90, "left   -45")
move( 90, 90, "centre")

print("\n[3] Tilt test")
move(90, 120, "down   +30")
move(90,  60, "up     -30")
move(90,  90, "centre")

print("\n[4] Corner test")
move(135, 120, "right-down")
move( 45,  60, "left-up")
move(135,  60, "right-up")
move( 45, 120, "left-down")
move( 90,  90, "centre")

print("\n" + "=" * 50)
print("  All moves complete.")
print("  If servos reached each position -> PASS")
print("=" * 50 + "\n")
