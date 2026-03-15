"""
AEGIS-Cloud — PCA9685 Fiziksel Servo Testi
===========================================
Nano'da çalıştır. Servo'lar fiziksel olarak hareket etmeli.

Bağlantı:
  PCA9685 SDA → Nano pin 3
  PCA9685 SCL → Nano pin 5
  PCA9685 VCC → Nano pin 1 (3.3V)
  PCA9685 GND → Nano pin 6
  PCA9685 V+  → Harici 5V güç kaynağı (servo için)
  Pan  servo  → kanal 0
  Tilt servo  → kanal 1

Çalıştır:
  pip install adafruit-circuitpython-servokit
  python tests/test_servo_hardware.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── PCA9685 bağlantısını kontrol et ─────────────────────────────────────────
try:
    from adafruit_servokit import ServoKit
except ImportError:
    print("\n[HATA] adafruit-servokit kurulu değil.")
    print("  Çözüm: pip install adafruit-circuitpython-servokit\n")
    sys.exit(1)

try:
    kit = ServoKit(channels=16, address=0x40)
    kit.frequency = 50
    print("\n[OK] PCA9685 bağlandı — I²C 0x40\n")
except Exception as e:
    print(f"\n[HATA] PCA9685 bulunamadı: {e}")
    print("  Kontrol et: SDA/SCL kabloları, 3.3V VCC, GND\n")
    sys.exit(1)

PAN  = 0   # kanal 0
TILT = 1   # kanal 1
WAIT = 0.6 # her hareket arası bekleme (saniye)


def move(pan_deg, tilt_deg, label):
    pan_deg  = max(0, min(180, pan_deg))
    tilt_deg = max(0, min(180, tilt_deg))
    kit.servo[PAN].angle  = pan_deg
    kit.servo[TILT].angle = tilt_deg
    print(f"  → {label:20s}  pan={pan_deg:5.1f}°  tilt={tilt_deg:5.1f}°")
    time.sleep(WAIT)


# ── Test sekansı ─────────────────────────────────────────────────────────────
print("=" * 50)
print("  PCA9685 Fiziksel Servo Testi")
print("=" * 50)

print("\n[1] Merkez (home)")
move(90, 90, "merkez")

print("\n[2] Pan testi")
move(135, 90, "sağa  +45°")
move( 45, 90, "sola  -45°")
move( 90, 90, "merkez")

print("\n[3] Tilt testi")
move(90, 120, "aşağı +30°")
move(90,  60, "yukarı -30°")
move(90,  90, "merkez")

print("\n[4] Köşe testi")
move(135, 120, "sağ-aşağı")
move( 45,  60, "sol-yukarı")
move(135,  60, "sağ-yukarı")
move( 45, 120, "sol-aşağı")
move( 90,  90, "merkez")

print("\n" + "=" * 50)
print("  Tüm hareketler tamamlandı.")
print("  Servo'lar yukarıdaki pozisyonlara gittiyse → BAŞARILI")
print("=" * 50 + "\n")
