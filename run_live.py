"""
AEGIS-Cloud — Gerçek Pipeline
==============================
Gerçek video → Gerçek YOLOv8 tespiti → Gerçek servo takibi

Kullanım:
  # USB webcam ile:
  python run_live.py --camera 0

  # Video dosyası ile:
  python run_live.py --video data/sim_samples/drone_flyby.mp4

  # IMX219 CSI kamera (Nano):
  python run_live.py --gstreamer
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── PCA9685 bağlantısı ──────────────────────────────────────────────────────
try:
    from adafruit_servokit import ServoKit
    _kit = ServoKit(channels=16, address=0x40)
    _kit.frequency = 50
    SERVO_OK = True
    print("[SERVO] PCA9685 bağlandı ✓")
except Exception as e:
    SERVO_OK = False
    print(f"[SERVO] PCA9685 yok → servo komutları loglanacak ({e})")

# ── YOLOv8 yükle ─────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    model_path = Path("edge/models/yolov8n.pt")
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print("[MODEL] yolov8n.pt indiriliyor...")
    model = YOLO(str(model_path))
    MODEL_OK = True
    print("[MODEL] YOLOv8 yüklendi ✓")
except Exception as e:
    MODEL_OK = False
    model = None
    print(f"[MODEL] YOLOv8 yok → mock tespit kullanılacak ({e})")

# ── Servo fonksiyonları ──────────────────────────────────────────────────────
_pan_deg  = 90.0
_tilt_deg = 90.0
_prev_pan_err  = 0.0
_prev_tilt_err = 0.0
_last_t = time.perf_counter()

KP = 0.4
KD = 0.05
DEADZONE = 0.03


def set_servo(pan, tilt):
    pan  = max(0.0,  min(180.0, pan))
    tilt = max(45.0, min(135.0, tilt))
    if SERVO_OK:
        _kit.servo[0].angle = pan
        _kit.servo[1].angle = tilt


def track_bbox(bbox):
    """bbox merkezine göre pan/tilt hesapla ve servo'ya gönder."""
    global _pan_deg, _tilt_deg, _prev_pan_err, _prev_tilt_err, _last_t

    now = time.perf_counter()
    dt  = max(now - _last_t, 0.001)
    _last_t = now

    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    pan_err  = cx - 0.5
    tilt_err = cy - 0.5

    if abs(pan_err) > DEADZONE:
        d_pan = KP * pan_err + KD * (pan_err - _prev_pan_err) / dt
        _pan_deg = max(0.0, min(180.0, _pan_deg + d_pan))
        _prev_pan_err = pan_err

    if abs(tilt_err) > DEADZONE:
        d_tilt = KP * tilt_err + KD * (tilt_err - _prev_tilt_err) / dt
        _tilt_deg = max(45.0, min(135.0, _tilt_deg + d_tilt))
        _prev_tilt_err = tilt_err

    set_servo(_pan_deg, _tilt_deg)
    return _pan_deg, _tilt_deg


def center_servo():
    global _pan_deg, _tilt_deg
    _pan_deg, _tilt_deg = 90.0, 90.0
    set_servo(90.0, 90.0)


# ── Video kaynağı ─────────────────────────────────────────────────────────────
def open_capture(args):
    if args.gstreamer:
        pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=640, height=360, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        label = "IMX219 CSI"
    elif args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        label = f"Webcam {args.camera}"
    else:
        cap = cv2.VideoCapture(str(args.video))
        label = str(args.video)

    if not cap.isOpened():
        print(f"[HATA] Kaynak açılamadı: {label}")
        sys.exit(1)

    print(f"[KAMERA] Açıldı: {label} ✓\n")
    return cap, label


# ── Tespit ───────────────────────────────────────────────────────────────────
TARGET_CLASSES = {"drone", "person", "vehicle", "car", "truck", "bus",
                  "motorcycle", "boat", "ship"}
CONF_THRESHOLD = 0.45

def detect(frame, frame_id):
    if MODEL_OK:
        results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls.item())].lower()
                if cls_name not in TARGET_CLASSES:
                    continue
                conf = float(box.conf.item())
                bbox = box.xyxyn.tolist()[0]
                detections.append((cls_name, conf, bbox))
        return detections
    else:
        # Mock: kayan drone
        t = frame_id % 200
        cx = 0.15 + t * 0.003
        return [("drone", 0.85, [cx - 0.05, 0.4, cx + 0.05, 0.6])]


# ── Görsel çizim ─────────────────────────────────────────────────────────────
def draw(frame, detections, pan_deg, tilt_deg):
    h, w = frame.shape[:2]
    for cls_name, conf, bbox in detections:
        x1 = int(bbox[0] * w); y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w); y2 = int(bbox[3] * h)
        color = (0, 0, 255) if cls_name == "drone" else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Servo durumu
    servo_txt = f"PAN:{pan_deg:.1f}  TILT:{tilt_deg:.1f}"
    cv2.putText(frame, servo_txt, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    return frame


# ── Ana döngü ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera",     type=int,        help="Webcam index (ör. 0)")
    src.add_argument("--video",      type=Path,
                     default=Path("data/sim_samples/drone_flyby.mp4"),
                     help="Video dosyası")
    src.add_argument("--gstreamer",  action="store_true", help="IMX219 CSI kamera")
    args = parser.parse_args()

    cap, label = open_capture(args)
    center_servo()

    print("=" * 55)
    print("  AEGIS Canlı Pipeline — Başladı")
    print(f"  Kaynak  : {label}")
    print(f"  Model   : {'YOLOv8' if MODEL_OK else 'Mock'}")
    print(f"  Servo   : {'PCA9685 AKTIF' if SERVO_OK else 'LOG MODU'}")
    print("  Durdurmak için: Ctrl+C veya 'q'")
    print("=" * 55 + "\n")

    frame_id   = 0
    pan_deg    = 90.0
    tilt_deg   = 90.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Video dosyası bittiyse başa sar
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            t0 = time.perf_counter()

            # ── ADIM 1: Tespit ──────────────────────────────────────────
            detections = detect(frame, frame_id)

            # ── ADIM 2: Servo takibi ────────────────────────────────────
            if detections:
                best = max(detections, key=lambda d: d[1])
                pan_deg, tilt_deg = track_bbox(best[2])
                print(
                    f"Frame {frame_id:4d} | "
                    f"{best[0]:8s} conf={best[1]:.2f} | "
                    f"bbox_cx={( best[2][0]+best[2][2])/2:.2f} | "
                    f"pan={pan_deg:6.1f}°  tilt={tilt_deg:6.1f}°  "
                    f"[{(time.perf_counter()-t0)*1000:.0f}ms]"
                )
            else:
                center_servo()

            # ── ADIM 3: Görüntü göster ──────────────────────────────────
            out = draw(frame.copy(), detections, pan_deg, tilt_deg)
            cv2.imshow("AEGIS — Gerçek Takip", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("\n[DURDURULDU] Ctrl+C")
    finally:
        center_servo()
        cap.release()
        cv2.destroyAllWindows()
        print("[BITTI] Servo merkeze döndü, kamera kapatıldı.")


if __name__ == "__main__":
    main()
